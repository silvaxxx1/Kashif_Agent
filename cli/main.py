"""
cli/main.py — Typer CLI for Kashif (Step 4g)

Usage:
    uv run python -m cli.main run --csv data.csv --target survived
    uv run python -m cli.main run --csv data.csv --target price --rounds 4
    uv run python -m cli.main run --csv data.csv --target churn --no-agent
    uv run python -m cli.main run --csv data.csv --target churn --llm anthropic
    uv run python -m cli.main run --csv data.csv --target churn --no-eda

This module:
  1. Reads config.yaml
  2. Loads the CSV
  3. Runs profiler
  4. Resolves LLM provider from config (or --llm override)
  5. Runs fe_agent loop (or skips with --no-agent)
  6. Runs reporter
  7. Prints JSON output contract to stdout
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
import yaml

app = typer.Typer(
    name="kashif",
    help="Kashif — LLM-powered feature engineering for tabular ML",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def _load_program_md(program_path: Path) -> str:
    if not program_path.exists():
        return ""
    return program_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# LLM resolver
# ---------------------------------------------------------------------------

def _resolve_llm(provider: str, cfg: dict):
    """Instantiate the correct BaseLLM subclass from config."""
    llm_cfg = cfg.get("llm", {})
    model = llm_cfg.get("model")
    temperature = float(llm_cfg.get("temperature", 0.2))
    max_tokens = int(llm_cfg.get("max_tokens", 4096))
    api_key_env = llm_cfg.get("api_key_env")

    if provider == "groq":
        from core.llm.groq import GroqLLM
        return GroqLLM(
            model=model or "llama-3.3-70b-versatile",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env or "GROQ_API_KEY",
        )
    elif provider == "anthropic":
        from core.llm.anthropic import AnthropicLLM
        return AnthropicLLM(
            model=model or "claude-sonnet-4-6",
            temperature=temperature,
            max_tokens=max_tokens,
            api_key_env=api_key_env or "ANTHROPIC_API_KEY",
        )
    else:
        typer.echo(f"[kashif] Unknown provider '{provider}'. Choose: groq | anthropic", err=True)
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Output contract builder
# ---------------------------------------------------------------------------

def _sanitize(value):
    """Replace float nan/inf with None so json.dumps produces valid JSON."""
    import math
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _build_output(
    log,
    profile_json: dict,
    report_path: Optional[str],
    output_dir: str,
) -> dict:
    """Build the JSON output contract from the experiment log."""
    from core.reporter import _baseline, _best_round

    baseline = _baseline(log)
    best = _best_round(log)

    baseline_score = baseline.cv_score if baseline else 0.0
    best_score = best.cv_score if best else baseline_score
    best_round_num = best.round_num if best else 0
    delta = best_score - baseline_score

    model_path = None
    if best and best.train_result:
        model_path = getattr(best.train_result, "model_path", None)

    top_features = (best.shap_top[:10] if best else []) or []
    dead_features = (best.shap_dead[:10] if best else []) or []

    rounds_list = []
    for r in log:
        if r.round_num == 0:
            continue
        rounds_list.append({
            "round": r.round_num,
            "cv_score": _sanitize(round(r.cv_score, 4)),
            "delta": _sanitize(round(r.delta, 4)),
            "improved": r.improved,
            "fe_code": r.fe_code or "",
            "top_features": r.shap_top[:5],
            "executor_error": r.executor_error or "",
        })

    return {
        "status": "complete" if best_round_num > 0 or baseline_score > 0 else "failed",
        "best_round": best_round_num,
        "cv_score": _sanitize(round(best_score, 4)),
        "baseline_score": _sanitize(round(baseline_score, 4)),
        "delta": _sanitize(round(delta, 4)),
        "model_path": model_path,
        "top_features": top_features,
        "dead_features": dead_features,
        "rounds": rounds_list,
        "report_path": report_path,
        "output_dir": output_dir,
    }


# ---------------------------------------------------------------------------
# run command
# ---------------------------------------------------------------------------

@app.command()
def run(
    csv: Path = typer.Option(..., help="Path to input CSV file"),
    target: str = typer.Option(..., help="Name of the target column"),
    rounds: Optional[int] = typer.Option(None, help="Max FE rounds (overrides config.yaml)"),
    no_agent: bool = typer.Option(False, "--no-agent", help="Skip LLM FE loop — static pipeline only"),
    llm: Optional[str] = typer.Option(None, help="LLM provider override: groq | anthropic"),
    output_dir: str = typer.Option("./outputs", help="Directory for model artifacts and reports"),
    no_eda: bool = typer.Option(False, "--no-eda", help="Skip EDA HTML report generation"),
    config_path: Path = typer.Option(_DEFAULT_CONFIG_PATH, "--config", help="Path to config.yaml"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed logs"),
):
    """
    Run the Kashif ML pipeline on a CSV file.

    Produces a JSON output contract on stdout and saves artifacts to --output-dir.
    """
    # Logging
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")

    # Load config
    cfg = _load_config(config_path)
    program_md_path = config_path.parent / "program.md"
    program_md = _load_program_md(program_md_path)

    # Validate CSV
    if not csv.exists():
        typer.echo(f"[kashif] CSV not found: {csv}", err=True)
        raise typer.Exit(code=1)

    # Load data
    typer.echo(f"[kashif] Loading {csv} ...", err=True)
    try:
        import pandas as pd
        df = pd.read_csv(csv)
    except Exception as e:
        typer.echo(f"[kashif] Failed to load CSV: {e}", err=True)
        raise typer.Exit(code=1)

    if target not in df.columns:
        typer.echo(f"[kashif] Target column '{target}' not found in CSV. "
                   f"Available columns: {list(df.columns)}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"[kashif] Dataset: {df.shape[0]} rows × {df.shape[1]} cols  target={target}", err=True)

    # Profiler
    typer.echo("[kashif] Profiling data ...", err=True)
    from core.profiler import run as profiler_run
    profile_json, eda_path = profiler_run(
        df, target,
        output_dir=output_dir,
        save_eda=not no_eda,
    )
    typer.echo(
        f"[kashif] Task: {profile_json['task_type']} (confidence {profile_json['task_confidence']:.2f})",
        err=True,
    )
    if eda_path:
        typer.echo(f"[kashif] EDA report: {eda_path}", err=True)

    # Resolve provider
    provider = llm or cfg.get("llm", {}).get("provider", "groq")

    # Build fe_agent config
    fe_cfg = dict(cfg.get("fe_agent", {}))
    if rounds is not None:
        fe_cfg["max_rounds"] = rounds

    # Training config
    train_cfg = cfg.get("training", None)

    # Run pipeline
    if no_agent:
        typer.echo("[kashif] Running static pipeline (--no-agent) ...", err=True)
        from core.trainer import train as trainer_run
        result = trainer_run(
            df=df,
            target_col=target,
            task_type=profile_json.get("task_type"),
            config=train_cfg,
            save_model=True,
            output_dir=output_dir,
        )
        # Wrap single result into a minimal experiment log
        from core.fe_agent import RoundResult
        log = [RoundResult(
            round_num=0,
            fe_code=None,
            cv_score=result.cv_score,
            cv_loss=result.cv_loss,
            delta=0.0,
            improved=False,
            executor_error=None,
            shap_top=[],
            shap_dead=[],
            train_result=result,
        )]
    else:
        typer.echo(f"[kashif] LLM provider: {provider}", err=True)
        llm_instance = _resolve_llm(provider, cfg)

        typer.echo(
            f"[kashif] Running FE loop (max {fe_cfg.get('max_rounds', 10)} rounds) ...",
            err=True,
        )
        from core.fe_agent import run as agent_run
        log = agent_run(
            df=df,
            target_col=target,
            profile_json=profile_json,
            llm=llm_instance,
            program_md=program_md,
            config=fe_cfg,
            train_config=train_cfg,
            output_dir=output_dir,
        )

    # Report
    typer.echo("[kashif] Generating report ...", err=True)
    from core.reporter import run as reporter_run
    report_md, report_path = reporter_run(
        log, profile_json, output_dir=output_dir, save_report=True
    )
    if report_path:
        typer.echo(f"[kashif] Report: {report_path}", err=True)

    # Narration (skip if --no-agent and no LLM instance available)
    narration = None
    if not no_agent:
        typer.echo("[kashif] Generating plain-English summary ...", err=True)
        try:
            from core.narrator import narrate
            narration_result = narrate(log, profile_json, llm_instance)
            narration = {
                "executive_summary": narration_result.executive_summary,
                "accuracy_statement": narration_result.accuracy_statement,
                "key_factors": narration_result.key_factors,
                "what_improved": narration_result.what_improved,
                "next_steps": narration_result.next_steps,
            }
        except Exception as e:
            typer.echo(f"[kashif] Narration skipped: {e}", err=True)

    # Build and print JSON output
    output = _build_output(log, profile_json, report_path, output_dir)
    if narration:
        output["narration"] = narration
    typer.echo(json.dumps(output, indent=2))


# ---------------------------------------------------------------------------
# info command — show config without running
# ---------------------------------------------------------------------------

@app.command()
def info(
    config_path: Path = typer.Option(_DEFAULT_CONFIG_PATH, "--config"),
):
    """Show current config.yaml settings."""
    cfg = _load_config(config_path)
    typer.echo(yaml.dump(cfg, default_flow_style=False))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
