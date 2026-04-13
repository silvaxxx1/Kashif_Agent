"""
api/server.py — FastAPI server for Kashif (Step 5)

Endpoints:
  GET  /health      — liveness check
  GET  /providers   — available LLM providers and notes
  POST /run         — upload CSV + form options → JSON output contract

Run with:
  uv run uvicorn api.server:app --reload --port 8000

The JSON output contract is identical to the CLI output so any client
that works with the CLI output works unchanged with the API.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Generator, Optional

# Ensure kashif_core/ is on sys.path so `core.*` and `cli.*` are importable
# regardless of how the server is launched (uvicorn entry point vs python -m).
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import yaml
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

app = FastAPI(
    title="Kashif",
    description="LLM-powered feature engineering for tabular ML",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(base: str) -> str:
    """Create a timestamped subdirectory under *base* for this run's artifacts."""
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_{uuid.uuid4().hex[:6]}"
    run_dir = os.path.join(base, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _load_config(path: Path = _DEFAULT_CONFIG_PATH) -> dict:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_program_md() -> str:
    path = _DEFAULT_CONFIG_PATH.parent / "program.md"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


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
    elif provider == "ollama":
        from core.llm.ollama import OllamaLLM
        base_url = llm_cfg.get("base_url", "http://localhost:11434/v1")
        return OllamaLLM(
            model=model or "llama3.2",
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    else:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown provider '{provider}'. Choose: groq | anthropic | ollama",
        )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness check."""
    return {"status": "ok", "version": "1.0.0"}


@app.get("/providers")
def providers():
    """List available LLM providers."""
    return {
        "providers": ["groq", "anthropic", "ollama"],
        "default": "groq",
        "notes": {
            "groq":      "Requires GROQ_API_KEY env var — fastest, recommended",
            "anthropic": "Requires ANTHROPIC_API_KEY env var",
            "ollama":    "No API key — requires Ollama running locally (ollama serve)",
        },
    }


@app.post("/run")
def run(
    file: UploadFile = File(..., description="CSV file to run the pipeline on"),
    target: str = Form(..., description="Target column name"),
    rounds: Optional[int] = Form(None, description="Max FE rounds (overrides config.yaml)"),
    provider: Optional[str] = Form(None, description="LLM provider: groq | anthropic | ollama"),
    no_agent: bool = Form(False, description="Skip LLM FE loop — static pipeline only"),
    output_dir: str = Form("./outputs", description="Directory for model artifacts and reports"),
):
    """
    Run the Kashif ML pipeline on an uploaded CSV file.

    Returns the JSON output contract identical to the CLI.
    Blocking — completes before returning. Allow a generous client timeout (600s+).
    """
    cfg = _load_config()

    # --- Load CSV ---
    tmp_path = None
    try:
        contents = file.file.read()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        df = pd.read_csv(tmp_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to read CSV: {exc}")
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    if target not in df.columns:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Target column '{target}' not found. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    # --- Isolated run directory ---
    run_dir = _make_run_dir(output_dir)

    # --- Profiler ---
    try:
        from core.profiler import run as profiler_run
        profile_json, _eda_path = profiler_run(
            df, target, output_dir=run_dir, save_eda=True
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Profiler error: {exc}")

    # --- Build configs ---
    resolved_provider = provider or cfg.get("llm", {}).get("provider", "groq")
    fe_cfg = dict(cfg.get("fe_agent", {}))
    if rounds is not None:
        fe_cfg["max_rounds"] = rounds
    train_cfg = cfg.get("training", None)

    # --- Run pipeline ---
    llm_instance = None
    try:
        if no_agent:
            from core.trainer import train as trainer_run
            from core.fe_agent import RoundResult
            result = trainer_run(
                df=df,
                target_col=target,
                task_type=profile_json.get("task_type"),
                config=train_cfg,
                save_model=True,
                output_dir=run_dir,
            )
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
            llm_instance = _resolve_llm(resolved_provider, cfg)
            program_md = _load_program_md()
            from core.fe_agent import run as agent_run
            log = agent_run(
                df=df,
                target_col=target,
                profile_json=profile_json,
                llm=llm_instance,
                program_md=program_md,
                config=fe_cfg,
                train_config=train_cfg,
                output_dir=run_dir,
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}")

    # --- Reporter ---
    report_path = None
    try:
        from core.reporter import run as reporter_run
        _report_md, report_path = reporter_run(
            log, profile_json, output_dir=run_dir, save_report=True
        )
    except Exception:
        pass

    # --- Build output ---
    from cli.main import _build_output
    output = _build_output(log, profile_json, report_path, run_dir)

    # --- Narration ---
    if not no_agent and llm_instance is not None:
        try:
            from core.narrator import narrate
            narration_result = narrate(log, profile_json, llm_instance)
            output["narration"] = {
                "executive_summary": narration_result.executive_summary,
                "accuracy_statement": narration_result.accuracy_statement,
                "key_factors": narration_result.key_factors,
                "what_improved": narration_result.what_improved,
                "next_steps": narration_result.next_steps,
            }
        except Exception:
            pass

    return output


@app.post("/run/stream")
def run_stream(
    file: UploadFile = File(...),
    target: str = Form(...),
    rounds: Optional[int] = Form(None),
    provider: Optional[str] = Form(None),
    no_agent: bool = Form(False),
    output_dir: str = Form("./outputs"),
):
    """
    Streaming version of POST /run.

    Yields NDJSON progress events, then the final result as the last line:
      {"stage": "profiling",  "progress": 0.15, "message": "Profiling data..."}
      {"stage": "training",   "progress": 0.30, "message": "Training baseline..."}
      ...
      {"stage": "complete",   "progress": 1.0,  "result": {...}}
      {"stage": "error",      "progress": 1.0,  "error":  "..."}
    """
    def _event(stage: str, progress: float, message: str, **extra) -> str:
        return json.dumps({"stage": stage, "progress": round(progress, 2),
                           "message": message, **extra}) + "\n"

    def generate() -> Generator[str, None, None]:
        cfg = _load_config()

        # ── Load CSV ──
        yield _event("loading", 0.05, "Loading CSV...")
        tmp_path = None
        try:
            contents = file.file.read()
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                tmp.write(contents)
                tmp_path = tmp.name
            df = pd.read_csv(tmp_path)
        except Exception as exc:
            yield _event("error", 1.0, str(exc), error=str(exc))
            return
        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        if target not in df.columns:
            msg = (f"Target column '{target}' not found. "
                   f"Available: {list(df.columns)}")
            yield _event("error", 1.0, msg, error=msg)
            return

        run_dir = _make_run_dir(output_dir)

        # ── Profile ──
        yield _event("profiling", 0.12, f"Profiling {df.shape[0]:,} rows × {df.shape[1]} cols...")
        try:
            from core.profiler import run as profiler_run
            profile_json, _ = profiler_run(df, target, output_dir=run_dir, save_eda=True)
        except Exception as exc:
            yield _event("error", 1.0, str(exc), error=str(exc))
            return
        yield _event("profiling", 0.18,
                     f"Task detected: {profile_json.get('task_type')} "
                     f"(confidence {profile_json.get('task_confidence', 0):.0%})")

        resolved_provider = provider or cfg.get("llm", {}).get("provider", "groq")
        fe_cfg = dict(cfg.get("fe_agent", {}))
        if rounds is not None:
            fe_cfg["max_rounds"] = rounds
        max_rounds = fe_cfg.get("max_rounds", 10)
        train_cfg = cfg.get("training", None)

        # ── Pipeline ──
        log = None
        llm_instance = None
        run_error = None

        if no_agent:
            yield _event("training", 0.25, "Training baseline models (5-fold CV)...")
            try:
                from core.trainer import train as trainer_run
                from core.fe_agent import RoundResult
                result = trainer_run(
                    df=df, target_col=target,
                    task_type=profile_json.get("task_type"),
                    config=train_cfg, save_model=True, output_dir=run_dir,
                )
                log = [RoundResult(
                    round_num=0, fe_code=None,
                    cv_score=result.cv_score, cv_loss=result.cv_loss,
                    delta=0.0, improved=False, executor_error=None,
                    shap_top=[], shap_dead=[], train_result=result,
                )]
                yield _event("training", 0.82,
                             f"Baseline complete — CV score: {result.cv_score:.4f}")
            except Exception as exc:
                yield _event("error", 1.0, str(exc), error=str(exc))
                return
        else:
            # Resolve LLM
            yield _event("setup", 0.22, f"Initialising {resolved_provider} LLM...")
            try:
                llm_instance = _resolve_llm(resolved_provider, cfg)
            except HTTPException as exc:
                yield _event("error", 1.0, exc.detail, error=exc.detail)
                return

            # Run fe_agent in a background thread; yield estimated progress per second
            yield _event("baseline", 0.25, "Training baseline model...")
            result_holder: dict = {}

            def _run_agent():
                try:
                    program_md = _load_program_md()
                    from core.fe_agent import run as agent_run
                    result_holder["log"] = agent_run(
                        df=df, target_col=target, profile_json=profile_json,
                        llm=llm_instance, program_md=program_md,
                        config=fe_cfg, train_config=train_cfg, output_dir=run_dir,
                    )
                except Exception as exc:
                    result_holder["error"] = str(exc)

            t = threading.Thread(target=_run_agent, daemon=True)
            t.start()

            # Yield one progress tick per second while the agent runs
            elapsed = 0
            while t.is_alive():
                time.sleep(1)
                elapsed += 1
                est_round = min(elapsed / 30, max_rounds)  # ~30s per round estimate
                progress = 0.25 + 0.55 * (est_round / max(max_rounds, 1))
                round_label = f"~round {int(est_round) + 1}/{max_rounds}"
                yield _event("fe_loop", progress,
                             f"LLM feature engineering loop running ({round_label})...")

            t.join()

            if "error" in result_holder:
                yield _event("error", 1.0, result_holder["error"],
                             error=result_holder["error"])
                return

            log = result_holder.get("log", [])
            best = max((r.cv_score for r in log), default=0)
            yield _event("fe_loop", 0.82,
                         f"FE loop complete — best CV: {best:.4f} over {len(log)-1} rounds")

        # ── Reporter ──
        yield _event("report", 0.87, "Generating markdown report...")
        report_path = None
        try:
            from core.reporter import run as reporter_run
            _, report_path = reporter_run(log, profile_json,
                                          output_dir=run_dir, save_report=True)
        except Exception:
            pass

        # ── Narration ──
        if not no_agent and llm_instance is not None:
            yield _event("narrating", 0.93, "Writing plain-English summary...")
            output_extra = {}
            try:
                from core.narrator import narrate
                narration_result = narrate(log, profile_json, llm_instance)
                output_extra["narration"] = {
                    "executive_summary": narration_result.executive_summary,
                    "accuracy_statement": narration_result.accuracy_statement,
                    "key_factors": narration_result.key_factors,
                    "what_improved": narration_result.what_improved,
                    "next_steps": narration_result.next_steps,
                }
            except Exception:
                pass
        else:
            output_extra = {}

        # ── Final result ──
        from cli.main import _build_output
        output = _build_output(log, profile_json, report_path, run_dir)
        output.update(output_extra)

        yield _event("complete", 1.0, "Pipeline complete!", result=output)

    return StreamingResponse(generate(), media_type="application/x-ndjson")
