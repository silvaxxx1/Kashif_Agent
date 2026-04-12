"""
reporter.py — Markdown report from experiment log (Step 4f)

Contract:
  IN : experiment_log  — list of RoundResult from fe_agent.run()
       profile_json    — from profiler.run() (optional, enriches the report)
       output_dir      — where to write report.md (default: ./outputs)
  OUT: report_md str   — full markdown report
       report_path     — path of written file (or None if save=False)

Public API:
  report(experiment_log, profile_json=None) -> str
  save(report_md, output_dir)               -> str (path)
  run(experiment_log, profile_json, output_dir, save) -> (report_md, path|None)
"""

from __future__ import annotations

import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.fe_agent import RoundResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_round(log: List[RoundResult]) -> Optional[RoundResult]:
    """Return the round with the lowest cv_loss (highest score)."""
    completed = [r for r in log if r.train_result is not None or r.round_num == 0]
    if not completed:
        return None
    return min(completed, key=lambda r: r.cv_loss)


def _baseline(log: List[RoundResult]) -> Optional[RoundResult]:
    """Return round 0 (baseline)."""
    for r in log:
        if r.round_num == 0:
            return r
    return None


def _score_bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for a 0–1 score."""
    filled = max(0, min(width, round(score * width)))
    return f"[{'█' * filled}{'░' * (width - filled)}] {score:.4f}"


def _delta_arrow(delta: float) -> str:
    if delta > 0.001:
        return f"▲ +{delta:.4f}"
    if delta < -0.001:
        return f"▼ {delta:.4f}"
    return f"  {delta:+.4f}"


def _indent_code(code: str, indent: int = 4) -> str:
    """Indent a code block for markdown fencing."""
    return textwrap.indent(code.strip(), " " * indent)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _header(log: List[RoundResult], profile_json: Optional[Dict[str, Any]]) -> str:
    baseline = _baseline(log)
    best = _best_round(log)

    target_col = (profile_json or {}).get("target_col", "target")
    task_type = (profile_json or {}).get("task_type", "unknown")
    n_rows = (profile_json or {}).get("n_rows", "?")
    n_cols = (profile_json or {}).get("n_cols", "?")

    baseline_score = baseline.cv_score if baseline else 0.0
    best_score = best.cv_score if best else 0.0
    best_round = best.round_num if best else 0
    total_delta = best_score - baseline_score
    total_rounds = len([r for r in log if r.round_num > 0])

    lines = [
        "# Kashif — Experiment Report",
        "",
        f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "---",
        "",
        "## Summary",
        "",
        f"| | |",
        f"|---|---|",
        f"| Dataset | {n_rows} rows × {n_cols} features |",
        f"| Target | `{target_col}` ({task_type}) |",
        f"| Baseline CV score | {baseline_score:.4f} |",
        f"| Best CV score | {best_score:.4f} (round {best_round}) |",
        f"| Total improvement | {total_delta:+.4f} ({total_delta / max(baseline_score, 1e-9) * 100:.1f}%) |",
        f"| FE rounds run | {total_rounds} |",
        "",
    ]
    return "\n".join(lines)


def _score_progression(log: List[RoundResult]) -> str:
    best_so_far = float("inf")
    lines = [
        "## Score Progression",
        "",
        "| Round | CV Score | Delta | Bar | Status |",
        "|---|---|---|---|---|",
    ]
    for r in log:
        if r.cv_loss < best_so_far:
            best_so_far = r.cv_loss
            status = "**best**" if r.round_num > 0 else "baseline"
        else:
            status = "no improvement" if r.round_num > 0 else "baseline"

        label = "baseline" if r.round_num == 0 else f"round {r.round_num}"
        bar = _score_bar(r.cv_score)
        arrow = "—" if r.round_num == 0 else _delta_arrow(r.delta)
        lines.append(f"| {label} | {r.cv_score:.4f} | {arrow} | `{bar}` | {status} |")

    lines.append("")
    return "\n".join(lines)


def _feature_analysis(log: List[RoundResult]) -> str:
    best = _best_round(log)
    if best is None or not best.shap_top:
        return ""

    lines = [
        "## Feature Analysis",
        "",
        "### Top Features (best round SHAP)",
        "",
    ]
    for i, feat in enumerate(best.shap_top[:10], 1):
        lines.append(f"{i}. `{feat}`")
    lines.append("")

    if best.shap_dead:
        lines += [
            "### Dead Features (near-zero SHAP — consider dropping)",
            "",
        ]
        for feat in best.shap_dead[:10]:
            lines.append(f"- `{feat}`")
        lines.append("")

    return "\n".join(lines)


def _round_details(log: List[RoundResult]) -> str:
    fe_rounds = [r for r in log if r.round_num > 0]
    if not fe_rounds:
        return ""

    lines = ["## Round Details", ""]

    for r in fe_rounds:
        icon = "✓" if r.improved else "✗"
        lines.append(f"### {icon} Round {r.round_num} — CV {r.cv_score:.4f} ({_delta_arrow(r.delta)})")
        lines.append("")

        if r.executor_error:
            lines.append(f"**Executor error:**")
            lines.append("```")
            lines.append(r.executor_error[:500])
            lines.append("```")
            lines.append("")

        if r.fe_code:
            lines.append("**Feature engineering code:**")
            lines.append("```python")
            lines.append(r.fe_code.strip())
            lines.append("```")
            lines.append("")

        if r.shap_top:
            lines.append(f"**Top features:** {', '.join(f'`{f}`' for f in r.shap_top[:5])}")
            lines.append("")

    return "\n".join(lines)


def _best_code_section(log: List[RoundResult]) -> str:
    best = _best_round(log)
    if best is None or best.fe_code is None:
        return ""

    lines = [
        "## Best Feature Engineering Code",
        "",
        f"> Round {best.round_num} — CV score {best.cv_score:.4f}",
        "",
        "```python",
        best.fe_code.strip(),
        "```",
        "",
    ]
    return "\n".join(lines)


def _recommendation(log: List[RoundResult], profile_json: Optional[Dict[str, Any]]) -> str:
    baseline = _baseline(log)
    best = _best_round(log)

    if baseline is None or best is None:
        return ""

    task_type = (profile_json or {}).get("task_type", "unknown")
    metric_name = "accuracy" if task_type == "classification" else "RMSE"
    total_delta = best.cv_score - baseline.cv_score
    pct = total_delta / max(baseline.cv_score, 1e-9) * 100

    lines = ["## Recommendation", ""]

    if best.round_num == 0 or total_delta < 0.001:
        lines += [
            "The LLM feature engineering loop did not improve upon the static baseline.",
            "",
            "**Recommendation:** Use the static pipeline (round 0) for production.",
            "Consider:",
            "- Adding domain hints to `program.md`",
            "- Increasing `max_rounds` in `config.yaml`",
            "- Switching to a more capable LLM provider",
            "",
        ]
    else:
        lines += [
            f"LLM feature engineering improved {metric_name} by **{total_delta:+.4f}** "
            f"({pct:.1f}%) over the static baseline.",
            "",
            f"**Recommendation:** Use the Round {best.round_num} feature engineering code in production.",
            "",
            "To apply this FE transform in your own pipeline:",
            "",
            "```python",
            "from core.fe_agent import FETransformer",
            "from core.trainer import train",
            "",
            "fe_step = FETransformer(fe_code=BEST_FE_CODE)  # paste code from Round Details above",
            "result = train(df, target_col, fe_step=fe_step)",
            "```",
            "",
        ]

    best_model_name = ""
    if best.train_result and hasattr(best.train_result, "best_model_name"):
        best_model_name = best.train_result.best_model_name
    if best_model_name:
        lines.append(f"Best model: **{best_model_name}**")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def report(
    experiment_log: List[RoundResult],
    profile_json: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Build a full markdown report from the experiment log.

    Parameters
    ----------
    experiment_log : list[RoundResult]
        Output of fe_agent.run().
    profile_json : dict, optional
        Output of profiler.run(). Enriches the report with dataset metadata.

    Returns
    -------
    str — full markdown document
    """
    if not experiment_log:
        return "# Kashif Report\n\nNo experiment data available.\n"

    sections = [
        _header(experiment_log, profile_json),
        _score_progression(experiment_log),
        _feature_analysis(experiment_log),
        _best_code_section(experiment_log),
        _round_details(experiment_log),
        _recommendation(experiment_log, profile_json),
    ]

    return "\n".join(s for s in sections if s)


def save(report_md: str, output_dir: str = "./outputs") -> str:
    """
    Write report_md to output_dir/report.md.

    Returns the path of the written file.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.md"
    path.write_text(report_md, encoding="utf-8")
    return str(path)


def run(
    experiment_log: List[RoundResult],
    profile_json: Optional[Dict[str, Any]] = None,
    output_dir: str = "./outputs",
    save_report: bool = True,
) -> Tuple[str, Optional[str]]:
    """
    Build and optionally save the markdown report.

    Returns
    -------
    (report_md, path)  — path is None if save_report=False
    """
    report_md = report(experiment_log, profile_json)
    path = save(report_md, output_dir) if save_report else None
    return report_md, path
