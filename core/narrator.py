"""
narrator.py — LLM-powered plain-English result narration (Step 4h)

Contract:
  IN : experiment_log (List[RoundResult])
       profile_json   (from profiler.run)
       llm            (BaseLLM instance)
  OUT: NarratorResult dataclass with:
       - executive_summary  : 2-3 sentence plain-English summary
       - key_factors        : bullet list of top features in plain English
       - what_improved      : what the AI found (if FE helped), else None
       - accuracy_statement : single sentence with accuracy framed for humans
       - next_steps         : 2-3 actionable suggestions

RULE: Only imports BaseLLM from core.llm.base — same isolation rule as fe_agent.
"""

from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from core.llm.base import BaseLLM, LLMError
from core.fe_agent import RoundResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class NarratorResult:
    executive_summary: str
    accuracy_statement: str
    key_factors: List[str]          # plain-English factor descriptions
    what_improved: Optional[str]    # None if no FE improvement
    next_steps: List[str]
    raw_response: str               # full LLM output for debugging


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_NARRATOR_SYSTEM = textwrap.dedent("""\
    You are an AI that explains machine learning results to non-technical business users.
    Your job is to translate numbers and feature names into clear, actionable language.

    RULES:
    1. Never use jargon: no "cross-validation", "SHAP", "cv_score", "delta", "AUC"
    2. Frame accuracy as confidence: "correctly predicts 9 out of 10 cases"
    3. Explain features in terms of what they mean, not their column names
    4. Be specific — use actual numbers from the data
    5. Be concise — executives read, not skim
    6. Return ONLY valid JSON matching the schema provided — no markdown, no explanation
""")


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_narrator_prompt(
    experiment_log: List[RoundResult],
    profile_json: Dict[str, Any],
) -> str:
    baseline = experiment_log[0]
    best_round = max(experiment_log, key=lambda r: r.cv_score)
    improved_rounds = [r for r in experiment_log[1:] if r.improved]

    task_type = profile_json.get("task_type", "classification")
    target_col = profile_json.get("target_col", "target")
    n_rows = profile_json.get("n_rows", "?")
    target_dist = profile_json.get("target", {}).get("distribution", {})

    # Format accuracy for humans
    accuracy_pct = round(best_round.cv_score * 100, 1)
    baseline_pct = round(baseline.cv_score * 100, 1)
    delta_pct = round((best_round.cv_score - baseline.cv_score) * 100, 2)

    # Top SHAP features from best round
    shap_features = best_round.shap_top[:8] if best_round.shap_top else []
    # Strip pipeline prefixes (num__, cat__) for readability
    clean_features = [f.replace("num__", "").replace("cat__", "") for f in shap_features]

    # Best FE code summary
    fe_summary = None
    if improved_rounds:
        best_fe = max(improved_rounds, key=lambda r: r.cv_score)
        if best_fe.fe_code:
            # Extract new column names from fe_code
            import re
            new_cols = re.findall(r"df\[['\"]([\w_]+)['\"]\]\s*=", best_fe.fe_code)
            fe_summary = {
                "round": best_fe.round_num,
                "improvement_pct": round((best_fe.cv_score - baseline.cv_score) * 100, 2),
                "new_features_created": new_cols[:6],
            }

    # Model that won
    winning_model = None
    if best_round.train_result and best_round.train_result.leaderboard:
        winning_model = best_round.train_result.leaderboard[0].get("model_name", "")

    data = {
        "task_type": task_type,
        "target_column": target_col,
        "dataset_size": n_rows,
        "target_distribution": target_dist,
        "accuracy_percent": accuracy_pct,
        "baseline_accuracy_percent": baseline_pct,
        "improvement_percent": delta_pct,
        "winning_model": winning_model,
        "top_predictive_features": clean_features,
        "ai_feature_engineering": fe_summary,
        "total_rounds_run": len(experiment_log) - 1,
    }

    schema = {
        "executive_summary": "2-3 sentences. What did we predict, how accurate, is it useful?",
        "accuracy_statement": "1 sentence. Frame accuracy as '9 out of 10 cases' style.",
        "key_factors": [
            "Plain-English description of factor 1 and why it matters",
            "Plain-English description of factor 2 and why it matters",
            "... up to 5 factors"
        ],
        "what_improved": "1-2 sentences about what the AI discovered, or null if no improvement",
        "next_steps": [
            "Actionable suggestion 1",
            "Actionable suggestion 2",
            "Actionable suggestion 3"
        ],
    }

    return (
        f"Here are the results of an automated ML analysis:\n\n"
        f"{json.dumps(data, indent=2)}\n\n"
        f"Return a JSON object with exactly this structure:\n"
        f"{json.dumps(schema, indent=2)}"
    )


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def _parse_narrator_response(raw: str) -> Optional[Dict[str, Any]]:
    """Extract and parse JSON from LLM response."""
    import re

    # Strip markdown fences
    clean = re.sub(r"```(?:json)?\s*\n?", "", raw)
    clean = re.sub(r"```\s*$", "", clean, flags=re.MULTILINE).strip()

    try:
        parsed = json.loads(clean)
        # Validate required keys
        required = {"executive_summary", "accuracy_statement", "key_factors",
                    "what_improved", "next_steps"}
        if required.issubset(parsed.keys()):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try to find JSON block within the text
    match = re.search(r"\{[\s\S]*\}", clean)
    if match:
        try:
            parsed = json.loads(match.group())
            required = {"executive_summary", "accuracy_statement"}
            if required.issubset(parsed.keys()):
                return parsed
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Main narrate() function
# ---------------------------------------------------------------------------

_FALLBACK_SUMMARY = (
    "The model was trained and evaluated successfully. "
    "See the accuracy score and feature importance chart for details."
)


def narrate(
    experiment_log: List[RoundResult],
    profile_json: Dict[str, Any],
    llm: BaseLLM,
) -> NarratorResult:
    """
    Generate plain-English narration of experiment results.

    Parameters
    ----------
    experiment_log : list[RoundResult]  — from fe_agent.run()
    profile_json   : dict               — from profiler.run()
    llm            : BaseLLM            — same instance used for FE

    Returns
    -------
    NarratorResult
        Falls back to minimal narration if LLM call fails — never raises.
    """
    if not experiment_log:
        return _fallback_result("No experiment results to narrate.")

    prompt = _build_narrator_prompt(experiment_log, profile_json)

    try:
        raw = llm.complete_with_system(system=_NARRATOR_SYSTEM, user=prompt)
    except LLMError as e:
        logger.warning("narrator: LLM call failed: %s", e)
        return _fallback_result(_FALLBACK_SUMMARY)

    parsed = _parse_narrator_response(raw)
    if parsed is None:
        logger.warning("narrator: failed to parse LLM response — using fallback")
        return _fallback_result(_FALLBACK_SUMMARY, raw_response=raw)

    best = max(experiment_log, key=lambda r: r.cv_score)
    accuracy_pct = round(best.cv_score * 100, 1)

    return NarratorResult(
        executive_summary=parsed.get("executive_summary", _FALLBACK_SUMMARY),
        accuracy_statement=parsed.get(
            "accuracy_statement",
            f"The model correctly predicts {accuracy_pct}% of cases.",
        ),
        key_factors=parsed.get("key_factors", []) or [],
        what_improved=parsed.get("what_improved") or None,
        next_steps=parsed.get("next_steps", []) or [],
        raw_response=raw,
    )


def _fallback_result(summary: str, raw_response: str = "") -> NarratorResult:
    return NarratorResult(
        executive_summary=summary,
        accuracy_statement="",
        key_factors=[],
        what_improved=None,
        next_steps=[],
        raw_response=raw_response,
    )
