"""
fe_agent.py — LLM-powered feature engineering loop (Step 4e)

Contract:
  IN : profile_json (from profiler.run)
       program_md   (user domain directive text, from program.md)
       llm          (BaseLLM instance — resolved at runtime, never imported here)
       df           (raw DataFrame)
       target_col   (str)
  OUT: experiment_log (list of round dicts)
       best_fe_code (str | None — None if FE never improved baseline)

The agent runs for up to max_rounds:
  - Round 0: static baseline (no FE)
  - Round N: LLM writes engineer_features(df) -> df
             executor runs it
             trainer scores it (cross-validation)
             if delta > delta_threshold: keep, update best
             else: loop continues with failure context

Stopping conditions (any of):
  - delta < delta_threshold for delta_rounds consecutive rounds (default 3)
  - max_rounds reached
  - LLM returns an unparseable / empty response 3 times

RULE: This module only imports BaseLLM from core.llm.base.
      Never import GroqLLM, AnthropicLLM, or any provider SDK directly.
"""

from __future__ import annotations

import logging
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from core.executor import execute, summarise_error
from core.llm.base import BaseLLM, LLMError
from core.trainer import TrainResult, train

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults (overridden by config.yaml values passed from CLI)
# ---------------------------------------------------------------------------

_DEFAULT_MAX_ROUNDS: int = 10
_DEFAULT_DELTA_THRESHOLD: float = 0.005   # 0.5% improvement minimum
_DEFAULT_STALL_ROUNDS: int = 3            # stop after N consecutive non-improving rounds
_DEFAULT_MAX_EMPTY_RESPONSES: int = 3     # stop after N LLM parse failures


# ---------------------------------------------------------------------------
# FETransformer — wraps executor inside an sklearn pipeline step
# ---------------------------------------------------------------------------

class FETransformer(BaseEstimator, TransformerMixin):
    """
    sklearn transformer that wraps executor.execute().

    Inserted into the pipeline between cleaning and processing:
        Pipeline(cleaning → FETransformer → processing → model)

    Because it's inside the Pipeline, CrossValidator.run_cv() clones and
    re-fits the full chain on each CV fold — no leakage.

    If execution fails (executor returns error), transform() returns X unchanged
    so CV can still complete and score the round (it will just score poorly).
    """

    def __init__(self, fe_code: str) -> None:
        self.fe_code = fe_code

    def fit(self, X: pd.DataFrame, y=None) -> "FETransformer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        result, err = execute(self.fe_code, X)
        if err is not None:
            logger.warning("FETransformer.transform error (returning X unchanged): %s", err[:200])
            return X
        # Replace inf/-inf so downstream scalers don't choke
        result = result.replace([np.inf, -np.inf], np.nan)
        return result


# ---------------------------------------------------------------------------
# Round result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RoundResult:
    round_num: int
    fe_code: Optional[str]
    cv_score: float
    cv_loss: float
    delta: float              # vs previous best
    improved: bool
    executor_error: Optional[str]
    shap_top: List[str]       # top-5 feature names by SHAP
    shap_dead: List[str]      # features with near-zero SHAP
    train_result: Optional[TrainResult]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

_SYSTEM_PREAMBLE = textwrap.dedent("""\
    You are an expert ML feature engineer. Your job is to write a single Python
    function called engineer_features(df) that takes a pandas DataFrame and
    returns an enriched DataFrame with new features added.

    RULES:
    1. Function signature must be exactly: def engineer_features(df):
    2. Always start with df = df.copy()
    3. Return the full DataFrame — never drop the original columns
    4. Do NOT add the target column
    5. Do NOT import os, sys, subprocess, or open files
    6. Allowed imports: math, re, datetime, collections, itertools, functools, statistics
    7. pandas (pd) and numpy (np) are always available — do not import them
    8. Output only the Python function — no markdown, no explanation, no backtick fences
""")


def _build_prompt(
    profile_json: Dict[str, Any],
    program_md: str,
    experiment_log: List[RoundResult],
    surviving_cols: List[str],
) -> str:
    """
    Assemble the user-portion of the LLM prompt for round N.

    The system preamble (_SYSTEM_PREAMBLE) is sent separately via
    complete_with_system() so providers that support a system role
    (Groq, Anthropic) enforce the rules at the API level rather than
    relying on the model to follow instructions buried in a user message.
    """

    parts: List[str] = [""]

    # --- Dataset schema ---
    parts.append("=== DATASET SCHEMA (post-cleaning columns available to you) ===")
    target_col = profile_json.get("target_col", "target")
    task_type = profile_json.get("task_type", "unknown")
    n_rows = profile_json.get("n_rows", "?")
    parts.append(f"Task: {task_type}  |  Target: {target_col}  |  Rows: {n_rows}")
    parts.append(f"Available columns (excluding target): {', '.join(surviving_cols)}")
    parts.append("")

    # Column details
    col_stats = profile_json.get("columns", {})
    if col_stats:
        parts.append("Column statistics:")
        for col in surviving_cols:
            if col in col_stats:
                stats = col_stats[col]
                dtype = stats.get("dtype", "?")
                null_rate = stats.get("null_rate", 0.0)
                if dtype in ("float64", "int64", "float32", "int32"):
                    mean = stats.get("mean", "?")
                    std = stats.get("std", "?")
                    skew = stats.get("skew", "?")
                    parts.append(
                        f"  {col}: {dtype}, mean={mean:.3g}, std={std:.3g}, "
                        f"skew={skew:.3g}, null_rate={null_rate:.2%}"
                        if isinstance(mean, float) else
                        f"  {col}: {dtype}, null_rate={null_rate:.2%}"
                    )
                else:
                    top = stats.get("top_values", {})
                    top_str = ", ".join(f"{k}({v})" for k, v in list(top.items())[:3]) if top else "?"
                    parts.append(f"  {col}: {dtype}, top=[{top_str}], null_rate={null_rate:.2%}")
        parts.append("")

    # --- Pipeline OHE note ---
    # AutoDFColumnTransformer OHE-encodes all categorical columns downstream.
    # Tell the LLM which columns will be encoded (and what names they produce)
    # so it does NOT create duplicate one-hot columns manually.
    cat_cols = [
        col for col in surviving_cols
        if col in col_stats and col_stats[col].get("dtype") in ("object", "category", "string", "bool")
    ]
    if cat_cols:
        parts.append("=== PIPELINE NOTE (read carefully) ===")
        parts.append(
            "After your function returns, the pipeline automatically one-hot-encodes "
            "every categorical (object/string) column. Do NOT manually create binary "
            "or one-hot columns for the following — they will already exist:"
        )
        for col in cat_cols:
            top = col_stats[col].get("top_values", {})
            if top:
                encoded = [f"{col}_{v}" for v in list(top.keys())[:6]]
                parts.append(f"  '{col}' → {', '.join(encoded)}")
            else:
                parts.append(f"  '{col}' will be one-hot-encoded automatically")
        parts.append(
            "Instead, engineer higher-order features FROM the raw categorical values "
            "(e.g., groupby aggregates, interaction terms with numeric columns)."
        )
        parts.append("")

    # --- User directive ---
    if program_md.strip():
        parts.append("=== DOMAIN DIRECTIVE (from program.md) ===")
        parts.append(program_md.strip())
        parts.append("")

    # --- Previous rounds ---
    if experiment_log:
        parts.append("=== PREVIOUS ROUNDS ===")
        for r in experiment_log:
            status = "IMPROVED" if r.improved else "no improvement"
            parts.append(f"Round {r.round_num}: cv_score={r.cv_score:.4f}  delta={r.delta:+.4f}  [{status}]")
            if r.fe_code:
                # Show abbreviated code
                code_lines = r.fe_code.strip().splitlines()
                preview = "\n".join(code_lines[:15])
                if len(code_lines) > 15:
                    preview += f"\n    ... ({len(code_lines) - 15} more lines)"
                parts.append(f"  Code:\n{textwrap.indent(preview, '    ')}")
            if r.executor_error:
                parts.append(f"  Error: {summarise_error(r.executor_error, max_lines=5)}")
            if r.shap_top:
                parts.append(f"  Top features (SHAP): {', '.join(r.shap_top[:5])}")
            if r.shap_dead:
                parts.append(f"  Dead features (low SHAP): {', '.join(r.shap_dead[:10])}")
            parts.append("")

    # --- Task ---
    best_score = max((r.cv_score for r in experiment_log if r.improved), default=None)
    if best_score is not None:
        parts.append(f"=== TASK ===")
        parts.append(
            f"The best CV score so far is {best_score:.4f}. "
            f"Write an improved engineer_features(df) that scores higher. "
            f"Focus on new angles not tried before. Avoid dead features."
        )
    else:
        parts.append("=== TASK ===")
        parts.append(
            "Write engineer_features(df) that adds useful features for this dataset. "
            "Think about interactions, transforms, ratios, and domain-relevant derivations."
        )
    parts.append("")
    parts.append("Return ONLY the Python function — no markdown, no explanation.")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Code extraction (strip markdown fences + redundant imports)
# ---------------------------------------------------------------------------

_REDUNDANT_IMPORTS = re.compile(
    r"^\s*import\s+(pandas|numpy)\s*.*$|"
    r"^\s*from\s+(pandas|numpy)\s+import\s+.*$",
    re.MULTILINE,
)


def _extract_code(raw: str) -> Optional[str]:
    """
    Strip markdown fences and redundant pandas/numpy imports from LLM output.
    Returns None if no function definition found.
    """
    # Remove ```python ... ``` or ``` ... ``` fences
    raw = re.sub(r"```(?:python)?\s*\n?", "", raw)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)

    # Remove standalone pandas/numpy imports (pre-injected)
    raw = _REDUNDANT_IMPORTS.sub("", raw)

    # Verify we have the required function
    if "def engineer_features" not in raw:
        return None

    return raw.strip()


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def _extract_shap_features(
    train_result: TrainResult,
    top_n: int = 10,
    dead_threshold: float = 0.001,
) -> Tuple[List[str], List[str]]:
    """Extract top and dead feature names from TrainResult.shap_dict."""
    shap_dict = getattr(train_result, "shap_dict", None) or {}
    if not shap_dict:
        return [], []

    # shap_dict maps feature_name -> mean_abs_shap
    sorted_features = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    top = [name for name, _ in sorted_features[:top_n]]
    dead = [name for name, val in sorted_features if abs(val) < dead_threshold]
    return top, dead


# ---------------------------------------------------------------------------
# FEAgent — main class
# ---------------------------------------------------------------------------

class FEAgent:
    """
    LLM-powered feature engineering loop.

    Usage:
        agent = FEAgent(llm=groq_llm, config={"max_rounds": 5})
        log = agent.run(df, target_col, profile_json, program_md)
    """

    def __init__(
        self,
        llm: BaseLLM,
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        cfg = config or {}
        self.llm = llm
        self.max_rounds: int = cfg.get("max_rounds", _DEFAULT_MAX_ROUNDS)
        self.delta_threshold: float = cfg.get("delta_threshold", _DEFAULT_DELTA_THRESHOLD)
        self.stall_rounds: int = cfg.get("stall_rounds", _DEFAULT_STALL_ROUNDS)
        self.max_empty_responses: int = cfg.get("max_empty_responses", _DEFAULT_MAX_EMPTY_RESPONSES)

    # ------------------------------------------------------------------
    def run(
        self,
        df: pd.DataFrame,
        target_col: str,
        profile_json: Dict[str, Any],
        program_md: str = "",
        train_config: Optional[Dict[str, Any]] = None,
        output_dir: str = "./outputs",
        save_model: bool = True,
    ) -> List[RoundResult]:
        """
        Run the FE loop.

        Returns the full experiment_log (list of RoundResult, round 0 = baseline).
        """
        task_type = profile_json.get("task_type")
        experiment_log: List[RoundResult] = []
        best_loss = float("inf")
        stall_count = 0
        empty_count = 0

        # Compute surviving columns once (after cleaning, before FE)
        surviving_cols = self._surviving_cols(df, target_col, profile_json)
        logger.info("FEAgent: surviving cols = %s", surviving_cols)

        # ── Round 0: static baseline ─────────────────────────────────
        logger.info("FEAgent round 0: static baseline")
        r0 = self._run_train(
            df=df,
            target_col=target_col,
            task_type=task_type,
            fe_code=None,
            round_num=0,
            prev_loss=float("inf"),
            train_config=train_config,
            output_dir=output_dir,
            save_model=save_model,
        )
        experiment_log.append(r0)
        best_loss = r0.cv_loss
        logger.info("FEAgent round 0: cv_score=%.4f", r0.cv_score)

        # ── Rounds 1..max_rounds ─────────────────────────────────────
        for round_num in range(1, self.max_rounds + 1):
            logger.info("FEAgent round %d/%d", round_num, self.max_rounds)

            # Build prompt
            prompt = _build_prompt(
                profile_json=profile_json,
                program_md=program_md,
                experiment_log=experiment_log,
                surviving_cols=surviving_cols,
            )

            # Call LLM — Fix A: LLMError (API/key) is a hard stop; None = parse failure
            try:
                fe_code = self._call_llm(prompt)
            except LLMError as e:
                logger.error("FEAgent round %d: LLM API error — stopping loop: %s", round_num, e)
                break  # hard stop — no point retrying auth/rate-limit errors

            if fe_code is None:
                empty_count += 1
                logger.warning("FEAgent round %d: LLM output unparseable after retry (%d/%d)",
                               round_num, empty_count, self.max_empty_responses)
                if empty_count >= self.max_empty_responses:
                    logger.info("FEAgent: too many unparseable responses — stopping")
                    break
                # Log a no-op round so the LLM sees its own failure in next prompt
                experiment_log.append(RoundResult(
                    round_num=round_num,
                    fe_code=None,
                    cv_score=experiment_log[-1].cv_score,
                    cv_loss=experiment_log[-1].cv_loss,
                    delta=0.0,
                    improved=False,
                    executor_error="LLM output unparseable after retry",
                    shap_top=[],
                    shap_dead=[],
                    train_result=None,
                ))
                continue

            # Run train with FE
            rN = self._run_train(
                df=df,
                target_col=target_col,
                task_type=task_type,
                fe_code=fe_code,
                round_num=round_num,
                prev_loss=best_loss,
                train_config=train_config,
                output_dir=output_dir,
                save_model=save_model,
            )
            experiment_log.append(rN)

            if rN.improved:
                best_loss = rN.cv_loss
                stall_count = 0
                logger.info("FEAgent round %d: IMPROVED cv_score=%.4f (delta=%+.4f)",
                            round_num, rN.cv_score, rN.delta)
            else:
                stall_count += 1
                logger.info("FEAgent round %d: no improvement cv_score=%.4f (stall %d/%d)",
                            round_num, rN.cv_score, stall_count, self.stall_rounds)
                if stall_count >= self.stall_rounds:
                    logger.info("FEAgent: %d consecutive stall rounds — stopping", self.stall_rounds)
                    break

        # ── Save the best model (baseline or best FE round) ──────────
        if save_model:
            best_fe_round = max(
                (r for r in experiment_log if r.improved and r.fe_code is not None),
                key=lambda r: r.cv_score,
                default=None,
            )
            if best_fe_round is not None:
                # An FE round improved — re-fit and save that model
                logger.info("FEAgent: saving best FE model from round %d", best_fe_round.round_num)
                best_fe_step = FETransformer(best_fe_round.fe_code)
                final_result = train(
                    df=df,
                    target_col=target_col,
                    task_type=task_type,
                    fe_step=best_fe_step,
                    config=train_config,
                    save_model=True,
                    output_dir=output_dir,
                    compute_shap=True,
                )
                best_fe_round.train_result = final_result
            else:
                # Baseline was best — save baseline model
                logger.info("FEAgent: saving baseline model (no FE improvement)")
                baseline_result = train(
                    df=df,
                    target_col=target_col,
                    task_type=task_type,
                    fe_step=None,
                    config=train_config,
                    save_model=True,
                    output_dir=output_dir,
                    compute_shap=True,
                )
                experiment_log[0].train_result = baseline_result

        return experiment_log

    # ------------------------------------------------------------------
    def _call_llm(self, user_prompt: str) -> Optional[str]:
        """
        Call LLM via system/user split and extract fe_code.

        Fix A: LLMError (API/key failures) propagates to the caller — the loop
               catches it and stops immediately. Only parse failures return None.

        Fix B: On first parse failure, retry once with a strict structural reminder
               before counting the round as empty.

        Fix C: Rules are sent as the system message; data+task as the user message.
               Providers that support a native system role enforce it at the API
               level, which is far more reliable than embedding rules in the user
               turn for chat-tuned models.

        Raises
        ------
        LLMError
            Re-raised from the provider on API/auth/rate-limit failure.
            Caller must catch this and stop the loop — do not retry API errors.
        """
        # First attempt — system/user split (Fix C)
        raw = self.llm.complete_with_system(system=_SYSTEM_PREAMBLE, user=user_prompt)
        code = _extract_code(raw)

        if code is not None:
            return code

        # Parse failed — one retry with an explicit structural reminder (Fix B)
        logger.warning("FEAgent: parse failed on first attempt — retrying with strict reminder")
        retry_suffix = (
            "\n\nYour previous response did not contain a valid Python function. "
            "Return ONLY the following structure — no explanation, no markdown:\n\n"
            "def engineer_features(df):\n"
            "    df = df.copy()\n"
            "    # your new features here\n"
            "    return df"
        )
        raw = self.llm.complete_with_system(
            system=_SYSTEM_PREAMBLE,
            user=user_prompt + retry_suffix,
        )
        code = _extract_code(raw)
        if code is None:
            logger.warning("FEAgent: parse failed after retry — counting as empty response")
        return code

    # ------------------------------------------------------------------
    def _run_train(
        self,
        df: pd.DataFrame,
        target_col: str,
        task_type: Optional[str],
        fe_code: Optional[str],
        round_num: int,
        prev_loss: float,
        train_config: Optional[Dict[str, Any]],
        output_dir: str,
        save_model: bool,
    ) -> RoundResult:
        """Run trainer for one round. fe_code=None means static baseline."""
        fe_step = FETransformer(fe_code) if fe_code is not None else None
        executor_error: Optional[str] = None

        # Pre-validate code before handing to trainer
        if fe_code is not None:
            _, err = execute(fe_code, df.drop(columns=[target_col]).head(5))
            if err is not None:
                executor_error = err
                logger.warning("FEAgent round %d: executor pre-check failed: %s", round_num, err[:200])
                # Still let trainer run — FETransformer will fall back to X unchanged
                # This means the round will essentially be re-running baseline

        result: TrainResult = train(
            df=df,
            target_col=target_col,
            task_type=task_type,
            fe_step=fe_step,
            config=train_config,
            save_model=False,  # explicit final save after loop selects best round
            output_dir=output_dir,
            compute_shap=True,  # SHAP every round — feeds LLM reflection
        )

        delta = prev_loss - result.cv_loss   # positive = improvement (lower loss is better)
        improved = delta >= self.delta_threshold and result.status == "complete"

        shap_top, shap_dead = _extract_shap_features(result)

        return RoundResult(
            round_num=round_num,
            fe_code=fe_code,
            cv_score=result.cv_score,
            cv_loss=result.cv_loss,
            delta=delta,
            improved=improved,
            executor_error=executor_error,
            shap_top=shap_top,
            shap_dead=shap_dead,
            train_result=result,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _surviving_cols(
        df: pd.DataFrame,
        target_col: str,
        profile_json: Dict[str, Any],
    ) -> List[str]:
        """
        Return feature column names that are likely to survive cleaning.

        Uses profile_json warnings to exclude high-cardinality columns,
        and null_rate to exclude columns that would be dropped by UniversalDropper.
        Falls back to all non-target columns if profile data is unavailable.
        """
        warnings_text = " ".join(profile_json.get("warnings", []))
        col_stats = profile_json.get("columns", {})
        excluded: set = set()

        for col, stats in col_stats.items():
            # Skip columns flagged as high cardinality
            if col in warnings_text and "high cardinality" in warnings_text:
                # More precise: check if this specific col appears in a warning
                if any(col in w and "cardinality" in w.lower() for w in profile_json.get("warnings", [])):
                    excluded.add(col)
            # Skip columns with >50% nulls (UniversalDropper threshold)
            null_rate = stats.get("null_rate", 0.0)
            if null_rate > 0.5:
                excluded.add(col)

        all_feature_cols = [c for c in df.columns if c != target_col]
        surviving = [c for c in all_feature_cols if c not in excluded]

        # Fallback: never return empty list
        if not surviving:
            surviving = all_feature_cols

        return surviving


# ---------------------------------------------------------------------------
# Convenience function — matches the module-level pattern of profiler/trainer
# ---------------------------------------------------------------------------

def run(
    df: pd.DataFrame,
    target_col: str,
    profile_json: Dict[str, Any],
    llm: BaseLLM,
    program_md: str = "",
    config: Optional[Dict[str, Any]] = None,
    train_config: Optional[Dict[str, Any]] = None,
    output_dir: str = "./outputs",
    save_model: bool = True,
) -> List[RoundResult]:
    """
    Top-level entry point for the FE agent loop.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    profile_json : dict  — from profiler.run()
    llm : BaseLLM        — resolved by cli/main.py from config.yaml
    program_md : str     — contents of kashif_core/program.md
    config : dict        — fe_agent section of config.yaml
    train_config : dict  — training section of config.yaml
    output_dir : str
    save_model : bool

    Returns
    -------
    list[RoundResult]
    """
    agent = FEAgent(llm=llm, config=config)
    return agent.run(
        df=df,
        target_col=target_col,
        profile_json=profile_json,
        program_md=program_md,
        train_config=train_config,
        output_dir=output_dir,
        save_model=save_model,
    )
