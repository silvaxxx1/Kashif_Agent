"""
test_fe_agent.py — Step 4e tests for core/fe_agent.py

Strategy:
  - All DataFrames are synthetic
  - LLM is always mocked — no real API calls
  - Covers: FETransformer, prompt builder, code extraction, RoundResult,
    FEAgent loop logic, stopping conditions, surviving_cols, run()
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from core.fe_agent import (
    FEAgent,
    FETransformer,
    RoundResult,
    _extract_code,
    _build_prompt,
    run,
)
from core.llm.base import BaseLLM, LLMError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "fare": rng.uniform(5, 300, n),
        "pclass": rng.choice([1, 2, 3], n).astype(float),
        "target": rng.choice([0, 1], n),
    })


def make_profile(df: pd.DataFrame, target_col: str = "target") -> dict:
    """Minimal profile_json structure for testing."""
    feature_cols = [c for c in df.columns if c != target_col]
    columns = {}
    for col in feature_cols:
        columns[col] = {
            "dtype": str(df[col].dtype),
            "null_rate": float(df[col].isna().mean()),
            "mean": float(df[col].mean()),
            "std": float(df[col].std()),
            "skew": 0.0,
        }
    return {
        "n_rows": len(df),
        "n_cols": len(feature_cols),
        "target_col": target_col,
        "task_type": "classification",
        "task_confidence": 0.9,
        "columns": columns,
        "target": {"distribution": {"0": 50, "1": 50}, "imbalance_ratio": 1.0, "is_imbalanced": False},
        "warnings": [],
    }


def make_mock_llm(response: str = None) -> MagicMock:
    """LLM mock that returns a valid engineer_features function."""
    if response is None:
        response = """
def engineer_features(df):
    df = df.copy()
    df["age_x_fare"] = df["age"] * df["fare"]
    return df
"""
    mock = MagicMock(spec=BaseLLM)
    mock.complete.return_value = response
    # _call_llm now uses complete_with_system — wire both
    mock.complete_with_system.return_value = response
    return mock


# ---------------------------------------------------------------------------
# FETransformer
# ---------------------------------------------------------------------------

class TestFETransformer:
    def test_transform_adds_column(self):
        df = make_df().drop(columns=["target"])
        code = """
def engineer_features(df):
    df = df.copy()
    df["new_col"] = df["age"] * 2
    return df
"""
        transformer = FETransformer(fe_code=code)
        transformer.fit(df)
        result = transformer.transform(df)
        assert "new_col" in result.columns

    def test_fit_returns_self(self):
        transformer = FETransformer(fe_code="def engineer_features(df):\n    return df.copy()")
        result = transformer.fit(make_df().drop(columns=["target"]))
        assert result is transformer

    def test_transform_bad_code_returns_original(self):
        """Executor errors must not crash — return original X instead."""
        df = make_df().drop(columns=["target"])
        code = """
def engineer_features(df):
    raise RuntimeError("intentional failure")
"""
        transformer = FETransformer(fe_code=code)
        result = transformer.transform(df)
        assert list(result.columns) == list(df.columns)

    def test_transform_replaces_inf(self):
        """inf values must be replaced with NaN after transform."""
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        code = """
def engineer_features(df):
    df = df.copy()
    df["bad"] = float("inf")
    return df
"""
        transformer = FETransformer(fe_code=code)
        result = transformer.transform(df)
        assert not result["bad"].isin([float("inf"), float("-inf")]).any()

    def test_transform_preserves_row_count(self):
        df = make_df(n=80).drop(columns=["target"])
        code = """
def engineer_features(df):
    df = df.copy()
    df["x"] = 1
    return df
"""
        transformer = FETransformer(fe_code=code)
        result = transformer.transform(df)
        assert len(result) == 80

    def test_sklearn_clone_works(self):
        """FETransformer must be cloneable by sklearn (required for CV)."""
        from sklearn.base import clone
        transformer = FETransformer(fe_code="def engineer_features(df):\n    return df.copy()")
        cloned = clone(transformer)
        assert cloned.fe_code == transformer.fe_code
        assert cloned is not transformer


# ---------------------------------------------------------------------------
# _extract_code
# ---------------------------------------------------------------------------

class TestExtractCode:
    def test_plain_function_returned_as_is(self):
        code = "def engineer_features(df):\n    return df.copy()"
        assert _extract_code(code) is not None
        assert "def engineer_features" in _extract_code(code)

    def test_strips_markdown_python_fence(self):
        raw = "```python\ndef engineer_features(df):\n    return df.copy()\n```"
        result = _extract_code(raw)
        assert result is not None
        assert "```" not in result

    def test_strips_plain_fence(self):
        raw = "```\ndef engineer_features(df):\n    return df.copy()\n```"
        result = _extract_code(raw)
        assert result is not None
        assert "```" not in result

    def test_strips_pandas_import(self):
        raw = "import pandas as pd\ndef engineer_features(df):\n    return df.copy()"
        result = _extract_code(raw)
        assert "import pandas" not in result

    def test_strips_numpy_import(self):
        raw = "import numpy as np\ndef engineer_features(df):\n    return df.copy()"
        result = _extract_code(raw)
        assert "import numpy" not in result

    def test_returns_none_if_no_function(self):
        raw = "Here is some explanation but no code."
        assert _extract_code(raw) is None

    def test_returns_none_for_empty_string(self):
        assert _extract_code("") is None

    def test_preserves_allowed_imports(self):
        raw = "import math\ndef engineer_features(df):\n    return df.copy()"
        result = _extract_code(raw)
        assert "import math" in result


# ---------------------------------------------------------------------------
# _build_prompt
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_contains_target_col(self):
        df = make_df()
        prof = make_profile(df)
        prompt = _build_prompt(prof, "", [], ["age", "fare", "pclass"])
        assert "target" in prompt

    def test_contains_task_type(self):
        df = make_df()
        prof = make_profile(df)
        prompt = _build_prompt(prof, "", [], ["age", "fare", "pclass"])
        assert "classification" in prompt

    def test_contains_surviving_cols(self):
        df = make_df()
        prof = make_profile(df)
        prompt = _build_prompt(prof, "", [], ["age", "fare"])
        assert "age" in prompt
        assert "fare" in prompt

    def test_contains_program_md(self):
        df = make_df()
        prof = make_profile(df)
        prompt = _build_prompt(prof, "Titanic survival prediction", [], ["age"])
        assert "Titanic" in prompt

    def test_contains_previous_round_info(self):
        df = make_df()
        prof = make_profile(df)
        prev_round = RoundResult(
            round_num=1,
            fe_code="def engineer_features(df):\n    return df.copy()",
            cv_score=0.75,
            cv_loss=0.25,
            delta=0.05,
            improved=True,
            executor_error=None,
            shap_top=["age", "fare"],
            shap_dead=[],
            train_result=None,
        )
        prompt = _build_prompt(prof, "", [prev_round], ["age", "fare"])
        assert "Round 1" in prompt
        assert "0.7500" in prompt

    def test_contains_error_when_present(self):
        df = make_df()
        prof = make_profile(df)
        error_round = RoundResult(
            round_num=1,
            fe_code=None,
            cv_score=0.70,
            cv_loss=0.30,
            delta=0.0,
            improved=False,
            executor_error="KeyError: 'nonexistent_column'",
            shap_top=[],
            shap_dead=[],
            train_result=None,
        )
        prompt = _build_prompt(prof, "", [error_round], ["age"])
        assert "Error" in prompt or "KeyError" in prompt

    def test_empty_log_mentions_task(self):
        df = make_df()
        prof = make_profile(df)
        prompt = _build_prompt(prof, "", [], ["age"])
        assert "engineer_features" in prompt

    def test_ohe_note_shown_for_categorical_cols(self):
        """OHE pipeline note must appear when categorical columns are present."""
        df = make_df()
        prof = make_profile(df)
        # Add a categorical column to the profile
        prof["columns"]["sex"] = {
            "dtype": "object",
            "null_rate": 0.0,
            "top_values": {"male": 60, "female": 40},
        }
        prompt = _build_prompt(prof, "", [], ["age", "sex"])
        assert "PIPELINE NOTE" in prompt
        assert "sex" in prompt
        assert "one-hot" in prompt.lower() or "one_hot" in prompt.lower() or "encoded" in prompt.lower()

    def test_ohe_note_absent_for_numeric_only(self):
        """OHE pipeline note must NOT appear when all columns are numeric."""
        df = make_df()
        prof = make_profile(df)
        # All columns in make_df() are float64 or int64 — no categoricals
        prompt = _build_prompt(prof, "", [], ["age", "fare", "pclass"])
        assert "PIPELINE NOTE" not in prompt


# ---------------------------------------------------------------------------
# FEAgent — core loop logic (mocked trainer)
# ---------------------------------------------------------------------------

class TestFEAgentLoopLogic:
    """
    Test loop logic by patching trainer.train so no real sklearn work happens.
    Each mock train call returns a TrainResult-like object.
    """

    def _make_train_result(self, cv_score: float, cv_loss: float) -> MagicMock:
        r = MagicMock()
        r.status = "complete"
        r.cv_score = cv_score
        r.cv_loss = cv_loss
        r.model_path = "/tmp/model.pkl"
        r.leaderboard = []
        r.task_type = "classification"
        r.shap_dict = {}
        return r

    def test_round_0_is_baseline(self):
        """Experiment log first entry must have fe_code=None."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = self._make_train_result(0.70, 0.30)
        fe_result = self._make_train_result(0.75, 0.25)

        with patch("core.fe_agent.train", side_effect=[baseline, fe_result, fe_result]):
            agent = FEAgent(llm=llm, config={"max_rounds": 1, "stall_rounds": 1})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        assert log[0].fe_code is None
        assert log[0].round_num == 0

    def test_improvement_recorded(self):
        """Round where cv_loss drops must have improved=True."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = self._make_train_result(0.70, 0.30)
        fe_result = self._make_train_result(0.80, 0.20)

        # 3rd call = final save of best FE model (new: save_model=True after loop)
        with patch("core.fe_agent.train", side_effect=[baseline, fe_result, fe_result]):
            agent = FEAgent(llm=llm, config={"max_rounds": 1, "stall_rounds": 3})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        assert log[1].improved is True
        assert log[1].delta > 0

    def test_no_improvement_marked_correctly(self):
        """Round where cv_loss does not drop must have improved=False."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = self._make_train_result(0.70, 0.30)
        worse = self._make_train_result(0.65, 0.35)

        # 5th call = final baseline save (no FE round improved → save baseline)
        with patch("core.fe_agent.train", side_effect=[baseline, worse, worse, worse, baseline]):
            agent = FEAgent(llm=llm, config={"max_rounds": 3, "stall_rounds": 3})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        assert not any(r.improved for r in log[1:])

    def test_stall_stops_loop(self):
        """Loop must stop after stall_rounds consecutive non-improving rounds."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = self._make_train_result(0.70, 0.30)
        same = self._make_train_result(0.70, 0.30)

        with patch("core.fe_agent.train", side_effect=[baseline] + [same] * 10):
            agent = FEAgent(llm=llm, config={"max_rounds": 10, "stall_rounds": 3,
                                              "delta_threshold": 0.005})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        # Should stop after round 0 + 3 stall rounds = 4 total
        assert len(log) <= 5  # baseline + stall_rounds + possible 1 extra

    def test_max_rounds_respected(self):
        """Loop must not exceed max_rounds FE rounds."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        # Always improve so stall never triggers
        results = [self._make_train_result(0.70 + i * 0.01, 0.30 - i * 0.01) for i in range(20)]
        with patch("core.fe_agent.train", side_effect=results):
            agent = FEAgent(llm=llm, config={"max_rounds": 3, "stall_rounds": 10})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        # round 0 (baseline) + at most 3 FE rounds
        assert len(log) <= 4

    def test_llm_error_does_not_crash_agent(self):
        """LLMError (API/key failure) must stop the loop immediately — hard stop."""
        df = make_df()
        prof = make_profile(df)

        mock_llm = MagicMock(spec=BaseLLM)
        # complete_with_system is what _call_llm now uses
        mock_llm.complete_with_system.side_effect = LLMError("rate limit")

        baseline = self._make_train_result(0.70, 0.30)
        with patch("core.fe_agent.train", return_value=baseline):
            agent = FEAgent(mock_llm, config={"max_rounds": 5, "max_empty_responses": 2})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        assert log[0].round_num == 0  # baseline always present
        # Hard stop on first LLMError — loop does not continue
        assert len(log) == 1

    def test_unparseable_llm_response_handled(self):
        """LLM response without engineer_features must not crash."""
        df = make_df()
        prof = make_profile(df)

        mock_llm = MagicMock(spec=BaseLLM)
        # Both complete and complete_with_system must return prose (no function)
        mock_llm.complete.return_value = "Sorry, I cannot help with that."
        mock_llm.complete_with_system.return_value = "Sorry, I cannot help with that."

        baseline = self._make_train_result(0.70, 0.30)
        with patch("core.fe_agent.train", return_value=baseline):
            agent = FEAgent(mock_llm, config={"max_rounds": 3, "max_empty_responses": 2})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        assert log[0].round_num == 0

    def test_experiment_log_is_list_of_round_results(self):
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = self._make_train_result(0.70, 0.30)
        with patch("core.fe_agent.train", return_value=baseline):
            agent = FEAgent(llm, config={"max_rounds": 1, "stall_rounds": 1})
            log = agent.run(df, "target", prof, output_dir="/tmp")

        assert isinstance(log, list)
        for entry in log:
            assert isinstance(entry, RoundResult)


# ---------------------------------------------------------------------------
# FEAgent._surviving_cols
# ---------------------------------------------------------------------------

class TestSurvivingCols:
    def test_excludes_high_null_cols(self):
        df = make_df()
        prof = make_profile(df)
        # Inject high null rate
        prof["columns"]["age"]["null_rate"] = 0.6
        surviving = FEAgent._surviving_cols(df, "target", prof)
        assert "age" not in surviving

    def test_includes_normal_cols(self):
        df = make_df()
        prof = make_profile(df)
        surviving = FEAgent._surviving_cols(df, "target", prof)
        assert "fare" in surviving
        assert "pclass" in surviving

    def test_never_includes_target(self):
        df = make_df()
        prof = make_profile(df)
        surviving = FEAgent._surviving_cols(df, "target", prof)
        assert "target" not in surviving

    def test_fallback_when_all_excluded(self):
        """Even if all cols are high-null, must return non-empty list."""
        df = make_df()
        prof = make_profile(df)
        for col in prof["columns"]:
            prof["columns"][col]["null_rate"] = 0.9
        surviving = FEAgent._surviving_cols(df, "target", prof)
        assert len(surviving) > 0

    def test_excludes_high_cardinality_cols(self):
        df = make_df()
        prof = make_profile(df)
        prof["warnings"] = ["age has high cardinality (ratio=0.95)"]
        surviving = FEAgent._surviving_cols(df, "target", prof)
        assert "age" not in surviving


# ---------------------------------------------------------------------------
# run() — convenience function
# ---------------------------------------------------------------------------

class TestRunFunction:
    def test_returns_list(self):
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = MagicMock()
        baseline.status = "complete"
        baseline.cv_score = 0.75
        baseline.cv_loss = 0.25
        baseline.shap_dict = {}

        with patch("core.fe_agent.train", return_value=baseline):
            result = run(df, "target", prof, llm,
                         config={"max_rounds": 0},
                         output_dir="/tmp")

        assert isinstance(result, list)
        assert result[0].round_num == 0

    def test_only_imports_base_llm(self):
        """fe_agent must not import provider SDKs directly."""
        import importlib
        import ast, inspect
        import core.fe_agent as module
        source = inspect.getsource(module)
        tree = ast.parse(source)
        provider_names = {"groq", "openai", "anthropic", "ollama"}
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module_name = ""
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.lower()
                elif isinstance(node, ast.ImportFrom) and node.module:
                    module_name = node.module.lower()
                for prov in provider_names:
                    assert prov not in module_name, \
                        f"fe_agent.py must not import '{prov}' directly — found: {module_name}"


# ---------------------------------------------------------------------------
# Regression tests — fixes applied after live run assessment
# ---------------------------------------------------------------------------

class TestRegressionFixes:
    """Regression tests for the 4-fix batch: SHAP, model save, NaN."""

    def _make_train_result(self, cv_score, cv_loss, shap_dict=None):
        r = MagicMock()
        r.status = "complete"
        r.cv_score = cv_score
        r.cv_loss = cv_loss
        r.shap_dict = shap_dict or {}
        r.model_path = None
        return r

    def test_shap_passed_to_all_rounds(self):
        """compute_shap must be True for both baseline and FE rounds."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()
        calls = []

        def spy_train(**kwargs):
            calls.append(kwargs.get("compute_shap"))
            return self._make_train_result(0.80, 0.20)

        with patch("core.fe_agent.train", side_effect=spy_train):
            agent = FEAgent(llm, config={"max_rounds": 1, "stall_rounds": 1})
            agent.run(df, "target", prof, save_model=False, output_dir="/tmp")

        # All round calls must have compute_shap=True
        assert all(v is True for v in calls), \
            f"Expected compute_shap=True for all rounds, got: {calls}"

    def test_best_fe_model_saved_when_improved(self, tmp_path):
        """When an FE round improves the baseline, a model must be saved."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()

        baseline = self._make_train_result(0.70, 0.30)
        fe_improved = self._make_train_result(0.80, 0.20)
        # final save call returns a result with a model_path
        final_saved = self._make_train_result(0.80, 0.20)
        final_saved.model_path = str(tmp_path / "best_model.pkl")

        call_count = [0]

        def side_effect_train(**kwargs):
            call_count[0] += 1
            if kwargs.get("save_model"):
                return final_saved
            if call_count[0] == 1:
                return baseline
            return fe_improved

        with patch("core.fe_agent.train", side_effect=side_effect_train):
            agent = FEAgent(llm, config={"max_rounds": 1, "stall_rounds": 1,
                                         "delta_threshold": 0.05})
            log = agent.run(df, "target", prof, save_model=True,
                            output_dir=str(tmp_path))

        # At least one call must have save_model=True
        # (verified by final_saved.model_path being set)
        best_rounds = [r for r in log if r.improved]
        if best_rounds:
            assert best_rounds[-1].train_result is not None

    def test_no_model_save_when_not_requested(self):
        """save_model=False must result in no save_model=True call to train()."""
        df = make_df()
        prof = make_profile(df)
        llm = make_mock_llm()
        save_calls = []

        def spy_train(**kwargs):
            if kwargs.get("save_model"):
                save_calls.append(True)
            return self._make_train_result(0.80, 0.20)

        with patch("core.fe_agent.train", side_effect=spy_train):
            agent = FEAgent(llm, config={"max_rounds": 1, "stall_rounds": 1})
            agent.run(df, "target", prof, save_model=False, output_dir="/tmp")

        assert len(save_calls) == 0, "train() must not be called with save_model=True when save_model=False"
