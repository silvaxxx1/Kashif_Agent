"""
test_reporter.py — Step 4f tests for core/reporter.py

Strategy:
  - All inputs are synthetic RoundResult lists
  - Tests cover: empty log, baseline-only, improved runs, no-improvement runs,
    all section contents, save/load, run() entry point
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from core.fe_agent import RoundResult
from core.reporter import report, run, save


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_round(
    round_num: int,
    cv_score: float,
    improved: bool = False,
    fe_code: str = None,
    executor_error: str = None,
    shap_top: list = None,
    shap_dead: list = None,
    best_model_name: str = "Random Forest",
) -> RoundResult:
    train_result = MagicMock()
    train_result.best_model_name = best_model_name
    train_result.status = "complete"

    # cv_loss is inverse of cv_score for classification (lower = better means higher acc)
    cv_loss = 1.0 - cv_score

    return RoundResult(
        round_num=round_num,
        fe_code=fe_code,
        cv_score=cv_score,
        cv_loss=cv_loss,
        delta=cv_score - 0.70 if round_num > 0 else 0.0,
        improved=improved,
        executor_error=executor_error,
        shap_top=shap_top or [],
        shap_dead=shap_dead or [],
        train_result=train_result,
    )


def make_baseline(cv_score: float = 0.70) -> RoundResult:
    return make_round(0, cv_score)


def make_profile() -> dict:
    return {
        "target_col": "survived",
        "task_type": "classification",
        "n_rows": 891,
        "n_cols": 7,
    }


def make_good_log() -> list:
    """Baseline + 2 rounds, second improves."""
    code = "def engineer_features(df):\n    df = df.copy()\n    df['x'] = 1\n    return df"
    return [
        make_baseline(0.70),
        make_round(1, 0.72, improved=False, fe_code=code,
                   shap_top=["age", "fare", "pclass"],
                   shap_dead=["embarked"]),
        make_round(2, 0.78, improved=True, fe_code=code,
                   shap_top=["age_x_fare", "family_size", "age"],
                   shap_dead=["ticket"]),
    ]


# ---------------------------------------------------------------------------
# report() — return value
# ---------------------------------------------------------------------------

class TestReportReturnType:
    def test_returns_string(self):
        log = make_good_log()
        result = report(log)
        assert isinstance(result, str)

    def test_empty_log_returns_string(self):
        result = report([])
        assert isinstance(result, str)
        assert len(result) > 0

    def test_baseline_only_returns_string(self):
        result = report([make_baseline()])
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# report() — header / summary section
# ---------------------------------------------------------------------------

class TestReportHeader:
    def test_contains_title(self):
        md = report(make_good_log())
        assert "Kashif" in md

    def test_contains_baseline_score(self):
        md = report(make_good_log())
        assert "0.7000" in md

    def test_contains_best_score(self):
        md = report(make_good_log())
        assert "0.7800" in md

    def test_contains_target_col_from_profile(self):
        md = report(make_good_log(), make_profile())
        assert "survived" in md

    def test_contains_task_type(self):
        md = report(make_good_log(), make_profile())
        assert "classification" in md

    def test_contains_row_count(self):
        md = report(make_good_log(), make_profile())
        assert "891" in md

    def test_contains_round_count(self):
        md = report(make_good_log())
        # Should mention number of FE rounds run
        assert "2" in md

    def test_contains_improvement_pct(self):
        md = report(make_good_log())
        # 0.78 - 0.70 = 0.08 → ~11.4%
        assert "%" in md


# ---------------------------------------------------------------------------
# report() — score progression table
# ---------------------------------------------------------------------------

class TestScoreProgression:
    def test_contains_progression_heading(self):
        md = report(make_good_log())
        assert "Score Progression" in md or "Progression" in md

    def test_contains_baseline_row(self):
        md = report(make_good_log())
        assert "baseline" in md.lower()

    def test_contains_round_rows(self):
        md = report(make_good_log())
        assert "round 1" in md.lower() or "Round 1" in md

    def test_contains_all_cv_scores(self):
        md = report(make_good_log())
        assert "0.7000" in md
        assert "0.7200" in md
        assert "0.7800" in md

    def test_no_improvement_run_shows_correct_status(self):
        log = [make_baseline(0.70), make_round(1, 0.68, improved=False)]
        md = report(log)
        assert "no improvement" in md.lower() or "0.6800" in md


# ---------------------------------------------------------------------------
# report() — feature analysis section
# ---------------------------------------------------------------------------

class TestFeatureAnalysis:
    def test_top_features_present(self):
        md = report(make_good_log())
        assert "age_x_fare" in md or "family_size" in md

    def test_dead_features_present(self):
        md = report(make_good_log())
        assert "ticket" in md

    def test_no_shap_data_skips_section_gracefully(self):
        log = [make_baseline(), make_round(1, 0.75, improved=True)]
        md = report(log)
        assert isinstance(md, str)

    def test_feature_analysis_heading(self):
        md = report(make_good_log())
        assert "Feature" in md


# ---------------------------------------------------------------------------
# report() — best code section
# ---------------------------------------------------------------------------

class TestBestCodeSection:
    def test_best_code_present(self):
        code = "def engineer_features(df):\n    df = df.copy()\n    df['x'] = 1\n    return df"
        log = [make_baseline(), make_round(1, 0.80, improved=True, fe_code=code)]
        md = report(log)
        assert "engineer_features" in md

    def test_best_code_in_fenced_block(self):
        code = "def engineer_features(df):\n    return df.copy()"
        log = [make_baseline(), make_round(1, 0.80, improved=True, fe_code=code)]
        md = report(log)
        assert "```python" in md or "```" in md

    def test_no_fe_code_skips_section_gracefully(self):
        log = [make_baseline()]
        md = report(log)
        assert isinstance(md, str)


# ---------------------------------------------------------------------------
# report() — round details section
# ---------------------------------------------------------------------------

class TestRoundDetails:
    def test_round_details_heading(self):
        md = report(make_good_log())
        assert "Round Details" in md or "Details" in md

    def test_executor_error_shown(self):
        log = [
            make_baseline(),
            make_round(1, 0.70, improved=False,
                       executor_error="KeyError: 'nonexistent_column'"),
        ]
        md = report(log)
        assert "KeyError" in md or "executor" in md.lower() or "error" in md.lower()

    def test_baseline_not_in_round_details(self):
        """Round 0 has no FE code — should not appear in Round Details section."""
        log = [make_baseline()]
        md = report(log)
        # No round details section when only baseline
        assert "Round Details" not in md or "round 0" not in md.lower()

    def test_improved_round_marked(self):
        code = "def engineer_features(df):\n    return df.copy()"
        log = [make_baseline(), make_round(1, 0.80, improved=True, fe_code=code)]
        md = report(log)
        assert "✓" in md or "best" in md.lower()

    def test_failed_round_marked(self):
        code = "def engineer_features(df):\n    return df.copy()"
        log = [make_baseline(), make_round(1, 0.65, improved=False, fe_code=code)]
        md = report(log)
        assert "✗" in md or "no improvement" in md.lower()


# ---------------------------------------------------------------------------
# report() — recommendation section
# ---------------------------------------------------------------------------

class TestRecommendation:
    def test_recommendation_heading(self):
        md = report(make_good_log())
        assert "Recommendation" in md

    def test_improvement_acknowledged(self):
        md = report(make_good_log())
        # Should say something positive about improvement
        assert "improved" in md.lower() or "improvement" in md.lower()

    def test_no_improvement_advises_static(self):
        log = [make_baseline(0.70), make_round(1, 0.70, improved=False)]
        md = report(log)
        assert "static" in md.lower() or "baseline" in md.lower()

    def test_best_model_name_in_recommendation(self):
        log = [make_baseline(), make_round(1, 0.80, improved=True,
                                           best_model_name="Gradient Boosting")]
        md = report(log)
        assert "Gradient Boosting" in md

    def test_code_usage_hint_when_improved(self):
        code = "def engineer_features(df):\n    return df.copy()"
        log = [make_baseline(), make_round(1, 0.80, improved=True, fe_code=code)]
        md = report(log)
        assert "FETransformer" in md or "fe_step" in md


# ---------------------------------------------------------------------------
# save() and run()
# ---------------------------------------------------------------------------

class TestSave:
    def test_saves_file(self, tmp_path):
        md = "# Test Report\n\nHello."
        path = save(md, str(tmp_path))
        assert os.path.exists(path)

    def test_saved_file_is_md(self, tmp_path):
        path = save("# hi", str(tmp_path))
        assert path.endswith(".md")

    def test_saved_content_matches(self, tmp_path):
        md = "# Kashif\n\nTest content here."
        path = save(md, str(tmp_path))
        assert Path(path).read_text() == md

    def test_creates_output_dir(self, tmp_path):
        nested = str(tmp_path / "deep" / "nested")
        path = save("# hi", nested)
        assert os.path.exists(path)


class TestRun:
    def test_returns_tuple(self, tmp_path):
        log = make_good_log()
        result = run(log, make_profile(), output_dir=str(tmp_path))
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_first_element_is_string(self, tmp_path):
        md, _ = run(make_good_log(), output_dir=str(tmp_path))
        assert isinstance(md, str)

    def test_save_true_returns_path(self, tmp_path):
        _, path = run(make_good_log(), output_dir=str(tmp_path), save_report=True)
        assert path is not None
        assert os.path.exists(path)

    def test_save_false_returns_none(self, tmp_path):
        _, path = run(make_good_log(), output_dir=str(tmp_path), save_report=False)
        assert path is None

    def test_report_content_not_empty(self, tmp_path):
        md, _ = run(make_good_log(), make_profile(), output_dir=str(tmp_path))
        assert len(md) > 500
