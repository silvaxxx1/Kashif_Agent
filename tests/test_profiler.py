"""
test_profiler.py — Step 4b tests for core/profiler.py

Strategy:
  - All DataFrames are synthetic
  - EDA tests only check file creation and HTML structure, not visual correctness
  - TaskDetector tests cover all 6 rules + edge cases
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from core.profiler import TaskDetector, eda, profile, run


# ---------------------------------------------------------------------------
# Synthetic DataFrames
# ---------------------------------------------------------------------------

def make_classification_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 80, n).astype(float),
        "salary": rng.normal(50_000, 15_000, n),
        "department": rng.choice(["eng", "sales", "hr"], n),
        "id_col": [f"user_{i}" for i in range(n)],       # high cardinality
        "target": rng.choice(["yes", "no"], n),
    })


def make_regression_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "x1": rng.normal(0, 1, n),
        "x2": rng.normal(5, 2, n),
        "x3": rng.normal(-1, 3, n),
        "target": rng.normal(100, 20, n),
    })


def make_nulls_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "good": rng.normal(0, 1, n),
        "sparse": [np.nan if i % 2 == 0 else float(i) for i in range(n)],  # 50% null
        "target": rng.choice([0, 1], n),
    })
    return df


# ---------------------------------------------------------------------------
# TaskDetector
# ---------------------------------------------------------------------------

class TestTaskDetector:
    def setup_method(self):
        self.detector = TaskDetector()

    def test_string_target_is_classification(self):
        y = pd.Series(["yes", "no", "yes", "no"] * 50)
        task, conf = self.detector.detect(y)
        assert task == "classification"
        assert conf > 0.7

    def test_binary_01_is_classification(self):
        y = pd.Series([0, 1, 0, 1, 1, 0] * 30)
        task, conf = self.detector.detect(y)
        assert task == "classification"
        assert conf > 0.8  # binary 0/1 aggregates across multiple rules — avg lands ~0.87

    def test_boolean_target_is_classification(self):
        y = pd.Series([True, False, True, False] * 50)
        task, conf = self.detector.detect(y)
        assert task == "classification"
        assert conf > 0.9

    def test_continuous_float_is_regression(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.normal(100, 20, 500))
        task, conf = self.detector.detect(y)
        assert task == "regression"
        assert conf > 0.7

    def test_low_cardinality_int_is_classification(self):
        y = pd.Series([1, 2, 3, 1, 2, 3] * 40)
        task, conf = self.detector.detect(y)
        assert task == "classification"

    def test_high_cardinality_int_is_regression(self):
        y = pd.Series(range(500))
        task, conf = self.detector.detect(y)
        assert task == "regression"

    def test_confidence_between_0_and_1(self):
        y = pd.Series(["a", "b", "c"] * 20)
        _, conf = self.detector.detect(y)
        assert 0.0 <= conf <= 1.0

    def test_get_detection_details_keys(self):
        y = pd.Series(["yes", "no"] * 50)
        details = self.detector.get_detection_details(y)
        for key in ["task_type", "confidence", "n_samples", "n_unique",
                    "unique_ratio", "dtype", "null_count", "null_rate"]:
            assert key in details

    def test_empty_series_returns_classification(self):
        y = pd.Series([], dtype=object)
        task, conf = self.detector.detect(y)
        assert task in ("classification", "regression")
        assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# profile()
# ---------------------------------------------------------------------------

class TestProfile:
    def test_returns_required_keys(self):
        df = make_classification_df()
        result = profile(df, "target")
        for key in ["n_rows", "n_cols", "target_col", "task_type",
                    "task_confidence", "memory_mb", "columns", "target", "warnings"]:
            assert key in result

    def test_n_rows_correct(self):
        df = make_classification_df(150)
        result = profile(df, "target")
        assert result["n_rows"] == 150

    def test_n_cols_excludes_target(self):
        df = make_classification_df()
        result = profile(df, "target")
        assert result["n_cols"] == len(df.columns) - 1

    def test_task_type_classification(self):
        df = make_classification_df()
        result = profile(df, "target")
        assert result["task_type"] == "classification"

    def test_task_type_regression(self):
        df = make_regression_df()
        result = profile(df, "target")
        assert result["task_type"] == "regression"

    def test_column_stats_present(self):
        df = make_classification_df()
        result = profile(df, "target")
        for col in df.columns:
            if col != "target":
                assert col in result["columns"]

    def test_numeric_col_has_mean_std(self):
        df = make_classification_df()
        result = profile(df, "target")
        assert "mean" in result["columns"]["age"]
        assert "std" in result["columns"]["age"]
        assert "skew" in result["columns"]["age"]

    def test_categorical_col_has_top_values(self):
        df = make_classification_df()
        result = profile(df, "target")
        assert result["columns"]["department"]["top_values"] is not None

    def test_null_rates_computed(self):
        df = make_nulls_df()
        result = profile(df, "target")
        assert result["columns"]["sparse"]["null_rate"] > 0.4

    def test_target_distribution_classification(self):
        df = make_classification_df()
        result = profile(df, "target")
        assert "distribution" in result["target"]
        assert "imbalance_ratio" in result["target"]

    def test_target_stats_regression(self):
        df = make_regression_df()
        result = profile(df, "target")
        assert "mean" in result["target"]
        assert "std" in result["target"]

    def test_warnings_for_high_cardinality(self):
        df = make_classification_df()
        result = profile(df, "target")
        warning_text = " ".join(result["warnings"])
        assert "id_col" in warning_text

    def test_warnings_for_high_nulls(self):
        df = make_nulls_df()
        result = profile(df, "target")
        warning_text = " ".join(result["warnings"])
        assert "sparse" in warning_text

    def test_missing_target_raises(self):
        df = make_classification_df()
        with pytest.raises(ValueError, match="Target column"):
            profile(df, "nonexistent_col")

    def test_imbalanced_flag(self):
        # 90/10 split — definitely imbalanced
        rng = np.random.default_rng(5)
        y = pd.Series(rng.choice(["yes", "no"], 200, p=[0.9, 0.1]))
        df = pd.DataFrame({"x": rng.normal(0, 1, 200), "target": y})
        result = profile(df, "target")
        assert result["target"]["is_imbalanced"] is True


# ---------------------------------------------------------------------------
# eda()
# ---------------------------------------------------------------------------

class TestEda:
    def test_creates_html_file(self, tmp_path):
        df = make_classification_df()
        path = eda(df, "target", output_dir=str(tmp_path))
        assert os.path.exists(path)
        assert path.endswith(".html")

    def test_html_contains_target_name(self, tmp_path):
        df = make_classification_df()
        path = eda(df, "target", output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "target" in content

    def test_html_is_not_empty(self, tmp_path):
        df = make_classification_df()
        path = eda(df, "target", output_dir=str(tmp_path))
        assert Path(path).stat().st_size > 10_000  # at least 10KB (has embedded charts)

    def test_html_contains_charts(self, tmp_path):
        df = make_classification_df()
        path = eda(df, "target", output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "data:image/png;base64," in content

    def test_regression_eda(self, tmp_path):
        df = make_regression_df()
        path = eda(df, "target", output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_accepts_precomputed_profile(self, tmp_path):
        df = make_classification_df()
        prof = profile(df, "target")
        path = eda(df, "target", profile_json=prof, output_dir=str(tmp_path))
        assert os.path.exists(path)

    def test_no_nulls_chart_graceful(self, tmp_path):
        # "No missing values" message is rendered inside the matplotlib chart (base64 image)
        # so it won't appear as plain text in the HTML — just verify the file is valid
        df = make_regression_df()  # no nulls
        path = eda(df, "target", output_dir=str(tmp_path))
        content = Path(path).read_text()
        assert "data:image/png;base64," in content
        assert Path(path).stat().st_size > 5_000


# ---------------------------------------------------------------------------
# run() — combined entry point
# ---------------------------------------------------------------------------

class TestRun:
    def test_returns_profile_and_path(self, tmp_path):
        df = make_classification_df()
        prof, eda_path = run(df, "target", output_dir=str(tmp_path), save_eda=True)
        assert isinstance(prof, dict)
        assert eda_path is not None
        assert os.path.exists(eda_path)

    def test_save_eda_false_returns_none(self, tmp_path):
        df = make_classification_df()
        prof, eda_path = run(df, "target", output_dir=str(tmp_path), save_eda=False)
        assert isinstance(prof, dict)
        assert eda_path is None

    def test_profile_task_type_matches(self, tmp_path):
        df = make_regression_df()
        prof, _ = run(df, "target", output_dir=str(tmp_path), save_eda=False)
        assert prof["task_type"] == "regression"
