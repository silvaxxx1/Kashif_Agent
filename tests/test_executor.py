"""
test_executor.py — Step 4c tests for core/executor.py

Strategy:
  - All DataFrames are synthetic — no external CSV dependency
  - Tests cover: happy path, all validation failures, all error types,
    namespace isolation, original-df immutability, summarise_error helper
  - Timeout test is skipped on platforms without SIGALRM (Windows)
"""

import math
import signal

import numpy as np
import pandas as pd
import pytest

from core.executor import DEFAULT_TIMEOUT, execute, summarise_error


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_df(n: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "a": rng.standard_normal(n),
        "b": rng.standard_normal(n),
        "c": rng.integers(0, 5, size=n).astype(float),
    })


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_basic_feature_added(self):
        df = make_df()
        code = """
def engineer_features(df):
    df = df.copy()
    df["a_plus_b"] = df["a"] + df["b"]
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert result is not None
        assert "a_plus_b" in result.columns
        assert len(result) == len(df)

    def test_log_transform(self):
        df = make_df()
        df["a"] = df["a"].abs() + 1  # ensure positive
        code = """
import math
def engineer_features(df):
    df = df.copy()
    df["log_a"] = np.log1p(df["a"])
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "log_a" in result.columns

    def test_multiple_new_columns(self):
        df = make_df()
        code = """
def engineer_features(df):
    df = df.copy()
    df["ab_ratio"] = df["a"] / (df["b"].abs() + 1e-6)
    df["c_sq"] = df["c"] ** 2
    df["ab_sum"] = df["a"] + df["b"]
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert set(["ab_ratio", "c_sq", "ab_sum"]).issubset(result.columns)

    def test_pandas_operations(self):
        df = make_df()
        code = """
def engineer_features(df):
    df = df.copy()
    df["a_zscore"] = (df["a"] - df["a"].mean()) / df["a"].std()
    df["b_rank"] = df["b"].rank(pct=True)
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "a_zscore" in result.columns
        assert "b_rank" in result.columns

    def test_numpy_operations(self):
        df = make_df()
        code = """
def engineer_features(df):
    df = df.copy()
    df["a_clip"] = np.clip(df["a"], -2, 2)
    df["ab_dot"] = np.multiply(df["a"], df["b"])
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "a_clip" in result.columns

    def test_uses_math_module(self):
        df = make_df()
        code = """
def engineer_features(df):
    df = df.copy()
    df["a_sin"] = df["a"].apply(lambda x: math.sin(x))
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "a_sin" in result.columns

    def test_uses_re_module(self):
        df = pd.DataFrame({"text": ["abc_123", "def_456", "ghi_789"]})
        code = """
def engineer_features(df):
    df = df.copy()
    df["digits"] = df["text"].str.extract(r"(\\d+)").astype(float)
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "digits" in result.columns

    def test_helper_function_allowed(self):
        """LLM may define helper functions alongside engineer_features."""
        df = make_df()
        code = """
def _clip(series, lo, hi):
    return series.clip(lo, hi)

def engineer_features(df):
    df = df.copy()
    df["a_safe"] = _clip(df["a"], -3, 3)
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "a_safe" in result.columns

    def test_original_df_not_mutated(self):
        """executor must pass df.copy() — original must be untouched."""
        df = make_df()
        original_cols = list(df.columns)
        original_values = df["a"].copy()

        code = """
def engineer_features(df):
    df["injected"] = 999
    df["a"] = -1
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert "injected" not in df.columns
        assert list(df.columns) == original_cols
        pd.testing.assert_series_equal(df["a"], original_values)

    def test_returns_more_columns_than_input(self):
        df = make_df()
        code = """
def engineer_features(df):
    df = df.copy()
    for i in range(5):
        df[f"feat_{i}"] = df["a"] * i
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert result.shape[1] == df.shape[1] + 5

    def test_result_row_count_matches(self):
        df = make_df(n=200)
        code = """
def engineer_features(df):
    df = df.copy()
    df["x"] = 1
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert len(result) == 200


# ---------------------------------------------------------------------------
# Syntax errors
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    def test_syntax_error_caught(self):
        code = "def engineer_features(df):\n    return df[["  # broken bracket
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None
        assert "SyntaxError" in err

    def test_invalid_indentation(self):
        code = "def engineer_features(df):\nreturn df"  # no indent
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_empty_code_string(self):
        result, err = execute("", make_df())
        assert result is None
        assert err is not None
        assert "engineer_features" in err


# ---------------------------------------------------------------------------
# Definition errors (exec fails at import/definition time)
# ---------------------------------------------------------------------------

class TestDefinitionErrors:
    def test_import_forbidden_module(self):
        """os is not in the safe namespace — NameError at definition time."""
        code = """
import os
def engineer_features(df):
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_missing_engineer_features(self):
        code = "x = 1 + 1"
        result, err = execute(code, make_df())
        assert result is None
        assert "engineer_features" in err

    def test_not_callable(self):
        code = "engineer_features = 42"
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None
        assert "not callable" in err


# ---------------------------------------------------------------------------
# Runtime errors
# ---------------------------------------------------------------------------

class TestRuntimeErrors:
    def test_key_error(self):
        code = """
def engineer_features(df):
    df = df.copy()
    df["x"] = df["nonexistent_column"]
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None
        assert "RuntimeError" in err

    def test_division_by_zero(self):
        code = """
def engineer_features(df):
    df = df.copy()
    df["bad"] = 1 / 0
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert "RuntimeError" in err

    def test_type_error_at_runtime(self):
        code = """
def engineer_features(df):
    return df + "string"
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_raises_inside_function(self):
        code = """
def engineer_features(df):
    raise ValueError("intentional error from test")
"""
        result, err = execute(code, make_df())
        assert result is None
        assert "intentional error from test" in err


# ---------------------------------------------------------------------------
# Validation failures
# ---------------------------------------------------------------------------

class TestValidation:
    def test_returns_none(self):
        code = """
def engineer_features(df):
    return None
"""
        result, err = execute(code, make_df())
        assert result is None
        assert "DataFrame" in err

    def test_returns_list(self):
        code = """
def engineer_features(df):
    return [1, 2, 3]
"""
        result, err = execute(code, make_df())
        assert result is None
        assert "DataFrame" in err

    def test_returns_empty_columns(self):
        code = """
def engineer_features(df):
    return df.iloc[:, :0]   # zero columns
"""
        result, err = execute(code, make_df())
        assert result is None
        assert "no columns" in err

    def test_returns_empty_rows(self):
        code = """
def engineer_features(df):
    return df.iloc[:0]   # zero rows
"""
        result, err = execute(code, make_df())
        assert result is None
        assert "empty" in err.lower() or "0 rows" in err

    def test_row_count_mismatch(self):
        code = """
def engineer_features(df):
    return df.head(10)   # filters rows
"""
        result, err = execute(code, make_df(n=50))
        assert result is None
        assert "Row count mismatch" in err

    def test_all_nan_rejected(self):
        code = """
def engineer_features(df):
    import pandas as pd
    return pd.DataFrame({"x": [float("nan")] * len(df)})
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------

class TestNamespaceIsolation:
    def test_os_not_accessible(self):
        code = """
def engineer_features(df):
    import os
    os.system("echo pwned")
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_sys_not_accessible(self):
        code = """
def engineer_features(df):
    import sys
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_subprocess_not_accessible(self):
        code = """
def engineer_features(df):
    import subprocess
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_open_not_accessible(self):
        code = """
def engineer_features(df):
    f = open("/etc/passwd", "r")
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_exec_not_accessible(self):
        code = """
def engineer_features(df):
    exec("import os")
    return df
"""
        result, err = execute(code, make_df())
        assert result is None
        assert err is not None

    def test_builtins_safe_set_available(self):
        """len, range, zip etc. must work inside LLM code."""
        code = """
def engineer_features(df):
    df = df.copy()
    df["n"] = len(df)
    cols = list(df.columns)
    return df
"""
        result, err = execute(code, make_df())
        assert err is None
        assert result is not None


# ---------------------------------------------------------------------------
# Timeout (SIGALRM platforms only)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not hasattr(signal, "SIGALRM"),
    reason="SIGALRM not available on this platform",
)
class TestTimeout:
    def test_infinite_loop_is_killed(self):
        code = """
def engineer_features(df):
    while True:
        pass
"""
        result, err = execute(code, make_df(), timeout=2)
        assert result is None
        assert "Timeout" in err or "timed out" in err.lower()

    def test_fast_code_completes_within_timeout(self):
        code = """
def engineer_features(df):
    df = df.copy()
    df["x"] = 1
    return df
"""
        result, err = execute(code, make_df(), timeout=DEFAULT_TIMEOUT)
        assert err is None
        assert result is not None


# ---------------------------------------------------------------------------
# summarise_error helper
# ---------------------------------------------------------------------------

class TestSummariseError:
    def test_short_error_unchanged(self):
        err = "SyntaxError: bad syntax"
        assert summarise_error(err, max_lines=10) == err

    def test_long_error_truncated(self):
        long_err = "\n".join([f"line {i}" for i in range(50)])
        summary = summarise_error(long_err, max_lines=10)
        assert len(summary.splitlines()) < 50
        assert "line 0" in summary   # head preserved
        assert "line 49" in summary  # tail preserved

    def test_exactly_max_lines_unchanged(self):
        err = "\n".join(["x"] * 10)
        assert summarise_error(err, max_lines=10) == err


# ---------------------------------------------------------------------------
# Audit fixes — new tests for previously uncovered edge cases
# ---------------------------------------------------------------------------

class TestExecutorAuditFixes:
    def test_itertools_pre_injected(self):
        """Issue #6: itertools must be available without import."""
        code = """
def engineer_features(df):
    df = df.copy()
    # use itertools.chain directly (pre-injected)
    cols = list(itertools.chain(["a"], ["b"]))
    df["test"] = 1
    return df
"""
        result, err = execute(code, make_df())
        assert err is None, f"Unexpected error: {err}"
        assert "test" in result.columns

    def test_itertools_import_also_works(self):
        """Issue #6: explicit import itertools must also work."""
        code = """
import itertools
def engineer_features(df):
    df = df.copy()
    df["x"] = 1
    return df
"""
        result, err = execute(code, make_df())
        assert err is None

    def test_duplicate_column_names_handled(self):
        """Issue #10: LLM accidentally creates duplicate column — must not crash executor."""
        code = """
def engineer_features(df):
    df = df.copy()
    df["a"] = df["a"] * 2   # overwrite existing col — allowed
    return df
"""
        result, err = execute(code, make_df())
        assert err is None
        assert result.shape[1] == make_df().shape[1]

    def test_non_default_index_preserved(self):
        """Issue #10: DataFrame with non-default index must still pass row count check."""
        df = make_df()
        df.index = range(100, 100 + len(df))   # non-default index
        code = """
def engineer_features(df):
    df = df.copy()
    df["x"] = 1
    return df
"""
        result, err = execute(code, df)
        assert err is None
        assert len(result) == len(df)
