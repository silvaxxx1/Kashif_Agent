"""
executor.py — Sandboxed feature engineering code runner (Step 4c)

Written from scratch. Imports nothing from kashif_core — hard isolation rule.

Public API:
    execute(fe_code, df, timeout)  ->  (transformed_df | None, error_str | None)

The LLM must produce code that defines exactly one function:
    engineer_features(df: pd.DataFrame) -> pd.DataFrame

executor.execute() runs that function in a restricted namespace, enforces a
timeout, validates the output, and returns either the transformed DataFrame or
a structured error string. It never raises — all failures become error strings
so fe_agent.py can log them and feed them back to the LLM.
"""

from __future__ import annotations

import ast
import collections
import datetime
import itertools
import math
import re
import signal
import textwrap
import threading
import traceback
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_TIMEOUT: int = 30  # seconds

# Globals exposed to LLM-generated code — whitelist only
# Modules the LLM is allowed to import inside fe_code
_ALLOWED_IMPORTS: frozenset = frozenset({
    "math", "re", "datetime", "collections", "itertools",
    "functools", "statistics", "string",
})


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    """Custom __import__ that only allows safe standard-library modules."""
    if name in _ALLOWED_IMPORTS:
        return __import__(name, globals, locals, fromlist, level)
    raise ImportError(
        f"Import of '{name}' is not allowed in fe_code. "
        f"Allowed: {sorted(_ALLOWED_IMPORTS)}. "
        "Data science libraries (pd, np) are pre-injected — no import needed."
    )


_SAFE_GLOBALS: dict = {
    "__builtins__": {
        # safe built-ins
        "abs": abs, "all": all, "any": any, "bool": bool,
        "dict": dict, "enumerate": enumerate, "filter": filter,
        "float": float, "frozenset": frozenset, "getattr": getattr,
        "hasattr": hasattr, "int": int, "isinstance": isinstance,
        "issubclass": issubclass, "iter": iter, "len": len, "list": list,
        "map": map, "max": max, "min": min, "next": next, "object": object,
        "print": print, "range": range, "repr": repr, "reversed": reversed,
        "round": round, "set": set, "setattr": setattr, "slice": slice,
        "sorted": sorted, "str": str, "sum": sum, "tuple": tuple,
        "type": type, "zip": zip,
        # allow restricted imports (math, re, etc.)
        "__import__": _restricted_import,
        # exceptions LLM code may need to catch
        "ValueError": ValueError, "TypeError": TypeError,
        "KeyError": KeyError, "IndexError": IndexError,
        "AttributeError": AttributeError, "Exception": Exception,
    },
    # data science stack pre-injected — LLM should use these directly
    "pd": pd,
    "np": np,
    # standard library modules pre-injected for convenience
    "math": math,
    "re": re,
    "datetime": datetime,
    "collections": collections,
    "itertools": itertools,
}


# ---------------------------------------------------------------------------
# Timeout helpers
#
# Two strategies:
#   - SIGALRM (Linux/macOS, main thread only) — preferred, zero overhead
#   - threading.Thread join (fallback) — works from any thread (e.g. FastAPI
#     thread pool), but cannot forcibly kill a hung thread; the thread runs
#     to completion as a daemon after the timeout is declared.
# ---------------------------------------------------------------------------

class _TimeoutError(Exception):
    """Raised when fe_code exceeds the allowed wall-clock time."""


def _alarm_handler(signum, frame):  # noqa: ANN001
    raise _TimeoutError("fe_code timed out")


def _supports_sigalrm() -> bool:
    """SIGALRM is available AND we are on the main thread."""
    return (
        hasattr(signal, "SIGALRM")
        and threading.current_thread() is threading.main_thread()
    )


def _run_with_thread_timeout(
    fn,
    df_in: "pd.DataFrame",
    timeout: int,
) -> "Tuple[Optional[pd.DataFrame], Optional[str]]":
    """
    Run fn(df_in) in a daemon thread with a join-based timeout.

    Used when SIGALRM is unavailable (Windows) or when called from a
    non-main thread (e.g. FastAPI thread-pool workers).
    """
    result_holder: list = [None]
    error_holder: list = [None]

    def _target():
        try:
            result_holder[0] = fn(df_in)
        except Exception:
            error_holder[0] = f"RuntimeError:\n{traceback.format_exc()}"

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        return None, f"TimeoutError: fe_code exceeded {timeout}s wall-clock limit."

    return result_holder[0], error_holder[0]


# ---------------------------------------------------------------------------
# Core execute() function
# ---------------------------------------------------------------------------

def execute(
    fe_code: str,
    df: pd.DataFrame,
    timeout: int = DEFAULT_TIMEOUT,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Execute LLM-generated feature engineering code against *df*.

    Parameters
    ----------
    fe_code : str
        Python source code that defines ``engineer_features(df) -> df``.
    df : pd.DataFrame
        Input features (target column already removed by the caller).
    timeout : int
        Max wall-clock seconds allowed. Enforced via SIGALRM on Linux/macOS;
        silently skipped on Windows.

    Returns
    -------
    (transformed_df, None)   on success
    (None, error_str)        on any failure — never raises
    """

    # ------------------------------------------------------------------
    # Step 1 — syntax check (no execution, cheap)
    # ------------------------------------------------------------------
    try:
        ast.parse(fe_code)
    except SyntaxError as exc:
        return None, f"SyntaxError: {exc}"

    # ------------------------------------------------------------------
    # Step 2 — exec the code in a restricted namespace to define the fn
    #
    # Use a single dict as both globals and locals so that helper
    # functions defined alongside engineer_features are visible when
    # engineer_features calls them (they share the same __globals__).
    # ------------------------------------------------------------------
    ns: dict = _SAFE_GLOBALS.copy()

    try:
        exec(fe_code, ns)  # noqa: S102
    except Exception:
        return None, f"DefinitionError:\n{traceback.format_exc()}"

    # ------------------------------------------------------------------
    # Step 3 — verify engineer_features is defined and callable
    # ------------------------------------------------------------------
    fn = ns.get("engineer_features")
    if fn is None:
        return None, (
            "fe_code must define a function named `engineer_features(df) -> df`. "
            "No such function found."
        )
    if not callable(fn):
        return None, "`engineer_features` is defined but is not callable."

    # ------------------------------------------------------------------
    # Step 4 — call with a copy; enforce timeout
    #
    # Use SIGALRM when on the main thread (CLI path), thread-join timeout
    # when called from a worker thread (FastAPI path).
    # ------------------------------------------------------------------
    df_in = df.copy()
    result: Optional[pd.DataFrame] = None
    error: Optional[str] = None

    if _supports_sigalrm():
        old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
        signal.alarm(timeout)
        try:
            result = fn(df_in)
        except _TimeoutError:
            error = f"TimeoutError: fe_code exceeded {timeout}s wall-clock limit."
        except Exception:
            error = f"RuntimeError:\n{traceback.format_exc()}"
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    else:
        result, error = _run_with_thread_timeout(fn, df_in, timeout)

    if error is not None:
        return None, error

    # ------------------------------------------------------------------
    # Step 5 — validate the returned DataFrame
    # ------------------------------------------------------------------
    validation_error = _validate_result(result, df)
    if validation_error:
        return None, validation_error

    return result, None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def _validate_result(result: object, original_df: pd.DataFrame) -> Optional[str]:
    """
    Return an error string if *result* is not a valid transformed DataFrame,
    or None if everything looks good.
    """
    if not isinstance(result, pd.DataFrame):
        return (
            f"engineer_features must return a pandas DataFrame, "
            f"got {type(result).__name__!r}."
        )

    if result.shape[1] == 0:
        return "engineer_features returned a DataFrame with no columns."

    if len(result) == 0:
        return "engineer_features returned an empty DataFrame (0 rows)."

    if len(result) != len(original_df):
        return (
            f"Row count mismatch: input had {len(original_df)} rows, "
            f"engineer_features returned {len(result)} rows. "
            "Do not filter or resample rows."
        )

    if result.isnull().all(axis=None):
        return "engineer_features returned a DataFrame where every cell is NaN."

    return None


# ---------------------------------------------------------------------------
# Convenience: extract a clean one-liner error summary for LLM prompts
# ---------------------------------------------------------------------------

def summarise_error(error_str: str, max_lines: int = 10) -> str:
    """
    Trim a full traceback down to the last *max_lines* lines.
    Used by fe_agent.py when building the reflection prompt.
    """
    lines = error_str.strip().splitlines()
    if len(lines) <= max_lines:
        return error_str
    head = lines[0]
    tail = lines[-max_lines:]
    return head + "\n  ...\n" + "\n".join(tail)
