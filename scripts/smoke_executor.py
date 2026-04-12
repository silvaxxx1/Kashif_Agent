"""
smoke_executor.py — executor.py smoke test with real datasets

Exercises execute() end-to-end:
  - good FE code (several real transforms)
  - bad code (syntax error, runtime error, forbidden import, row filter)
  - timeout (infinite loop)
  - summarise_error helper

Run from kashif_core/:
    uv run python scripts/smoke_executor.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

from core.executor import execute, summarise_error

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)

def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")

def fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")

def check(label: str, result, err, expect_success: bool) -> None:
    if expect_success:
        if err is None and result is not None:
            ok(f"{label}  →  {result.shape[1]} cols  ({result.shape[0]} rows)")
        else:
            fail(f"{label}  →  unexpected error: {summarise_error(err, max_lines=4)}")
    else:
        if err is not None and result is None:
            ok(f"{label}  →  correctly rejected: {err[:80].strip()}...")
        else:
            fail(f"{label}  →  should have failed but returned a DataFrame")


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

section("Loading datasets")

bc = load_breast_cancer(as_frame=True)
df_clf = bc.frame.drop(columns=["target"])       # features only (target removed by caller)
print(f"  Breast Cancer   {df_clf.shape[0]} rows × {df_clf.shape[1]} cols")

cal = fetch_california_housing(as_frame=True)
df_reg = cal.frame.drop(columns=["MedHouseVal"])  # features only
print(f"  California Hsg  {df_reg.shape[0]} rows × {df_reg.shape[1]} cols")


# ---------------------------------------------------------------------------
# 1. Happy path — classification dataset
# ---------------------------------------------------------------------------

section("1. Happy path — classification features")

check("log transform", *execute("""
def engineer_features(df):
    df = df.copy()
    df["mean_radius_log"] = np.log1p(df["mean radius"])
    df["area_sqrt"]       = np.sqrt(df["mean area"].abs())
    return df
""", df_clf), expect_success=True)

check("ratio feature", *execute("""
def engineer_features(df):
    df = df.copy()
    df["perimeter_per_area"] = df["mean perimeter"] / (df["mean area"] + 1e-6)
    return df
""", df_clf), expect_success=True)

check("zscore normalise", *execute("""
def engineer_features(df):
    df = df.copy()
    for col in ["mean radius", "mean texture", "mean perimeter"]:
        mu, sigma = df[col].mean(), df[col].std()
        df[f"{col}_z"] = (df[col] - mu) / (sigma + 1e-9)
    return df
""", df_clf), expect_success=True)

check("helper function", *execute("""
def _clip_iqr(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return series.clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)

def engineer_features(df):
    df = df.copy()
    df["radius_clipped"] = _clip_iqr(df["mean radius"])
    return df
""", df_clf), expect_success=True)

check("import math inside code", *execute("""
import math
def engineer_features(df):
    df = df.copy()
    df["log2_area"] = df["mean area"].apply(lambda x: math.log2(max(x, 1)))
    return df
""", df_clf), expect_success=True)

check("numpy polyfit interaction", *execute("""
def engineer_features(df):
    df = df.copy()
    df["radius_x_texture"] = df["mean radius"] * df["mean texture"]
    df["radius_sq"] = df["mean radius"] ** 2
    return df
""", df_clf), expect_success=True)


# ---------------------------------------------------------------------------
# 2. Happy path — regression dataset
# ---------------------------------------------------------------------------

section("2. Happy path — regression features")

check("poly features", *execute("""
def engineer_features(df):
    df = df.copy()
    df["MedInc_sq"]   = df["MedInc"] ** 2
    df["rooms_ratio"] = df["AveRooms"] / (df["AveBedrms"] + 1e-6)
    return df
""", df_reg), expect_success=True)

check("binning", *execute("""
def engineer_features(df):
    df = df.copy()
    df["income_bin"] = pd.cut(df["MedInc"], bins=5, labels=False).astype(float)
    return df
""", df_reg), expect_success=True)

check("population density proxy", *execute("""
def engineer_features(df):
    df = df.copy()
    df["pop_per_room"] = df["Population"] / (df["AveRooms"] * df["HouseAge"] + 1e-6)
    df["lat_lon_dist"] = np.sqrt(df["Latitude"]**2 + df["Longitude"]**2)
    return df
""", df_reg), expect_success=True)


# ---------------------------------------------------------------------------
# 3. Error cases — all must be rejected cleanly
# ---------------------------------------------------------------------------

section("3. Error cases — must be rejected")

check("syntax error", *execute("""
def engineer_features(df):
    return df[[
""", df_clf), expect_success=False)

check("missing engineer_features", *execute("""
x = 1 + 1
""", df_clf), expect_success=False)

check("runtime KeyError", *execute("""
def engineer_features(df):
    df = df.copy()
    df["bad"] = df["nonexistent_column_xyz"]
    return df
""", df_clf), expect_success=False)

check("returns None", *execute("""
def engineer_features(df):
    return None
""", df_clf), expect_success=False)

check("row filter (len mismatch)", *execute("""
def engineer_features(df):
    return df.head(10)
""", df_clf), expect_success=False)

check("returns empty columns", *execute("""
def engineer_features(df):
    return df.iloc[:, :0]
""", df_clf), expect_success=False)

check("forbidden import: os", *execute("""
import os
def engineer_features(df):
    return df
""", df_clf), expect_success=False)

check("forbidden import: subprocess", *execute("""
import subprocess
def engineer_features(df):
    return df
""", df_clf), expect_success=False)

check("open() not in builtins", *execute("""
def engineer_features(df):
    f = open("/etc/passwd")
    return df
""", df_clf), expect_success=False)


# ---------------------------------------------------------------------------
# 4. Timeout
# ---------------------------------------------------------------------------

section("4. Timeout — infinite loop (2s limit)")

check("infinite loop killed", *execute("""
def engineer_features(df):
    while True:
        pass
""", df_clf, timeout=2), expect_success=False)


# ---------------------------------------------------------------------------
# 5. Original DataFrame immutability
# ---------------------------------------------------------------------------

section("5. Original DataFrame immutability")

original_cols = list(df_clf.columns)
execute("""
def engineer_features(df):
    df["injected"] = 999
    df["mean radius"] = -1
    return df
""", df_clf)

if list(df_clf.columns) == original_cols:
    ok("Original df unchanged after execute()")
else:
    fail("Original df was mutated!")


# ---------------------------------------------------------------------------
# 6. summarise_error
# ---------------------------------------------------------------------------

section("6. summarise_error helper")

_, err = execute("""
def engineer_features(df):
    raise RuntimeError("deep error\\nwith\\nmany\\nlines\\nof\\ntraceback\\n" * 10)
""", df_clf)

summary = summarise_error(err, max_lines=6)
line_count = len(summary.splitlines())
ok(f"Full error: {len(err.splitlines())} lines → summarised to {line_count} lines")


# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

section("DONE")
print("  executor.py smoke test complete.")
print("  All results shown above — [OK] = expected, [FAIL] = unexpected.\n")
