"""
smoke_profiler.py — profiler.py smoke test on real datasets

Runs profile() + eda() on both a classification and a regression dataset.
Opens the EDA HTML report path so you can inspect it in a browser.

Run from kashif_core/:
    uv run python scripts/smoke_profiler.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing

from core.profiler import run

OUTPUT_DIR = "./outputs"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def print_profile(prof: dict) -> None:
    print(f"\n  Shape         {prof['n_rows']:,} rows × {prof['n_cols']} features")
    print(f"  Task          {prof['task_type']}  (confidence {prof['task_confidence']:.0%})")
    print(f"  Memory        {prof['memory_mb']} MB")

    # target
    t = prof["target"]
    if prof["task_type"] == "classification":
        dist = "  ".join(f"{k}: {v}" for k, v in t["distribution"].items())
        print(f"  Target        {prof['target_col']}  →  {dist}")
        if t["is_imbalanced"]:
            print(f"  ⚠ Imbalanced  ratio {t['imbalance_ratio']}x")
    else:
        print(
            f"  Target        {prof['target_col']}  "
            f"mean={t['mean']}  std={t['std']}  "
            f"min={t['min']}  max={t['max']}"
        )

    # nulls
    null_cols = {c: s for c, s in prof["columns"].items() if s["null_rate"] > 0}
    if null_cols:
        print(f"  Nulls         " + "  ".join(
            f"{c}: {s['null_rate']*100:.1f}%" for c, s in null_cols.items()
        ))
    else:
        print("  Nulls         none")

    # top skewed features
    skewed = sorted(
        [(c, s["skew"]) for c, s in prof["columns"].items()
         if s["is_numeric"] and abs(s.get("skew", 0)) > 1.0],
        key=lambda x: abs(x[1]), reverse=True
    )[:5]
    if skewed:
        print("  High skew     " + "  ".join(f"{c}: {sk:.2f}" for c, sk in skewed))

    # warnings
    if prof["warnings"]:
        for w in prof["warnings"]:
            print(f"  ⚠ {w}")


# ---------------------------------------------------------------------------
# classification — breast cancer
# ---------------------------------------------------------------------------

print("=" * 60)
print("KASHIF — Profiler Smoke Test")
print("=" * 60)

print("\n[1/2] Breast Cancer Wisconsin  (classification)")
print("-" * 60)

data = load_breast_cancer(as_frame=True)
df_clf = data.frame
df_clf["target"] = df_clf["target"].map({0: "malignant", 1: "benign"})

prof_clf, eda_clf = run(
    df_clf,
    target_col="target",
    output_dir=OUTPUT_DIR,
    save_eda=True,
    csv_name="breast_cancer.csv",
)

print_profile(prof_clf)
print(f"\n  EDA report    → {eda_clf}")

# ---------------------------------------------------------------------------
# regression — california housing
# ---------------------------------------------------------------------------

print("\n[2/2] California Housing  (regression)")
print("-" * 60)

data2 = fetch_california_housing(as_frame=True)
df_reg = data2.frame

# rename EDA output so it doesn't overwrite classification report
prof_reg, eda_reg = run(
    df_reg,
    target_col="MedHouseVal",
    output_dir=OUTPUT_DIR,
    save_eda=True,
    csv_name="california_housing.csv",
)

# save regression report under a separate name
import shutil, os
reg_path = os.path.join(OUTPUT_DIR, "eda_report_regression.html")
shutil.move(eda_reg, reg_path)

print_profile(prof_reg)
print(f"\n  EDA report    → {reg_path}")

# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"\n  Classification EDA  →  {eda_clf}")
print(f"  Regression EDA      →  {reg_path}")
print("\n  Open either file in a browser to inspect the full report.")
