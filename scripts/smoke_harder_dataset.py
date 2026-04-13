"""
smoke_harder_dataset.py — Diagnostic on a dataset with real headroom for FE

Wine at 97.19% was a ceiling problem (Random Forest already near-perfect).
This script uses breast cancer (~91% baseline) where engineered features
can meaningfully contribute. If the agent improves here, the workflow works.

Run from kashif_core/:
    source .env && uv run python scripts/smoke_harder_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  DIAGNOSTIC: Dataset with real FE headroom")
print("=" * 60)

bc = load_breast_cancer(as_frame=True)
df = bc.frame.copy()
# target: 0=malignant, 1=benign — rename for clarity
df["target"] = df["target"].map({0: "malignant", 1: "benign"})

print(f"\n  Dataset       : Breast Cancer (sklearn)")
print(f"  Shape         : {df.shape}")
print(f"  All numeric   : {all(df.drop(columns=['target']).dtypes != object)}")
print(f"  Target dist   : {df['target'].value_counts().to_dict()}")

csv_path = Path("data/breast_cancer_diagnostic.csv")
csv_path.parent.mkdir(exist_ok=True)
df.to_csv(csv_path, index=False)
print(f"  Saved         : {csv_path}")

# ---------------------------------------------------------------------------
# 2. Baseline
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  STEP 1: Baseline — static pipeline, no LLM")
print("=" * 60)

from core.trainer import train

FAST_CFG = {
    "cleaning": {
        "cardinality": {"max_unique_share": 0.9},
        "variance":    {"min_threshold": 0.0},
        "nan_thresholds": {"numeric": 0.5, "categorical": 0.5},
    },
    "model_selection": {"models": {"classification": ["Random Forest"]}},
    "settings": {"cv_folds": 5, "random_state": 42},
}

baseline = train(df, "target", task_type="classification",
                 config=FAST_CFG, save_model=False, compute_shap=True)

print(f"\n  Baseline accuracy : {baseline.cv_score:.4f}")
print(f"  Baseline CV loss  : {baseline.cv_loss:.4f}")
print(f"  SHAP populated    : {len(baseline.shap_dict) > 0} ({len(baseline.shap_dict)} features)")
if baseline.shap_dict:
    top5 = sorted(baseline.shap_dict.items(), key=lambda x: x[1], reverse=True)[:5]
    clean = [(k.replace("num__", ""), v) for k, v in top5]
    print(f"  Top features      : {', '.join(f'{k}({v:.3f})' for k, v in clean)}")

# ---------------------------------------------------------------------------
# 3. LLM FE agent — 5 rounds
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  STEP 2: LLM FE agent — 5 rounds with SHAP feedback")
print("=" * 60)

from core.profiler import run as profiler_run
from core.llm.groq import GroqLLM
from core.fe_agent import run as agent_run
from core.reporter import run as reporter_run

profile_json, _ = profiler_run(df, "target",
                                output_dir="outputs/breast_cancer_diagnostic",
                                save_eda=False)
print(f"\n  Task detected : {profile_json['task_type']} (conf {profile_json['task_confidence']:.2f})")
print(f"  Features      : {profile_json['n_cols']} columns")

llm = GroqLLM(model="llama-3.3-70b-versatile", temperature=0.3)

PROGRAM_MD = """
## Goal
Predict whether a breast mass is malignant or benign from cell nucleus measurements.

## Feature groups (10 measurements, each with mean/SE/worst):
- radius, texture, perimeter, area: size/shape
- smoothness, compactness, concavity, concave_points: shape irregularity
- symmetry, fractal_dimension: structural complexity

## Good feature ideas
- Ratios: compactness/area, concavity/radius_mean, perimeter/area
- Volume approximation: (4/3) * pi * (radius_mean^3)
- Shape irregularity score: concavity_mean * concave_points_mean
- Worst-to-mean ratios for each feature (radius_worst/radius_mean, etc.)
- compactness = (perimeter^2 / area - 1.0) — the actual formula; verify against raw
- Size-normalized concavity: concavity_mean / area_mean

## Note
ALL features are numeric. Focus on ratios, normalizations, and worst-to-mean comparisons.
Malignant cells tend to be larger, less smooth, and more irregular.
"""

agent_config = {"max_rounds": 5, "stall_rounds": 5, "delta_threshold": 0.003}

log = agent_run(
    df=df,
    target_col="target",
    profile_json=profile_json,
    llm=llm,
    program_md=PROGRAM_MD,
    config=agent_config,
    train_config=FAST_CFG,
    output_dir="outputs/breast_cancer_diagnostic",
    save_model=False,
)

# ---------------------------------------------------------------------------
# 4. Report
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  RESULTS")
print("=" * 60)

for r in log:
    tag = "baseline" if r.round_num == 0 else f"round {r.round_num}"
    status = ""
    if r.round_num > 0:
        if r.executor_error:
            status = f"  ← ERROR"
        elif r.improved:
            status = f"  ← IMPROVED +{r.delta:.4f}"
        else:
            status = f"  ← no improvement ({r.delta:+.4f})"
    print(f"  {tag:12s}  cv_score={r.cv_score:.4f}{status}")

baseline_score = log[0].cv_score
best_score = max(r.cv_score for r in log)
delta = best_score - baseline_score
improved_rounds = [r for r in log[1:] if r.improved]

print(f"\n  Baseline  : {baseline_score:.4f}")
print(f"  Best      : {best_score:.4f}")
print(f"  Delta     : {delta:+.4f}")

if improved_rounds:
    best = max(improved_rounds, key=lambda r: r.cv_score)
    print(f"\n  Best round features (SHAP top-5): {best.shap_top[:5]}")
    print(f"\n  VERDICT: Agent IMPROVED on numeric data with real headroom.")
    print(f"           The workflow is sound. LLM FE adds measurable value.")
else:
    print(f"\n  VERDICT: No improvement on breast cancer either.")
    print(f"           Look at round code below — are features reasonable?")
    for r in log[1:]:
        if r.fe_code:
            lines = r.fe_code.strip().splitlines()
            print(f"\n  Round {r.round_num} ({len(lines)} lines, score={r.cv_score:.4f}):")
            for line in lines[:10]:
                print(f"    {line}")

report_md, report_path = reporter_run(log, profile_json,
                                       output_dir="outputs/breast_cancer_diagnostic",
                                       save_report=True)
if report_path:
    print(f"\n  Report saved: {report_path}")

print("\n" + "=" * 60 + "\n")
