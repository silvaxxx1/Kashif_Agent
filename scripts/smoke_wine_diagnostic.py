"""
smoke_wine_diagnostic.py — Workflow diagnostic on a pure-numeric dataset

Purpose: Determine whether the zero-improvement problem on Titanic is
a DATA problem (pipeline OHE pre-empts LLM features) or a WORKFLOW problem
(the agent loop itself doesn't work).

Dataset: sklearn wine (178 rows, 13 numeric features, 3 classes → binarized)
  - All numeric — pipeline OHE does NOTHING to these features
  - LLM-generated features cannot be pre-empted by preprocessing
  - If the agent still fails to improve, the workflow itself is the problem
  - If the agent improves, Titanic failure was a data problem

Run from kashif_core/:
    uv run python scripts/smoke_wine_diagnostic.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine

# ---------------------------------------------------------------------------
# 1. Load and prepare dataset
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  DIAGNOSTIC: Does the agent work on a numeric-only dataset?")
print("=" * 60)

wine = load_wine(as_frame=True)
df = wine.frame.copy()

# Binarize: class 0 (high-quality) vs classes 1+2 (other)
df["target"] = (df["target"] == 0).astype(str).map({"True": "premium", "False": "other"})

print(f"\n  Dataset       : Wine (sklearn) — binarized")
print(f"  Shape         : {df.shape}")
print(f"  Features      : {list(df.columns[:-1])}")
print(f"  All numeric   : {all(df.drop(columns=['target']).dtypes != object)}")
print(f"  Target dist   : {df['target'].value_counts().to_dict()}")

# Save CSV
csv_path = Path("data/wine_diagnostic.csv")
df.to_csv(csv_path, index=False)
print(f"  Saved         : {csv_path}")

# ---------------------------------------------------------------------------
# 2. Baseline (no agent)
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
    print(f"  Top features      : {', '.join(f'{k.replace('num__','')}({v:.3f})' for k,v in top5)}")

# ---------------------------------------------------------------------------
# 3. LLM FE agent — 3 rounds with full SHAP feedback
# ---------------------------------------------------------------------------

print("\n" + "=" * 60)
print("  STEP 2: LLM FE agent — 3 rounds with SHAP feedback")
print("=" * 60)

from core.profiler import run as profiler_run
from core.llm.groq import GroqLLM
from core.fe_agent import run as agent_run
from core.reporter import run as reporter_run

profile_json, _ = profiler_run(df, "target", output_dir="outputs/wine_diagnostic", save_eda=False)
print(f"\n  Task detected : {profile_json['task_type']} (conf {profile_json['task_confidence']:.2f})")

llm = GroqLLM(model="llama-3.3-70b-versatile", temperature=0.3)

# Program.md hint for wine domain
PROGRAM_MD = """
## Goal
Predict whether a wine is premium class (class 0) or other (classes 1 or 2).

## Column notes
- alcohol: % alcohol by volume — higher tends to indicate premium
- malic_acid: secondary acid — affects taste profile
- ash / alcalinity_of_ash: mineral content
- magnesium: mineral content
- total_phenols / flavanoids: antioxidant compounds — high in premium wines
- nonflavanoid_phenols: lowers quality when high
- proanthocyanins: polyphenol compound — linked to quality
- color_intensity: visual quality indicator
- hue: color ratio — differs strongly by class
- od280_od315_of_diluted_wines: protein content proxy
- proline: amino acid — very high in premium class

## Good feature ideas
- ratios: flavanoids/nonflavanoid_phenols, total_phenols/ash, alcohol/malic_acid
- interactions: alcohol * proline, flavanoids * total_phenols
- compound quality score: weighted combination of top features
- log transforms on skewed features (proline, color_intensity)

## Note
ALL features are numeric. Do NOT create categorical features.
The pipeline will NOT modify your features — what you output is what the model sees.
"""

agent_config = {"max_rounds": 3, "stall_rounds": 3, "delta_threshold": 0.005}

log = agent_run(
    df=df,
    target_col="target",
    profile_json=profile_json,
    llm=llm,
    program_md=PROGRAM_MD,
    config=agent_config,
    train_config=FAST_CFG,
    output_dir="outputs/wine_diagnostic",
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
            status = f"  ← ERROR: {r.executor_error[:50]}"
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
print(f"  Delta     : {delta:+.4f}  {'↑ AGENT IMPROVED — workflow works' if delta > 0.005 else '→ no improvement — deeper problem'}")

if improved_rounds:
    best = max(improved_rounds, key=lambda r: r.cv_score)
    print(f"\n  Best FE features: {best.shap_top[:5]}")
    print(f"\n  VERDICT: Titanic failure was a DATA problem (OHE interference).")
    print(f"           The agent workflow works correctly on numeric data.")
else:
    print(f"\n  VERDICT: Agent did not improve even on pure-numeric data.")
    print(f"           This points to a WORKFLOW problem (prompt, parsing, or model).")
    for r in log[1:]:
        if r.fe_code:
            lines = r.fe_code.strip().splitlines()
            print(f"\n  Round {r.round_num} code preview ({len(lines)} lines):")
            for line in lines[:8]:
                print(f"    {line}")
        elif r.executor_error:
            print(f"\n  Round {r.round_num}: {r.executor_error}")

# Save report
report_md, report_path = reporter_run(log, profile_json,
                                       output_dir="outputs/wine_diagnostic",
                                       save_report=True)
if report_path:
    print(f"\n  Report saved: {report_path}")

print("\n" + "=" * 60 + "\n")
