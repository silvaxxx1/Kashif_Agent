"""
smoke_classification.py — trainer.py smoke test on breast cancer dataset

Run from kashif_core/:
    uv run python scripts/smoke_classification.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.datasets import load_breast_cancer

from core.trainer import train

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("KASHIF — Classification Smoke Test")
print("Dataset: Breast Cancer Wisconsin (sklearn built-in)")
print("=" * 60)

data = load_breast_cancer(as_frame=True)
df = data.frame
df["target"] = df["target"].map({0: "malignant", 1: "benign"})

print(f"\nShape       : {df.shape}")
print(f"Target      : 'target'  ({df['target'].value_counts().to_dict()})")
print(f"Features    : {df.shape[1] - 1} numeric columns")
print(f"Nulls       : {df.isnull().sum().sum()}")

# ---------------------------------------------------------------------------
# Run trainer
# ---------------------------------------------------------------------------
print("\nRunning pipeline...\n")

result = train(
    df,
    target_col="target",
    task_type="classification",
    save_model=False,
    compute_shap=False,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("=" * 60)
print("LEADERBOARD")
print("=" * 60)
print(f"{'Model':<28} {'Loss (1-acc)':<16} {'Std':<10} {'Accuracy':<10} {'F1':<10} {'Time(s)'}")
print("-" * 60)
for row in result.leaderboard:
    print(
        f"{row['model']:<28} "
        f"{row['cv_loss_mean']:<16.4f} "
        f"{row['cv_loss_std']:<10.4f} "
        f"{row.get('accuracy', 0):<10.4f} "
        f"{row.get('f1', 0):<10.4f} "
        f"{row['time_s']}"
    )

print()
print("=" * 60)
print("BEST RESULT")
print("=" * 60)
print(f"Model       : {result.best_model_name}")
print(f"CV Accuracy : {result.cv_score:.4f}")
print(f"CV Loss     : {result.cv_loss:.4f}")
print(f"Status      : {result.status}")
