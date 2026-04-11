"""
smoke_regression.py — trainer.py smoke test on California Housing dataset

Run from kashif_core/:
    uv run python scripts/smoke_regression.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from sklearn.datasets import fetch_california_housing

from core.trainer import train

# ---------------------------------------------------------------------------
# Load dataset
# ---------------------------------------------------------------------------
print("=" * 60)
print("KASHIF — Regression Smoke Test")
print("Dataset: California Housing (sklearn built-in)")
print("=" * 60)

data = fetch_california_housing(as_frame=True)
df = data.frame

print(f"\nShape       : {df.shape}")
print(f"Target      : 'MedHouseVal'  (median house value in $100k)")
print(f"Features    : {df.shape[1] - 1} numeric columns")
print(f"Target range: {df['MedHouseVal'].min():.2f} – {df['MedHouseVal'].max():.2f}")
print(f"Nulls       : {df.isnull().sum().sum()}")

# ---------------------------------------------------------------------------
# Run trainer
# ---------------------------------------------------------------------------
print("\nRunning pipeline...\n")

result = train(
    df,
    target_col="MedHouseVal",
    task_type="regression",
    save_model=False,
    compute_shap=False,
)

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print("=" * 60)
print("LEADERBOARD")
print("=" * 60)
print(f"{'Model':<28} {'RMSE (loss)':<16} {'Std':<10} {'MAE':<10} {'R2':<10} {'Time(s)'}")
print("-" * 60)
for row in result.leaderboard:
    print(
        f"{row['model']:<28} "
        f"{row['cv_loss_mean']:<16.4f} "
        f"{row['cv_loss_std']:<10.4f} "
        f"{row.get('mae', 0):<10.4f} "
        f"{row.get('r2', 0):<10.4f} "
        f"{row['time_s']}"
    )

print()
print("=" * 60)
print("BEST RESULT")
print("=" * 60)
print(f"Model       : {result.best_model_name}")
print(f"CV RMSE     : {result.cv_score:.4f}")
print(f"CV Loss     : {result.cv_loss:.4f}")
print(f"Status      : {result.status}")
