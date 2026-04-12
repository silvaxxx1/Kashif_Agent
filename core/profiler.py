"""
profiler.py — Data profiling + task detection + EDA report (Step 4b)

Sources:
  - TaskDetector: AutoML/src/data_processing/task_detector.py (verbatim logic,
    Config dependency removed, plain params used instead)
  - Profiling wrapper + EDA: written from scratch

Public API:
  profile(df, target_col)  →  profile_json dict
  eda(df, target_col, output_dir)  →  saves outputs/eda_report.html, returns path
  run(df, target_col, output_dir, save_eda)  →  (profile_json, eda_path | None)
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TASK DETECTOR — verbatim logic from AutoML/src/data_processing/task_detector.py
# (removed: sys.path hacks, Config import, logging.info noise)
# (changed: constructor takes plain thresholds instead of Config object)
# ---------------------------------------------------------------------------

class TaskDetector:
    """
    Detect classification vs regression from target variable.
    Six-rule weighted voting with confidence score.
    Ported from AutoML/src/data_processing/task_detector.py.
    """

    def __init__(
        self,
        unique_ratio_threshold: float = 0.05,
        max_unique_for_classification: int = 20,
    ):
        self.unique_ratio_threshold = unique_ratio_threshold
        self.max_unique_for_classification = max_unique_for_classification

    def detect(self, y: pd.Series) -> Tuple[str, float]:
        """Return (task_type, confidence) — 'classification' or 'regression'."""
        rules = [
            self._check_data_type(y),
            self._check_binary(y),
            self._check_unique_count(y),
            self._check_unique_ratio(y),
            self._check_integer_pattern(y),
            self._check_continuous_distribution(y),
        ]
        valid = [r for r in rules if r is not None]
        if not valid:
            return "classification", 0.5
        return self._aggregate(valid)

    def get_detection_details(self, y: pd.Series) -> Dict[str, Any]:
        task_type, confidence = self.detect(y)
        n_total = len(y.dropna())
        return {
            "task_type": task_type,
            "confidence": round(confidence, 4),
            "n_samples": len(y),
            "n_unique": int(y.nunique()),
            "unique_ratio": round(y.nunique() / n_total, 4) if n_total > 0 else 0,
            "dtype": str(y.dtype),
            "null_count": int(y.isnull().sum()),
            "null_rate": round(y.isnull().mean(), 4),
            "sample_values": y.dropna().head(10).tolist(),
        }

    # --- six rules (verbatim logic) ---

    def _check_data_type(self, y: pd.Series) -> Optional[Tuple[str, float]]:
        if y.dtype == "object" or y.dtype.name == "category":
            return "classification", 0.95
        if y.dtype == "bool":
            return "classification", 0.99
        return None

    def _check_binary(self, y: pd.Series) -> Optional[Tuple[str, float]]:
        unique = set(y.dropna().unique())
        if unique.issubset({0, 1}) or unique.issubset({0.0, 1.0}):
            return "classification", 0.99
        if unique.issubset({True, False}):
            return "classification", 0.99
        if len(unique) == 2:
            return "classification", 0.90
        return None

    def _check_unique_count(self, y: pd.Series) -> Optional[Tuple[str, float]]:
        n_unique = y.nunique()
        n_total = len(y.dropna())
        if n_unique <= 10 and n_total > 20 and n_unique / n_total < 0.3:
            return "classification", 0.95
        if n_unique <= self.max_unique_for_classification:
            return "classification", 0.80
        if n_unique > self.max_unique_for_classification:
            return "regression", 0.75
        return None

    def _check_unique_ratio(self, y: pd.Series) -> Optional[Tuple[str, float]]:
        n_total = len(y.dropna())
        if n_total == 0:
            return None
        ratio = y.nunique() / n_total
        if ratio < self.unique_ratio_threshold:
            return "classification", 0.85
        if ratio > 0.5:
            return "regression", 0.80
        return None

    def _check_integer_pattern(self, y: pd.Series) -> Optional[Tuple[str, float]]:
        if y.dtype not in ["int64", "int32", "int16", "int8"]:
            return None
        n_unique = y.nunique()
        if n_unique < 20:
            return "classification", 0.70
        y_sorted = sorted(y.dropna().unique())
        if len(y_sorted) > 1 and np.all(np.diff(y_sorted) == 1):
            return None  # sequential — inconclusive
        return None

    def _check_continuous_distribution(self, y: pd.Series) -> Optional[Tuple[str, float]]:
        if y.dtype not in ["float64", "float32", "int64", "int32"]:
            return None
        n_total = len(y.dropna())
        if n_total == 0:
            return None
        ratio = y.nunique() / n_total
        if ratio > 0.8:
            return "regression", 0.95
        if y.dtype in ["int64", "int32"] and ratio > 0.5:
            return "regression", 0.90
        return None

    def _aggregate(self, rules: List[Tuple[str, float]]) -> Tuple[str, float]:
        clf = [c for t, c in rules if t == "classification"]
        reg = [c for t, c in rules if t == "regression"]
        clf_score = sum(clf)
        reg_score = sum(reg)
        if clf_score >= reg_score:
            return "classification", float(np.mean(clf)) if clf else 0.5
        return "regression", float(np.mean(reg)) if reg else 0.5


# ---------------------------------------------------------------------------
# PROFILING — written from scratch
# ---------------------------------------------------------------------------

def _col_stats(series: pd.Series) -> Dict[str, Any]:
    """Compute per-column statistics."""
    is_numeric = pd.api.types.is_numeric_dtype(series)
    stats: Dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(series.isnull().sum()),
        "null_rate": round(float(series.isnull().mean()), 4),
        "cardinality": int(series.nunique()),
        "cardinality_rate": round(float(series.nunique() / len(series)) if len(series) > 0 else 0, 4),
        "is_numeric": is_numeric,
    }
    if is_numeric:
        stats.update({
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "min": round(float(series.min()), 4),
            "max": round(float(series.max()), 4),
            "median": round(float(series.median()), 4),
            "skew": round(float(series.skew()), 4),
            "top_values": None,
        })
    else:
        top = series.value_counts().head(5).to_dict()
        stats["top_values"] = {str(k): int(v) for k, v in top.items()}
    return stats


def _target_stats(series: pd.Series, task_type: str) -> Dict[str, Any]:
    """Compute target-specific statistics."""
    stats: Dict[str, Any] = {
        "dtype": str(series.dtype),
        "null_count": int(series.isnull().sum()),
        "null_rate": round(float(series.isnull().mean()), 4),
        "cardinality": int(series.nunique()),
    }
    if task_type == "classification":
        dist = series.value_counts().to_dict()
        dist = {str(k): int(v) for k, v in dist.items()}
        counts = list(dist.values())
        imbalance_ratio = round(max(counts) / min(counts), 4) if counts and min(counts) > 0 else None
        stats["distribution"] = dist
        stats["imbalance_ratio"] = imbalance_ratio
        stats["is_imbalanced"] = imbalance_ratio is not None and imbalance_ratio > 3.0
    else:
        stats["mean"] = round(float(series.mean()), 4)
        stats["std"] = round(float(series.std()), 4)
        stats["min"] = round(float(series.min()), 4)
        stats["max"] = round(float(series.max()), 4)
        stats["skew"] = round(float(series.skew()), 4)
        stats["is_imbalanced"] = False   # not applicable for regression
    return stats


def _build_warnings(df: pd.DataFrame, target_col: str, profile: Dict) -> List[str]:
    """Generate human-readable warnings from profile data."""
    warnings: List[str] = []
    for col, stats in profile["columns"].items():
        if stats["null_rate"] > 0.3:
            warnings.append(
                f"'{col}' has {stats['null_rate']*100:.1f}% missing values — may be dropped by cleaning"
            )
        if stats["cardinality_rate"] > 0.9 and not stats["is_numeric"]:
            warnings.append(
                f"'{col}' has {stats['cardinality_rate']*100:.1f}% unique values — likely an ID column, will be dropped"
            )
    if profile["target"].get("is_imbalanced"):
        ratio = profile["target"]["imbalance_ratio"]
        warnings.append(
            f"Target '{target_col}' is imbalanced (ratio {ratio:.1f}x) — consider class weighting"
        )
    return warnings


def profile(df: pd.DataFrame, target_col: str) -> Dict[str, Any]:
    """
    Profile a DataFrame and return a structured dict.

    Parameters
    ----------
    df : pd.DataFrame
        Full input DataFrame including target column.
    target_col : str
        Name of the target column.

    Returns
    -------
    dict with keys:
        n_rows, n_cols, target_col, task_type, task_confidence,
        memory_mb, columns (per-col stats), target (target stats), warnings
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")

    detector = TaskDetector()
    task_type, confidence = detector.detect(df[target_col])

    feature_cols = [c for c in df.columns if c != target_col]

    col_stats: Dict[str, Any] = {}
    for col in feature_cols:
        col_stats[col] = _col_stats(df[col])

    prof: Dict[str, Any] = {
        "n_rows": int(len(df)),
        "n_cols": int(len(feature_cols)),
        "target_col": target_col,
        "task_type": task_type,
        "task_confidence": round(confidence, 4),
        "memory_mb": round(float(df.memory_usage(deep=True).sum() / 1024 ** 2), 3),
        "columns": col_stats,
        "target": _target_stats(df[target_col], task_type),
        "warnings": [],
    }
    prof["warnings"] = _build_warnings(df, target_col, prof)

    logger.info(
        "profiler: %d rows × %d cols | task=%s (conf=%.2f) | warnings=%d",
        prof["n_rows"], prof["n_cols"], task_type, confidence, len(prof["warnings"])
    )
    return prof


# ---------------------------------------------------------------------------
# EDA REPORT — written from scratch
# ---------------------------------------------------------------------------

def _fig_to_b64(fig: plt.Figure) -> str:
    """Convert a matplotlib figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def _chart_target(series: pd.Series, task_type: str, target_col: str) -> str:
    fig, ax = plt.subplots(figsize=(6, 4))
    if task_type == "classification":
        counts = series.value_counts()
        ax.bar(counts.index.astype(str), counts.values, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Target distribution — {target_col}")
        ax.set_xlabel("Class")
        ax.set_ylabel("Count")
    else:
        ax.hist(series.dropna(), bins=40, color="#4C72B0", edgecolor="white")
        ax.set_title(f"Target distribution — {target_col}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_null_rates(col_stats: Dict[str, Any]) -> str:
    cols = [c for c, s in col_stats.items() if s["null_rate"] > 0]
    if not cols:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center", fontsize=14)
        ax.axis("off")
        return _fig_to_b64(fig)

    rates = [col_stats[c]["null_rate"] * 100 for c in cols]
    fig, ax = plt.subplots(figsize=(max(5, len(cols) * 0.6), 4))
    ax.barh(cols, rates, color="#DD8452", edgecolor="white")
    ax.set_xlabel("Missing (%)")
    ax.set_title("Missing value rates")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_correlation(df: pd.DataFrame, target_col: str) -> str:
    numeric = df.select_dtypes(include="number")
    if len(numeric.columns) < 2:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "Not enough numeric columns", ha="center", va="center")
        ax.axis("off")
        return _fig_to_b64(fig)

    corr = numeric.corr()
    n = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.5), max(5, n * 0.45)))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(corr.columns, fontsize=7)
    plt.colorbar(im, ax=ax)
    ax.set_title("Feature correlation matrix")
    fig.tight_layout()
    return _fig_to_b64(fig)


def _chart_feature_distributions(df: pd.DataFrame, target_col: str, max_cols: int = 12) -> str:
    numeric_cols = [c for c in df.select_dtypes(include="number").columns if c != target_col][:max_cols]
    if not numeric_cols:
        fig, ax = plt.subplots(figsize=(4, 2))
        ax.text(0.5, 0.5, "No numeric features", ha="center", va="center")
        ax.axis("off")
        return _fig_to_b64(fig)

    n = len(numeric_cols)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
    axes = np.array(axes).flatten() if n > 1 else [axes]

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=30, color="#55A868", edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=8)
        axes[i].tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Numeric feature distributions", fontsize=10, y=1.01)
    fig.tight_layout()
    return _fig_to_b64(fig)


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Kashif EDA — {target_col}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: #0f1117; color: #e0e0e0; margin: 0; padding: 24px; }}
    h1   {{ color: #ffffff; font-size: 1.6rem; margin-bottom: 4px; }}
    h2   {{ color: #a0a8b8; font-size: 1.1rem; margin: 32px 0 12px; border-bottom: 1px solid #2a2d3a; padding-bottom: 6px; }}
    .sub {{ color: #6b7280; font-size: 0.85rem; margin-bottom: 28px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; margin-bottom: 24px; }}
    .card {{ background: #1a1d26; border-radius: 8px; padding: 16px 20px; }}
    .card .label {{ font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
    .card .value {{ font-size: 1.5rem; font-weight: 600; color: #ffffff; margin-top: 4px; }}
    .card .sub    {{ font-size: 0.8rem; color: #4C72B0; margin-top: 2px; }}
    .warn  {{ background: #2a1f0e; border-left: 3px solid #f59e0b; padding: 10px 14px;
              border-radius: 4px; font-size: 0.85rem; color: #fbbf24; margin: 6px 0; }}
    .chart {{ background: #1a1d26; border-radius: 8px; padding: 16px; text-align: center; margin-bottom: 20px; }}
    .chart img {{ max-width: 100%; border-radius: 4px; }}
    .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
    @media (max-width: 700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>

<h1>Kashif &mdash; EDA Report</h1>
<p class="sub">Dataset: <strong>{csv_name}</strong> &nbsp;|&nbsp; Target: <strong>{target_col}</strong> &nbsp;|&nbsp; Task: <strong>{task_type}</strong> (confidence {task_conf})</p>

<div class="grid">
  <div class="card"><div class="label">Rows</div><div class="value">{n_rows:,}</div></div>
  <div class="card"><div class="label">Features</div><div class="value">{n_cols}</div></div>
  <div class="card"><div class="label">Memory</div><div class="value">{memory_mb} MB</div></div>
  <div class="card"><div class="label">Total nulls</div><div class="value">{total_nulls:,}</div><div class="sub">{null_pct}% of all cells</div></div>
</div>

{warnings_html}

<h2>Target distribution</h2>
<div class="chart"><img src="data:image/png;base64,{chart_target}" /></div>

<div class="two-col">
  <div>
    <h2>Missing values</h2>
    <div class="chart"><img src="data:image/png;base64,{chart_nulls}" /></div>
  </div>
  <div>
    <h2>Correlation matrix</h2>
    <div class="chart"><img src="data:image/png;base64,{chart_corr}" /></div>
  </div>
</div>

<h2>Feature distributions</h2>
<div class="chart"><img src="data:image/png;base64,{chart_dists}" /></div>

</body>
</html>
"""


def eda(
    df: pd.DataFrame,
    target_col: str,
    profile_json: Optional[Dict[str, Any]] = None,
    output_dir: str = "./outputs",
    csv_name: str = "dataset",
) -> str:
    """
    Generate a self-contained EDA HTML report.

    Parameters
    ----------
    df : pd.DataFrame
    target_col : str
    profile_json : dict, optional
        Pre-computed profile. If None, profile() is called internally.
    output_dir : str
        Where to save eda_report.html.
    csv_name : str
        Display name of the source file (shown in the report header).

    Returns
    -------
    str — absolute path to the saved eda_report.html
    """
    if profile_json is None:
        profile_json = profile(df, target_col)

    col_stats = profile_json["columns"]
    total_nulls = sum(s["null_count"] for s in col_stats.values())
    total_cells = profile_json["n_rows"] * profile_json["n_cols"]
    null_pct = round(total_nulls / total_cells * 100, 1) if total_cells > 0 else 0

    warnings_html = ""
    for w in profile_json["warnings"]:
        warnings_html += f'<div class="warn">&#9888; {w}</div>\n'

    html = _HTML_TEMPLATE.format(
        csv_name=csv_name,
        target_col=target_col,
        task_type=profile_json["task_type"],
        task_conf=f"{profile_json['task_confidence']:.0%}",
        n_rows=profile_json["n_rows"],
        n_cols=profile_json["n_cols"],
        memory_mb=profile_json["memory_mb"],
        total_nulls=total_nulls,
        null_pct=null_pct,
        warnings_html=warnings_html,
        chart_target=_chart_target(df[target_col], profile_json["task_type"], target_col),
        chart_nulls=_chart_null_rates(col_stats),
        chart_corr=_chart_correlation(df, target_col),
        chart_dists=_chart_feature_distributions(df, target_col),
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "eda_report.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("profiler: EDA report saved → %s", out_path)
    return out_path


def run(
    df: pd.DataFrame,
    target_col: str,
    output_dir: str = "./outputs",
    save_eda: bool = True,
    csv_name: str = "dataset",
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Full profiler entry point: profile + optional EDA.

    Returns
    -------
    (profile_json, eda_path | None)
    """
    prof = profile(df, target_col)
    eda_path = None
    if save_eda:
        eda_path = eda(df, target_col, profile_json=prof,
                       output_dir=output_dir, csv_name=csv_name)
    return prof, eda_path
