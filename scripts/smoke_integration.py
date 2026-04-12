"""
smoke_integration.py — Live end-to-end integration test

Wires all 4 completed modules together with a real Groq API call:

  profiler  →  profile dataset
  GroqLLM   →  generate engineer_features(df) code from profile
  executor  →  run the LLM code safely
  trainer   →  score baseline vs FE-enhanced pipeline

Dataset: Titanic (OpenML) — 1309 rows, mixed types, missing values,
         categorical features. Much more challenging than Breast Cancer.

This is a manual preview of what fe_agent.py will automate in step 4e.

Run from kashif_core/:
    uv run python scripts/smoke_integration.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.base import BaseEstimator, TransformerMixin

from core.profiler import profile
from core.llm.groq import GroqLLM
from core.llm.base import LLMError
from core.executor import execute, summarise_error
from core.trainer import train

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TARGET    = "survived"
MODEL_CFG = {
    "cleaning":        {"cardinality": {"max_unique_share": 0.5},
                        "variance":    {"min_threshold": 0.0},
                        "nan_thresholds": {"numeric": 0.6, "categorical": 0.6}},
    "model_selection": {"models": {"classification": ["Random Forest"]}},
    "settings":        {"cv_folds": 3, "random_state": 42},
}

def sep(title=""):
    print(f"\n{'=' * 60}")
    if title:
        print(f"  {title}")
        print('=' * 60)

# ---------------------------------------------------------------------------
# 1. Load dataset
# ---------------------------------------------------------------------------

sep("1. Dataset — Titanic (OpenML)")

raw = fetch_openml("titanic", version=1, as_frame=True).frame

# Drop leakage columns (known only after survival outcome)
# boat  = lifeboat number — only survivors have this
# body  = body recovery number — only fatalities have this
df = raw.drop(columns=["boat", "body"], errors="ignore")

# Convert target to string labels
df[TARGET] = df[TARGET].astype(str).map({"0": "died", "1": "survived"})

print(f"  Shape       {df.shape}")
print(f"  Target      {df[TARGET].value_counts().to_dict()}")
nulls = df.isnull().sum()
nulls = nulls[nulls > 0]
print(f"  Missing     {dict(nulls)}")
print(f"  Dtypes      {df.dtypes.value_counts().to_dict()}")

# ---------------------------------------------------------------------------
# 2. Profile
# ---------------------------------------------------------------------------

sep("2. Profile")
prof = profile(df, TARGET)
print(f"  Task        {prof['task_type']}  (conf {prof['task_confidence']:.0%})")
print(f"  Rows/cols   {prof['n_rows']} × {prof['n_cols']}")
for w in prof["warnings"]:
    print(f"  ⚠ {w}")

# Build column summary for the prompt
col_lines = []
for col, s in prof["columns"].items():
    if s["is_numeric"]:
        col_lines.append(f"  {col}: numeric, skew={s['skew']}, null_rate={s['null_rate']}")
    else:
        top = list(s.get("top_values", {}).keys())[:3]
        col_lines.append(f"  {col}: categorical, top={top}, null_rate={s['null_rate']}")

# ---------------------------------------------------------------------------
# 3. Baseline score (no FE)
# ---------------------------------------------------------------------------

sep("3. Baseline — no feature engineering")
baseline = train(df, TARGET, config=MODEL_CFG, save_model=False, compute_shap=False)
print(f"  Model     {baseline.best_model_name}")
print(f"  Accuracy  {baseline.cv_score:.4f}")
print(f"  CV loss   {baseline.cv_loss:.4f}")

# ---------------------------------------------------------------------------
# Determine which columns survive cleaning (FE runs AFTER cleaning)
# ---------------------------------------------------------------------------

from core.trainer import CardinalityStripper, VarianceStripper, UniversalDropper
from sklearn.pipeline import Pipeline as SkPipeline

_cleaning_pipe = SkPipeline([
    ("cardinality", CardinalityStripper(threshold=MODEL_CFG["cleaning"]["cardinality"]["max_unique_share"])),
    ("variance",    VarianceStripper(min_threshold=0.0)),
    ("nan_drop",    UniversalDropper(thresholds=MODEL_CFG["cleaning"]["nan_thresholds"])),
])
X_all = df.drop(columns=[TARGET])
_cleaning_pipe.fit(X_all)
surviving_cols = list(_cleaning_pipe.transform(X_all).columns)
print(f"\n  Columns surviving cleaning ({len(surviving_cols)}): {surviving_cols}")

# Build column summary for surviving columns only
col_lines_clean = []
for col in surviving_cols:
    s = prof["columns"].get(col, {})
    if s.get("is_numeric"):
        col_lines_clean.append(f"  {col}: numeric, skew={s['skew']}, null_rate={s['null_rate']}")
    else:
        top = list(s.get("top_values", {}).keys())[:3]
        col_lines_clean.append(f"  {col}: categorical, top={top}, null_rate={s['null_rate']}")

# ---------------------------------------------------------------------------
# 4. Build the prompt and call Groq
# ---------------------------------------------------------------------------

sep("4. Groq — generating feature engineering code")

PROMPT = f"""You are a senior data scientist writing feature engineering code for the Titanic survival dataset.

IMPORTANT: Feature engineering runs AFTER the cleaning pipeline.
You only have access to these {len(surviving_cols)} columns (others were dropped by cleaning):
{chr(10).join(col_lines_clean)}

CURRENT BASELINE:
- Model: Random Forest, CV accuracy: {baseline.cv_score:.4f}

DOMAIN KNOWLEDGE:
- pclass: passenger class (1=first, 2=second, 3=third)
- sex: male/female
- age: passenger age — has missing values, fill with median
- sibsp: number of siblings/spouses aboard
- parch: number of parents/children aboard
- fare: ticket price
- embarked: port of embarkation (C=Cherbourg, Q=Queenstown, S=Southampton)

GOOD FEATURE IDEAS:
- family_size = sibsp + parch + 1
- is_alone = (family_size == 1)
- fare_per_person = fare / family_size
- log_fare = log(fare + 1)
- age_group bins (child/adult/senior)
- pclass * fare interaction
- age * pclass interaction

TASK:
Write a Python function called `engineer_features(df)` that creates new features.

RULES:
- Signature must be exactly: def engineer_features(df):
- Must return a DataFrame with the same number of rows as input
- Do NOT drop or rename existing columns — only add new ones
- Use only: pandas (pd), numpy (np), math — NO imports needed, they are pre-loaded
- Handle missing values safely — use fillna(), never dropna()
- Safe division only — always add 1e-6 or use np.where()

Return ONLY the Python code. No explanation. No imports. No markdown.
"""

llm = GroqLLM(model="llama-3.3-70b-versatile", temperature=0.2)

try:
    fe_code = llm.complete(PROMPT)
    # Strip markdown fences and redundant imports — LLMs add these habitually
    # pd, np, math, re are pre-injected into the executor namespace
    _STRIP_PREFIXES = (
        "```", "import pandas", "import numpy",
        "import math", "import re", "from pandas", "from numpy",
    )
    lines = fe_code.splitlines()
    lines = [l for l in lines if not any(l.strip().startswith(p) for p in _STRIP_PREFIXES)]
    fe_code = "\n".join(lines).strip()
    print(f"  Response length: {len(fe_code)} chars")
    print(f"\n--- Generated code ---")
    print(fe_code)
    print("--- End of code ---")
except LLMError as e:
    print(f"  [ERROR] LLM call failed: {e}")
    sys.exit(1)

# ---------------------------------------------------------------------------
# 5. Execute the LLM code — with one retry on failure
# ---------------------------------------------------------------------------

sep("5. Executor — running LLM code")

def clean_code(raw: str) -> str:
    """Strip markdown fences and pre-injected imports."""
    _STRIP = ("```", "import pandas", "import numpy",
              "import math", "import re", "from pandas", "from numpy")
    lines = [l for l in raw.splitlines()
             if not any(l.strip().startswith(p) for p in _STRIP)]
    return "\n".join(lines).strip()

X = df.drop(columns=[TARGET])
result, err = execute(fe_code, X)

if err:
    print(f"  [FAIL] Round 1 error — sending back to LLM for correction:")
    print(f"  {summarise_error(err, max_lines=6)}")

    fix_prompt = f"""The following feature engineering code failed with this error:

ERROR:
{summarise_error(err, max_lines=10)}

ORIGINAL CODE:
{fe_code}

Fix the code so it handles all edge cases safely:
- Use `.apply(lambda x: ... if pd.notna(x) else default)` for nullable columns
- Use `re.search(pattern, x).group() if re.search(pattern, x) else 'Unknown'`
- Safe division: always add 1e-6 or use np.where
- Do NOT add imports — pd, np, re, math are already available

Return ONLY the corrected Python code, no explanation, no markdown fences.
"""
    sep("5b. Groq — fixing the code")
    try:
        fe_code = clean_code(llm.complete(fix_prompt))
        print(f"  Fixed code length: {len(fe_code)} chars")
        print(f"\n--- Fixed code ---")
        print(fe_code)
        print("--- End ---")
        result, err = execute(fe_code, X)
    except LLMError as e:
        print(f"  [ERROR] LLM fix call failed: {e}")
        sys.exit(1)

if err:
    print(f"  [FAIL] Still failing after retry:")
    print(summarise_error(err, max_lines=10))
    sys.exit(1)

new_cols = [c for c in result.columns if c not in X.columns]
print(f"  [OK] Columns before FE : {X.shape[1]}")
print(f"  [OK] Columns after  FE : {result.shape[1]}")
print(f"  [OK] New features added: {len(new_cols)}")
print(f"  New cols: {new_cols}")

# ---------------------------------------------------------------------------
# 6. Score with FE
# ---------------------------------------------------------------------------

sep("6. Trainer — scoring with engineered features")

df_fe = result.copy()
df_fe[TARGET] = df[TARGET].values

class FETransformer(BaseEstimator, TransformerMixin):
    """Sklearn wrapper — applies LLM FE code inside the CV pipeline."""
    def __init__(self, fe_code: str): self.fe_code = fe_code
    def fit(self, X, y=None): return self
    def transform(self, X):
        out, err = execute(self.fe_code, X)
        if err: raise RuntimeError(err)
        return out.replace([np.inf, -np.inf], np.nan)

fe_result = train(
    df_fe, TARGET,
    fe_step=FETransformer(fe_code),
    config=MODEL_CFG,
    save_model=False,
    compute_shap=False,
)
print(f"  Model     {fe_result.best_model_name}")
print(f"  Accuracy  {fe_result.cv_score:.4f}")
print(f"  CV loss   {fe_result.cv_loss:.4f}")

# ---------------------------------------------------------------------------
# 7. Delta
# ---------------------------------------------------------------------------

sep("7. Result")
delta = fe_result.cv_score - baseline.cv_score
direction = "+" if delta >= 0 else ""
print(f"  Baseline accuracy  : {baseline.cv_score:.4f}")
print(f"  FE accuracy        : {fe_result.cv_score:.4f}")
print(f"  Delta              : {direction}{delta:.4f}  {'↑ improved' if delta > 0.001 else '↓ regressed' if delta < -0.001 else '→ no change'}")
print(f"\n  {'LLM FE helped!' if delta > 0.001 else 'No improvement this round — fe_agent will iterate.'}")

sep("DONE")
print("  Full stack: profiler → Groq LLM → executor → trainer\n")
