"""
trainer.py — Static sklearn pipeline (Step 4a)

Sources:
  - Cleaning block: AutoFlowML/src/cleaning.py (verbatim, logger removed)
  - Processing block: AutoFlowML/src/processing.py (verbatim)
  - CrossValidator: AutoFlowML/src/evaluation.py (verbatim fold structure)
  - ModelInfo/ModelRegistry: AutoML/src/models/registry.py (adapted, sys.path hacks removed)
  - PipelineArchitect: AutoFlowML/src/pipeline.py (adapted, fe_step param added)

The fe_step param in build_pipeline() is the LLM injection point (Step 4e).
It inserts between cleaning and processing as: cleaning → fe_step → processing → model.
"""

from __future__ import annotations

import inspect
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler

# Optional boosting libraries
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

# sklearn models
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CLEANING BLOCK — verbatim from AutoFlowML/src/cleaning.py
# (only change: replaced `from src.utils.logger import logger` with stdlib logging)
# ---------------------------------------------------------------------------

class VarianceStripper(BaseEstimator, TransformerMixin):
    """Remove columns with zero variance or variance below a threshold."""

    def __init__(self, min_threshold: float = 0.0):
        self.min_threshold = min_threshold

    def fit(self, X: pd.DataFrame, y=None):
        variances = X.var(numeric_only=True)
        low_var_cols = variances[variances <= self.min_threshold].index.tolist()
        static_cols = [col for col in X.columns if X[col].nunique() <= 1]
        to_drop = list(set(low_var_cols + static_cols))
        self.columns_to_keep_: List[str] = [c for c in X.columns if c not in to_drop]
        self.n_features_in_: int = X.shape[1]
        if to_drop:
            logger.debug("VarianceStripper: dropped %d columns: %s", len(to_drop), to_drop)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_to_keep_)


class UniversalDropper(BaseEstimator, TransformerMixin):
    """Drop columns that exceed the null-rate threshold per dtype."""

    def __init__(self, thresholds: Dict[str, float] = None):
        self.thresholds = thresholds or {"numeric": 0.5, "categorical": 0.5}

    def fit(self, X: pd.DataFrame, y=None):
        self.n_features_in_: int = X.shape[1]
        to_drop = []
        for col in X.columns:
            null_ratio = X[col].isnull().mean()
            dtype_key = "numeric" if pd.api.types.is_any_real_numeric_dtype(X[col]) else "categorical"
            limit = self.thresholds.get(dtype_key, 0.5)
            if null_ratio > limit:
                to_drop.append(col)
        self.columns_to_keep_: List[str] = [c for c in X.columns if c not in to_drop]
        if to_drop:
            logger.debug("UniversalDropper: dropped %d high-null columns: %s", len(to_drop), to_drop)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self)
        return X[self.columns_to_keep_]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(self.columns_to_keep_)


class CardinalityStripper(BaseEstimator, TransformerMixin):
    """Drop columns where unique-value share exceeds threshold (e.g. ID columns)."""

    def __init__(self, threshold: float = 0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        n_rows = len(X)
        self.cols_to_drop_: List[str] = []
        for col in X.columns:
            # Skip float columns — continuous values are expected to have high uniqueness
            if pd.api.types.is_float_dtype(X[col]):
                continue
            share_unique = X[col].nunique() / n_rows if n_rows > 0 else 0
            if share_unique >= self.threshold:
                self.cols_to_drop_.append(col)
        if self.cols_to_drop_:
            logger.debug("CardinalityStripper: dropped %d columns: %s", len(self.cols_to_drop_), self.cols_to_drop_)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return X.drop(columns=self.cols_to_drop_)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if input_features is None:
            return np.array([c for c in self.cols_to_drop_])
        return np.array([f for f in input_features if f not in self.cols_to_drop_])


# ---------------------------------------------------------------------------
# PROCESSING BLOCK — verbatim from AutoFlowML/src/processing.py
# ---------------------------------------------------------------------------

def get_numeric_transformer(scaler_type: str = "standard") -> Pipeline:
    """Impute then scale numeric columns."""
    scaler = RobustScaler() if scaler_type == "robust" else StandardScaler()
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", scaler),
    ])


def get_categorical_transformer() -> Pipeline:
    """Impute then OHE categorical columns."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])


class AutoDFColumnTransformer(ColumnTransformer):
    """ColumnTransformer that always returns pandas DataFrames."""

    def __init__(
        self,
        transformers,
        *,
        remainder="drop",
        sparse_threshold=0.3,
        n_jobs=None,
        transformer_weights=None,
        verbose=False,
        verbose_feature_names_out=True,
    ):
        super().__init__(
            transformers=transformers,
            remainder=remainder,
            sparse_threshold=sparse_threshold,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights,
            verbose=verbose,
            verbose_feature_names_out=verbose_feature_names_out,
        )
        self.set_output(transform="pandas")


class TargetEncodedModelWrapper(BaseEstimator):
    """
    Transparent proxy that handles label encoding/decoding internally.

    Resolves the y_train vs y_train_encoded inconsistency present in
    AutoML/src/models/trainers/classification.py:144 — the wrapper owns
    encoding so the outer pipeline never sees raw vs encoded labels.
    """

    def __init__(self, model: Any, task_type: str = "classification"):
        self.model = model
        self.task_type = task_type
        self.label_encoder = LabelEncoder()

    def fit(self, X, y):
        self.model_ = clone(self.model)
        if self.task_type == "classification":
            y_enc = self.label_encoder.fit_transform(y)
            self.classes_ = self.label_encoder.classes_
            self.model_.fit(X, y_enc)
        else:
            self.model_.fit(X, y)
        return self

    def predict(self, X):
        preds = self.model_.predict(X)
        if self.task_type == "classification":
            return self.label_encoder.inverse_transform(preds)
        return preds

    def predict_proba(self, X):
        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        raise AttributeError(f"{type(self.model_).__name__} does not support predict_proba")

    def __getattr__(self, name: str):
        if "model_" in self.__dict__:
            return getattr(self.model_, name)
        return getattr(self.model, name)

    def __dir__(self):
        base = list(object.__dir__(self))
        model_dir = dir(self.model_ if hasattr(self, "model_") else self.model)
        return list(set(base + model_dir))


# ---------------------------------------------------------------------------
# MODEL REGISTRY — adapted from AutoML/src/models/registry.py
# (sys.path hacks removed; Config dependency replaced with plain params)
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """Metadata container for a model."""

    name: str
    model_class: Any
    default_params: Dict
    task_types: List[str]
    strengths: List[str]
    weaknesses: List[str]
    complexity: str           # 'fast' | 'medium' | 'slow'
    requires_scaling: bool
    handles_missing: bool
    handles_categorical: bool
    interpretable: bool


class ModelRegistry:
    """Central model factory for AutoML model selection."""

    def __init__(self, random_state: int = 42, n_jobs: int = -1):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._classification = self._build_classification()
        self._regression = self._build_regression()

    def _build_classification(self) -> Dict[str, ModelInfo]:
        rs = self.random_state
        nj = self.n_jobs
        reg: Dict[str, ModelInfo] = {}

        reg["Logistic Regression"] = ModelInfo(
            name="Logistic Regression", model_class=LogisticRegression,
            default_params={"random_state": rs, "max_iter": 1000},
            task_types=["classification"],
            strengths=["Fast", "Interpretable", "Probabilistic"],
            weaknesses=["Linear boundary only", "Requires scaling"],
            complexity="fast", requires_scaling=True,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["Random Forest"] = ModelInfo(
            name="Random Forest", model_class=RandomForestClassifier,
            default_params={"n_estimators": 100, "random_state": rs, "n_jobs": nj,
                            "max_depth": 10, "min_samples_split": 10, "min_samples_leaf": 4},
            task_types=["classification"],
            strengths=["Excellent OOB performance", "Feature importance", "No scaling"],
            weaknesses=["Slow on large data", "High memory"],
            complexity="medium", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=False,
        )
        reg["Gradient Boosting"] = ModelInfo(
            name="Gradient Boosting", model_class=GradientBoostingClassifier,
            default_params={"n_estimators": 100, "learning_rate": 0.1,
                            "max_depth": 3, "random_state": rs},
            task_types=["classification"],
            strengths=["Often best accuracy", "Handles complex patterns"],
            weaknesses=["Slow training", "Prone to overfit"],
            complexity="slow", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=False,
        )
        reg["Decision Tree"] = ModelInfo(
            name="Decision Tree", model_class=DecisionTreeClassifier,
            default_params={"random_state": rs, "max_depth": 10},
            task_types=["classification"],
            strengths=["Interpretable", "Fast"],
            weaknesses=["Prone to overfit"],
            complexity="fast", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["K-Nearest Neighbors"] = ModelInfo(
            name="K-Nearest Neighbors", model_class=KNeighborsClassifier,
            default_params={"n_neighbors": 5, "n_jobs": nj},
            task_types=["classification"],
            strengths=["Simple", "No training"],
            weaknesses=["Slow prediction", "Requires scaling"],
            complexity="medium", requires_scaling=True,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["Naive Bayes"] = ModelInfo(
            name="Naive Bayes", model_class=GaussianNB,
            default_params={},
            task_types=["classification"],
            strengths=["Extremely fast", "Works with small datasets"],
            weaknesses=["Assumes feature independence"],
            complexity="fast", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["AdaBoost"] = ModelInfo(
            name="AdaBoost", model_class=AdaBoostClassifier,
            default_params={"n_estimators": 50, "learning_rate": 1.0, "random_state": rs},
            task_types=["classification"],
            strengths=["Less prone to overfit than single trees"],
            weaknesses=["Sensitive to noise"],
            complexity="medium", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=False,
        )
        if XGBOOST_AVAILABLE:
            reg["XGBoost"] = ModelInfo(
                name="XGBoost", model_class=XGBClassifier,
                default_params={"n_estimators": 100, "learning_rate": 0.1, "max_depth": 6,
                                "random_state": rs, "n_jobs": nj, "eval_metric": "logloss",
                                "verbosity": 0},
                task_types=["classification"],
                strengths=["Competition winner", "Handles missing values"],
                weaknesses=["Many hyperparams"],
                complexity="medium", requires_scaling=False,
                handles_missing=True, handles_categorical=False, interpretable=False,
            )
        if LIGHTGBM_AVAILABLE:
            reg["LightGBM"] = ModelInfo(
                name="LightGBM", model_class=LGBMClassifier,
                default_params={"n_estimators": 100, "learning_rate": 0.05,
                                "num_leaves": 31, "max_depth": 7, "verbose": -1},
                task_types=["classification"],
                strengths=["Extremely fast", "Low memory", "Handles categoricals"],
                weaknesses=["Can overfit on small data"],
                complexity="fast", requires_scaling=False,
                handles_missing=True, handles_categorical=True, interpretable=False,
            )
        return reg

    def _build_regression(self) -> Dict[str, ModelInfo]:
        rs = self.random_state
        nj = self.n_jobs
        reg: Dict[str, ModelInfo] = {}

        reg["Linear Regression"] = ModelInfo(
            name="Linear Regression", model_class=LinearRegression,
            default_params={},
            task_types=["regression"],
            strengths=["Extremely fast", "Interpretable"],
            weaknesses=["Linear only", "Sensitive to outliers"],
            complexity="fast", requires_scaling=True,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["Ridge"] = ModelInfo(
            name="Ridge", model_class=Ridge,
            default_params={"alpha": 1.0},
            task_types=["regression"],
            strengths=["L2 regularization", "Handles multicollinearity"],
            weaknesses=["Linear only"],
            complexity="fast", requires_scaling=True,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["Lasso"] = ModelInfo(
            name="Lasso", model_class=Lasso,
            default_params={"alpha": 1.0},
            task_types=["regression"],
            strengths=["Feature selection via L1"],
            weaknesses=["Unstable with correlated features"],
            complexity="fast", requires_scaling=True,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["Random Forest"] = ModelInfo(
            name="Random Forest", model_class=RandomForestRegressor,
            default_params={"n_estimators": 100, "random_state": rs, "n_jobs": nj},
            task_types=["regression"],
            strengths=["Excellent OOB performance", "Feature importance"],
            weaknesses=["Slow on large data", "High memory"],
            complexity="medium", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=False,
        )
        reg["Gradient Boosting"] = ModelInfo(
            name="Gradient Boosting", model_class=GradientBoostingRegressor,
            default_params={"n_estimators": 100, "learning_rate": 0.1,
                            "max_depth": 3, "random_state": rs},
            task_types=["regression"],
            strengths=["Often best accuracy"],
            weaknesses=["Slow training", "Prone to overfit"],
            complexity="slow", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=False,
        )
        reg["Decision Tree"] = ModelInfo(
            name="Decision Tree", model_class=DecisionTreeRegressor,
            default_params={"random_state": rs, "max_depth": 10},
            task_types=["regression"],
            strengths=["Interpretable", "Fast"],
            weaknesses=["Prone to overfit"],
            complexity="fast", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=True,
        )
        reg["AdaBoost"] = ModelInfo(
            name="AdaBoost", model_class=AdaBoostRegressor,
            default_params={"n_estimators": 50, "learning_rate": 1.0, "random_state": rs},
            task_types=["regression"],
            strengths=["Less prone to overfit"],
            weaknesses=["Sensitive to noise"],
            complexity="medium", requires_scaling=False,
            handles_missing=False, handles_categorical=False, interpretable=False,
        )
        if XGBOOST_AVAILABLE:
            reg["XGBoost"] = ModelInfo(
                name="XGBoost", model_class=XGBRegressor,
                default_params={"n_estimators": 100, "learning_rate": 0.1,
                                "max_depth": 6, "random_state": rs, "n_jobs": nj,
                                "verbosity": 0},
                task_types=["regression"],
                strengths=["Excellent accuracy", "Handles missing values"],
                weaknesses=["Many hyperparams"],
                complexity="medium", requires_scaling=False,
                handles_missing=True, handles_categorical=False, interpretable=False,
            )
        if LIGHTGBM_AVAILABLE:
            reg["LightGBM"] = ModelInfo(
                name="LightGBM", model_class=LGBMRegressor,
                default_params={"n_estimators": 100, "learning_rate": 0.1,
                                "verbose": -1},
                task_types=["regression"],
                strengths=["Very fast", "Low memory"],
                weaknesses=["Can overfit on small data"],
                complexity="fast", requires_scaling=False,
                handles_missing=True, handles_categorical=True, interpretable=False,
            )
        return reg

    def get_models_for_task(self, task_type: str) -> List[str]:
        if task_type == "classification":
            return list(self._classification.keys())
        if task_type == "regression":
            return list(self._regression.keys())
        raise ValueError(f"Unknown task type: {task_type!r}")

    def get_model(self, model_name: str, task_type: str, custom_params: Dict = None) -> Any:
        registry = (self._classification if task_type == "classification"
                    else self._regression if task_type == "regression"
                    else None)
        if registry is None:
            raise ValueError(f"Unknown task type: {task_type!r}")
        if model_name not in registry:
            raise ValueError(
                f"Model {model_name!r} not found for {task_type}. "
                f"Available: {list(registry.keys())}"
            )
        info = registry[model_name]
        params = info.default_params.copy()
        if custom_params:
            params.update(custom_params)
        return info.model_class(**params)

    def get_model_info(self, model_name: str, task_type: str) -> ModelInfo:
        registry = self._classification if task_type == "classification" else self._regression
        if model_name not in registry:
            raise ValueError(f"Model {model_name!r} not found for {task_type}")
        return registry[model_name]


# ---------------------------------------------------------------------------
# EVALUATION BLOCK — verbatim fold structure from AutoFlowML/src/evaluation.py
# LeaderboardEngine adapted to return list of dicts instead of DataFrame
# ---------------------------------------------------------------------------

class ModelEvaluator:
    """Compute metrics for a given task type."""

    _METRIC_MAP = {
        "accuracy": metrics.accuracy_score,
        "precision": lambda y, p: metrics.precision_score(y, p, average="weighted", zero_division=0),
        "recall": lambda y, p: metrics.recall_score(y, p, average="weighted", zero_division=0),
        "f1": lambda y, p: metrics.f1_score(y, p, average="weighted", zero_division=0),
        "rmse": lambda y, p: float(np.sqrt(metrics.mean_squared_error(y, p))),
        "mae": metrics.mean_absolute_error,
        "r2": metrics.r2_score,
    }

    def __init__(self, task_type: str, metrics_list: List[str]):
        self.task_type = task_type
        self.metrics_list = metrics_list

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        return {
            m: round(float(self._METRIC_MAP[m](y_true, y_pred)), 4)
            for m in self.metrics_list
            if m in self._METRIC_MAP
        }


class CrossValidator:
    """
    Stratified K-Fold cross-validator.

    Crown jewel: clone(pipeline) per fold — full pipeline re-fit on fold
    train data only. Preprocessing never sees held-out data. No leakage.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state

    def run_cv(
        self,
        pipeline: Any,
        X: pd.DataFrame,
        y: pd.Series,
        task_type: str,
    ) -> Dict[str, Any]:
        logger.info("CV: starting %d-fold validation", self.n_splits)
        oof_preds = np.empty(len(y), dtype=object if task_type == "classification" else float)
        scores: List[float] = []

        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        # Regression: bin target for stratification
        if task_type == "regression":
            n_bins = min(10, len(y) // self.n_splits)
            target_split = pd.qcut(y, q=max(2, n_bins), labels=False, duplicates="drop")
        else:
            target_split = y

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, target_split)):
            fold_pipe = clone(pipeline)
            fold_pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            preds = fold_pipe.predict(X.iloc[val_idx])
            oof_preds[val_idx] = preds

            if task_type == "regression":
                score = float(np.sqrt(metrics.mean_squared_error(y.iloc[val_idx], preds)))
            else:
                score = float(1 - metrics.accuracy_score(y.iloc[val_idx], preds))
            scores.append(score)
            logger.debug("   fold %d loss: %.4f", fold + 1, score)

        return {
            "mean_loss": float(np.mean(scores)),
            "std_loss": float(np.std(scores)),
            "oof_predictions": oof_preds,
        }


# ---------------------------------------------------------------------------
# PIPELINE ARCHITECT — adapted from AutoFlowML/src/pipeline.py
# Added: fe_step optional param (LLM injection point for Step 4e)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG: Dict[str, Any] = {
    "cleaning": {
        "cardinality": {"max_unique_share": 0.9},
        "variance": {"min_threshold": 0.0},
        "nan_thresholds": {"numeric": 0.5, "categorical": 0.5},
    },
    "model_selection": {
        "models": {
            "classification": ["Random Forest", "Logistic Regression", "Decision Tree"],
            "regression": ["Random Forest", "Ridge", "Decision Tree"],
        }
    },
    "settings": {
        "cv_folds": 5,
        "random_state": 42,
    },
}


class PipelineArchitect:
    """Assembles cleaning → [fe_step] → processing → model into one Pipeline."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or _DEFAULT_CONFIG

    def build_pipeline(
        self,
        model_instance: Any,
        task_type: str,
        fe_step: Optional[BaseEstimator] = None,
    ) -> Pipeline:
        """
        Build the end-to-end sklearn Pipeline.

        Parameters
        ----------
        model_instance : sklearn estimator
            The raw (unwrapped) model. TargetEncodedModelWrapper is applied here.
        task_type : str
            'classification' or 'regression'
        fe_step : sklearn transformer, optional
            LLM-generated feature engineering step. Inserted between cleaning
            and processing. Must implement fit/transform and return a DataFrame.
            Used by fe_agent.py (Step 4e). Pass None for the static baseline.
        """
        cleaning_cfg = self.config["cleaning"]

        cleaning = Pipeline(steps=[
            ("cardinality", CardinalityStripper(
                threshold=cleaning_cfg["cardinality"]["max_unique_share"]
            )),
            ("variance", VarianceStripper(
                min_threshold=cleaning_cfg["variance"]["min_threshold"]
            )),
            ("nan_drop", UniversalDropper(
                thresholds=cleaning_cfg["nan_thresholds"]
            )),
        ])

        processing = AutoDFColumnTransformer(transformers=[
            ("num", get_numeric_transformer(), make_column_selector(dtype_include=["number"])),
            ("cat", get_categorical_transformer(), make_column_selector(dtype_exclude=["number"])),
        ])

        wrapped_model = TargetEncodedModelWrapper(model_instance, task_type=task_type)

        steps = [("cleaning", cleaning)]
        if fe_step is not None:
            steps.append(("fe_transform", fe_step))
        steps.append(("processing", processing))
        steps.append(("model", wrapped_model))

        logger.debug("PipelineArchitect: built pipeline with steps %s", [s[0] for s in steps])
        return Pipeline(steps=steps)


# ---------------------------------------------------------------------------
# TOP-LEVEL API
# ---------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Return contract from train(). Matches the Kashif JSON output spec."""

    status: str
    best_model_name: str
    cv_score: float           # primary metric (accuracy or RMSE depending on task)
    cv_loss: float            # raw loss used for comparison (lower = better)
    model_path: Optional[str]
    leaderboard: List[Dict[str, Any]]
    task_type: str
    # Fields populated by fe_agent (Step 4e) — empty for static baseline
    baseline_score: float = field(default=0.0)
    delta: float = field(default=0.0)


def _detect_task_type(y: pd.Series, threshold: int = 15) -> str:
    """
    Simple heuristic task detector (fallback when profiler is not available).
    Full TaskDetector with 6-rule weighted voting lives in profiler.py (Step 4b).
    """
    if y.dtype == object or y.dtype.name == "category":
        return "classification"
    if y.nunique() <= threshold:
        return "classification"
    return "regression"


def train(
    df: pd.DataFrame,
    target_col: str,
    task_type: Optional[str] = None,
    fe_step: Optional[BaseEstimator] = None,
    config: Optional[Dict[str, Any]] = None,
    save_model: bool = True,
    output_dir: str = "./outputs",
    compute_shap: bool = True,
) -> TrainResult:
    """
    Run the full static AutoML pipeline on *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Input data including the target column.
    target_col : str
        Name of the target column.
    task_type : str, optional
        'classification' or 'regression'. Auto-detected if None.
    fe_step : sklearn transformer, optional
        LLM FE step from fe_agent (Step 4e). None = static baseline.
    config : dict, optional
        Override default pipeline config. See _DEFAULT_CONFIG for schema.
    save_model : bool
        Persist best pipeline to output_dir/best_model.pkl.
    output_dir : str
        Directory for model artifacts.
    compute_shap : bool
        Compute SHAP values for the best model (slow on large data).

    Returns
    -------
    TrainResult
    """
    cfg = config or _DEFAULT_CONFIG
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if task_type is None:
        task_type = _detect_task_type(y)
    logger.info("trainer.train: task_type=%s  shape=%s", task_type, X.shape)

    # Metrics to compute
    primary_metric = "accuracy" if task_type == "classification" else "rmse"
    metric_list = (
        ["accuracy", "f1", "precision", "recall"]
        if task_type == "classification"
        else ["rmse", "mae", "r2"]
    )

    registry = ModelRegistry(
        random_state=cfg["settings"]["random_state"],
        n_jobs=-1,
    )
    evaluator = ModelEvaluator(task_type=task_type, metrics_list=metric_list)
    cv_engine = CrossValidator(
        n_splits=cfg["settings"]["cv_folds"],
        random_state=cfg["settings"]["random_state"],
    )
    architect = PipelineArchitect(config=cfg)

    model_slugs: List[str] = cfg["model_selection"]["models"][task_type]
    leaderboard: List[Dict[str, Any]] = []
    best_loss = float("inf")
    best_pipeline = None
    best_model_name = ""

    for slug in model_slugs:
        try:
            model_instance = registry.get_model(slug, task_type)
            pipeline = architect.build_pipeline(model_instance, task_type, fe_step=fe_step)

            t0 = time.time()
            cv_res = cv_engine.run_cv(pipeline, X, y, task_type)
            duration = round(time.time() - t0, 2)

            metrics_res = evaluator.evaluate(y, cv_res["oof_predictions"])

            entry: Dict[str, Any] = {
                "model": slug,
                "cv_loss_mean": round(cv_res["mean_loss"], 4),
                "cv_loss_std": round(cv_res["std_loss"], 4),
                "time_s": duration,
                **metrics_res,
            }
            leaderboard.append(entry)
            logger.info("  %-30s  loss=%.4f ± %.4f", slug, cv_res["mean_loss"], cv_res["std_loss"])

            if cv_res["mean_loss"] < best_loss:
                best_loss = cv_res["mean_loss"]
                best_model_name = slug
                # Re-fit best pipeline on full data for saving
                best_pipeline = clone(pipeline)
                best_pipeline.fit(X, y)

        except Exception as exc:
            logger.warning("trainer: skipped %s — %s", slug, exc)

    leaderboard.sort(key=lambda r: r["cv_loss_mean"])

    # Primary metric score from leaderboard entry
    best_entry = next((r for r in leaderboard if r["model"] == best_model_name), {})
    cv_score = best_entry.get(primary_metric, 0.0)

    # Save model
    model_path: Optional[str] = None
    if save_model and best_pipeline is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        model_path = str(Path(output_dir) / "best_model.pkl")
        joblib.dump(best_pipeline, model_path)
        logger.info("trainer: best model saved → %s", model_path)

    return TrainResult(
        status="complete",
        best_model_name=best_model_name,
        cv_score=cv_score,
        cv_loss=round(best_loss, 4),
        model_path=model_path,
        leaderboard=leaderboard,
        task_type=task_type,
    )
