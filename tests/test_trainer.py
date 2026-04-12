"""
test_trainer.py — Step 4a tests for core/trainer.py

Strategy:
  - All DataFrames are synthetic (no external CSV dependency)
  - Test config uses cv_folds=3, single fast model (Decision Tree) for speed
  - compute_shap=False in all train() calls
  - Covers: cleaning transformers, processing, pipeline assembly, CV, train()
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from core.trainer import (
    AutoDFColumnTransformer,
    CardinalityStripper,
    CrossValidator,
    ModelEvaluator,
    ModelRegistry,
    PipelineArchitect,
    TargetEncodedModelWrapper,
    TrainResult,
    UniversalDropper,
    VarianceStripper,
    _detect_task_type,
    get_categorical_transformer,
    get_numeric_transformer,
    train,
)

# ---------------------------------------------------------------------------
# Shared config — fast, deterministic
# ---------------------------------------------------------------------------

FAST_CONFIG = {
    "cleaning": {
        "cardinality": {"max_unique_share": 0.9},
        "variance": {"min_threshold": 0.0},
        "nan_thresholds": {"numeric": 0.5, "categorical": 0.5},
    },
    "model_selection": {
        "models": {
            "classification": ["Decision Tree"],
            "regression": ["Decision Tree"],
        }
    },
    "settings": {"cv_folds": 3, "random_state": 42},
}


# ---------------------------------------------------------------------------
# Synthetic DataFrames
# ---------------------------------------------------------------------------

def make_classification_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "age": rng.integers(18, 80, size=n).astype(float),
        "salary": rng.normal(50_000, 15_000, size=n),
        "department": rng.choice(["eng", "sales", "hr"], size=n),
        "target": rng.choice(["yes", "no"], size=n),
    })


def make_regression_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, 3))
    return pd.DataFrame({
        "x1": X[:, 0],
        "x2": X[:, 1],
        "x3": X[:, 2],
        "target": X[:, 0] * 2 + X[:, 1] - X[:, 2] + rng.normal(0, 0.1, size=n),
    })


# ---------------------------------------------------------------------------
# VarianceStripper
# ---------------------------------------------------------------------------

class TestVarianceStripper:
    def test_drops_constant_column(self):
        df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        vs = VarianceStripper()
        result = vs.fit_transform(df)
        assert "a" not in result.columns
        assert "b" in result.columns

    def test_keeps_all_when_no_low_variance(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        vs = VarianceStripper()
        result = vs.fit_transform(df)
        assert list(result.columns) == ["a", "b"]

    def test_transform_uses_fit_columns(self):
        df_train = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
        df_test = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        vs = VarianceStripper().fit(df_train)
        result = vs.transform(df_test)
        assert "a" not in result.columns


# ---------------------------------------------------------------------------
# UniversalDropper
# ---------------------------------------------------------------------------

class TestUniversalDropper:
    def test_drops_high_null_numeric(self):
        df = pd.DataFrame({
            "good": [1.0, 2.0, 3.0, 4.0],
            "bad": [np.nan, np.nan, np.nan, 1.0],  # 75% null
        })
        ud = UniversalDropper(thresholds={"numeric": 0.5, "categorical": 0.5})
        result = ud.fit_transform(df)
        assert "bad" not in result.columns
        assert "good" in result.columns

    def test_keeps_low_null_columns(self):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0, np.nan],  # 25% null — below threshold
        })
        ud = UniversalDropper(thresholds={"numeric": 0.5})
        result = ud.fit_transform(df)
        assert "a" in result.columns


# ---------------------------------------------------------------------------
# CardinalityStripper
# ---------------------------------------------------------------------------

class TestCardinalityStripper:
    def test_drops_id_like_column(self):
        df = pd.DataFrame({
            "id": list(range(100)),          # 100% unique
            "cat": ["a", "b"] * 50,
        })
        cs = CardinalityStripper(threshold=0.9)
        result = cs.fit_transform(df)
        assert "id" not in result.columns
        assert "cat" in result.columns

    def test_keeps_low_cardinality(self):
        df = pd.DataFrame({
            "color": ["red", "blue", "green"] * 10,
        })
        cs = CardinalityStripper(threshold=0.9)
        result = cs.fit_transform(df)
        assert "color" in result.columns


# ---------------------------------------------------------------------------
# AutoDFColumnTransformer
# ---------------------------------------------------------------------------

class TestAutoDFColumnTransformer:
    def test_returns_dataframe(self):
        from sklearn.compose import make_column_selector

        df = pd.DataFrame({
            "num": [1.0, 2.0, 3.0],
            "cat": ["a", "b", "c"],
        })
        ct = AutoDFColumnTransformer(transformers=[
            ("num", get_numeric_transformer(), make_column_selector(dtype_include=["number"])),
            ("cat", get_categorical_transformer(), make_column_selector(dtype_exclude=["number"])),
        ])
        result = ct.fit_transform(df)
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3


# ---------------------------------------------------------------------------
# TargetEncodedModelWrapper
# ---------------------------------------------------------------------------

class TestTargetEncodedModelWrapper:
    def test_classification_encodes_and_decodes(self):
        df = make_classification_df(60)
        X = df[["age", "salary"]].to_numpy()
        y = df["target"]
        wrapper = TargetEncodedModelWrapper(
            RandomForestClassifier(n_estimators=10, random_state=0),
            task_type="classification",
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert set(preds).issubset({"yes", "no"})

    def test_regression_passes_through(self):
        from sklearn.tree import DecisionTreeRegressor

        df = make_regression_df(60)
        X = df[["x1", "x2", "x3"]].to_numpy()
        y = df["target"]
        wrapper = TargetEncodedModelWrapper(
            DecisionTreeRegressor(random_state=0),
            task_type="regression",
        )
        wrapper.fit(X, y)
        preds = wrapper.predict(X)
        assert preds.dtype in (np.float32, np.float64, float)

    def test_proxy_exposes_feature_importances(self):
        df = make_classification_df(60)
        X = df[["age", "salary"]].to_numpy()
        y = df["target"]
        wrapper = TargetEncodedModelWrapper(
            RandomForestClassifier(n_estimators=10, random_state=0),
            task_type="classification",
        )
        wrapper.fit(X, y)
        assert hasattr(wrapper, "feature_importances_")


# ---------------------------------------------------------------------------
# ModelRegistry
# ---------------------------------------------------------------------------

class TestModelRegistry:
    def test_classification_models_listed(self):
        reg = ModelRegistry()
        names = reg.get_models_for_task("classification")
        assert "Random Forest" in names
        assert "Decision Tree" in names

    def test_regression_models_listed(self):
        reg = ModelRegistry()
        names = reg.get_models_for_task("regression")
        assert "Random Forest" in names
        assert "Ridge" in names

    def test_get_model_returns_instance(self):
        reg = ModelRegistry()
        model = reg.get_model("Decision Tree", "classification")
        assert hasattr(model, "fit")

    def test_unknown_model_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError):
            reg.get_model("Flux Capacitor", "classification")

    def test_unknown_task_raises(self):
        reg = ModelRegistry()
        with pytest.raises(ValueError):
            reg.get_models_for_task("time_travel")


# ---------------------------------------------------------------------------
# PipelineArchitect
# ---------------------------------------------------------------------------

class TestPipelineArchitect:
    def test_builds_pipeline_without_fe_step(self):
        from sklearn.tree import DecisionTreeClassifier

        arch = PipelineArchitect(config=FAST_CONFIG)
        pipe = arch.build_pipeline(DecisionTreeClassifier(), "classification")
        assert isinstance(pipe, Pipeline)
        step_names = [s[0] for s in pipe.steps]
        assert "cleaning" in step_names
        assert "processing" in step_names
        assert "model" in step_names
        assert "fe_transform" not in step_names

    def test_builds_pipeline_with_fe_step(self):
        from sklearn.base import BaseEstimator, TransformerMixin
        from sklearn.tree import DecisionTreeClassifier

        class NoOpTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X

        arch = PipelineArchitect(config=FAST_CONFIG)
        pipe = arch.build_pipeline(
            DecisionTreeClassifier(),
            "classification",
            fe_step=NoOpTransformer(),
        )
        step_names = [s[0] for s in pipe.steps]
        assert "fe_transform" in step_names
        # fe_transform should come between cleaning and processing
        assert step_names.index("fe_transform") == step_names.index("cleaning") + 1

    def test_pipeline_fits_and_predicts(self):
        from sklearn.tree import DecisionTreeClassifier

        df = make_classification_df(80)
        X = df.drop(columns=["target"])
        y = df["target"]
        arch = PipelineArchitect(config=FAST_CONFIG)
        pipe = arch.build_pipeline(DecisionTreeClassifier(random_state=0), "classification")
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert len(preds) == len(y)
        assert set(preds).issubset({"yes", "no"})


# ---------------------------------------------------------------------------
# CrossValidator
# ---------------------------------------------------------------------------

class TestCrossValidator:
    def test_cv_returns_expected_keys(self):
        from sklearn.tree import DecisionTreeClassifier

        df = make_classification_df(90)
        X = df.drop(columns=["target"])
        y = df["target"]
        arch = PipelineArchitect(config=FAST_CONFIG)
        pipe = arch.build_pipeline(DecisionTreeClassifier(random_state=0), "classification")
        cv = CrossValidator(n_splits=3, random_state=42)
        result = cv.run_cv(pipe, X, y, "classification")
        assert "mean_loss" in result
        assert "std_loss" in result
        assert "oof_predictions" in result
        assert len(result["oof_predictions"]) == len(y)

    def test_cv_loss_is_positive(self):
        from sklearn.tree import DecisionTreeClassifier

        df = make_classification_df(90)
        X = df.drop(columns=["target"])
        y = df["target"]
        arch = PipelineArchitect(config=FAST_CONFIG)
        pipe = arch.build_pipeline(DecisionTreeClassifier(random_state=0), "classification")
        cv = CrossValidator(n_splits=3, random_state=42)
        result = cv.run_cv(pipe, X, y, "classification")
        assert result["mean_loss"] >= 0.0

    def test_cv_regression(self):
        from sklearn.tree import DecisionTreeRegressor

        df = make_regression_df(90)
        X = df.drop(columns=["target"])
        y = df["target"]
        arch = PipelineArchitect(config=FAST_CONFIG)
        pipe = arch.build_pipeline(DecisionTreeRegressor(random_state=0), "regression")
        cv = CrossValidator(n_splits=3, random_state=42)
        result = cv.run_cv(pipe, X, y, "regression")
        assert result["mean_loss"] >= 0.0


# ---------------------------------------------------------------------------
# ModelEvaluator
# ---------------------------------------------------------------------------

class TestModelEvaluator:
    def test_accuracy_metric(self):
        y_true = pd.Series(["a", "b", "a", "b"])
        y_pred = np.array(["a", "b", "a", "a"])
        ev = ModelEvaluator("classification", ["accuracy", "f1"])
        res = ev.evaluate(y_true, y_pred)
        assert "accuracy" in res
        assert 0.0 <= res["accuracy"] <= 1.0

    def test_rmse_metric(self):
        y_true = pd.Series([1.0, 2.0, 3.0, 4.0])
        y_pred = np.array([1.1, 1.9, 3.0, 3.8])
        ev = ModelEvaluator("regression", ["rmse", "mae", "r2"])
        res = ev.evaluate(y_true, y_pred)
        assert "rmse" in res
        assert res["rmse"] >= 0.0


# ---------------------------------------------------------------------------
# Task detection
# ---------------------------------------------------------------------------

class TestDetectTaskType:
    def test_string_target_is_classification(self):
        y = pd.Series(["yes", "no", "yes"])
        assert _detect_task_type(y) == "classification"

    def test_low_cardinality_numeric_is_classification(self):
        y = pd.Series([0, 1, 0, 1, 0])
        assert _detect_task_type(y, threshold=15) == "classification"

    def test_high_cardinality_numeric_is_regression(self):
        rng = np.random.default_rng(0)
        y = pd.Series(rng.uniform(0, 1000, size=200))
        assert _detect_task_type(y, threshold=15) == "regression"


# ---------------------------------------------------------------------------
# train() — integration
# ---------------------------------------------------------------------------

class TestTrain:
    def test_classification_end_to_end(self):
        df = make_classification_df(120)
        result = train(
            df,
            target_col="target",
            task_type="classification",
            config=FAST_CONFIG,
            save_model=False,
            compute_shap=False,
        )
        assert isinstance(result, TrainResult)
        assert result.status == "complete"
        assert result.task_type == "classification"
        assert result.best_model_name != ""
        assert 0.0 <= result.cv_score <= 1.0
        assert len(result.leaderboard) == 1  # only Decision Tree in FAST_CONFIG

    def test_regression_end_to_end(self):
        df = make_regression_df(120)
        result = train(
            df,
            target_col="target",
            task_type="regression",
            config=FAST_CONFIG,
            save_model=False,
            compute_shap=False,
        )
        assert result.status == "complete"
        assert result.task_type == "regression"
        assert result.cv_score >= 0.0

    def test_auto_detect_task_type(self):
        df = make_classification_df(120)
        result = train(
            df,
            target_col="target",
            task_type=None,  # auto-detect
            config=FAST_CONFIG,
            save_model=False,
            compute_shap=False,
        )
        assert result.task_type == "classification"

    def test_save_model(self, tmp_path):
        df = make_classification_df(120)
        result = train(
            df,
            target_col="target",
            task_type="classification",
            config=FAST_CONFIG,
            save_model=True,
            output_dir=str(tmp_path),
            compute_shap=False,
        )
        assert result.model_path is not None
        import os
        assert os.path.exists(result.model_path)

    def test_leaderboard_sorted_by_loss(self):
        # Single model: leaderboard should have exactly one entry
        df = make_classification_df(120)
        result = train(
            df,
            target_col="target",
            task_type="classification",
            config=FAST_CONFIG,
            save_model=False,
            compute_shap=False,
        )
        losses = [r["cv_loss_mean"] for r in result.leaderboard]
        assert losses == sorted(losses)

    def test_fe_step_noop_is_accepted(self):
        from sklearn.base import BaseEstimator, TransformerMixin

        class NoOpTransformer(BaseEstimator, TransformerMixin):
            def fit(self, X, y=None):
                return self
            def transform(self, X):
                return X

        df = make_classification_df(120)
        result = train(
            df,
            target_col="target",
            task_type="classification",
            fe_step=NoOpTransformer(),
            config=FAST_CONFIG,
            save_model=False,
            compute_shap=False,
        )
        assert result.status == "complete"

    def test_mixed_dtypes_handled(self):
        """Ensures mixed numeric + categorical columns don't crash."""
        df = make_classification_df(120)
        result = train(
            df,
            target_col="target",
            task_type="classification",
            config=FAST_CONFIG,
            save_model=False,
            compute_shap=False,
        )
        assert result.status == "complete"


# ---------------------------------------------------------------------------
# Audit fixes — new tests for previously uncovered edge cases
# ---------------------------------------------------------------------------

from sklearn.base import BaseEstimator, TransformerMixin


class TestTrainerAuditFixes:
    def test_all_models_fail_returns_failed_status(self):
        """Issue #2: when every model raises, status must be 'failed' not 'complete'."""
        df = make_classification_df(30)
        bad_config = {
            "cleaning":        FAST_CONFIG["cleaning"],
            "model_selection": {"models": {"classification": ["Nonexistent Model XYZ"]}},
            "settings":        FAST_CONFIG["settings"],
        }
        result = train(df, "target", task_type="classification",
                       config=bad_config, save_model=False, compute_shap=False)
        assert result.status == "failed"
        assert result.cv_loss == float("inf")
        assert result.best_model_name == ""

    def test_small_regression_dataset_cv_does_not_crash(self):
        """Issue #8: regression CV binning must not crash on tiny datasets."""
        df = make_regression_df(n=12)
        result = train(df, "target", task_type="regression",
                       config=FAST_CONFIG, save_model=False, compute_shap=False)
        assert result.status == "complete"

    def test_get_model_info_invalid_task_type_raises(self):
        """Issue #9: get_model_info must raise on unknown task_type."""
        reg = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown task type"):
            reg.get_model_info("Random Forest", "time_travel")

    def test_numeric_only_pipeline(self):
        """Issue #11: pipeline with no categorical columns must not crash."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "a": rng.normal(size=60),
            "b": rng.normal(size=60),
            "c": rng.normal(size=60),
            "target": rng.choice(["yes", "no"], size=60),
        })
        result = train(df, "target", task_type="classification",
                       config=FAST_CONFIG, save_model=False, compute_shap=False)
        assert result.status == "complete"

    def test_categorical_only_pipeline(self):
        """Issue #11: pipeline with only categorical columns must not crash."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "color": rng.choice(["red", "blue", "green"], size=60),
            "size":  rng.choice(["small", "medium", "large"], size=60),
            "target": rng.choice(["yes", "no"], size=60),
        })
        result = train(df, "target", task_type="classification",
                       config=FAST_CONFIG, save_model=False, compute_shap=False)
        assert result.status == "complete"
