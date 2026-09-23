# tests/test_models.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import ModelConfig, PreprocessingConfig
from src.models.factory import build_model_pipeline, compute_scale_pos_weight


def make_synthetic_df(n=300, weights=(0.7, 0.3), random_state=42):
    X, y = make_classification(
        n_samples=n, n_features=5, n_informative=3, n_redundant=1,
        weights=list(weights), random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(5)])
    rng = np.random.RandomState(random_state)
    df["category"] = rng.choice(["A", "B", "C"], size=n)
    df["target"] = y
    return df


def split_xy(df):
    return df.drop(columns=["target"]), df["target"]


@pytest.mark.parametrize("model_name", ["logistic_regression", "xgboost"])
def test_factory_builds_valid_pipeline(model_name):
    df = make_synthetic_df()
    X, y = split_xy(df)
    numerical = [c for c in X.columns if c.startswith("num_")]
    categorical = ["category"]

    pipeline = build_model_pipeline(
        model_name, numerical, categorical, PreprocessingConfig(), ModelConfig(), y_train=y
    )
    pipeline.fit(X, y)

    preds = pipeline.predict(X)
    proba = pipeline.predict_proba(X)

    assert preds.shape == (len(X),)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


def test_scale_pos_weight_computation():
    y = pd.Series([0, 0, 0, 1])
    assert compute_scale_pos_weight(y) == pytest.approx(3.0)


def test_scale_pos_weight_raises_without_positive_class():
    y = pd.Series([0, 0, 0])
    with pytest.raises(ValueError):
        compute_scale_pos_weight(y)


def test_class_imbalance_strategy_applies_class_weight():
    df = make_synthetic_df()
    X, y = split_xy(df)
    numerical = [c for c in X.columns if c.startswith("num_")]
    config = ModelConfig(class_imbalance_strategy="balanced")

    pipeline = build_model_pipeline(
        "logistic_regression", numerical, ["category"], PreprocessingConfig(), config, y_train=y
    )
    assert pipeline.named_steps["classifier"].class_weight == "balanced"


def test_xgboost_scale_pos_weight_strategy():
    df = make_synthetic_df()
    X, y = split_xy(df)
    numerical = [c for c in X.columns if c.startswith("num_")]
    config = ModelConfig(class_imbalance_strategy="scale_pos_weight")

    pipeline = build_model_pipeline(
        "xgboost", numerical, ["category"], PreprocessingConfig(), config, y_train=y
    )
    assert pipeline.named_steps["classifier"].scale_pos_weight is not None


def test_unknown_model_name_raises():
    df = make_synthetic_df()
    X, y = split_xy(df)
    numerical = [c for c in X.columns if c.startswith("num_")]
    with pytest.raises(ValueError):
        build_model_pipeline("random_forest", numerical, ["category"], PreprocessingConfig(), ModelConfig())