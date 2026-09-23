# tests/test_tuning.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import ModelConfig, PreprocessingConfig, TuningConfig
from src.models.factory import build_model_pipeline
from src.models.tune import run_randomized_search


def make_data(n=150):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3, random_state=0)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    rng = np.random.RandomState(0)
    df["category"] = rng.choice(["A", "B"], size=n)
    return df, pd.Series(y)


def small_tuning_config():
    return TuningConfig(
        n_iter=3, n_splits=3, random_state=42, n_jobs=1, scoring="roc_auc",
        param_distributions={
            "classifier__n_estimators": [50, 100],
            "classifier__max_depth": [2, 3],
            "classifier__learning_rate": [0.1, 0.2],
        },
    )


def build_pipeline(X):
    return build_model_pipeline(
        "xgboost", [c for c in X.columns if c.startswith("num_")], ["category"],
        PreprocessingConfig(), ModelConfig(),
    )


def test_randomized_search_produces_best_estimator():
    X, y = make_data()
    result = run_randomized_search(build_pipeline(X), X, y, small_tuning_config())

    assert result.best_estimator is not None
    assert 0.0 <= result.best_score <= 1.0
    assert isinstance(result.best_params, dict)
    assert len(result.cv_results) > 0
    assert result.best_estimator.predict(X).shape[0] == len(X)


def test_randomized_search_reproducibility():
    X, y = make_data()
    config = small_tuning_config()
    result_1 = run_randomized_search(build_pipeline(X), X, y, config)
    result_2 = run_randomized_search(build_pipeline(X), X, y, config)

    assert result_1.best_params == result_2.best_params
    assert result_1.best_score == pytest.approx(result_2.best_score)


def test_cv_results_have_expected_fields():
    X, y = make_data()
    result = run_randomized_search(build_pipeline(X), X, y, small_tuning_config())
    assert "mean_test_score" in result.cv_results.columns
    assert "mean_train_score" in result.cv_results.columns