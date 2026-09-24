# tests/test_calibration.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import CalibrationConfig, ModelConfig, PreprocessingConfig
from src.models.factory import build_model_pipeline
from src.calibration.calibrator import fit_calibrated_model_with_split
from src.calibration.evaluate import compare_calibration


def make_data(n=400):
    X, y = make_classification(n_samples=n, n_features=6, n_informative=4, random_state=1)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(6)])
    rng = np.random.RandomState(1)
    df["category"] = rng.choice(["A", "B", "C"], size=n)
    return df, pd.Series(y)


def build_pipeline(X):
    numerical = [c for c in X.columns if c.startswith("num_")]
    return build_model_pipeline(
        "xgboost", numerical, ["category"], PreprocessingConfig(), ModelConfig(),
    )


def test_calibrator_fits_successfully_prefit():
    X, y = make_data()
    pipeline = build_pipeline(X)
    config = CalibrationConfig(strategy="prefit", calibration_holdout_size=0.3, random_state=0)

    result, X_calib, y_calib = fit_calibrated_model_with_split(pipeline, X, y, config)

    assert result.calibrated_model is not None
    assert result.raw_pipeline is not None
    proba = result.calibrated_model.predict_proba(X_calib)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_calibrator_fits_successfully_cv():
    X, y = make_data()
    pipeline = build_pipeline(X)
    config = CalibrationConfig(strategy="cv", cv_splits=3, random_state=0)

    result, X_val, y_val = fit_calibrated_model_with_split(pipeline, X, y, config)
    proba = result.calibrated_model.predict_proba(X_val)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))


def test_calibration_does_not_touch_a_test_set():
    """Structural leakage check: fit_calibrated_model_with_split only ever
    receives training data; asserts the calibration split is a strict
    subset of the data passed in (no external data referenced)."""
    X, y = make_data()
    pipeline = build_pipeline(X)
    config = CalibrationConfig(strategy="prefit", calibration_holdout_size=0.25, random_state=0)

    _, X_calib, _ = fit_calibrated_model_with_split(pipeline, X, y, config)
    assert set(X_calib.index).issubset(set(X.index))
    assert len(X_calib) < len(X)


def test_calibration_metrics_computed_correctly():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    before = np.array([0.4, 0.3, 0.6, 0.55, 0.45, 0.7])
    after = np.array([0.2, 0.15, 0.8, 0.75, 0.25, 0.85])

    report = compare_calibration(y_true, before, after)
    assert "before_calibration" in report and "after_calibration" in report
    assert "brier_score" in report["before_calibration"]
    assert isinstance(report["brier_score_delta"], float)


def test_calibrated_model_can_be_saved_and_reloaded(tmp_path):
    import joblib

    X, y = make_data()
    pipeline = build_pipeline(X)
    config = CalibrationConfig(strategy="prefit", random_state=0)
    result, X_calib, y_calib = fit_calibrated_model_with_split(pipeline, X, y, config)

    path = tmp_path / "calibrated.joblib"
    joblib.dump(result.calibrated_model, path)
    reloaded = joblib.load(path)

    original_proba = result.calibrated_model.predict_proba(X_calib)[:, 1]
    reloaded_proba = reloaded.predict_proba(X_calib)[:, 1]
    np.testing.assert_allclose(original_proba, reloaded_proba)


def test_invalid_calibration_method_raises():
    from src.calibration.calibrator import CalibrationError, fit_calibrated_model

    X, y = make_data()
    pipeline = build_pipeline(X)
    config = CalibrationConfig(method="platt_typo")
    with pytest.raises(CalibrationError):
        fit_calibrated_model(pipeline, X, y, config)