# tests/test_artifacts.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

from src.config import ArtifactConfig, CalibrationConfig, ModelConfig, PreprocessingConfig
from src.features.preprocessing import identify_column_types
from src.models.factory import build_model_pipeline
from src.calibration.calibrator import fit_calibrated_model_with_split
from src.artifacts.save import build_model_metadata, save_model_package
from src.artifacts.load import ArtifactLoadError, load_model_artifacts


def make_data(n=200):
    X, y = make_classification(n_samples=n, n_features=4, n_informative=3, random_state=3)
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(4)])
    rng = np.random.RandomState(3)
    df["category"] = rng.choice(["A", "B"], size=n)
    return df, pd.Series(y)


@pytest.fixture
def packaged_artifacts(tmp_path):
    X, y = make_data()
    numerical, categorical = identify_column_types(X)
    pipeline = build_model_pipeline("xgboost", numerical, categorical, PreprocessingConfig(), ModelConfig())

    calib_config = CalibrationConfig(strategy="prefit", random_state=0)
    fit_result, X_calib, y_calib = fit_calibrated_model_with_split(pipeline, X, y, calib_config)

    artifact_config = ArtifactConfig(
        models_dir=tmp_path / "models", thresholds_dir=tmp_path / "thresholds",
        metadata_dir=tmp_path / "metadata",
    )

    shap_metadata = {"numerical_features": numerical, "categorical_features": categorical}
    threshold_payload = {"threshold": 0.42, "strategy": "maximize_f1", "rationale": "test"}
    metadata = build_model_metadata(
        model_name="credit_risk_model", model_version="v1_test", model_type="XGBClassifier",
        calibration_method="sigmoid", calibration_strategy="prefit", threshold=0.42,
        threshold_strategy="maximize_f1", training_random_state=42,
        numerical_features=numerical, categorical_features=categorical, target_column="target",
        best_hyperparameters={"max_depth": 4}, evaluation_metrics={"roc_auc": 0.9},
        calibration_metrics={"brier_score": 0.1},
    )

    paths = save_model_package(
        "v1_test", fit_result.calibrated_model, fit_result.raw_pipeline,
        shap_metadata, threshold_payload, metadata, artifact_config,
    )
    return artifact_config, paths, X


def test_artifacts_can_be_saved(packaged_artifacts):
    _, paths, _ = packaged_artifacts
    for p in paths.values():
        assert p.exists()


def test_artifacts_can_be_loaded(packaged_artifacts):
    artifact_config, _, _ = packaged_artifacts
    artifacts = load_model_artifacts("v1_test", artifact_config)
    assert artifacts.model is not None
    assert artifacts.raw_pipeline is not None


def test_metadata_is_valid_and_has_version(packaged_artifacts):
    artifact_config, _, _ = packaged_artifacts
    artifacts = load_model_artifacts("v1_test", artifact_config)
    assert artifacts.metadata["model_version"] == "v1_test"
    assert "python_version" in artifacts.metadata
    assert "package_versions" in artifacts.metadata


def test_threshold_within_valid_range(packaged_artifacts):
    artifact_config, _, _ = packaged_artifacts
    artifacts = load_model_artifacts("v1_test", artifact_config)
    assert 0.0 <= artifacts.threshold <= 1.0


def test_loaded_artifacts_can_predict(packaged_artifacts):
    artifact_config, _, X = packaged_artifacts
    artifacts = load_model_artifacts("v1_test", artifact_config)
    proba = artifacts.predict_proba(X)
    preds = artifacts.predict(X)
    assert proba.shape[0] == len(X)
    assert set(np.unique(preds)).issubset({0, 1})


def test_missing_version_raises_clear_error(packaged_artifacts):
    artifact_config, _, _ = packaged_artifacts
    with pytest.raises(ArtifactLoadError):
        load_model_artifacts("v999_does_not_exist", artifact_config)


def test_corrupted_package_missing_file_raises(packaged_artifacts):
    artifact_config, paths, _ = packaged_artifacts
    paths["shap_metadata_path"].unlink()  # simulate a missing/corrupted artifact
    with pytest.raises(ArtifactLoadError):
        load_model_artifacts("v1_test", artifact_config)