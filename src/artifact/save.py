# src/artifacts/save.py
"""
Model packaging: saves a versioned bundle of
    - model.joblib          (CalibratedClassifierCV -- deployable)
    - raw_pipeline.joblib   (uncalibrated Pipeline -- SHAP only)
    - shap_metadata.json    (feature schema + transformed feature names)
    - threshold.json        (Phase 5 artifact, copied into the version folder)
    - model_metadata.json   (full reproducibility metadata, no PII)

SECURITY: only ever write to paths under this repository's own
`artifacts/` tree. This module never downloads or accepts external URLs.
"""
from __future__ import annotations

import json
import logging
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version as pkg_version
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import joblib

from src.config import ArtifactConfig

logger = logging.getLogger(__name__)

TRACKED_PACKAGES = ["scikit-learn", "xgboost", "shap", "pandas", "numpy"]


def get_git_commit_hash() -> "str | None":
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        logger.info("Git commit hash unavailable (not a git repo or git not installed).")
        return None


def collect_environment_metadata() -> Dict[str, Any]:
    packages = {}
    for pkg in TRACKED_PACKAGES:
        try:
            packages[pkg] = pkg_version(pkg)
        except PackageNotFoundError:
            packages[pkg] = None
    return {
        "python_version": platform.python_version(),
        "package_versions": packages,
        "git_commit_hash": get_git_commit_hash(),
        "packaged_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_model_metadata(
    model_name: str,
    model_version: str,
    model_type: str,
    calibration_method: str,
    calibration_strategy: str,
    threshold: float,
    threshold_strategy: str,
    training_random_state: int,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    target_column: str,
    best_hyperparameters: Dict[str, Any],
    evaluation_metrics: Dict[str, Any],
    calibration_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Builds metadata EXCLUSIVELY from values passed in by the caller (which
    must be real objects from the training run) -- no field here is a
    literal invented placeholder. No applicant-level data is included.
    """
    metadata = {
        "model_name": model_name,
        "model_version": model_version,
        "model_type": model_type,
        "calibration_method": calibration_method,
        "calibration_strategy": calibration_strategy,
        "threshold": threshold,
        "threshold_selection_strategy": threshold_strategy,
        "training_random_state": training_random_state,
        "feature_count": len(numerical_features) + len(categorical_features),
        "numerical_features": list(numerical_features),
        "categorical_features": list(categorical_features),
        "target_column": target_column,
        "best_hyperparameters": best_hyperparameters,
        "evaluation_metrics": evaluation_metrics,
        "calibration_metrics": calibration_metrics,
    }
    metadata.update(collect_environment_metadata())
    return metadata


def save_model_package(
    version: str,
    calibrated_model: Any,
    raw_pipeline: Any,
    shap_metadata: Dict[str, Any],
    threshold_payload: Dict[str, Any],
    metadata: Dict[str, Any],
    artifact_config: ArtifactConfig,
) -> Dict[str, Path]:
    model_dir = artifact_config.models_dir / artifact_config.model_name / version
    threshold_dir = artifact_config.thresholds_dir / version
    metadata_dir = artifact_config.metadata_dir / version
    for d in (model_dir, threshold_dir, metadata_dir):
        d.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    raw_pipeline_path = model_dir / "raw_pipeline.joblib"
    shap_metadata_path = model_dir / "shap_metadata.json"
    threshold_path = threshold_dir / "threshold.json"
    metadata_path = metadata_dir / "model_metadata.json"

    joblib.dump(calibrated_model, model_path)
    joblib.dump(raw_pipeline, raw_pipeline_path)
    with open(shap_metadata_path, "w") as f:
        json.dump(shap_metadata, f, indent=2, default=str)
    with open(threshold_path, "w") as f:
        json.dump(threshold_payload, f, indent=2, default=str)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    logger.info("Model package v%s saved under %s", version, model_dir)
    return {
        "model_path": model_path,
        "raw_pipeline_path": raw_pipeline_path,
        "shap_metadata_path": shap_metadata_path,
        "threshold_path": threshold_path,
        "metadata_path": metadata_path,
    }