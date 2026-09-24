# src/artifacts/load.py
"""
Artifact loading for future inference services (FastAPI, Phase 9).

SECURITY: joblib.load can execute arbitrary code if given an untrusted
file. This module ONLY loads paths constructed from the local, repo-owned
`artifacts/` directory tree -- it never accepts or fetches a URL, and
should never be pointed at artifacts from an untrusted source.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from src.config import ArtifactConfig
from src.features.preprocessing import FeatureSchema
from src.optimization.threshold import apply_threshold

logger = logging.getLogger(__name__)


class ArtifactLoadError(Exception):
    """Raised when a requested model version or a required file is missing/invalid."""


@dataclass
class ModelArtifacts:
    model: Any                 # calibrated, deployable
    raw_pipeline: Any          # uncalibrated, SHAP-only
    threshold: float
    threshold_metadata: Dict[str, Any]
    metadata: Dict[str, Any]
    shap_metadata: Dict[str, Any]
    feature_schema: FeatureSchema

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_schema.validate(X)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return apply_threshold(self.predict_proba(X), self.threshold)


def _require_file(path: Path, what: str) -> None:
    if not path.exists():
        raise ArtifactLoadError(f"Missing {what} at expected path: {path}")


def load_model_artifacts(version: str, artifact_config: ArtifactConfig) -> ModelArtifacts:
    model_dir = artifact_config.models_dir / artifact_config.model_name / version
    threshold_path = artifact_config.thresholds_dir / version / "threshold.json"
    metadata_path = artifact_config.metadata_dir / version / "model_metadata.json"

    model_path = model_dir / "model.joblib"
    raw_pipeline_path = model_dir / "raw_pipeline.joblib"
    shap_metadata_path = model_dir / "shap_metadata.json"

    for path, label in [
        (model_path, "model.joblib"), (raw_pipeline_path, "raw_pipeline.joblib"),
        (shap_metadata_path, "shap_metadata.json"), (threshold_path, "threshold.json"),
        (metadata_path, "model_metadata.json"),
    ]:
        _require_file(path, label)

    # SECURITY: trusted, repo-local path only -- see module docstring.
    model = joblib.load(model_path)
    raw_pipeline = joblib.load(raw_pipeline_path)

    with open(shap_metadata_path) as f:
        shap_metadata = json.load(f)
    with open(threshold_path) as f:
        threshold_payload = json.load(f)
    with open(metadata_path) as f:
        metadata = json.load(f)

    threshold = threshold_payload.get("threshold")
    if threshold is None or not (0.0 <= threshold <= 1.0):
        raise ArtifactLoadError(f"Invalid threshold in {threshold_path}: {threshold}")

    numerical_features = metadata.get("numerical_features")
    categorical_features = metadata.get("categorical_features")
    if numerical_features is None or categorical_features is None:
        raise ArtifactLoadError("model_metadata.json is missing feature schema information.")

    feature_schema = FeatureSchema(tuple(numerical_features), tuple(categorical_features))

    logger.info("Loaded model artifacts for version=%s", version)
    return ModelArtifacts(
        model=model,
        raw_pipeline=raw_pipeline,
        threshold=threshold,
        threshold_metadata=threshold_payload,
        metadata=metadata,
        shap_metadata=shap_metadata,
        feature_schema=feature_schema,
    )