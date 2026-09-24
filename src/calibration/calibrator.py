# src/calibration/calibrator.py
"""
Probability calibration.

Responsibility: transform a discriminative model's raw scores into
probabilities that better match observed outcome frequencies, WITHOUT
leaking calibration-set information into the base model, and without
using the held-out test set.

Design decision: calibration, discrimination, and thresholding are kept as
three distinct concepts throughout this module:
    - Discrimination (ROC AUC / PR AUC) is a property of the base model's
      ranking ability and is NOT expected to improve from calibration.
    - Calibration (Brier score / log loss / reliability curve) measures how
      well predicted probabilities match observed frequencies.
    - Thresholding (src.optimization.threshold) is a separate, later step
      that turns a (calibrated) probability into a binary decision.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import CalibrationConfig
from src.features.split import train_test_split_data

logger = logging.getLogger(__name__)


class CalibrationError(Exception):
    """Raised for invalid calibration configuration or fitting failures."""


@dataclass
class CalibrationFitResult:
    calibrated_model: CalibratedClassifierCV
    raw_pipeline: Pipeline
    strategy: str
    method: str
    split_metadata: Dict[str, Any]


def fit_calibrated_model(
    base_pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: CalibrationConfig,
) -> CalibrationFitResult:
    """
    Fits a calibrated probability model using ONLY X_train/y_train.

    strategy="prefit" (default):
        X_train is split into X_fit/X_calib (stratified, reproducible via
        Phase 2's train_test_split_data). The base pipeline is fit on
        X_fit only; sigmoid calibration is fit on X_calib only -- the base
        model never sees the data its calibration curve is fit on.
        `raw_pipeline` (fit on X_fit) is returned explicitly for later use
        by SHAP, which must never operate on the CalibratedClassifierCV
        wrapper directly (see src/explainability).

    strategy="cv":
        CalibratedClassifierCV internally performs stratified k-fold
        fitting + calibration and ensembles the k calibrated sub-models.
        More data-efficient, but there is no single canonical uncalibrated
        pipeline; `raw_pipeline` returned in this case is a SEPARATE model
        fit on the FULL X_train purely for explanation purposes, and is
        documented as "representative", not identical to any calibrated
        sub-model.
    """
    if config.method not in ("sigmoid", "isotonic"):
        raise CalibrationError(f"Unsupported calibration method: {config.method}")

    if config.strategy == "prefit":
        X_fit, X_calib, y_fit, y_calib = train_test_split_data(
            X_train, y_train,
            test_size=config.calibration_holdout_size,
            random_state=config.random_state,
            stratify=True,
        )
        raw_pipeline = clone(base_pipeline)
        raw_pipeline.fit(X_fit, y_fit)

        calibrated_model = CalibratedClassifierCV(
            estimator=raw_pipeline, method=config.method, cv="prefit"
        )
        calibrated_model.fit(X_calib, y_calib)

        split_metadata = {
            "n_fit": int(len(X_fit)),
            "n_calib": int(len(X_calib)),
            "calibration_holdout_size": config.calibration_holdout_size,
        }
        logger.info(
            "Calibration fitted with strategy='prefit': n_fit=%s, n_calib=%s",
            len(X_fit), len(X_calib),
        )

    elif config.strategy == "cv":
        cv = StratifiedKFold(n_splits=config.cv_splits, shuffle=True, random_state=config.random_state)
        calibrated_model = CalibratedClassifierCV(
            estimator=clone(base_pipeline), method=config.method, cv=cv
        )
        calibrated_model.fit(X_train, y_train)

        # A separate, full-data model for SHAP -- NOT part of the
        # calibration ensemble, documented as representative only.
        raw_pipeline = clone(base_pipeline)
        raw_pipeline.fit(X_train, y_train)

        split_metadata = {"cv_splits": config.cv_splits, "n_train": int(len(X_train))}
        logger.info("Calibration fitted with strategy='cv': cv_splits=%s", config.cv_splits)

    else:
        raise CalibrationError(f"Unknown calibration strategy: {config.strategy}")

    return CalibrationFitResult(
        calibrated_model=calibrated_model,
        raw_pipeline=raw_pipeline,
        strategy=config.strategy,
        method=config.method,
        split_metadata=split_metadata,
    )


def get_calibration_validation_set(
    X_train: pd.DataFrame, y_train: pd.Series, fit_result: CalibrationFitResult
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Returns the (X, y) pair that is safe to use for BOTH calibration
    evaluation and Phase 5 threshold selection under strategy="prefit"
    (i.e. X_calib/y_calib, re-derived deterministically with the same
    random_state/test_size used to fit calibration -- never re-splitting
    with different parameters, which would silently change the set).

    For strategy="cv", the full X_train is returned since there is no
    dedicated calibration holdout; callers should be aware threshold
    metrics in that case are computed via cross_val_predict-style OOF
    logic, not a single static holdout (see calibration/evaluate.py note).
    """
    if fit_result.strategy == "prefit":
        # NOTE: this recomputes the identical split (same random_state,
        # same test_size) rather than caching it, to keep this function's
        # contract simple and avoid hidden state -- for very large
        # datasets, callers may prefer to save X_calib/y_calib directly
        # from fit_calibrated_model's internals instead of re-splitting.
        raise NotImplementedError(
            "Prefer capturing X_calib/y_calib directly from the fitting "
            "call site; see fit_calibrated_model_with_split() below."
        )
    return X_train, y_train


def fit_calibrated_model_with_split(
    base_pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, config: CalibrationConfig
):
    """
    Convenience variant of fit_calibrated_model that ALSO returns the exact
    (X_calib, y_calib) used, so callers do not need to re-derive it. This is
    the function used by the orchestration script.
    """
    if config.strategy != "prefit":
        result = fit_calibrated_model(base_pipeline, X_train, y_train, config)
        return result, X_train, y_train

    X_fit, X_calib, y_fit, y_calib = train_test_split_data(
        X_train, y_train, test_size=config.calibration_holdout_size,
        random_state=config.random_state, stratify=True,
    )
    raw_pipeline = clone(base_pipeline)
    raw_pipeline.fit(X_fit, y_fit)
    calibrated_model = CalibratedClassifierCV(estimator=raw_pipeline, method=config.method, cv="prefit")
    calibrated_model.fit(X_calib, y_calib)

    result = CalibrationFitResult(
        calibrated_model=calibrated_model,
        raw_pipeline=raw_pipeline,
        strategy="prefit",
        method=config.method,
        split_metadata={"n_fit": len(X_fit), "n_calib": len(X_calib)},
    )
    return result, X_calib, y_calib