# src/models/train.py
"""
Model training and cross-validation orchestration.

Responsibility: run StratifiedKFold cross-validation on TRAINING data only,
fit the final pipeline, and produce validation probabilities for downstream
threshold optimization -- all without ever touching the held-out test set.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.config import CVConfig
from src.models.evaluate import compute_classification_metrics

logger = logging.getLogger(__name__)


@dataclass
class CVResult:
    model_name: str
    fold_metrics: pd.DataFrame
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    training_time_seconds: float
    oof_probabilities: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "mean_metrics": self.mean_metrics,
            "std_metrics": self.std_metrics,
            "training_time_seconds": self.training_time_seconds,
            "fold_metrics": self.fold_metrics.to_dict(orient="records"),
        }


def make_cv_splitter(cv_config: CVConfig) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=cv_config.n_splits, shuffle=cv_config.shuffle, random_state=cv_config.random_state
    )


def cross_validate_model(
    model_name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_config: CVConfig,
    compute_oof: bool = True,
) -> CVResult:
    """
    Runs stratified k-fold CV on TRAINING data only.

    Each fold clones the full pipeline (preprocessing + classifier) and
    fits it on that fold's training portion only. This is what prevents
    preprocessing leakage across folds.
    """
    cv = make_cv_splitter(cv_config)

    X_reset = X_train.reset_index(drop=True)
    y_reset = y_train.reset_index(drop=True)

    fold_rows: List[Dict[str, Any]] = []
    oof_proba = np.full(len(y_reset), np.nan)

    start = time.time()
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_reset, y_reset)):
        fold_pipeline = clone(pipeline)

        X_fold_train, X_fold_val = X_reset.iloc[train_idx], X_reset.iloc[val_idx]
        y_fold_train, y_fold_val = y_reset.iloc[train_idx], y_reset.iloc[val_idx]

        fold_pipeline.fit(X_fold_train, y_fold_train)

        y_fold_proba = fold_pipeline.predict_proba(X_fold_val)[:, 1]
        y_fold_pred = (y_fold_proba >= 0.5).astype(int)

        if compute_oof:
            oof_proba[val_idx] = y_fold_proba

        metrics = compute_classification_metrics(y_fold_val.values, y_fold_pred, y_fold_proba)
        metrics["fold"] = fold_idx
        fold_rows.append(metrics)

    elapsed = time.time() - start

    fold_metrics_df = pd.DataFrame(fold_rows)
    numeric_cols = [c for c in fold_metrics_df.columns if c != "fold"]
    mean_metrics = fold_metrics_df[numeric_cols].mean().to_dict()
    std_metrics = fold_metrics_df[numeric_cols].std().to_dict()

    logger.info(
        "CV complete for %s: roc_auc=%.4f (+/-%.4f), pr_auc=%.4f, time=%.1fs",
        model_name, mean_metrics.get("roc_auc", float("nan")),
        std_metrics.get("roc_auc", float("nan")), mean_metrics.get("pr_auc", float("nan")), elapsed,
    )

    return CVResult(
        model_name=model_name,
        fold_metrics=fold_metrics_df,
        mean_metrics=mean_metrics,
        std_metrics=std_metrics,
        training_time_seconds=elapsed,
        oof_probabilities=oof_proba if compute_oof else None,
    )


def fit_final_pipeline(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fits a clone of the given pipeline on the FULL training set. This is
    the artifact used for final test-set evaluation and later deployment."""
    fitted = clone(pipeline)
    fitted.fit(X_train, y_train)
    return fitted


def get_validation_probabilities(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_config: CVConfig,
    strategy: str = "oof",
    holdout_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produces (y_val_true, y_val_proba) for threshold optimization WITHOUT
    touching the held-out test set.

    strategy="oof": cross_val_predict across the full training set -- every
        training sample gets a probability from a fold that never trained
        on it. Data-efficient; recommended default.
    strategy="holdout": carves an internal validation split out of the
        training set. Simpler mental model, costs some training data.
    """
    if strategy == "oof":
        cv = make_cv_splitter(cv_config)
        y_proba = cross_val_predict(pipeline, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        return y_train.to_numpy(), y_proba

    if strategy == "holdout":
        from src.features.split import train_test_split_data  # reuse Phase 2 utility

        X_tr, X_val, y_tr, y_val = train_test_split_data(
            X_train, y_train, test_size=holdout_size, random_state=random_state, stratify=True
        )
        fitted = clone(pipeline)
        fitted.fit(X_tr, y_tr)
        y_proba = fitted.predict_proba(X_val)[:, 1]
        return y_val.to_numpy(), y_proba

    raise ValueError(f"Unknown validation strategy: {strategy}. Use 'oof' or 'holdout'.")