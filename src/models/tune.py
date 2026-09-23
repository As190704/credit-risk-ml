# src/models/tune.py
"""
XGBoost hyperparameter tuning via RandomizedSearchCV.

Responsibility: search a configurable hyperparameter space over the full
(preprocessing + classifier) pipeline using StratifiedKFold, without ever
touching the held-out test set.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import joblib
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config import TuningConfig

logger = logging.getLogger(__name__)

OVERFITTING_GAP_WARNING_THRESHOLD = 0.10


@dataclass
class TuningResult:
    best_params: Dict[str, Any]
    best_score: float
    cv_results: pd.DataFrame
    best_estimator: Pipeline
    scoring: str
    search_time_seconds: float
    train_val_gap: float

    def to_metadata_dict(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "scoring": self.scoring,
            "search_time_seconds": self.search_time_seconds,
            "train_val_gap": self.train_val_gap,
        }


def run_randomized_search(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tuning_config: TuningConfig,
) -> TuningResult:
    """
    Runs RandomizedSearchCV over the full pipeline. Every candidate
    configuration is evaluated with fold-specific preprocessing; the
    held-out test set is never referenced here.
    """
    cv = StratifiedKFold(
        n_splits=tuning_config.n_splits, shuffle=True, random_state=tuning_config.random_state
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=tuning_config.param_distributions,
        n_iter=tuning_config.n_iter,
        scoring=tuning_config.scoring,
        cv=cv,
        random_state=tuning_config.random_state,
        n_jobs=tuning_config.n_jobs,
        refit=True,
        return_train_score=True,
        verbose=1,
    )

    start = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - start

    cv_results = pd.DataFrame(search.cv_results_)
    train_score = float(cv_results.loc[search.best_index_, "mean_train_score"])
    val_score = float(cv_results.loc[search.best_index_, "mean_test_score"])
    gap = train_score - val_score

    if gap > OVERFITTING_GAP_WARNING_THRESHOLD:
        logger.warning(
            "Train/validation score gap of %.4f detected for the best XGBoost "
            "configuration (train=%.4f, val=%.4f) -- possible overfitting. "
            "Consider stronger regularization (reg_alpha/reg_lambda), lower "
            "max_depth, or fewer estimators.", gap, train_score, val_score,
        )

    logger.info(
        "RandomizedSearchCV complete: best_score=%.4f (%s), best_params=%s, time=%.1fs",
        search.best_score_, tuning_config.scoring, search.best_params_, elapsed,
    )

    return TuningResult(
        best_params=search.best_params_,
        best_score=search.best_score_,
        cv_results=cv_results,
        best_estimator=search.best_estimator_,
        scoring=tuning_config.scoring,
        search_time_seconds=elapsed,
        train_val_gap=gap,
    )


def save_tuning_artifacts(
    result: TuningResult,
    models_dir: Union[str, Path],
    metadata_dir: Union[str, Path],
    version: str,
) -> Dict[str, Path]:
    """Saves versioned artifacts: best estimator (joblib), full cv_results
    (csv), and metadata (json)."""
    models_dir = Path(models_dir); metadata_dir = Path(metadata_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"xgboost_tuned_{version}.joblib"
    cv_results_path = metadata_dir / f"xgboost_tuning_cv_results_{version}.csv"
    metadata_path = metadata_dir / f"xgboost_tuning_metadata_{version}.json"

    joblib.dump(result.best_estimator, model_path)
    result.cv_results.to_csv(cv_results_path, index=False)
    with open(metadata_path, "w") as f:
        json.dump(result.to_metadata_dict(), f, indent=2, default=str)

    logger.info("Tuning artifacts saved: %s | %s | %s", model_path, cv_results_path, metadata_path)
    return {"model_path": model_path, "cv_results_path": cv_results_path, "metadata_path": metadata_path}