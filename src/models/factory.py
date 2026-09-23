# src/models/factory.py
"""
Model factory.

Responsibility: construct UNFITTED end-to-end pipelines (preprocessing +
estimator) for each supported model family. Training, evaluation, and
tuning are handled by other modules -- this module has no knowledge of
cross-validation, metrics, or search strategies.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional, Sequence

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import ModelConfig, PreprocessingConfig
from src.features.preprocessing import build_preprocessor

logger = logging.getLogger(__name__)

ModelName = Literal["logistic_regression", "xgboost"]


def compute_scale_pos_weight(y: pd.Series) -> float:
    """Computes the negative/positive class ratio from training labels.

    This is only ever called when the caller explicitly opts into
    `class_imbalance_strategy="scale_pos_weight"` -- imbalance correction
    is never applied silently.
    """
    y = pd.Series(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    if pos == 0:
        raise ValueError("Cannot compute scale_pos_weight: no positive class samples found.")
    ratio = neg / pos
    logger.info("Training class distribution -> negative=%s, positive=%s, ratio=%.3f", neg, pos, ratio)
    return float(ratio)


def build_logistic_regression(
    config: ModelConfig, class_weight_override: Optional[str] = None
) -> LogisticRegression:
    params = dict(config.logistic_regression_params)
    params.setdefault("random_state", config.random_state)
    if class_weight_override is not None:
        params["class_weight"] = class_weight_override
    return LogisticRegression(**params)


def build_xgboost(
    config: ModelConfig, scale_pos_weight: Optional[float] = None
) -> XGBClassifier:
    params = dict(config.xgboost_params)
    params.setdefault("random_state", config.random_state)
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)


def build_model_pipeline(
    model_name: ModelName,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    preprocessing_config: PreprocessingConfig,
    model_config: ModelConfig,
    y_train: Optional[pd.Series] = None,
) -> Pipeline:
    """
    Builds an unfitted Pipeline([preprocessor, classifier]).

    `y_train` is only required when `model_config.class_imbalance_strategy`
    needs the actual training label distribution (i.e. "scale_pos_weight").
    """
    model_family = "linear" if model_name == "logistic_regression" else "tree"
    preprocessor = build_preprocessor(
        numerical_features, categorical_features, preprocessing_config, model_family=model_family
    )

    class_weight = None
    scale_pos_weight = None

    if model_config.class_imbalance_strategy == "balanced" and model_name == "logistic_regression":
        class_weight = "balanced"
    elif model_config.class_imbalance_strategy == "scale_pos_weight" and model_name == "xgboost":
        if y_train is None:
            raise ValueError("y_train is required to compute scale_pos_weight for XGBoost.")
        scale_pos_weight = compute_scale_pos_weight(y_train)
    elif model_config.class_imbalance_strategy is not None:
        logger.warning(
            "class_imbalance_strategy=%s is not applicable to model_name=%s; ignoring.",
            model_config.class_imbalance_strategy, model_name,
        )

    if model_name == "logistic_regression":
        estimator = build_logistic_regression(model_config, class_weight_override=class_weight)
    elif model_name == "xgboost":
        estimator = build_xgboost(model_config, scale_pos_weight=scale_pos_weight)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", estimator)])