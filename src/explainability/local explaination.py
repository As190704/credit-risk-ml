# src/explainability/local_explanation.py
"""
Local (single-prediction) SHAP explanation.

Distinguishes explicitly between:
  - `prediction_probability`: from the CALIBRATED model (the actual
    deployable probability).
  - `top_features` impacts: from SHAP on the UNCALIBRATED raw model's
    log-odds output (ranking/explanatory signal, not a decomposition of
    the calibrated probability itself).
This distinction is stated in the output's `explanation_note` field so
downstream consumers (e.g. the future FastAPI/frontend) never conflate
the two.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.explainability.shap_explainer import (
    build_feature_name_mapping, build_tree_explainer, transform_for_shap,
)
from src.features.preprocessing import FeatureSchema
from src.optimization.threshold import apply_threshold

logger = logging.getLogger(__name__)


def explain_prediction(
    raw_pipeline: Pipeline,
    calibrated_model: Any,
    threshold: float,
    input_df: pd.DataFrame,
    feature_schema: FeatureSchema,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    top_n: int = 5,
) -> Dict[str, Any]:
    """
    Produces a structured local explanation for exactly one row of input.

    Raises FeatureSchema.SchemaValidationError (via feature_schema.validate)
    if required columns are missing -- reuses the exact validation
    mechanism designed in Phase 2 for training/inference schema
    consistency, rather than duplicating validation logic here.
    """
    feature_schema.validate(input_df)
    if len(input_df) != 1:
        raise ValueError(f"explain_prediction expects exactly 1 row, got {len(input_df)}.")

    proba = float(calibrated_model.predict_proba(input_df)[:, 1][0])
    prediction = int(apply_threshold(np.array([proba]), threshold)[0])

    X_transformed, transformed_names = transform_for_shap(raw_pipeline, input_df)
    explainer = build_tree_explainer(raw_pipeline)
    shap_values = explainer(X_transformed)
    values = shap_values.values[0]

    mapping = build_feature_name_mapping(raw_pipeline, numerical_features, categorical_features)
    original_impacts: Dict[str, float] = {}
    for original_col, children in mapping.items():
        idxs = [transformed_names.index(c) for c in children if c in transformed_names]
        if idxs:
            original_impacts[original_col] = float(np.sum(values[idxs]))

    ranked: List[Any] = sorted(original_impacts.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
    top_features = [
        {
            "feature": name,
            "impact": round(impact, 6),
            "direction": "higher_risk" if impact > 0 else "lower_risk",
        }
        for name, impact in ranked
    ]

    return {
        "prediction_probability": round(proba, 6),
        "prediction": prediction,
        "threshold_used": float(threshold),
        "top_features": top_features,
        "explanation_note": (
            "`prediction_probability` comes from the CALIBRATED model. "
            "`top_features` impacts come from SHAP applied to the "
            "underlying UNCALIBRATED XGBoost model's log-odds output and "
            "describe relative feature contribution to model behavior, "
            "not a causal explanation and not an exact decomposition of "
            "the calibrated probability. 'higher_risk'/'lower_risk' "
            "assumes the positive class (1) represents higher credit risk "
            "per the project's documented target definition. This output "
            "is decision support, not an automated final lending decision."
        ),
    }