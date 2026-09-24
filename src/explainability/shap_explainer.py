# src/explainability/shap_explainer.py
"""
SHAP explainer construction and feature-name mapping.

Critical design decision (Phase 7.5): SHAP is built on the UNCALIBRATED
XGBoost pipeline (`raw_pipeline`), never on a CalibratedClassifierCV
wrapper. Calibration reshapes probabilities; it does not have a
tree-based structure that TreeExplainer can exploit, and mixing the two
concepts would make explanations misleading about which object produced
the final decision.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


class ShapExplainabilityError(Exception):
    """Raised for SHAP setup or computation failures."""


def build_tree_explainer(raw_pipeline: Pipeline) -> shap.TreeExplainer:
    """Builds a TreeExplainer from the fitted XGBClassifier step only."""
    if "classifier" not in raw_pipeline.named_steps:
        raise ShapExplainabilityError("raw_pipeline must contain a 'classifier' step.")
    xgb_model = raw_pipeline.named_steps["classifier"]
    return shap.TreeExplainer(xgb_model)


def transform_for_shap(raw_pipeline: Pipeline, X: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    """Transforms raw input into the representation the XGBoost step
    actually consumes, using the FITTED preprocessor's own feature names."""
    if "preprocessor" not in raw_pipeline.named_steps:
        raise ShapExplainabilityError("raw_pipeline must contain a 'preprocessor' step.")
    preprocessor = raw_pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X)
    feature_names = list(preprocessor.get_feature_names_out())
    return X_transformed, feature_names


def build_feature_name_mapping(
    raw_pipeline: Pipeline,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
) -> Dict[str, List[str]]:
    """
    Maps each ORIGINAL feature to the list of TRANSFORMED (post-
    ColumnTransformer) feature names it produced.

    Numerical features map 1:1. Categorical features expand to one entry
    per one-hot column. Matching is done against the fitted preprocessor's
    `get_feature_names_out()` output (the authoritative source), using the
    known list of original categorical column names to disambiguate
    prefixes -- never by guessing dataset-specific business names.
    """
    preprocessor = raw_pipeline.named_steps["preprocessor"]
    transformed_names = list(preprocessor.get_feature_names_out())

    mapping: Dict[str, List[str]] = {c: [] for c in list(numerical_features) + list(categorical_features)}
    sorted_categorical = sorted(categorical_features, key=len, reverse=True)  # longest-match-first

    for name in transformed_names:
        if name.startswith("numerical__"):
            original = name[len("numerical__"):]
            if original in mapping:
                mapping[original].append(name)
            else:
                logger.warning("Unmapped numerical transformed feature: %s", name)
            continue

        if name.startswith("categorical__"):
            remainder = name[len("categorical__"):]
            matched_col = None
            for col in sorted_categorical:
                if remainder == col or remainder.startswith(f"{col}_"):
                    matched_col = col
                    break
            if matched_col is not None:
                mapping[matched_col].append(name)
            else:
                logger.warning("Unmapped categorical transformed feature: %s", name)
            continue

        logger.warning("Transformed feature name with unrecognized prefix: %s", name)

    return mapping


def compute_shap_values(explainer: shap.TreeExplainer, X_transformed: np.ndarray) -> shap.Explanation:
    """Computes SHAP values. Deterministic for TreeExplainer's exact algorithm."""
    return explainer(X_transformed)