# src/explainability/global_explanation.py
"""
Global SHAP explanation: aggregate feature importance across a
representative sample of TRAINING data (never the held-out test set for
model-selection purposes -- global explanation is a diagnostic artifact,
not part of the model-selection loop, so this restriction is about
avoiding misleading "the model relies on X" claims sourced from data the
model was evaluated on, not a leakage concern per se).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.explainability.shap_explainer import (
    build_feature_name_mapping, build_tree_explainer, transform_for_shap,
)

logger = logging.getLogger(__name__)


def generate_global_shap_report(
    raw_pipeline: Pipeline,
    X_sample: pd.DataFrame,
    numerical_features: Sequence[str],
    categorical_features: Sequence[str],
    output_dir: Union[str, Path],
    max_display: int = 20,
    sample_size: int = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if sample_size is not None and len(X_sample) > sample_size:
        X_sample = X_sample.sample(n=sample_size, random_state=random_state)
        logger.info("Sampled %s rows for global SHAP computation.", sample_size)

    X_transformed, transformed_names = transform_for_shap(raw_pipeline, X_sample)
    explainer = build_tree_explainer(raw_pipeline)
    shap_values = explainer(X_transformed)  # shap.Explanation, shape (n, n_transformed_features)

    mean_abs_transformed = np.abs(shap_values.values).mean(axis=0)

    mapping = build_feature_name_mapping(raw_pipeline, numerical_features, categorical_features)
    aggregated_importance: Dict[str, float] = {}
    for original_col, children in mapping.items():
        idxs = [transformed_names.index(c) for c in children if c in transformed_names]
        aggregated_importance[original_col] = float(np.sum(mean_abs_transformed[idxs])) if idxs else 0.0

    importance_df = (
        pd.DataFrame({"feature": list(aggregated_importance.keys()),
                      "mean_abs_shap": list(aggregated_importance.values())})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(output_dir / "feature_importance.csv", index=False)

    # Beeswarm plot on the TRANSFORMED (exact SHAP computation) space --
    # aggregating one-hot children into a single beeswarm row is not
    # mathematically well-defined for per-sample SHAP dot plots, so the
    # beeswarm intentionally shows the transformed feature space while the
    # bar chart below shows the user-friendly aggregated view.
    plt.figure(figsize=(9, max(4, min(len(transformed_names), max_display) * 0.35)))
    shap.summary_plot(
        shap_values.values, X_transformed, feature_names=transformed_names,
        max_display=max_display, show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "shap_summary.png", bbox_inches="tight")
    plt.close()

    top = importance_df.head(max_display).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, max(4, len(top) * 0.35)))
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean |SHAP value| (log-odds space, aggregated across one-hot children)")
    ax.set_title("Global feature importance (original features)")
    fig.tight_layout()
    fig.savefig(output_dir / "shap_bar.png")
    plt.close(fig)

    summary = {
        "n_samples_explained": int(X_transformed.shape[0]),
        "n_transformed_features": int(X_transformed.shape[1]),
        "n_original_features": len(mapping),
        "expected_value": float(np.ravel(explainer.expected_value)[0]),
        "top_10_features": importance_df.head(10).to_dict(orient="records"),
        "note": (
            "SHAP values are computed on the uncalibrated XGBoost model's "
            "raw margin (log-odds) output and describe model behavior, not "
            "causal relationships. Importance for one-hot encoded "
            "categorical features is aggregated by summing the mean "
            "absolute SHAP value across their encoded categories."
        ),
    }
    logger.info("Global SHAP report written to %s", output_dir)
    return summary