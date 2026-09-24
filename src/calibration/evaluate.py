# src/calibration/evaluate.py
"""
Calibration evaluation: reliability curves, Brier score, log loss, and a
structured before/after comparison against the SAME evaluation set.

Reuses src.models.evaluate.compute_classification_metrics as the single
source of truth for ROC AUC / PR AUC / Brier score / log loss, so
calibration metrics are directly comparable to Phase 3-5 reporting.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

from src.models.evaluate import compute_classification_metrics

logger = logging.getLogger(__name__)


def compare_calibration(
    y_true: np.ndarray, y_proba_before: np.ndarray, y_proba_after: np.ndarray
) -> Dict[str, Any]:
    """
    Computes metrics before/after calibration on the SAME (y_true) set.

    IMPORTANT: ROC AUC / PR AUC are discrimination metrics and are not
    expected to improve from calibration (they may move slightly due to
    monotonic-but-not-rank-preserving effects of sigmoid fitting, but
    large changes would be unusual). Brier score / log loss measure
    reliability and are the metrics calibration is actually expected to
    improve.
    """
    y_true = np.asarray(y_true)
    pred_before = (np.asarray(y_proba_before) >= 0.5).astype(int)
    pred_after = (np.asarray(y_proba_after) >= 0.5).astype(int)

    before = compute_classification_metrics(y_true, pred_before, y_proba_before)
    after = compute_classification_metrics(y_true, pred_after, y_proba_after)

    return {
        "before_calibration": before,
        "after_calibration": after,
        "brier_score_delta": after["brier_score"] - before["brier_score"],
        "log_loss_delta": after["log_loss"] - before["log_loss"],
        "roc_auc_delta": after["roc_auc"] - before["roc_auc"],
        "interpretation": (
            "Negative brier_score_delta / log_loss_delta indicates improved "
            "probability reliability after calibration. roc_auc_delta is "
            "reported for transparency but discrimination is NOT the goal "
            "of this step; do not select a calibration method based on "
            "roc_auc_delta alone."
        ),
    }


def plot_calibration_curve(
    y_true: np.ndarray,
    y_proba_before: np.ndarray,
    y_proba_after: np.ndarray,
    output_path: Union[str, Path],
    n_bins: int = 10,
) -> None:
    frac_before, mean_before = calibration_curve(y_true, y_proba_before, n_bins=n_bins, strategy="uniform")
    frac_after, mean_after = calibration_curve(y_true, y_proba_after, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_before, frac_before, marker="o", label="Before calibration")
    ax.plot(mean_after, frac_after, marker="o", label="After calibration (sigmoid)")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency of positive class")
    ax.set_title("Reliability diagram")
    ax.legend()
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_probability_distribution(
    y_proba_before: np.ndarray, y_proba_after: np.ndarray, output_path: Union[str, Path]
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    axes[0].hist(y_proba_before, bins=20, color="steelblue")
    axes[0].set_title("Predicted probability (before)")
    axes[1].hist(y_proba_after, bins=20, color="darkorange")
    axes[1].set_title("Predicted probability (after)")
    for ax in axes:
        ax.set_xlabel("Predicted probability")
    axes[0].set_ylabel("Count")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def save_calibration_report(report: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Calibration report saved to %s", path)