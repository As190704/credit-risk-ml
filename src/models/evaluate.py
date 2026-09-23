# src/models/evaluate.py
"""
Evaluation utilities: classification metrics, curves, structured comparison.

This module is the single source of truth for how a "score" is computed
throughout the project (CV, tuning diagnostics, final test evaluation),
which keeps every reported number comparable.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Computes a standard suite of binary classification metrics.

    Probability-dependent metrics (ROC AUC, PR AUC, Brier score, log loss)
    are set to NaN (with a warning) when y_proba is not supplied or when
    y_true contains a single class, since those metrics are undefined in
    that scenario -- this avoids silently misleading results.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    metrics: Dict[str, float] = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    n_classes = len(np.unique(y_true))
    if y_proba is not None and n_classes > 1:
        y_proba = np.asarray(y_proba)
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
        metrics["pr_auc"] = average_precision_score(y_true, y_proba)
        metrics["brier_score"] = brier_score_loss(y_true, y_proba)
        try:
            metrics["log_loss"] = log_loss(y_true, y_proba, labels=[0, 1])
        except ValueError as exc:
            logger.warning("Could not compute log loss: %s", exc)
            metrics["log_loss"] = float("nan")
    else:
        if y_proba is not None and n_classes <= 1:
            logger.warning(
                "Only one class present in y_true; probability-based "
                "metrics (ROC AUC, PR AUC, Brier, log loss) are undefined "
                "and set to NaN."
            )
        metrics.update(
            {"roc_auc": float("nan"), "pr_auc": float("nan"),
             "brier_score": float("nan"), "log_loss": float("nan")}
        )

    return metrics


def compute_confusion_matrix(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def compute_roc_curve(y_true, y_proba) -> Dict[str, np.ndarray]:
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds}


def compute_pr_curve(y_true, y_proba) -> Dict[str, np.ndarray]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


def plot_confusion_matrix(cm: np.ndarray, output_path: Path, labels=("0", "1")) -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def plot_roc_curve(y_true, y_proba, output_path: Path) -> None:
    curve = compute_roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curve["fpr"], curve["tpr"], label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(); fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path); plt.close(fig)


def plot_pr_curve(y_true, y_proba, output_path: Path) -> None:
    curve = compute_pr_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(curve["recall"], curve["precision"], label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(); fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path); plt.close(fig)


@dataclass
class ModelComparisonEntry:
    model_name: str
    cv_mean_metrics: Dict[str, float]
    cv_std_metrics: Dict[str, float]
    validation_metrics: Optional[Dict[str, float]]
    training_time_seconds: float
    model_config: Dict[str, Any]
    limitations: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "cv_mean_metrics": self.cv_mean_metrics,
            "cv_std_metrics": self.cv_std_metrics,
            "validation_metrics": self.validation_metrics,
            "training_time_seconds": self.training_time_seconds,
            "model_config": self.model_config,
            "limitations": self.limitations,
        }


def build_comparison_report(entries: Sequence[ModelComparisonEntry]) -> pd.DataFrame:
    """Tabular multi-metric comparison. No model is selected here based on
    a single metric -- ranking is a human/documented decision (see README)."""
    rows = []
    for entry in entries:
        row = {"model_name": entry.model_name, "training_time_seconds": entry.training_time_seconds}
        row.update({f"cv_{k}": v for k, v in entry.cv_mean_metrics.items()})
        if entry.validation_metrics:
            row.update({f"val_{k}": v for k, v in entry.validation_metrics.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def save_evaluation_report(report: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved to %s", path)