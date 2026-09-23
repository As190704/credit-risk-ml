# src/optimization/threshold.py
"""
Classification threshold optimization.

A predicted probability is a risk-ranking signal, not a lending decision.
This module converts probabilities into binary predictions according to an
explicitly configured, auditable objective -- never an assumed 0.5 cutoff,
and never a hardcoded business cost.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = tuple(np.round(np.arange(0.01, 1.00, 0.01), 2))


class ThresholdSelectionError(Exception):
    """Raised when no threshold satisfies a requested constraint."""


def evaluate_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray, thresholds: Sequence[float] = DEFAULT_THRESHOLDS
) -> pd.DataFrame:
    """Builds a threshold-metric table: precision, recall, F1, and
    confusion-matrix counts at each candidate threshold."""
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    rows = []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        rows.append(
            {
                "threshold": float(t),
                "precision": precision_score(y_true, y_pred, zero_division=0),
                "recall": recall_score(y_true, y_pred, zero_division=0),
                "f1": f1_score(y_true, y_pred, zero_division=0),
                "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class ThresholdSelectionResult:
    threshold: float
    strategy: str
    rationale: str
    metrics_at_threshold: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "strategy": self.strategy,
            "rationale": self.rationale,
            "metrics_at_threshold": self.metrics_at_threshold,
        }


def _row_to_metrics(row: pd.Series) -> Dict[str, float]:
    return {k: float(row[k]) for k in ["precision", "recall", "f1", "tp", "fp", "tn", "fn"]}


def select_threshold_maximize_f1(table: pd.DataFrame) -> ThresholdSelectionResult:
    row = table.loc[table["f1"].idxmax()]
    return ThresholdSelectionResult(
        threshold=float(row["threshold"]),
        strategy="maximize_f1",
        rationale=(
            "Maximizes F1 (harmonic mean of precision and recall). Used as a "
            "neutral default when no explicit business cost or minimum "
            "recall/precision requirement has been documented."
        ),
        metrics_at_threshold=_row_to_metrics(row),
    )


def select_threshold_min_recall(table: pd.DataFrame, min_recall: float) -> ThresholdSelectionResult:
    candidates = table[table["recall"] >= min_recall]
    if candidates.empty:
        raise ThresholdSelectionError(f"No threshold in the evaluated range achieves recall >= {min_recall}.")
    row = candidates.loc[candidates["precision"].idxmax()]
    return ThresholdSelectionResult(
        threshold=float(row["threshold"]),
        strategy="min_recall",
        rationale=(
            f"Enforces a minimum recall of {min_recall} (limiting missed "
            "defaults / false negatives), then maximizes precision among "
            "thresholds that satisfy this floor."
        ),
        metrics_at_threshold=_row_to_metrics(row),
    )


def select_threshold_min_precision(table: pd.DataFrame, min_precision: float) -> ThresholdSelectionResult:
    candidates = table[table["precision"] >= min_precision]
    if candidates.empty:
        raise ThresholdSelectionError(f"No threshold in the evaluated range achieves precision >= {min_precision}.")
    row = candidates.loc[candidates["recall"].idxmax()]
    return ThresholdSelectionResult(
        threshold=float(row["threshold"]),
        strategy="min_precision",
        rationale=(
            f"Enforces a minimum precision of {min_precision} (limiting "
            "incorrectly flagged low-risk applicants / false positives), "
            "then maximizes recall among thresholds that satisfy this floor."
        ),
        metrics_at_threshold=_row_to_metrics(row),
    )


def select_threshold_cost_based(table: pd.DataFrame, cost_fp: float, cost_fn: float) -> ThresholdSelectionResult:
    """Minimizes total_cost = cost_fp * FP + cost_fn * FN.

    cost_fp/cost_fn MUST be supplied explicitly by the caller from a
    documented business policy -- this function never assumes default cost
    values, since that is a business decision outside ML scope.
    """
    costs = table["fp"] * cost_fp + table["fn"] * cost_fn
    row = table.loc[costs.idxmin()]
    return ThresholdSelectionResult(
        threshold=float(row["threshold"]),
        strategy="cost_based",
        rationale=(
            f"Minimizes total cost = {cost_fp} * FP + {cost_fn} * FN using "
            "explicitly supplied cost weights from a documented business policy."
        ),
        metrics_at_threshold=_row_to_metrics(row),
    )


STRATEGY_REGISTRY: Dict[str, Callable[..., ThresholdSelectionResult]] = {
    "maximize_f1": select_threshold_maximize_f1,
    "min_recall": select_threshold_min_recall,
    "min_precision": select_threshold_min_precision,
    "cost_based": select_threshold_cost_based,
}


def select_threshold(table: pd.DataFrame, strategy: str, **kwargs) -> ThresholdSelectionResult:
    if strategy not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown threshold strategy '{strategy}'. Available: {list(STRATEGY_REGISTRY)}")
    fn = STRATEGY_REGISTRY[strategy]
    return fn(table, **kwargs) if kwargs else fn(table)


def plot_threshold_metrics(table: pd.DataFrame, output_path: Path, selected_threshold: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(table["threshold"], table["precision"], label="Precision")
    ax.plot(table["threshold"], table["recall"], label="Recall")
    ax.plot(table["threshold"], table["f1"], label="F1")
    if selected_threshold is not None:
        ax.axvline(selected_threshold, color="black", linestyle="--", label=f"Selected = {selected_threshold:.2f}")
    ax.set_xlabel("Threshold"); ax.set_ylabel("Score")
    ax.set_title("Threshold analysis"); ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path); plt.close(fig)


def plot_precision_recall_tradeoff(table: pd.DataFrame, output_path: Path, selected_threshold: Optional[float] = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(table["recall"], table["precision"])
    if selected_threshold is not None:
        idx = (table["threshold"] - selected_threshold).abs().idxmin()
        row = table.loc[idx]
        ax.scatter([row["recall"]], [row["precision"]], color="red", zorder=5,
                   label=f"Selected threshold = {selected_threshold:.2f}")
        ax.legend()
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall trade-off")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path); plt.close(fig)


def save_threshold_artifact(
    result: ThresholdSelectionResult,
    table: pd.DataFrame,
    output_dir: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Path]:
    """Saves the selected threshold + full evaluation table + metadata so a
    future inference service can load `threshold.json` and apply it to
    fresh probability outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_path = output_dir / "threshold.json"
    table_path = output_dir / "threshold_evaluation_table.csv"

    payload = result.to_dict()
    payload["metadata"] = metadata or {}

    with open(threshold_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    table.to_csv(table_path, index=False)

    logger.info("Threshold artifact saved to %s", threshold_path)
    return {"threshold_path": threshold_path, "table_path": table_path}


def load_threshold_artifact(path: Union[str, Path]) -> Dict[str, Any]:
    with open(Path(path)) as f:
        return json.load(f)


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"Threshold must be within [0, 1], got {threshold}.")
    return (np.asarray(y_proba) >= threshold).astype(int)