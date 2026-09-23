# tests/test_threshold.py
import numpy as np
import pytest

from src.optimization.threshold import (
    ThresholdSelectionError, apply_threshold, evaluate_thresholds, select_threshold,
)


def make_probs():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.4, 0.9, 0.8, 0.6, 0.55, 0.3, 0.7, 0.45])
    return y_true, y_proba


def test_evaluate_thresholds_within_bounds():
    y_true, y_proba = make_probs()
    table = evaluate_thresholds(y_true, y_proba, thresholds=[0.1, 0.5, 0.9])
    assert table["threshold"].between(0, 1).all()
    assert set(table.columns) >= {"threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn"}


def test_predictions_change_with_threshold():
    _, y_proba = make_probs()
    assert apply_threshold(y_proba, 0.1).sum() >= apply_threshold(y_proba, 0.9).sum()


def test_apply_threshold_invalid_range():
    _, y_proba = make_probs()
    with pytest.raises(ValueError):
        apply_threshold(y_proba, 1.5)


def test_precision_recall_f1_manual_check():
    y_true, y_proba = make_probs()
    row = evaluate_thresholds(y_true, y_proba, thresholds=[0.5]).iloc[0]
    preds = (y_proba >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (y_true == 1))
    fp = np.sum((preds == 1) & (y_true == 0))
    fn = np.sum((preds == 0) & (y_true == 1))
    expected_precision = tp / (tp + fp) if (tp + fp) else 0.0
    expected_recall = tp / (tp + fn) if (tp + fn) else 0.0
    assert row["precision"] == pytest.approx(expected_precision)
    assert row["recall"] == pytest.approx(expected_recall)


def test_select_threshold_maximize_f1_within_range():
    y_true, y_proba = make_probs()
    table = evaluate_thresholds(y_true, y_proba)
    result = select_threshold(table, "maximize_f1")
    assert 0.0 <= result.threshold <= 1.0
    assert result.strategy == "maximize_f1"


def test_select_threshold_min_recall():
    y_true, y_proba = make_probs()
    table = evaluate_thresholds(y_true, y_proba)
    result = select_threshold(table, "min_recall", min_recall=0.8)
    assert result.metrics_at_threshold["recall"] >= 0.8


def test_select_threshold_min_recall_infeasible_raises():
    y_true, y_proba = make_probs()
    table = evaluate_thresholds(y_true, y_proba)
    with pytest.raises(ThresholdSelectionError):
        select_threshold(table, "min_recall", min_recall=1.01)


def test_select_threshold_cost_based():
    y_true, y_proba = make_probs()
    table = evaluate_thresholds(y_true, y_proba)
    result = select_threshold(table, "cost_based", cost_fp=1, cost_fn=5)
    assert 0.0 <= result.threshold <= 1.0
    assert result.strategy == "cost_based"


def test_threshold_functions_are_data_agnostic_by_contract():
    """Documents the leakage-prevention contract: evaluate_thresholds/
    select_threshold operate purely on whatever (y_true, y_proba) arrays
    are passed in. The caller (train.get_validation_probabilities) is
    responsible for ensuring these originate from training/validation
    folds only, never the test set."""
    y_true, y_proba = make_probs()
    table = evaluate_thresholds(y_true, y_proba)
    assert len(table) == len(np.arange(0.01, 1.0, 0.01))