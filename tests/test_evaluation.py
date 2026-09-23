# tests/test_evaluation.py
import numpy as np
import pytest

from src.models.evaluate import compute_classification_metrics, compute_confusion_matrix


def test_metrics_perfect_predictions():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_proba = np.array([0.05, 0.95, 0.1, 0.9])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_metrics_handle_single_class_gracefully():
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])
    y_proba = np.array([0.1, 0.2, 0.05, 0.3])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba)
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])


def test_metrics_without_probabilities():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    metrics = compute_classification_metrics(y_true, y_pred, y_proba=None)
    assert "precision" in metrics
    assert np.isnan(metrics["roc_auc"])


def test_confusion_matrix_shape_and_values():
    cm = compute_confusion_matrix([0, 1, 1, 0, 1], [0, 1, 0, 0, 1])
    assert cm.shape == (2, 2)
    assert cm.sum() == 5


def test_metrics_invalid_input_lengths_raise():
    with pytest.raises(ValueError):
        compute_classification_metrics(np.array([0, 1, 1]), np.array([0, 1]), None)