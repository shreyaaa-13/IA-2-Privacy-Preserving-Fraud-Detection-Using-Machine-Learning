"""
Unit Tests for Evaluation Metrics Suite.
Tests precision, recall, F1, PR-AUC, ROC-AUC, edge cases, and confusion matrix structures.
"""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation import compute_classification_metrics


def test_standard_metrics_calculation():
    """Verify known metric computations."""
    # 4 samples: 2 positive, 2 negative
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 0, 0, 1])  # TP=1, FN=1, TN=1, FP=1
    y_prob = np.array([0.9, 0.4, 0.1, 0.8])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, "Test_Exp")

    assert metrics["true_positives_tp"] == 1
    assert metrics["false_positives_fp"] == 1
    assert metrics["true_negatives_tn"] == 1
    assert metrics["false_negatives_fn"] == 1

    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 0.5
    assert metrics["f1_score"] == 0.5
    assert metrics["accuracy"] == 0.5
    assert isinstance(metrics["pr_auc"], float)
    assert isinstance(metrics["roc_auc"], float)


def test_all_negative_predictions_edge_case():
    """Verify that predicting all 0s (no positives) handles zero division safely."""
    y_true = np.array([1, 0, 0, 0])
    y_pred = np.array([0, 0, 0, 0])  # Naive majority classifier
    y_prob = np.array([0.1, 0.05, 0.02, 0.01])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, "Majority_Exp")

    assert metrics["accuracy"] == 0.75  # 75% accuracy!
    assert metrics["recall"] == 0.0      # Zero fraud caught
    assert metrics["precision"] == 0.0
    assert metrics["f1_score"] == 0.0
    assert "misleading" in metrics["accuracy_critique"].lower()


def test_perfect_classifier():
    """Verify perfect prediction metrics."""
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_prob = np.array([0.95, 0.90, 0.05, 0.02])

    metrics = compute_classification_metrics(y_true, y_pred, y_prob, "Perfect_Exp")

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1_score"] == 1.0
    assert metrics["accuracy"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["roc_auc"] == 1.0
