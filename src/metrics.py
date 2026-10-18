"""Code for computing metrics for CS-433 Project 1"""

# Binary metrics and numerically stable probability/loss calculations.

import numpy as np


def probabilities_from_logits(logits):
    """Convert linear scores to probabilities without exponential overflow."""
    logits = np.asarray(logits, dtype=float)
    return np.exp(-np.logaddexp(0.0, -logits))


def binary_log_loss(y, logits):
    """Mean binary cross-entropy from linear scores, without clipping probabilities."""
    y, logits = np.asarray(y), np.asarray(logits)
    # Choosing the signed score avoids subtracting two large positive values.
    return float(np.mean(np.logaddexp(0.0, np.where(y == 1, -logits, logits))))


def classification_metrics(y_true, y_pred):
    """Score 0/1 predictions; undefined precision/recall/F1 are reported as zero.

    Confusion-matrix rows are actual classes [0, 1]; columns are predictions [0, 1].
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim != 1 or y_true.shape != y_pred.shape or not len(y_true):
        raise ValueError("Expected matching, nonempty 1D labels and predictions.")
    if not np.isin(y_true, [0, 1]).all() or not np.isin(y_pred, [0, 1]).all():
        raise ValueError("Metrics require 0/1 labels.")
    tn = int(np.count_nonzero((y_true == 0) & (y_pred == 0)))
    fp = int(np.count_nonzero((y_true == 0) & (y_pred == 1)))
    fn = int(np.count_nonzero((y_true == 1) & (y_pred == 0)))
    tp = int(np.count_nonzero((y_true == 1) & (y_pred == 1)))
    return {
        "accuracy": (tn + tp) / len(y_true),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "predicted_positive_count": tp + fp,
    }
