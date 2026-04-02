# evaluate.py — Metrics, confusion matrix, and confidence analysis

import numpy as np
import pandas as pd


def compute_confusion_matrix(y_true, y_pred):
    """Compute a 2x2 confusion matrix.

    Convention (matching the spec):
        Positive class = '>50K'
        Negative class = '<=50K'

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        Dict with keys 'TP', 'FP', 'TN', 'FN'.
    """
    pass


def compute_metrics(y_true, y_pred):
    """Compute accuracy, precision, recall, and F1-score.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        Dict with 'accuracy' (float) and per-class dicts, e.g.
        {'accuracy': ..., '<=50K': {'precision': ..., 'recall': ..., 'f1': ...},
         '>50K': {'precision': ..., 'recall': ..., 'f1': ...}}.
    """
    pass


def posterior_ratio(log_proba):
    """Compute the posterior ratio R = P(c1|x) / P(c2|x) for each instance.

    Convention: c1 = '>50K' (column 1), c2 = '<=50K' (column 0) in log_proba.

    This measures prediction confidence:
        R >> 1 means strong confidence toward class 1
        R << 1 means strong confidence toward class 2
        R ~ 1 means borderline / uncertain

    Work in log-space first, then exponentiate:
        log R = log P(c1|x) - log P(c2|x)

    Args:
        log_proba: Array of shape (n_samples, 2) with log probabilities.

    Returns:
        Array of posterior ratios (one per instance).
    """
    pass


def find_high_confidence(X, log_proba, y_true, class_label, n=5):
    """Find the n instances with highest confidence for a given class.

    Useful for Q2: showing examples of high-confidence predictions.

    Args:
        X: Feature DataFrame (for displaying the instance details).
        log_proba: Array of shape (n_samples, 2) with log probabilities.
        y_true: True labels (to check if prediction is correct).
        class_label: Which class to find confident predictions for ('>50K' or '<=50K').
        n: Number of instances to return.

    Returns:
        DataFrame of the n most confident instances with their features and ratios.
    """
    pass


def find_borderline(X, log_proba, n=5):
    """Find the n instances closest to the decision boundary (R ~ 1).

    Useful for Q2: showing borderline/uncertain predictions.

    Args:
        X: Feature DataFrame.
        log_proba: Array of shape (n_samples, 2) with log probabilities.
        n: Number of instances to return.

    Returns:
        DataFrame of the n most borderline instances with their features and ratios.
    """
    pass


def print_classification_report(y_true, y_pred):
    """Print a formatted classification report with confusion matrix and all metrics.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
    """
    pass
