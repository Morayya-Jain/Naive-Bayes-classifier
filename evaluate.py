# evaluate.py — Metrics, confusion matrix, and confidence analysis

import numpy as np
from config import CLASSES


def compute_confusion_matrix(y_true, y_pred):
    """Compute a confusion matrix.

    Convention (matching the spec):
        Positive class = '>50K'
        Negative class = '<=50K'

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.

    Returns:
        Dict with keys 'TP', 'FP', 'TN', 'FN'.
    """

    confusion_matrix = {'TP': 0, 'FP': 0, 'TN': 0, 'FN': 0}

    for true, pred in zip(y_true, y_pred):
        # Negative class
        if true == CLASSES[0]:
            if pred == true:
                confusion_matrix['TN'] += 1
            else:
                confusion_matrix['FP'] += 1
        # Positive class
        else:
            if pred == true:
                confusion_matrix['TP'] += 1
            else:
                confusion_matrix['FN'] += 1
    
    return confusion_matrix


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
    
    confusion_matrix = compute_confusion_matrix(y_true, y_pred)
    TP, FP = confusion_matrix['TP'], confusion_matrix['FP']
    TN, FN = confusion_matrix['TN'], confusion_matrix['FN']

    correct_prediction = 0
    for true, pred in zip(y_true, y_pred):
        correct_prediction += (true == pred)
    accuracy = correct_prediction / len(y_pred) if len(y_pred) > 0 else -1

    # For '>50K' class label
    positive_precision = TP / (TP + FP)
    positive_recall = TP / (TP + FN)
    positive_f1 = (2 * positive_precision * positive_recall) / (positive_precision + positive_recall)
    positive_class_metrics = {'precision': positive_precision, 'recall': positive_recall, 'f1': positive_f1}
    
    # For '<=50K' class label
    negative_precision = TN / (TN + FN)
    negative_recall = TN / (TN + FP)
    negative_f1 = (2 * negative_precision * negative_recall) / (negative_precision + negative_recall)
    negative_class_metrics = {'precision': negative_precision, 'recall': negative_recall, 'f1': negative_f1}

    metrics = {'accuracy': accuracy, '<=50K': negative_class_metrics, '>50K': positive_class_metrics}
    return metrics


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
    
    negative = log_proba[:, 0]  # '<=50K'
    positive = log_proba[:, 1]  # '>50K'
    log_ratio = positive - negative
    
    ratio = np.exp(log_ratio)
    return ratio


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
    
    ratios = posterior_ratio(log_proba)

    if class_label == CLASSES[1]:  # >50K: high R = confident
        filtered = np.where(ratios > 1)[0]
        sorted_pos = filtered[np.argsort(-ratios[filtered])]
    else:                          # <=50K: low R = confident
        filtered = np.where(ratios < 1)[0]  
        sorted_pos = filtered[np.argsort(ratios[filtered])]

    top = sorted_pos[:n]
    result = X.iloc[top].copy()
    result['R'] = ratios[top]
    result['true_label'] = y_true.iloc[top].values

    return result


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
    
    ratios = posterior_ratio(log_proba)
    distance = np.abs(1 - ratios)
    top = np.argsort(distance)[:n]

    result = X.iloc[top].copy()
    result['R'] = ratios[top]

    return result
