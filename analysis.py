# analysis.py — Feature analysis and model comparison (Q1 + Q4)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def most_predictive_categories(model, feature_names, n=5):
    """Find the n most predictive category values per class.

    For each categorical feature value v, compute:
        R = P(xj=v | c1) / P(xj=v | c2)

    High R means v is predictive of c1 (>50K).
    Low R (or high 1/R) means v is predictive of c2 (<=50K).

    Q1 asks for the 5 most predictive category values for EACH class.

    Args:
        model: Trained MixedNaiveBayes instance.
        feature_names: List of categorical feature names (for labeling output).
        n: Number of top predictive values to return per class.

    Returns:
        Dict with keys for each class, values are lists of
        (feature_name, category_value, ratio) tuples.
    """
    pass


def compare_models(model1, model2, X_cont_test, X_cat_test, y_test):
    """Compare two models on the same test set (Q4).

    Reports side-by-side: accuracy, precision, recall, F1.

    Args:
        model1: First trained MixedNaiveBayes (supervised, from Q1).
        model2: Second trained MixedNaiveBayes (semi-supervised, from Q3).
        X_cont_test: Continuous test features.
        X_cat_test: Encoded categorical test features.
        y_test: True test labels.

    Returns:
        DataFrame with metrics for both models.
    """
    pass


def confidence_distribution(model, X_cont, X_cat):
    """Compute posterior ratio R for every instance in a dataset.

    Useful for Q4: comparing how model confidence changed after semi-supervised training.

    Args:
        model: Trained MixedNaiveBayes.
        X_cont: Continuous features.
        X_cat: Encoded categorical features.

    Returns:
        Array of posterior ratios.
    """
    pass


def plot_confidence_histogram(ratios_before, ratios_after, labels=None):
    """Plot overlaid histograms of confidence distributions for two models.

    Q4 asks to investigate how the confidence distribution changed.
    Consider using log(R) on the x-axis for better visualization.

    Args:
        ratios_before: Posterior ratios from the supervised model.
        ratios_after: Posterior ratios from the semi-supervised model.
        labels: Optional tuple of (label1, label2) for the legend.
    """
    pass


def parameter_changes(model_before, model_after, feature_names_cont, feature_names_cat):
    """Summarize how learned parameters changed between two models (Q4).

    Compare:
        - Gaussian means and stds for continuous features
        - Category probabilities for categorical features
        - Top predictive features per class

    Args:
        model_before: Supervised model from Q1.
        model_after: Semi-supervised model from Q3.
        feature_names_cont: List of continuous feature names.
        feature_names_cat: List of categorical feature names.

    Returns:
        Dict summarizing the changes (for reporting in the written answer).
    """
    pass
