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


