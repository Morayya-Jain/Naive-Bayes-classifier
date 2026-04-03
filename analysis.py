# analysis.py — Feature analysis and model comparison (Q1 + Q4)

import numpy as np


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
    log_probs = model.get_categorical_params()

    all_ratios = []
    for i, fname in enumerate(feature_names):
        # log_probs[i] shape: (2, n_categories) — row 0 = <=50K, row 1 = >50K
        log_r = log_probs[i][1, :] - log_probs[i][0, :]
        for v in range(len(log_r)):
            all_ratios.append((fname, v, np.exp(log_r[v])))

    # Sort by ratio descending — highest R = most predictive of >50K
    sorted_by_r = sorted(all_ratios, key=lambda x: x[2], reverse=True)

    return {
        '>50K': sorted_by_r[:n],
        '<=50K': sorted_by_r[-n:][::-1],  # lowest R = most predictive of <=50K
    }