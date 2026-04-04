# semi_supervised.py — Three semi-supervised approaches for Q3
#
# The assignment asks you to choose ONE of these three approaches.
# All three are scaffolded here so you can decide after exploring.
# Delete or ignore the approaches you don't use.

import numpy as np
import pandas as pd

from model import MixedNaiveBayes
from evaluate import posterior_ratio
from data import load_dataset, preprocess
from config import UNLABELLED_WITH_LABELS_FILE


def select_random(n_unlabelled, n=200, random_state=42):
    """Randomly select n indices from the unlabelled set (baseline strategy).

    Args:
        n_unlabelled: Total number of unlabelled instances.
        n: Number of instances to select.
        random_state: Random seed.

    Returns:
        Array of selected indices.
    """

    rng = np.random.default_rng(random_state)
    return rng.choice(n_unlabelled, size=n, replace=False)


def select_uncertain(model: MixedNaiveBayes, X_cont_unlabelled, X_cat_unlabelled, n=200):
    """Select the n most uncertain instances (uncertainty sampling).

    Most uncertain = closest to the decision boundary = posterior ratio R closest to 1.
    These are the most informative instances to label.

    Args:
        model: Trained MixedNaiveBayes.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.
        n: Number of instances to select.

    Returns:
        Array of selected indices.
    """
    
    log_proba = model.predict_log_proba(X_cont_unlabelled, X_cat_unlabelled)
    ratios = posterior_ratio(log_proba)
    distance = np.abs(1 - ratios)
    top = np.argsort(distance)[:n]

    return top


def reveal_labels(indices):
    """Look up the true labels for selected unlabelled instances.

    The file adult_unlabelled_with_labels.csv has the same rows as
    adult_unlabelled.csv but includes the income column.

    Args:
        indices: Array of row indices to reveal labels for.

    Returns:
        Array of true labels for the selected instances.
    """
    
    initial_dataset = load_dataset(UNLABELLED_WITH_LABELS_FILE)
    cleaned_dataset = preprocess(initial_dataset)

    return cleaned_dataset['income'].iloc[indices]


def active_learning_loop(model, X_cont_train, X_cat_train, y_train,
                         X_cont_unlabelled, X_cat_unlabelled,
                         strategy='uncertain', n=200):
    """Run the active learning loop.

    Each round:
        1. Select instances using the chosen strategy
        2. Reveal their true labels
        3. Add to training set
        4. Retrain model

    Compare random vs uncertainty sampling on a validation set.

    Args:
        model: Initial trained MixedNaiveBayes from Q1.
        X_cont_train: Continuous training features.
        X_cat_train: Encoded categorical training features.
        y_train: Training labels.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.
        strategy: 'random' or 'uncertain'.
        n: Instances to select per round.

    Returns:
        Final retrained MixedNaiveBayes model.
    """
    
    if strategy == 'uncertain':
        indices = select_uncertain(model, X_cont_unlabelled, X_cat_unlabelled, n)
    else:
        indices = select_random(len(X_cont_unlabelled), n)
    
    selected_y = reveal_labels(indices)
    selected_X_cont_unlabelled = np.array(X_cont_unlabelled)[indices]
    selected_X_cat_unlabelled = np.array(X_cat_unlabelled)[indices]

    new_X_cont = np.vstack([X_cont_train, selected_X_cont_unlabelled])
    new_X_cat = np.vstack([X_cat_train, selected_X_cat_unlabelled])
    new_y = np.concatenate([y_train, selected_y])

    new_model = MixedNaiveBayes(alpha=model.alpha)
    new_model.fit(new_X_cont, new_X_cat, new_y)

    return new_model
