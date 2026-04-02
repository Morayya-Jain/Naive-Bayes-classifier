# semi_supervised.py — Three semi-supervised approaches for Q3
#
# The assignment asks you to choose ONE of these three approaches.
# All three are scaffolded here so you can decide after exploring.
# Delete or ignore the approaches you don't use.

import numpy as np
import pandas as pd

from model import MixedNaiveBayes


# =============================================================================
# Approach 1: Label Propagation
# =============================================================================

def label_propagation(model, X_cont_unlabelled, X_cat_unlabelled, confidence_threshold=None):
    """Use a trained model to pseudo-label unlabelled data.

    Predict labels for all unlabelled instances. Optionally, only keep
    predictions above a confidence threshold (based on posterior ratio R).

    Args:
        model: Trained MixedNaiveBayes from Q1.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.
        confidence_threshold: If set, only return pseudo-labels where R > threshold.
                             This filters out uncertain predictions.

    Returns:
        Tuple of (pseudo_labels, mask) where mask indicates which instances passed
        the confidence threshold (all True if no threshold).
    """
    pass


def iterative_label_propagation(model, X_cont_train, X_cat_train, y_train,
                                X_cont_unlabelled, X_cat_unlabelled,
                                max_iter=5, confidence_threshold=None):
    """Iteratively pseudo-label and retrain.

    Each iteration:
        1. Pseudo-label unlabelled data with current model
        2. Add confident pseudo-labelled instances to training set
        3. Retrain model on expanded training set

    Use a validation set (from data.create_validation_split) to monitor
    whether performance improves or degrades across iterations.

    Args:
        model: Initial trained MixedNaiveBayes from Q1.
        X_cont_train: Continuous training features.
        X_cat_train: Encoded categorical training features.
        y_train: Training labels.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.
        max_iter: Maximum number of iterations.
        confidence_threshold: Minimum confidence for pseudo-labels.

    Returns:
        Final retrained MixedNaiveBayes model.
    """
    pass


# =============================================================================
# Approach 2: Active Learning
# =============================================================================

def select_random(n_unlabelled, n=200, random_state=42):
    """Randomly select n indices from the unlabelled set (baseline strategy).

    Args:
        n_unlabelled: Total number of unlabelled instances.
        n: Number of instances to select.
        random_state: Random seed.

    Returns:
        Array of selected indices.
    """
    pass


def select_uncertain(model, X_cont_unlabelled, X_cat_unlabelled, n=200):
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
    pass


def reveal_labels(indices, labels_file='adult_unlabelled_with_labels.csv'):
    """Look up the true labels for selected unlabelled instances.

    The file adult_unlabelled_with_labels.csv has the same rows as
    adult_unlabelled.csv but includes the income column.

    Args:
        indices: Array of row indices to reveal labels for.
        labels_file: Path to the file with true labels.

    Returns:
        Array of true labels for the selected instances.
    """
    pass


def active_learning_loop(model, X_cont_train, X_cat_train, y_train,
                         X_cont_unlabelled, X_cat_unlabelled,
                         true_labels, strategy='uncertain',
                         n_per_round=200, n_rounds=1):
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
        true_labels: All true labels from unlabelled_with_labels.csv.
        strategy: 'random' or 'uncertain'.
        n_per_round: Instances to select per round.
        n_rounds: Number of active learning rounds.

    Returns:
        Final retrained MixedNaiveBayes model.
    """
    pass


# =============================================================================
# Approach 3: Expectation-Maximisation (EM)
# =============================================================================

def e_step(model, X_cont_unlabelled, X_cat_unlabelled):
    """E-step: Compute soft (probabilistic) class assignments for unlabelled data.

    Use the current model to compute P(c|x) for each unlabelled instance.
    These are "soft" labels — probabilities, not hard 0/1 assignments.

    Args:
        model: Current MixedNaiveBayes model.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.

    Returns:
        Array of shape (n_unlabelled, n_classes) with soft class assignments.
        Each row sums to 1.
    """
    pass


def m_step(X_cont_labelled, X_cat_labelled, y_labelled,
           X_cont_unlabelled, X_cat_unlabelled, soft_assignments,
           alpha=1.0, weight=1.0):
    """M-step: Re-estimate model parameters using labelled data + soft assignments.

    Combine:
        - Labelled data (hard labels, full weight)
        - Unlabelled data (soft assignments, scaled by `weight`)

    The weight parameter controls how much influence unlabelled data has.
    Try different values on a validation set.

    Args:
        X_cont_labelled: Continuous features of labelled data.
        X_cat_labelled: Encoded categorical features of labelled data.
        y_labelled: Hard labels for labelled data.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.
        soft_assignments: Array of shape (n_unlabelled, n_classes) from E-step.
        alpha: Laplace smoothing.
        weight: Scaling factor for unlabelled data contribution.

    Returns:
        New MixedNaiveBayes model with updated parameters.
    """
    pass


def run_em(model, X_cont_labelled, X_cat_labelled, y_labelled,
           X_cont_unlabelled, X_cat_unlabelled,
           max_iter=20, tol=1e-4, alpha=1.0, weight=1.0):
    """Run EM algorithm until convergence.

    Initialize with the Q1 supervised model. Alternate E-step and M-step.
    Stop when log-likelihood change < tol or max_iter reached.

    Monitor convergence by tracking the log-likelihood at each iteration.

    Args:
        model: Initial MixedNaiveBayes from Q1 (used for first E-step).
        X_cont_labelled: Continuous features of labelled training data.
        X_cat_labelled: Encoded categorical features of labelled data.
        y_labelled: Hard labels.
        X_cont_unlabelled: Continuous features of unlabelled data.
        X_cat_unlabelled: Encoded categorical features of unlabelled data.
        max_iter: Maximum EM iterations.
        tol: Convergence tolerance on log-likelihood change.
        alpha: Laplace smoothing.
        weight: Scaling factor for unlabelled data.

    Returns:
        Tuple of (final_model, log_likelihood_history).
    """
    pass
