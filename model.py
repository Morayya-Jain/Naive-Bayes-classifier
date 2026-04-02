# model.py — Mixed Naive Bayes classifier wrapping sklearn

import numpy as np
from sklearn.naive_bayes import GaussianNB, CategoricalNB


class MixedNaiveBayes:
    """A Naive Bayes classifier that handles both continuous and categorical features.

    Internally uses sklearn's GaussianNB for continuous features and
    CategoricalNB for categorical features. Combines their log-likelihoods
    with a single set of class priors for prediction.

    Key sklearn attributes you'll need:
        GaussianNB:
            .class_prior_    — P(c) for each class
            .theta_          — means per class, shape (n_classes, n_continuous_features)
            .var_            — variances per class, same shape
        CategoricalNB:
            .class_log_prior_    — log P(c)
            .feature_log_prob_   — list of log P(feature=value|class) arrays
            .category_count_     — raw counts per category per class

    Important: When combining predictions from two separate models, be careful
    not to double-count the class prior. Each model adds its own prior internally,
    so you need to subtract one set when combining.
    """

    def __init__(self, alpha=1.0):
        """Initialize the mixed classifier.

        Args:
            alpha: Laplace smoothing parameter for CategoricalNB.
        """
        
        self.alpha = alpha


    def fit(self, X_continuous, X_categorical, y):
        """Train both sub-models on their respective feature subsets.

        Args:
            X_continuous: Array-like of continuous features.
            X_categorical: Array-like of encoded categorical features (integers).
            y: Array-like of class labels.

        Returns:
            self (for method chaining).
        """
        pass


    def predict(self, X_continuous, X_categorical):
        """Predict class labels for the given instances.

        Combine log-probabilities from both sub-models, pick the class
        with the highest combined log-posterior.

        Args:
            X_continuous: Array-like of continuous features.
            X_categorical: Array-like of encoded categorical features.

        Returns:
            Array of predicted class labels.
        """
        pass


    def predict_log_proba(self, X_continuous, X_categorical):
        """Compute combined log posterior probabilities for each class.

        This is the core combination step. Work in log-space:
            log P(c|x) = log P(c) + sum_j log P(x_j|c)  [continuous]
                                    + sum_k log P(x_k|c)  [categorical]

        Hint: sklearn models have a _joint_log_likelihood() method that
        returns log P(c) + sum log P(features|c) before normalization.

        Args:
            X_continuous: Array-like of continuous features.
            X_categorical: Array-like of encoded categorical features.

        Returns:
            Array of shape (n_samples, n_classes) with log probabilities.
        """
        pass


    def get_priors(self):
        """Return the learned prior probabilities P(c) for each class.

        Returns:
            Array of prior probabilities.
        """
        pass


    def get_gaussian_params(self):
        """Return learned Gaussian parameters for continuous features.

        Returns:
            Dict with 'means' and 'stds', each of shape (n_classes, n_continuous_features).
            Access via self.gaussian_nb.theta_ and np.sqrt(self.gaussian_nb.var_).
        """
        pass


    def get_categorical_params(self):
        """Return learned category probability tables for categorical features.

        Returns:
            The feature_log_prob_ from CategoricalNB — a list where each entry
            is an array of shape (n_classes, n_categories_for_that_feature).
        """
        pass
