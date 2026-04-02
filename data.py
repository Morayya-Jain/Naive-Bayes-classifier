# data.py — Data loading, preprocessing, and feature separation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, LABEL_COL, DROP_COLS


def load_dataset(filepath):
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with the loaded data.
    """
    pass


def preprocess(df):
    """Clean the dataset: drop the fnlwgt column and remove rows with missing values.

    The spec says missing values appear as empty strings (,, in the CSV).
    Convert those to NaN, then drop any row containing NaN.

    Args:
        df: Raw DataFrame from load_dataset.

    Returns:
        Cleaned DataFrame with no missing values and fnlwgt removed.
    """
    pass


def separate_features_label(df):
    """Split a DataFrame into feature matrix X and label vector y.

    Args:
        df: DataFrame containing both features and the income label.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features and y is a Series.
    """
    pass


def split_continuous_categorical(X):
    """Separate feature matrix into continuous and categorical subsets.

    Uses CONTINUOUS_FEATURES and CATEGORICAL_FEATURES from config.

    Args:
        X: DataFrame of all features.

    Returns:
        Tuple of (X_continuous, X_categorical) DataFrames.
    """
    pass


def encode_categorical(X_cat, encoder=None):
    """Encode categorical string features as integers for sklearn's CategoricalNB.

    If encoder is None, fit a new OrdinalEncoder. Otherwise, use the provided one
    (for transforming test/unlabelled data with the same encoding).

    Handle unseen categories in test data gracefully.

    Args:
        X_cat: DataFrame of categorical features (string values).
        encoder: Optional pre-fitted OrdinalEncoder.

    Returns:
        Tuple of (encoded_array, encoder).
    """
    pass


def create_validation_split(X_cont, X_cat, y, val_fraction=0.2, random_state=42):
    """Hold out a validation set from training data.

    Important: Q3 requires a validation set for hyperparameter selection.
    Do NOT use the test set for this.

    Args:
        X_cont: Continuous features DataFrame.
        X_cat: Categorical features DataFrame (or encoded array).
        y: Label Series.
        val_fraction: Fraction of data to hold out for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val).
    """
    pass
