# data.py — Data loading, preprocessing, and feature separation

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from config import CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, LABEL_COL


def load_dataset(filepath):
    """Load a CSV file into a pandas DataFrame.

    Args:
        filepath: Path to the CSV file.

    Returns:
        DataFrame with the loaded data.
    """

    df = pd.read_csv(filepath, na_values=['', '?', 'MISSING'])
    return df


def preprocess(df):
    """Clean the dataset: drop the fnlwgt column and remove rows with missing values.

    Args:
        df: Raw DataFrame from load_dataset.

    Returns:
        Cleaned DataFrame with no missing values and fnlwgt removed.
    """
    
    cleaned_df = df.dropna(subset = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES)  # Removing rows with null values 
    cleaned_df.drop(columns='fnlwgt', inplace=True)  # Removing 'fnlwgt' column as it is not used
    
    return cleaned_df


def separate_features_label(df):
    """Split a DataFrame into feature matrix X and label vector y.

    Args:
        df: DataFrame containing both features and the income label.

    Returns:
        Tuple of (X, y) where X is a DataFrame of features and y is a Series.
    """
    label_col = df[LABEL_COL]
    features = df.drop(columns=[LABEL_COL])
    
    return (features, label_col)


def split_continuous_categorical(X):
    """Separate feature matrix into continuous and categorical subsets.

    Uses CONTINUOUS_FEATURES and CATEGORICAL_FEATURES from config.

    Args:
        X: DataFrame of all features.

    Returns:
        Tuple of (X_continuous, X_categorical) DataFrames.
    """
    X_cont = X[CONTINUOUS_FEATURES]
    X_cat = X[CATEGORICAL_FEATURES]

    return (X_cont, X_cat)


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
    if encoder is None:
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        encoded_categorical = encoder.fit_transform(X_cat)
    else:
        encoded_categorical = encoder.transform(X_cat)  # Use encoder which knows about categorical values
        
    return (encoded_categorical, encoder)


def create_validation_split(X_cont, X_cat, y, val_fraction=0.2, r_state=42):
    """Hold out a validation set from training data (no test set).

    Args:
        X_cont: Continuous features DataFrame.
        X_cat: Categorical features DataFrame (or encoded array).
        y: Label Series.
        val_fraction: Fraction of data to hold out for validation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val).
    """
    X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = train_test_split(X_cont, X_cat, y, test_size=val_fraction, random_state=r_state)

    return (X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val)
