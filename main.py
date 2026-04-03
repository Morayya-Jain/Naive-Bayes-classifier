# main.py — Run and test your code here. Copy into notebook cells when done.

from config import (
    TRAIN_FILE, TEST_FILE, UNLABELLED_FILE, UNLABELLED_WITH_LABELS_FILE,
    CONTINUOUS_FEATURES, CATEGORICAL_FEATURES, CLASSES,
)
from data import (
    load_dataset, preprocess, separate_features_label,
    split_continuous_categorical, encode_categorical, create_validation_split,
)
from model import MixedNaiveBayes
from evaluate import (
    compute_confusion_matrix, compute_metrics, posterior_ratio,
    find_high_confidence, find_borderline,
)
from analysis import most_predictive_categories
from semi_supervised import (
    select_random, select_uncertain, reveal_labels, active_learning_loop
)


# =============================================================================
# Q1: Supervised Model Training
# =============================================================================

# Training
initial_dataset = load_dataset(TRAIN_FILE)
cleaned_dataset = preprocess(initial_dataset)

X, y = separate_features_label(cleaned_dataset)
X_cont, X_cat = split_continuous_categorical(X)
encoded_X_cat, encoder = encode_categorical(X_cat)

# Testing
testing_dataset = load_dataset(TEST_FILE)
cleaned_test_dataset = preprocess(testing_dataset)

test_X, test_y = separate_features_label(cleaned_test_dataset)
test_X_cont, test_X_cat = split_continuous_categorical(test_X)
test_encoded_X_cat, _ = encode_categorical(test_X_cat, encoder)

classifier = MixedNaiveBayes()
classifier.fit(X_cont, encoded_X_cat, y)


# =============================================================================
# Q2: Supervised Model Evaluation
# =============================================================================


# =============================================================================
# Q3: Semi-Supervised Extension (pick one approach)
# =============================================================================


# =============================================================================
# Q4: Semi-Supervised Model Evaluation
# =============================================================================
