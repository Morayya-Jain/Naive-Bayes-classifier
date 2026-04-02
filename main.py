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
    find_high_confidence, find_borderline, print_classification_report,
)
from analysis import (
    most_predictive_categories, compare_models, confidence_distribution,
    plot_confidence_histogram, parameter_changes,
)
from semi_supervised import (
    label_propagation, iterative_label_propagation,
    select_random, select_uncertain, reveal_labels, active_learning_loop,
    e_step, m_step, run_em,
)


# =============================================================================
# Q1: Supervised Model Training
# =============================================================================
cleaned_df = preprocess(load_dataset(TRAIN_FILE))
print(cleaned_df)

# =============================================================================
# Q2: Supervised Model Evaluation
# =============================================================================


# =============================================================================
# Q3: Semi-Supervised Extension (pick one approach)
# =============================================================================


# =============================================================================
# Q4: Semi-Supervised Model Evaluation
# =============================================================================
