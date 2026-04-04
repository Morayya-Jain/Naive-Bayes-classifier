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

# Training and Validation ---------------
initial_dataset = load_dataset(TRAIN_FILE)
cleaned_dataset = preprocess(initial_dataset)

X, y = separate_features_label(cleaned_dataset)
X_cont, X_cat = split_continuous_categorical(X)
X_cont_train, X_cont_val, X_cat_train, X_cat_val, y_train, y_val = create_validation_split(X_cont, X_cat, y)

encoded_X_cat_train, encoder = encode_categorical(X_cat_train)
encoded_X_cat_val, _ = encode_categorical(X_cat_val, encoder)

# Testing --------------------------------
testing_dataset = load_dataset(TEST_FILE)
cleaned_test_dataset = preprocess(testing_dataset)

test_X, test_y = separate_features_label(cleaned_test_dataset)
test_X_cont, test_X_cat = split_continuous_categorical(test_X)
test_encoded_X_cat, _ = encode_categorical(test_X_cat, encoder)

# Model ----------------------------------
classifier = MixedNaiveBayes()
classifier.fit(X_cont_train, encoded_X_cat_train, y_train)

# =============================================================================
# Q2: Supervised Model Evaluation
# =============================================================================


# =============================================================================
# Q3: Semi-Supervised Extension (pick one approach)
# =============================================================================

unlabelled_dataset = load_dataset(UNLABELLED_FILE)
cleaned_unlabelled_dataset = preprocess(unlabelled_dataset)

X_cont_unlabelled, X_cat_unlabelled = split_continuous_categorical(cleaned_unlabelled_dataset)
encoded_X_cat_unlabelled, _ = encode_categorical(X_cat_unlabelled, encoder)

random_model = active_learning_loop(classifier, X_cont_train, encoded_X_cat_train, y_train, X_cont_unlabelled, encoded_X_cat_unlabelled, 'random')
uncertain_model = active_learning_loop(classifier, X_cont_train, encoded_X_cat_train, y_train, X_cont_unlabelled, encoded_X_cat_unlabelled)

y_pred_supervised = classifier.predict(X_cont_val, encoded_X_cat_val)                                                                                                                                                                                               
print("Supervised:", compute_metrics(y_val, y_pred_supervised)['accuracy'])                                                                                                                                                                                                      
                                                                                                                                                                                                                                                                    
# Random selection model                                                                                                                                                                                                                                              
y_pred_random = random_model.predict(X_cont_val, encoded_X_cat_val)                                                                                                                                                                                                 
print("Random:", compute_metrics(y_val, y_pred_random)['accuracy'])                                                                                                                                                                                                              
                                                                                                                                                                                                                                                                    
# Uncertainty selection model                                                                                                                                                                                                                                         
y_pred_uncertain = uncertain_model.predict(X_cont_val, encoded_X_cat_val)                                                                                                                                                                                           
print("Uncertain:", compute_metrics(y_val, y_pred_uncertain)['accuracy'])    

# =============================================================================
# Q4: Semi-Supervised Model Evaluation
# =============================================================================
