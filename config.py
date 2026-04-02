# config.py — Constants and file paths for the Naive Bayes assignment

# Features as specified in the assignment
CONTINUOUS_FEATURES = [
    'age',
    'education-num',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]

CATEGORICAL_FEATURES = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]

LABEL_COL = 'income'
CLASSES = ['<=50K', '>50K']

# Data file paths
TRAIN_FILE = 'adult_supervised_train.csv'
TEST_FILE = 'adult_test.csv'
UNLABELLED_FILE = 'adult_unlabelled.csv'
UNLABELLED_WITH_LABELS_FILE = 'adult_unlabelled_with_labels.csv'
