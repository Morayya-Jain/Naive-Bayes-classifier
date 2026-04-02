# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Implementing supervised and semi-supervised Naive Bayes for income classification (>50K vs <=50K) on US Census data. The student fills in the function stubs themselves — do not write implementation code unless explicitly asked.

## Architecture

Six Python modules at root level, imported into the submission notebook:

- **config.py** — Feature lists (continuous vs categorical), file paths, class labels. All other modules import from here.
- **data.py** — Load CSVs, drop missing values + fnlwgt column, split features/labels, encode categoricals via OrdinalEncoder.
- **model.py** — `MixedNaiveBayes` class wrapping sklearn's `GaussianNB` (continuous) + `CategoricalNB` (categorical). Must combine log-likelihoods without double-counting priors.
- **evaluate.py** — Confusion matrix, accuracy/precision/recall/F1, posterior ratio R = P(c1|x)/P(c2|x) for confidence analysis.
- **analysis.py** — Most predictive category values (Q1), model comparison and confidence distribution plots (Q4).
- **semi_supervised.py** — Three approaches scaffolded (student picks one): Label Propagation, Active Learning (200 instances), Expectation-Maximisation.

**Submission artifact**: `Code Submission.ipynb` — 4 sections mapping to Q1–Q4.

## Data Files

All CSVs share the same schema (15 columns). Missing values appear as empty fields (`,,`).

| File | Rows | Has labels? |
|------|------|-------------|
| `adult_supervised_train.csv` | 16,280 | Yes |
| `adult_test.csv` | 16,281 | Yes |
| `adult_unlabelled.csv` | 16,281 | No |
| `adult_unlabelled_with_labels.csv` | 16,281 | Yes (ground truth for Q3) |

## Commands

```bash
# Run all modules as a syntax/import check
python3 -c "import config, data, model, evaluate, analysis, semi_supervised"

# Launch the notebook
jupyter notebook "Code Submission.ipynb"
```

## Dependencies

Python 3.12 with: numpy, pandas, scikit-learn, matplotlib, seaborn (optional).

## Key Constraints

- All functions are stubs (`pass`) — the student implements them. Do not fill in logic unless asked.
- Q3 requires a held-out validation set from training data — never use `adult_test.csv` for hyperparameter tuning.
- Work in log-space to avoid numerical underflow when multiplying probabilities.
- Drop `fnlwgt` column — it is a sampling weight, not a predictive feature.
