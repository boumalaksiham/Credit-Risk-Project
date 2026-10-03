# Credit Risk Classification — Siham Boumalak (R Version)

## Evaluation status

The scripts are available, but this checkout contains no committed dataset or run outputs. No performance result is verified by this README. `week1_data_preprocessing.R` fits scaling on all rows before splitting, which leaks holdout statistics. `improve_metrics.R` should also be reviewed for oversampling and model-selection placement before using its scores as independent test results. Rebuild preprocessing inside the training/CV workflow, keep a final holdout untouched, and rerun before publishing metrics. The scripts encode good credit as the positive class; label interpretation matters when reporting risk recall.


## Purpose and outputs

Compare classification models for German credit data and explore error patterns, interpretability and group-level fairness. This is an educational analysis workflow, not a validated lending decision system. The source is available, but no model performance is claimed without a reproducible run and evaluation-method fixes.

## Input contract

The first script expects `data/german_credit_data.csv`, including an index column read with `row.names = 1`. It expects `Risk` values `good`/`bad` and columns such as Age, Sex, Job, Housing, Saving accounts, Checking account, Credit amount, Duration and Purpose; R converts spaces in headers to dots. Check the downloaded schema before running. **Good credit is encoded as 1**, so a positive-class metric measures good-credit prediction rather than detection of bad credit.

## Working directory

Run all scripts from the cloned repository root so relative `data/` and `outputs/` paths resolve correctly. The first script creates preprocessing outputs; later scripts load those outputs and model artifacts. Running a later stage first produces missing-file errors. Some optional packages may be unavailable in a current R repository; record a verified R/package environment before claiming reproducibility.

## Project Structure

```
credit_risk_project/
│
├── data/
│   └── german_credit_data.csv        ← place your downloaded dataset here
│
├── outputs/                          ← auto-created when you run scripts
│   ├── *.png                         (all charts and plots)
│   ├── *.csv                         (results tables)
│   └── *.rds                         (saved models and data)
│
├── week1_data_preprocessing.R
├── week2_logistic_regression.R
├── week3_svm.R
├── week4_random_forest.R
├── week5_model_comparison.R
├── week6_fairness_interpretability.R
├── week7_final_evaluation.R
│
├── improve_metrics.R
├── mcnemar_test.R
└── README.md
```

## Setup

Install required packages in R:

```r
install.packages(c(
  "tidyverse",      # data manipulation & ggplot2
  "caret",          # unified ML training interface
  "e1071",          # SVM (used by caret)
  "randomForest",   # Random Forest
  "glmnet",         # Logistic Regression with regularisation
  "pROC",           # ROC curves and AUC
  "reshape2",       # data reshaping for heatmaps
  "gridExtra",      # multi-panel plots
  "grid",           # text grob for dashboard
  "DMwR2",          # SMOTE oversampling
  "xgboost",        # XGBoost (used by caret)
  "gbm"             # Gradient Boosting (used by caret)
))
```

## Dataset

Download from Kaggle:  
https://www.kaggle.com/datasets/kabure/german-credit-data-with-risk  
Save as: `data/german_credit_data.csv`

## Running the Project

Run scripts in order, one per week:

```r
source("week1_data_preprocessing.R")
source("week2_logistic_regression.R")
source("week3_svm.R")
source("week4_random_forest.R")
source("week5_model_comparison.R")
source("week6_fairness_interpretability.R")
source("week7_final_evaluation.R")

# Optional: advanced optimization
source("improve_metrics.R")

# Optional: statistical significance testing
source("mcnemar_test.R")
```

Scripts must be run in order — later scripts load `.rds` files saved by earlier ones.

## Key Python → R Equivalents

| Python (sklearn/pandas)             | R equivalent                             |
|-------------------------------------|------------------------------------------|
| `pd.read_csv()`                     | `read.csv()`                             |
| `train_test_split(stratify=y)`      | `caret::createDataPartition()`           |
| `StandardScaler`                    | `caret::preProcess(method=c("center","scale"))` |
| `pd.get_dummies()`                  | `model.matrix()`                         |
| `GridSearchCV`                      | `caret::train()` with `tuneGrid`         |
| `StratifiedKFold`                   | `caret::trainControl(method="cv")`       |
| `LogisticRegression`                | `glmnet` via `caret` (method="glmnet")   |
| `SVC`                               | `e1071` via `caret` (method="svmLinear"/"svmRadial") |
| `RandomForestClassifier`            | `randomForest::randomForest()`           |
| `XGBClassifier`                     | `xgboost` via `caret` (method="xgbTree")|
| `GradientBoostingClassifier`        | `gbm` via `caret` (method="gbm")        |
| `SMOTE` (imblearn)                  | `DMwR2::SMOTE()`                         |
| `VotingClassifier`                  | manual ensemble averaging of probabilities |
| `joblib.dump/load`                  | `saveRDS()` / `readRDS()`               |
| `confusion_matrix` + `seaborn`      | `caret::confusionMatrix()` + `ggplot2`   |
| `RocCurveDisplay`                   | `pROC::roc()` + `ggplot2`               |
| `mcnemar` (statsmodels)             | `stats::mcnemar.test()`                 |

## What Each Script Does

| Script   | Week | Description                                                     |
|----------|------|-----------------------------------------------------------------|
| week1    | 1    | Load data, EDA plots, preprocessing, train/test split           |
| week2    | 2    | Logistic Regression + hyperparameter tuning + coefficients      |
| week3    | 3    | SVM (linear + RBF) + tuning + comparison                        |
| week4    | 4    | Random Forest + tuning + feature importance                     |
| week5    | 5    | Side-by-side model comparison, ROC curves, CV stability         |
| week6    | 6    | Fairness by gender & age, interpretability analysis             |
| week7    | 7    | Final dashboard, consolidated results, summary                  |

## Notes

- Models are saved as `.rds` files (R's native serialisation format), replacing Python's `.pkl` files.
- The `caret` package provides a unified interface analogous to scikit-learn pipelines.
- `glmnet`'s `lambda` and scikit-learn's `C` control regularization in opposite directions, but they are not generally exact reciprocals across implementations and objective scaling.
- Tree models (RF, XGBoost, GBM) are trained on unscaled features (`X_train_raw`), consistent with the Python version.

## Evaluation requirements

Move scaling and learned preprocessing inside the training/CV workflow before reporting independent results. Use validation data for model/threshold selection, keep a final holdout untouched, and report bad-credit recall alongside good-credit metrics with the class convention explicit. Fairness estimates should include subgroup counts and uncertainty; small-group differences alone do not establish fairness.

Save `sessionInfo()`, data provenance/version, split settings, metrics and generated artifacts with any reported result. The optional McNemar script compares paired model errors on the same observations; statistical significance is not a measure of practical effect size.
