# Credit Risk Classification — Siham Boumalak (R Version)

## Analysis question

How do logistic regression, SVM, and Random Forest differ when predicting good versus bad credit, and how do their errors vary by age and sex?

The workflow separates model fitting, comparison, and subgroup analysis into seven stages. Logistic coefficients and Random Forest feature importance support different kinds of interpretation; neither establishes that an input causes credit risk. The practical question is which errors each model makes, rather than which one has the largest accuracy alone.

**Start here:** [week1_data_preprocessing.R](week1_data_preprocessing.R) defines the data preparation; [week5_model_comparison.R](week5_model_comparison.R) compares models; [week6_fairness_interpretability.R](week6_fairness_interpretability.R) examines subgroup behavior. The evaluation issues below must be addressed before treating generated scores as independent evidence.

## Evaluation status

The scripts now split on a categorical target before learning category levels or scaling. Scaled exports use only training statistics; Logistic Regression and SVM receive raw predictors and fit scaling inside each caret resampling fold. A shared metric function returns the F1 score requested by tuning, with **Good as the positive class**.

The optional optimization stage compares Logistic Regression and Random Forest, with and without **random oversampling inside training folds**. It selects a candidate by training CV F1, calibrates its threshold on separate training-partition rows, then evaluates that frozen candidate on the original holdout. It replaces the broken DMwR2/SMOTE path and avoids synthetic interpolation across one-hot categorical predictors.

No dataset or new run outputs are committed here. These fixes do not establish new model performance. The weekly comparison reports are descriptive: their reused holdout must not be used to choose another model or tune additional parameters. Training-CV tuning scores are not nested, unbiased estimates. Raw categorical encoding is learned on the outer training partition; a fully fold-local categorical recipe remains a possible extension.

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
  "e1071",          # supporting classification utilities
  "kernlab",        # caret svmLinear/svmRadial models
  "randomForest",   # Random Forest
  "glmnet",         # Logistic Regression with regularisation
  "pROC",           # ROC curves and AUC
  "reshape2",       # data reshaping for heatmaps
  "gridExtra",      # multi-panel plots
  "xgboost",        # XGBoost (used by caret)
  "gbm"             # Gradient Boosting (used by caret)
))
```

`grid` is included with R. The optional optimization stage no longer requires DMwR2, XGBoost, or GBM; it uses caret's built-in random oversampling. Install `kernlab` for caret's SVM implementations.

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
source("improve_metrics.R")  # CV selection + separate threshold calibration

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
| Random oversampling | `caret::trainControl(sampling="up")`, inside training folds |
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

Scaling is fit within caret training folds for LR/SVM. The optional stage separates threshold calibration from model fitting. Report bad-credit recall alongside good-credit metrics, and use a fresh external cohort if the existing holdout has influenced further decisions. Fairness estimates should include subgroup counts and uncertainty; small-group differences alone do not establish fairness.

Save `sessionInfo()`, data provenance/version, split settings, metrics and generated artifacts with any reported result. The optional McNemar script compares paired model errors on the same observations; statistical significance is not a measure of practical effect size.

## Regression checks

```bash
Rscript tests/test_evaluation.R
```

These checks exercise training-only scaler fitting, the positive-class metric convention, and threshold selection. GitHub Actions also parses every R file. They do not train the full workflow or replace dataset-level evaluation. After changing preprocessing, rerun week 1 and all downstream stages; older scaled exports/model files are incompatible with the new raw-input model contract.
