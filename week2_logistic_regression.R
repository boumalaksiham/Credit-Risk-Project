# =============================================================================
# WEEK 2: Logistic Regression Baseline
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(reshape2)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# -----------------------------------------------------------------------------

X_train <- read.csv(file.path(OUTPUT_DIR, "X_train_scaled.csv"))
X_test  <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
y_train <- read.csv(file.path(OUTPUT_DIR, "y_train.csv"))$Risk
y_test  <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk

cat(sprintf("X_train shape: %d x %d\n", nrow(X_train), ncol(X_train)))
cat(sprintf("X_test shape:  %d x %d\n", nrow(X_test),  ncol(X_test)))

# Convert target to factor for caret
y_train_f <- factor(y_train, levels = c(0, 1), labels = c("Bad", "Good"))
y_test_f  <- factor(y_test,  levels = c(0, 1), labels = c("Bad", "Good"))

train_df <- cbind(X_train, Risk = y_train_f)

# -----------------------------------------------------------------------------
# 2. CROSS-VALIDATION WITH HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("LOGISTIC REGRESSION - HYPERPARAMETER TUNING (GridSearchCV)\n")
cat(strrep("=", 60), "\n")

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  seeds           = lapply(1:6, function(i) 42)
)

# Tune regularisation (C in sklearn = 1/lambda in R's glmnet)
tune_grid <- expand.grid(
  alpha  = 0,          # ridge (L2), equivalent to penalty='l2'
  lambda = c(10, 1, 0.1, 0.01, 0.001, 0.0001)  # 1/C
)

set.seed(42)
gs_lr <- train(
  Risk ~ .,
  data      = train_df,
  method    = "glmnet",
  trControl = ctrl,
  tuneGrid  = tune_grid,
  metric    = "F",
  family    = "binomial"
)

cat(sprintf("\nBest parameters: alpha=%s, lambda=%s\n",
            gs_lr$bestTune$alpha, gs_lr$bestTune$lambda))
cat(sprintf("Best CV F1 score: %.4f\n", max(gs_lr$results$F)))

# -----------------------------------------------------------------------------
# 3. CROSS-VALIDATION METRICS (manual 5-fold)
# -----------------------------------------------------------------------------

set.seed(42)
folds <- createFolds(y_train_f, k = 5, list = TRUE)

cv_metrics <- lapply(folds, function(idx) {
  X_cv_train <- X_train[-idx, ]
  y_cv_train <- y_train_f[-idx]
  X_cv_val   <- X_train[idx, ]
  y_cv_val   <- y_train_f[idx]
  
  fit <- train(
    x = X_cv_train, y = y_cv_train,
    method    = "glmnet",
    trControl = trainControl(method = "none", classProbs = TRUE),
    tuneGrid  = gs_lr$bestTune,
    family    = "binomial"
  )
  
  preds <- predict(fit, X_cv_val)
  probs <- predict(fit, X_cv_val, type = "prob")[, "Good"]
  
  cm <- confusionMatrix(preds, y_cv_val, positive = "Good")
  roc_obj <- roc(as.numeric(y_cv_val == "Good"), probs, quiet = TRUE)
  
  c(
    accuracy  = cm$overall["Accuracy"],
    precision = cm$byClass["Precision"],
    recall    = cm$byClass["Recall"],
    f1        = cm$byClass["F1"],
    roc_auc   = auc(roc_obj)
  )
})

cv_df <- do.call(rbind, cv_metrics)
cat("\n", strrep("=", 60), "\n")
cat("CROSS-VALIDATION RESULTS (5-Fold)\n")
cat(strrep("=", 60), "\n")
for (metric in colnames(cv_df)) {
  cat(sprintf("%-12s: %.4f +/- %.4f\n",
              metric, mean(cv_df[, metric]), sd(cv_df[, metric])))
}

# -----------------------------------------------------------------------------
# 4. FINAL EVALUATION ON TEST SET
# -----------------------------------------------------------------------------

best_lr <- gs_lr$finalModel
best_lr_train <- train(
  x = X_train, y = y_train_f,
  method    = "glmnet",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid  = gs_lr$bestTune,
  family    = "binomial"
)

y_pred      <- predict(best_lr_train, X_test)
y_pred_prob <- predict(best_lr_train, X_test, type = "prob")[, "Good"]

cm      <- confusionMatrix(y_pred, y_test_f, positive = "Good")
roc_obj <- roc(as.numeric(y_test_f == "Good"), y_pred_prob, quiet = TRUE)

cat("\n", strrep("=", 60), "\n")
cat("TEST SET EVALUATION - LOGISTIC REGRESSION\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Accuracy:  %.4f\n", cm$overall["Accuracy"]))
cat(sprintf("Precision: %.4f\n", cm$byClass["Precision"]))
cat(sprintf("Recall:    %.4f\n", cm$byClass["Recall"]))
cat(sprintf("F1 Score:  %.4f\n", cm$byClass["F1"]))
cat(sprintf("ROC-AUC:   %.4f\n", auc(roc_obj)))
print(cm)

# Confusion Matrix plot
cm_df <- as.data.frame(cm$table)
p <- ggplot(cm_df, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile(color = "white") +
  geom_text(aes(label = Freq), size = 6) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(title = "Confusion Matrix - Logistic Regression",
       x = "Actual", y = "Predicted") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "lr_confusion_matrix.png"), p, dpi = 150, width = 5, height = 4)
cat(sprintf("Saved: %s/lr_confusion_matrix.png\n", OUTPUT_DIR))

# ROC Curve
roc_df <- data.frame(
  FPR = 1 - roc_obj$specificities,
  TPR = roc_obj$sensitivities
)
p <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "steelblue", linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = sprintf("ROC Curve - Logistic Regression (AUC = %.4f)", auc(roc_obj)),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "lr_roc_curve.png"), p, dpi = 150, width = 6, height = 5)
cat(sprintf("Saved: %s/lr_roc_curve.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 5. COEFFICIENT ANALYSIS
# -----------------------------------------------------------------------------

coef_mat <- coef(best_lr_train$finalModel, s = gs_lr$bestTune$lambda)
coef_vec <- as.vector(coef_mat)[-1]  # drop intercept
names(coef_vec) <- rownames(coef_mat)[-1]

coef_df <- data.frame(
  Feature         = names(coef_vec),
  Coefficient     = coef_vec,
  Abs_Coefficient = abs(coef_vec)
) %>% arrange(desc(Abs_Coefficient))

cat("\n", strrep("=", 60), "\n")
cat("LOGISTIC REGRESSION COEFFICIENTS (sorted by importance)\n")
cat(strrep("=", 60), "\n")
print(coef_df)

top_n    <- 15
top_coef <- head(coef_df, top_n)
p <- ggplot(top_coef, aes(x = reorder(Feature, Coefficient), y = Coefficient,
                           fill = Coefficient > 0)) +
  geom_bar(stat = "identity", color = "black") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato"), guide = "none") +
  geom_hline(yintercept = 0, linewidth = 0.8) +
  labs(title = sprintf("Top %d LR Coefficients\n(Blue=positive/good, Red=negative/bad)", top_n),
       x = "Feature", y = "Coefficient") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "lr_coefficients.png"), p, dpi = 150, width = 8, height = 6)
cat(sprintf("Saved: %s/lr_coefficients.png\n", OUTPUT_DIR))

# Save model and results
saveRDS(best_lr_train, file.path(OUTPUT_DIR, "logistic_regression_model.rds"))

lr_results <- list(
  model      = "Logistic Regression",
  best_params = gs_lr$bestTune,
  accuracy   = cm$overall["Accuracy"],
  precision  = cm$byClass["Precision"],
  recall     = cm$byClass["Recall"],
  f1         = cm$byClass["F1"],
  roc_auc    = as.numeric(auc(roc_obj))
)
saveRDS(lr_results, file.path(OUTPUT_DIR, "lr_results.rds"))

cat(sprintf("\nModel saved: %s/logistic_regression_model.rds\n", OUTPUT_DIR))
cat("Week 2 complete!\n")
