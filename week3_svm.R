# =============================================================================
# WEEK 3: Support Vector Machine Implementation
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(e1071)
library(pROC)
library(ggplot2)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD PREPROCESSED DATA
# -----------------------------------------------------------------------------

X_train <- read.csv(file.path(OUTPUT_DIR, "X_train_scaled.csv"))
X_test  <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
y_train <- read.csv(file.path(OUTPUT_DIR, "y_train.csv"))$Risk
y_test  <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk

y_train_f <- factor(y_train, levels = c(0, 1), labels = c("Bad", "Good"))
y_test_f  <- factor(y_test,  levels = c(0, 1), labels = c("Bad", "Good"))
train_df  <- cbind(X_train, Risk = y_train_f)

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)

# -----------------------------------------------------------------------------
# 2. LINEAR SVM
# -----------------------------------------------------------------------------

cat(strrep("=", 60), "\n")
cat("SVM - LINEAR KERNEL (GridSearchCV)\n")
cat(strrep("=", 60), "\n")

tune_linear <- expand.grid(C = c(0.01, 0.1, 1, 10, 100))

set.seed(42)
gs_linear <- train(
  Risk ~ .,
  data      = train_df,
  method    = "svmLinear",
  trControl = ctrl,
  tuneGrid  = tune_linear,
  metric    = "F",
  prob.model = TRUE
)

cat(sprintf("\nBest params (linear): C=%s\n", gs_linear$bestTune$C))
cat(sprintf("Best CV F1 (linear):  %.4f\n", max(gs_linear$results$F)))

# -----------------------------------------------------------------------------
# 3. RBF SVM
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("SVM - RBF KERNEL (GridSearchCV)\n")
cat(strrep("=", 60), "\n")

tune_rbf <- expand.grid(
  C     = c(0.1, 1, 10, 100),
  sigma = c(0.01, 0.05, 0.1, 0.5)
)

set.seed(42)
gs_rbf <- train(
  Risk ~ .,
  data      = train_df,
  method    = "svmRadial",
  trControl = ctrl,
  tuneGrid  = tune_rbf,
  metric    = "F",
  prob.model = TRUE
)

cat(sprintf("\nBest params (RBF): C=%s, sigma=%s\n",
            gs_rbf$bestTune$C, gs_rbf$bestTune$sigma))
cat(sprintf("Best CV F1 (RBF):  %.4f\n", max(gs_rbf$results$F)))

# -----------------------------------------------------------------------------
# 4. SELECT BEST SVM AND EVALUATE ON TEST SET
# -----------------------------------------------------------------------------

best_linear_f1 <- max(gs_linear$results$F)
best_rbf_f1    <- max(gs_rbf$results$F)

if (best_linear_f1 >= best_rbf_f1) {
  best_svm       <- gs_linear
  best_svm_label <- "Linear SVM"
} else {
  best_svm       <- gs_rbf
  best_svm_label <- "RBF SVM"
}
cat(sprintf("\nSelected best SVM: %s\n", best_svm_label))

# CV metrics
set.seed(42)
folds <- createFolds(y_train_f, k = 5, list = TRUE)

cv_metrics <- lapply(folds, function(idx) {
  X_cv_train <- X_train[-idx, ]
  y_cv_train <- y_train_f[-idx]
  X_cv_val   <- X_train[idx, ]
  y_cv_val   <- y_train_f[idx]
  
  method_name <- if (best_svm_label == "Linear SVM") "svmLinear" else "svmRadial"
  fit <- train(
    x = X_cv_train, y = y_cv_train,
    method    = method_name,
    trControl = trainControl(method = "none", classProbs = TRUE),
    tuneGrid  = best_svm$bestTune,
    prob.model = TRUE
  )
  
  preds <- predict(fit, X_cv_val)
  probs <- predict(fit, X_cv_val, type = "prob")[, "Good"]
  cm_cv <- confusionMatrix(preds, y_cv_val, positive = "Good")
  roc_cv <- roc(as.numeric(y_cv_val == "Good"), probs, quiet = TRUE)
  
  c(accuracy  = cm_cv$overall["Accuracy"],
    precision = cm_cv$byClass["Precision"],
    recall    = cm_cv$byClass["Recall"],
    f1        = cm_cv$byClass["F1"],
    roc_auc   = as.numeric(auc(roc_cv)))
})

cv_df <- do.call(rbind, cv_metrics)
cat("\n", strrep("=", 60), "\n")
cat(sprintf("CROSS-VALIDATION RESULTS (5-Fold) - %s\n", best_svm_label))
cat(strrep("=", 60), "\n")
for (metric in colnames(cv_df)) {
  cat(sprintf("%-12s: %.4f +/- %.4f\n", metric, mean(cv_df[, metric]), sd(cv_df[, metric])))
}

# Test set evaluation
y_pred      <- predict(best_svm, X_test)
y_pred_prob <- predict(best_svm, X_test, type = "prob")[, "Good"]

cm      <- confusionMatrix(y_pred, y_test_f, positive = "Good")
roc_obj <- roc(as.numeric(y_test_f == "Good"), y_pred_prob, quiet = TRUE)

cat("\n", strrep("=", 60), "\n")
cat(sprintf("TEST SET EVALUATION - %s\n", best_svm_label))
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
  scale_fill_gradient(low = "white", high = "mediumpurple") +
  labs(title = sprintf("Confusion Matrix - %s", best_svm_label),
       x = "Actual", y = "Predicted") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "svm_confusion_matrix.png"), p, dpi = 150, width = 5, height = 4)
cat(sprintf("Saved: %s/svm_confusion_matrix.png\n", OUTPUT_DIR))

# ROC Curve
roc_df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
p <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "mediumpurple", linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = sprintf("ROC Curve - %s (AUC = %.4f)", best_svm_label, auc(roc_obj)),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "svm_roc_curve.png"), p, dpi = 150, width = 6, height = 5)
cat(sprintf("Saved: %s/svm_roc_curve.png\n", OUTPUT_DIR))

cat("\n", strrep("=", 60), "\n")
cat("LINEAR vs RBF SVM COMPARISON (CV F1)\n")
cat(strrep("=", 60), "\n")
cat(sprintf("Linear SVM CV F1: %.4f\n", best_linear_f1))
cat(sprintf("RBF SVM    CV F1: %.4f\n", best_rbf_f1))
cat(sprintf("Selected:         %s\n", best_svm_label))

# Save model and results
saveRDS(best_svm, file.path(OUTPUT_DIR, "svm_model.rds"))

svm_results <- list(
  model       = best_svm_label,
  best_params = best_svm$bestTune,
  accuracy    = cm$overall["Accuracy"],
  precision   = cm$byClass["Precision"],
  recall      = cm$byClass["Recall"],
  f1          = cm$byClass["F1"],
  roc_auc     = as.numeric(auc(roc_obj))
)
saveRDS(svm_results, file.path(OUTPUT_DIR, "svm_results.rds"))

cat(sprintf("\nModel saved: %s/svm_model.rds\n", OUTPUT_DIR))
cat("Week 3 complete!\n")
