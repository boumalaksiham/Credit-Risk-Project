# =============================================================================
# WEEK 4: Random Forest Implementation and Tuning
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD DATA (unscaled — tree models don't need scaling)
# -----------------------------------------------------------------------------

X_train <- read.csv(file.path(OUTPUT_DIR, "X_train_raw.csv"))
X_test  <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
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
# 2. HYPERPARAMETER TUNING
# -----------------------------------------------------------------------------

cat(strrep("=", 60), "\n")
cat("RANDOM FOREST - HYPERPARAMETER TUNING (GridSearchCV)\n")
cat(strrep("=", 60), "\n")

tune_grid <- expand.grid(mtry = c(2, 4, 6, 8, 10))

set.seed(42)
gs_rf <- train(
  Risk ~ .,
  data      = train_df,
  method    = "rf",
  trControl = ctrl,
  tuneGrid  = tune_grid,
  metric    = "F",
  ntree     = 200
)

cat(sprintf("\nBest parameters: mtry=%d\n", gs_rf$bestTune$mtry))
cat(sprintf("Best CV F1 score: %.4f\n", max(gs_rf$results$F)))

# Additional manual grid for n_estimators, max_depth, min_samples_split
# We use randomForest directly for finer control
param_combos <- expand.grid(
  ntree          = c(100, 200, 300),
  maxnodes       = c(NA, 50, 100),      # NA = unlimited depth
  nodesize       = c(2, 5, 10)
)

set.seed(42)
best_f1 <- 0
best_params <- list()

# Simplified grid search (subset for runtime)
for (ntree in c(100, 200)) {
  for (nodesize in c(5, 10)) {
    folds <- createFolds(y_train_f, k = 5)
    fold_f1 <- sapply(folds, function(idx) {
      rf_cv <- randomForest(
        x        = X_train[-idx, ],
        y        = y_train_f[-idx],
        ntree    = ntree,
        nodesize = nodesize,
        mtry     = gs_rf$bestTune$mtry
      )
      preds <- predict(rf_cv, X_train[idx, ])
      cm_cv <- confusionMatrix(preds, y_train_f[idx], positive = "Good")
      cm_cv$byClass["F1"]
    })
    mean_f1 <- mean(fold_f1, na.rm = TRUE)
    if (mean_f1 > best_f1) {
      best_f1     <- mean_f1
      best_params <- list(ntree = ntree, nodesize = nodesize)
    }
  }
}

cat(sprintf("Best RF params (extended): ntree=%d, nodesize=%d\n",
            best_params$ntree, best_params$nodesize))

# -----------------------------------------------------------------------------
# 3. CROSS-VALIDATION METRICS
# -----------------------------------------------------------------------------

set.seed(42)
folds <- createFolds(y_train_f, k = 5, list = TRUE)

cv_metrics <- lapply(folds, function(idx) {
  rf_cv <- randomForest(
    x        = X_train[-idx, ],
    y        = y_train_f[-idx],
    ntree    = best_params$ntree,
    nodesize = best_params$nodesize,
    mtry     = gs_rf$bestTune$mtry
  )
  preds <- predict(rf_cv, X_train[idx, ])
  probs <- predict(rf_cv, X_train[idx, ], type = "prob")[, "Good"]
  cm_cv <- confusionMatrix(preds, y_train_f[idx], positive = "Good")
  roc_cv <- roc(as.numeric(y_train_f[idx] == "Good"), probs, quiet = TRUE)
  
  c(accuracy  = cm_cv$overall["Accuracy"],
    precision = cm_cv$byClass["Precision"],
    recall    = cm_cv$byClass["Recall"],
    f1        = cm_cv$byClass["F1"],
    roc_auc   = as.numeric(auc(roc_cv)))
})

cv_df <- do.call(rbind, cv_metrics)
cat("\n", strrep("=", 60), "\n")
cat("CROSS-VALIDATION RESULTS (5-Fold)\n")
cat(strrep("=", 60), "\n")
for (metric in colnames(cv_df)) {
  cat(sprintf("%-12s: %.4f +/- %.4f\n", metric, mean(cv_df[, metric]), sd(cv_df[, metric])))
}

# -----------------------------------------------------------------------------
# 4. TEST SET EVALUATION
# -----------------------------------------------------------------------------

set.seed(42)
best_rf <- randomForest(
  x        = X_train,
  y        = y_train_f,
  ntree    = best_params$ntree,
  nodesize = best_params$nodesize,
  mtry     = gs_rf$bestTune$mtry,
  importance = TRUE
)

y_pred      <- predict(best_rf, X_test)
y_pred_prob <- predict(best_rf, X_test, type = "prob")[, "Good"]

cm      <- confusionMatrix(y_pred, y_test_f, positive = "Good")
roc_obj <- roc(as.numeric(y_test_f == "Good"), y_pred_prob, quiet = TRUE)

cat("\n", strrep("=", 60), "\n")
cat("TEST SET EVALUATION - RANDOM FOREST\n")
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
  scale_fill_gradient(low = "white", high = "forestgreen") +
  labs(title = "Confusion Matrix - Random Forest", x = "Actual", y = "Predicted") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "rf_confusion_matrix.png"), p, dpi = 150, width = 5, height = 4)
cat(sprintf("Saved: %s/rf_confusion_matrix.png\n", OUTPUT_DIR))

# ROC Curve
roc_df <- data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities)
p <- ggplot(roc_df, aes(x = FPR, y = TPR)) +
  geom_line(color = "forestgreen", linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
  labs(title = sprintf("ROC Curve - Random Forest (AUC = %.4f)", auc(roc_obj)),
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "rf_roc_curve.png"), p, dpi = 150, width = 6, height = 5)
cat(sprintf("Saved: %s/rf_roc_curve.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 5. FEATURE IMPORTANCE ANALYSIS
# -----------------------------------------------------------------------------

fi      <- importance(best_rf, type = 2)  # Mean Decrease Gini
fi_df   <- data.frame(Feature = rownames(fi), Importance = fi[, 1]) %>%
  arrange(desc(Importance))

cat("\n", strrep("=", 60), "\n")
cat("FEATURE IMPORTANCE (Top 15)\n")
cat(strrep("=", 60), "\n")
print(head(fi_df, 15))

top_fi <- head(fi_df, 15)
p <- ggplot(top_fi, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "forestgreen", color = "black", alpha = 0.8) +
  coord_flip() +
  labs(title = "Random Forest - Top 15 Feature Importances",
       x = "Feature", y = "Mean Decrease Gini") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "rf_feature_importance.png"), p, dpi = 150, width = 8, height = 6)
cat(sprintf("Saved: %s/rf_feature_importance.png\n", OUTPUT_DIR))

# Effect of n_estimators
cat("\n", strrep("=", 60), "\n")
cat("EFFECT OF N_ESTIMATORS ON PERFORMANCE\n")
cat(strrep("=", 60), "\n")

n_range <- c(10, 50, 100, 150, 200, 250, 300)
oob_scores  <- numeric(length(n_range))
cv_f1_scores <- numeric(length(n_range))

for (i in seq_along(n_range)) {
  n <- n_range[i]
  rf_n <- randomForest(
    x        = X_train,
    y        = y_train_f,
    ntree    = n,
    nodesize = best_params$nodesize,
    mtry     = gs_rf$bestTune$mtry
  )
  oob_pred <- factor(
    levels(y_train_f)[apply(rf_n$votes, 1, which.max)],
    levels = levels(y_train_f)
  )
  oob_cm        <- confusionMatrix(oob_pred, y_train_f, positive = "Good")
  oob_scores[i] <- oob_cm$overall["Accuracy"]
  
  fold_f1 <- sapply(createFolds(y_train_f, k = 5), function(idx) {
    rf_cv <- randomForest(x = X_train[-idx, ], y = y_train_f[-idx],
                          ntree = n, nodesize = best_params$nodesize,
                          mtry = gs_rf$bestTune$mtry)
    preds <- predict(rf_cv, X_train[idx, ])
    cm_cv <- confusionMatrix(preds, y_train_f[idx], positive = "Good")
    cm_cv$byClass["F1"]
  })
  cv_f1_scores[i] <- mean(fold_f1, na.rm = TRUE)
  cat(sprintf("n=%3d | OOB Acc=%.4f | CV F1=%.4f\n", n, oob_scores[i], cv_f1_scores[i]))
}

plot_df <- data.frame(
  n    = rep(n_range, 2),
  Score = c(oob_scores, cv_f1_scores),
  Type  = rep(c("OOB Score", "CV F1"), each = length(n_range))
)
p <- ggplot(plot_df, aes(x = n, y = Score, color = Type, group = Type)) +
  geom_line() + geom_point() +
  labs(title = "Random Forest: Performance vs n_estimators",
       x = "Number of Trees", y = "Score") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "rf_n_estimators_effect.png"), p, dpi = 150, width = 7, height = 4)
cat(sprintf("Saved: %s/rf_n_estimators_effect.png\n", OUTPUT_DIR))

# Save model and results
saveRDS(best_rf, file.path(OUTPUT_DIR, "random_forest_model.rds"))

rf_results <- list(
  model       = "Random Forest",
  best_params = c(best_params, mtry = gs_rf$bestTune$mtry),
  accuracy    = cm$overall["Accuracy"],
  precision   = cm$byClass["Precision"],
  recall      = cm$byClass["Recall"],
  f1          = cm$byClass["F1"],
  roc_auc     = as.numeric(auc(roc_obj))
)
saveRDS(rf_results, file.path(OUTPUT_DIR, "rf_results.rds"))

cat(sprintf("\nModel saved: %s/random_forest_model.rds\n", OUTPUT_DIR))
cat("Week 4 complete!\n")
