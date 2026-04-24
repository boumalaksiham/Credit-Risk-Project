# =============================================================================
# WEEK 5: Structured Model Comparison
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(pROC)
library(ggplot2)
library(gridExtra)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD DATA AND MODELS
# -----------------------------------------------------------------------------

X_train_scaled <- read.csv(file.path(OUTPUT_DIR, "X_train_scaled.csv"))
X_test_scaled  <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
X_train_raw    <- read.csv(file.path(OUTPUT_DIR, "X_train_raw.csv"))
X_test_raw     <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
y_train        <- read.csv(file.path(OUTPUT_DIR, "y_train.csv"))$Risk
y_test         <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk

y_train_f <- factor(y_train, levels = c(0, 1), labels = c("Bad", "Good"))
y_test_f  <- factor(y_test,  levels = c(0, 1), labels = c("Bad", "Good"))

lr_model  <- readRDS(file.path(OUTPUT_DIR, "logistic_regression_model.rds"))
svm_model <- readRDS(file.path(OUTPUT_DIR, "svm_model.rds"))
rf_model  <- readRDS(file.path(OUTPUT_DIR, "random_forest_model.rds"))

lr_results  <- readRDS(file.path(OUTPUT_DIR, "lr_results.rds"))
svm_results <- readRDS(file.path(OUTPUT_DIR, "svm_results.rds"))
rf_results  <- readRDS(file.path(OUTPUT_DIR, "rf_results.rds"))

# Helper to extract predictions
get_probs <- function(model, X_new) {
  if (inherits(model, "randomForest")) {
    predict(model, X_new, type = "prob")[, "Good"]
  } else {
    predict(model, X_new, type = "prob")[, "Good"]
  }
}

# -----------------------------------------------------------------------------
# 2. CONSOLIDATED METRICS TABLE
# -----------------------------------------------------------------------------

metrics_keys <- c("accuracy", "precision", "recall", "f1", "roc_auc")

summary <- lapply(list(lr_results, svm_results, rf_results), function(res) {
  row <- data.frame(Model = res$model)
  for (k in metrics_keys) row[[toupper(k)]] <- round(as.numeric(res[[k]]), 4)
  row
})
summary_df <- do.call(rbind, summary)
rownames(summary_df) <- summary_df$Model
summary_df$Model <- NULL

cat(strrep("=", 70), "\n")
cat("MODEL COMPARISON - TEST SET PERFORMANCE\n")
cat(strrep("=", 70), "\n")
print(summary_df)

write.csv(summary_df, file.path(OUTPUT_DIR, "model_comparison.csv"))
cat(sprintf("\nSaved: %s/model_comparison.csv\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 3. PERFORMANCE BAR CHART
# -----------------------------------------------------------------------------

all_results <- list(lr_results, svm_results, rf_results)
plot_data <- do.call(rbind, lapply(all_results, function(res) {
  data.frame(
    Model  = res$model,
    Metric = toupper(metrics_keys),
    Score  = sapply(metrics_keys, function(k) as.numeric(res[[k]]))
  )
}))

p <- ggplot(plot_data, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
  geom_text(aes(label = sprintf("%.3f", Score)),
            position = position_dodge(width = 0.9),
            vjust = -0.3, size = 2.8) +
  scale_fill_manual(values = c(
    "Logistic Regression" = "steelblue",
    "SVM"                 = "mediumpurple",
    "Random Forest"       = "forestgreen"
  )) +
  ylim(0, 1.12) +
  labs(title = "Model Comparison: Test Set Performance Metrics",
       y = "Score", x = "Metric") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "model_comparison_bar.png"), p, dpi = 150, width = 10, height = 5)
cat(sprintf("Saved: %s/model_comparison_bar.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 4. ROC CURVES - ALL MODELS ON ONE PLOT
# -----------------------------------------------------------------------------

roc_lr  <- roc(as.numeric(y_test_f == "Good"),
               get_probs(lr_model, X_test_scaled), quiet = TRUE)
roc_svm <- roc(as.numeric(y_test_f == "Good"),
               get_probs(svm_model, X_test_scaled), quiet = TRUE)
roc_rf  <- roc(as.numeric(y_test_f == "Good"),
               predict(rf_model, X_test_raw, type = "prob")[, "Good"], quiet = TRUE)

roc_df <- bind_rows(
  data.frame(FPR = 1 - roc_lr$specificities,  TPR = roc_lr$sensitivities,
             Model = sprintf("LR (AUC=%.3f)", auc(roc_lr))),
  data.frame(FPR = 1 - roc_svm$specificities, TPR = roc_svm$sensitivities,
             Model = sprintf("SVM (AUC=%.3f)", auc(roc_svm))),
  data.frame(FPR = 1 - roc_rf$specificities,  TPR = roc_rf$sensitivities,
             Model = sprintf("RF (AUC=%.3f)", auc(roc_rf)))
)

p <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(title = "ROC Curves - All Models",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = c(0.7, 0.25))
ggsave(file.path(OUTPUT_DIR, "all_roc_curves.png"), p, dpi = 150, width = 7, height = 6)
cat(sprintf("Saved: %s/all_roc_curves.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 5. CROSS-VALIDATION STABILITY
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n")
cat("CROSS-VALIDATION STABILITY (5-Fold F1 Scores)\n")
cat(strrep("=", 70), "\n")

set.seed(42)
folds <- createFolds(y_train_f, k = 5, list = TRUE)

get_cv_f1 <- function(model, X_tr, is_rf = FALSE) {
  sapply(folds, function(idx) {
    X_tr_fold  <- X_tr[-idx, ]
    X_val_fold <- X_tr[idx, ]
    y_tr_fold  <- y_train_f[-idx]
    y_val_fold <- y_train_f[idx]
    
    if (is_rf) {
      library(randomForest)
      fit <- randomForest(x = X_tr_fold, y = y_tr_fold, ntree = 200)
      preds <- predict(fit, X_val_fold)
    } else {
      fit   <- update(model, x = X_tr_fold, y = y_tr_fold)
      preds <- predict(fit, X_val_fold)
    }
    cm_cv <- confusionMatrix(preds, y_val_fold, positive = "Good")
    cm_cv$byClass["F1"]
  })
}

cv_lr  <- get_cv_f1(lr_model,  X_train_scaled)
cv_svm <- get_cv_f1(svm_model, X_train_scaled)
cv_rf  <- get_cv_f1(rf_model,  X_train_raw, is_rf = TRUE)

for (pair in list(c("Logistic Regression", cv_lr),
                  c("SVM", cv_svm),
                  c("Random Forest", cv_rf))) {
  nm     <- pair[1]
  scores <- as.numeric(pair[-1])
  cat(sprintf("%-25s: [%s] | Mean=%.4f, Std=%.4f\n",
              nm,
              paste(round(scores, 4), collapse = ", "),
              mean(scores, na.rm = TRUE),
              sd(scores, na.rm = TRUE)))
}

# Box plot
cv_box_df <- data.frame(
  F1    = c(cv_lr, cv_svm, cv_rf),
  Model = rep(c("Logistic\nRegression", "SVM", "Random\nForest"), each = 5)
)
p <- ggplot(cv_box_df, aes(x = Model, y = F1, fill = Model)) +
  geom_boxplot(color = "navy", alpha = 0.6) +
  stat_summary(fun = median, geom = "crossbar", color = "red", linewidth = 0.6) +
  scale_fill_manual(values = c(
    "Logistic\nRegression" = "lightblue",
    "SVM"                  = "plum",
    "Random\nForest"       = "lightgreen"
  ), guide = "none") +
  ylim(0, 1) +
  labs(title = "Cross-Validation F1 Score Stability", y = "F1 Score") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "cv_stability_boxplot.png"), p, dpi = 150, width = 7, height = 5)
cat(sprintf("Saved: %s/cv_stability_boxplot.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 6. CONFUSION MATRIX COMPARISON
# -----------------------------------------------------------------------------

make_cm_plot <- function(model, X_te, label, fill_color) {
  preds <- predict(model, X_te)
  cm    <- confusionMatrix(preds, y_test_f, positive = "Good")
  df    <- as.data.frame(cm$table)
  ggplot(df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 5) +
    scale_fill_gradient(low = "white", high = fill_color) +
    labs(title = paste("Confusion Matrix\n", label),
         x = "Actual", y = "Predicted") +
    theme_minimal() + theme(legend.position = "none")
}

p1 <- make_cm_plot(lr_model,  X_test_scaled, "Logistic Regression", "steelblue")
p2 <- make_cm_plot(svm_model, X_test_scaled, "SVM",                 "mediumpurple")
p3 <- make_cm_plot(rf_model,  X_test_raw,    "Random Forest",       "forestgreen")

p_all <- grid.arrange(p1, p2, p3, ncol = 3,
                      top = "Confusion Matrix Comparison")
ggsave(file.path(OUTPUT_DIR, "confusion_matrix_comparison.png"), p_all,
       dpi = 150, width = 15, height = 4)
cat(sprintf("Saved: %s/confusion_matrix_comparison.png\n", OUTPUT_DIR))

# Best model
best_model_name <- rownames(summary_df)[which.max(summary_df$F1)]
cat("\n", strrep("=", 70), "\n")
cat(sprintf("RECOMMENDED BEST MODEL (by F1): %s\n", best_model_name))
cat(strrep("=", 70), "\n")
print(summary_df[best_model_name, ])
cat("Week 5 complete!\n")
