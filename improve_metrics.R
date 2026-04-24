# =============================================================================
# IMPROVED METRICS: All Optimization Techniques
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(DMwR2)        # SMOTE
library(xgboost)
library(gbm)
library(ggplot2)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------

X_train_scaled <- read.csv(file.path(OUTPUT_DIR, "X_train_scaled.csv"))
X_test_scaled  <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
X_train_raw    <- read.csv(file.path(OUTPUT_DIR, "X_train_raw.csv"))
X_test_raw     <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
y_train        <- read.csv(file.path(OUTPUT_DIR, "y_train.csv"))$Risk
y_test         <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk

y_train_f <- factor(y_train, levels = c(0, 1), labels = c("Bad", "Good"))
y_test_f  <- factor(y_test,  levels = c(0, 1), labels = c("Bad", "Good"))

ctrl <- trainControl(
  method          = "cv",
  number          = 5,
  classProbs      = TRUE,
  summaryFunction = twoClassSummary
)

evaluate <- function(name, y_true_f, y_pred_f, y_prob) {
  cm      <- confusionMatrix(y_pred_f, y_true_f, positive = "Good")
  roc_obj <- roc(as.numeric(y_true_f == "Good"), y_prob, quiet = TRUE)
  list(
    Model     = name,
    Accuracy  = round(cm$overall["Accuracy"],   4),
    Precision = round(cm$byClass["Precision"],  4),
    Recall    = round(cm$byClass["Recall"],     4),
    F1        = round(cm$byClass["F1"],         4),
    ROC_AUC   = round(as.numeric(auc(roc_obj)), 4)
  )
}

results <- list()

# -----------------------------------------------------------------------------
# 2. BASELINE (original best models for comparison)
# -----------------------------------------------------------------------------

cat(strrep("=", 60), "\n")
cat("BASELINE - Original Models\n")
cat(strrep("=", 60), "\n")

lr_base  <- readRDS(file.path(OUTPUT_DIR, "logistic_regression_model.rds"))
svm_base <- readRDS(file.path(OUTPUT_DIR, "svm_model.rds"))
rf_base  <- readRDS(file.path(OUTPUT_DIR, "random_forest_model.rds"))

for (item in list(
  list(name = "Baseline LR",  model = lr_base,  X_te = X_test_scaled),
  list(name = "Baseline SVM", model = svm_base, X_te = X_test_scaled),
  list(name = "Baseline RF",  model = rf_base,  X_te = X_test_raw)
)) {
  y_pred <- predict(item$model, item$X_te)
  y_prob <- predict(item$model, item$X_te, type = "prob")[, "Good"]
  r      <- evaluate(item$name, y_test_f, y_pred, y_prob)
  results[[length(results) + 1]] <- r
  cat(sprintf("%-20s | Acc=%.4f F1=%.4f AUC=%.4f\n",
              r$Model, r$Accuracy, r$F1, r$ROC_AUC))
}

# -----------------------------------------------------------------------------
# 3. TECHNIQUE 1: SMOTE
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("TECHNIQUE 1: SMOTE Oversampling\n")
cat(strrep("=", 60), "\n")

# SMOTE with DMwR2
train_scaled_df <- cbind(X_train_scaled, Risk = y_train_f)
train_raw_df    <- cbind(X_train_raw,    Risk = y_train_f)

set.seed(42)
smote_scaled <- SMOTE(Risk ~ ., data = train_scaled_df, perc.over = 200, perc.under = 150)
smote_raw    <- SMOTE(Risk ~ ., data = train_raw_df,    perc.over = 200, perc.under = 150)

X_train_smote       <- smote_scaled[, !names(smote_scaled) %in% "Risk"]
y_train_smote       <- smote_scaled$Risk
X_train_smote_raw   <- smote_raw[, !names(smote_raw) %in% "Risk"]
y_train_smote_raw   <- smote_raw$Risk

cat(sprintf("After SMOTE (scaled) - Class distribution: %s\n",
            paste(names(table(y_train_smote)), table(y_train_smote), sep = "=", collapse = ", ")))

# RF + SMOTE
set.seed(42)
rf_smote <- randomForest(x = X_train_smote_raw, y = y_train_smote_raw,
                          ntree = 200, nodesize = 10)
y_pred <- predict(rf_smote, X_test_raw)
y_prob <- predict(rf_smote, X_test_raw, type = "prob")[, "Good"]
r      <- evaluate("RF + SMOTE", y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("RF + SMOTE           | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# LR + SMOTE
set.seed(42)
lr_smote <- train(
  x = X_train_smote, y = y_train_smote,
  method    = "glmnet",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid  = data.frame(alpha = 0, lambda = 0.01),
  family    = "binomial"
)
y_pred <- predict(lr_smote, X_test_scaled)
y_prob <- predict(lr_smote, X_test_scaled, type = "prob")[, "Good"]
r      <- evaluate("LR + SMOTE", y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("LR + SMOTE           | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# -----------------------------------------------------------------------------
# 4. TECHNIQUE 2: THRESHOLD TUNING
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("TECHNIQUE 2: Optimal Threshold Tuning\n")
cat(strrep("=", 60), "\n")

find_best_threshold <- function(model, X_val, y_val_f, is_rf = FALSE) {
  probs <- if (is_rf) predict(model, X_val, type = "prob")[, "Good"] else
    predict(model, X_val, type = "prob")[, "Good"]
  best_t <- 0.5; best_f1 <- 0
  for (t in seq(0.2, 0.8, by = 0.01)) {
    preds    <- factor(ifelse(probs >= t, "Good", "Bad"), levels = c("Bad", "Good"))
    cm_t     <- confusionMatrix(preds, y_val_f, positive = "Good")
    f1_t     <- cm_t$byClass["F1"]
    if (!is.na(f1_t) && f1_t > best_f1) { best_f1 <- f1_t; best_t <- t }
  }
  list(threshold = best_t, f1 = best_f1)
}

# RF threshold
res_rf_t <- find_best_threshold(rf_base, X_train_raw, y_train_f, is_rf = TRUE)
cat(sprintf("RF best threshold: %.2f (CV F1=%.4f)\n", res_rf_t$threshold, res_rf_t$f1))

y_prob  <- predict(rf_base, X_test_raw, type = "prob")[, "Good"]
y_pred  <- factor(ifelse(y_prob >= res_rf_t$threshold, "Good", "Bad"), levels = c("Bad", "Good"))
r       <- evaluate(sprintf("RF + Threshold(%.2f)", res_rf_t$threshold), y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("RF + Threshold       | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# LR threshold
res_lr_t <- find_best_threshold(lr_base, X_train_scaled, y_train_f)
cat(sprintf("LR best threshold: %.2f\n", res_lr_t$threshold))

y_prob_lr <- predict(lr_base, X_test_scaled, type = "prob")[, "Good"]
y_pred_lr <- factor(ifelse(y_prob_lr >= res_lr_t$threshold, "Good", "Bad"), levels = c("Bad", "Good"))
r         <- evaluate(sprintf("LR + Threshold(%.2f)", res_lr_t$threshold), y_test_f, y_pred_lr, y_prob_lr)
results[[length(results) + 1]] <- r
cat(sprintf("LR + Threshold       | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# -----------------------------------------------------------------------------
# 5. TECHNIQUE 3: XGBOOST
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("TECHNIQUE 3: XGBoost\n")
cat(strrep("=", 60), "\n")

scale_pos_weight <- sum(y_train == 0) / sum(y_train == 1)

xgb_grid <- expand.grid(
  nrounds          = c(100, 200),
  max_depth        = c(3, 5),
  eta              = c(0.05, 0.1),
  gamma            = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample        = 0.8
)

set.seed(42)
gs_xgb <- train(
  x         = as.matrix(X_train_scaled),
  y         = y_train_f,
  method    = "xgbTree",
  trControl = ctrl,
  tuneGrid  = xgb_grid,
  metric    = "F",
  verbosity = 0
)

cat(sprintf("Best XGB params: nrounds=%d, max_depth=%d, eta=%.2f\n",
            gs_xgb$bestTune$nrounds, gs_xgb$bestTune$max_depth, gs_xgb$bestTune$eta))
cat(sprintf("Best CV F1: %.4f\n", max(gs_xgb$results$F, na.rm = TRUE)))

y_pred  <- predict(gs_xgb, as.matrix(X_test_scaled))
y_prob  <- predict(gs_xgb, as.matrix(X_test_scaled), type = "prob")[, "Good"]
r       <- evaluate("XGBoost", y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("XGBoost              | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# XGBoost + SMOTE
smote_mat <- cbind(X_train_smote, Risk = y_train_smote)
set.seed(42)
xgb_smote <- train(
  x         = as.matrix(X_train_smote),
  y         = y_train_smote,
  method    = "xgbTree",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid  = gs_xgb$bestTune,
  verbosity = 0
)
y_pred <- predict(xgb_smote, as.matrix(X_test_scaled))
y_prob <- predict(xgb_smote, as.matrix(X_test_scaled), type = "prob")[, "Good"]
r      <- evaluate("XGBoost + SMOTE", y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("XGBoost + SMOTE      | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# XGBoost + threshold
res_xgb_t <- find_best_threshold(gs_xgb, as.matrix(X_train_scaled), y_train_f)
y_prob_xgb <- predict(gs_xgb, as.matrix(X_test_scaled), type = "prob")[, "Good"]
y_pred_xgb <- factor(ifelse(y_prob_xgb >= res_xgb_t$threshold, "Good", "Bad"), levels = c("Bad", "Good"))
r          <- evaluate(sprintf("XGBoost + Threshold(%.2f)", res_xgb_t$threshold),
                        y_test_f, y_pred_xgb, y_prob_xgb)
results[[length(results) + 1]] <- r
cat(sprintf("XGBoost + Threshold  | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# -----------------------------------------------------------------------------
# 6. TECHNIQUE 4: GRADIENT BOOSTING (GBM)
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("TECHNIQUE 4: Gradient Boosting\n")
cat(strrep("=", 60), "\n")

gbm_grid <- expand.grid(
  n.trees           = c(100, 200),
  interaction.depth = c(3, 5),
  shrinkage         = c(0.05, 0.1),
  n.minobsinnode    = 10
)

set.seed(42)
gs_gb <- train(
  x         = X_train_scaled,
  y         = y_train_f,
  method    = "gbm",
  trControl = ctrl,
  tuneGrid  = gbm_grid,
  metric    = "F",
  verbose   = FALSE
)

cat(sprintf("Best GB params: n.trees=%d, depth=%d, shrinkage=%.2f\n",
            gs_gb$bestTune$n.trees, gs_gb$bestTune$interaction.depth,
            gs_gb$bestTune$shrinkage))

y_pred <- predict(gs_gb, X_test_scaled)
y_prob <- predict(gs_gb, X_test_scaled, type = "prob")[, "Good"]
r      <- evaluate("Gradient Boosting", y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("Gradient Boosting    | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# GBM + SMOTE
set.seed(42)
gb_smote <- train(
  x         = X_train_smote,
  y         = y_train_smote,
  method    = "gbm",
  trControl = trainControl(method = "none", classProbs = TRUE),
  tuneGrid  = gs_gb$bestTune,
  verbose   = FALSE
)
y_pred <- predict(gb_smote, X_test_scaled)
y_prob <- predict(gb_smote, X_test_scaled, type = "prob")[, "Good"]
r      <- evaluate("GradBoost + SMOTE", y_test_f, y_pred, y_prob)
results[[length(results) + 1]] <- r
cat(sprintf("GradBoost + SMOTE    | Acc=%.4f F1=%.4f AUC=%.4f\n", r$Accuracy, r$F1, r$ROC_AUC))

# -----------------------------------------------------------------------------
# 7. RESULTS COMPARISON
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("FULL RESULTS COMPARISON\n")
cat(strrep("=", 60), "\n")

results_df <- do.call(rbind, lapply(results, function(r) as.data.frame(r)))
rownames(results_df) <- results_df$Model
results_df$Model <- NULL

print(results_df)
write.csv(results_df, file.path(OUTPUT_DIR, "improved_results.csv"))

best_model_name <- rownames(results_df)[which.max(results_df$F1)]
cat(sprintf("\n>>> BEST MODEL: %s\n", best_model_name))
print(results_df[best_model_name, ])

# -----------------------------------------------------------------------------
# 8. VISUALIZATION
# -----------------------------------------------------------------------------

results_plot <- cbind(Model = rownames(results_df), results_df)
results_plot$is_baseline <- grepl("Baseline", results_plot$Model)
results_plot$Model <- factor(results_plot$Model, levels = rev(results_plot$Model))

baseline_rf_f1  <- results_df["Baseline RF", "F1"]
baseline_rf_auc <- results_df["Baseline RF", "ROC_AUC"]

p1 <- ggplot(results_plot, aes(x = F1, y = Model, fill = is_baseline)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.85) +
  geom_vline(xintercept = baseline_rf_f1, color = "red", linetype = "dashed",
             linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.4f", F1)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("TRUE" = "lightgray", "FALSE" = "steelblue"), guide = "none") +
  xlim(0.7, 1.02) +
  labs(title = "F1 Score Comparison — All Techniques", x = "F1 Score") +
  theme_minimal()

p2 <- ggplot(results_plot, aes(x = ROC_AUC, y = Model, fill = is_baseline)) +
  geom_bar(stat = "identity", color = "black", alpha = 0.85) +
  geom_vline(xintercept = baseline_rf_auc, color = "red", linetype = "dashed",
             linewidth = 1.2) +
  geom_text(aes(label = sprintf("%.4f", ROC_AUC)), hjust = -0.1, size = 3) +
  scale_fill_manual(values = c("TRUE" = "lightgray", "FALSE" = "forestgreen"), guide = "none") +
  xlim(0.65, 1.05) +
  labs(title = "ROC-AUC Comparison — All Techniques", x = "ROC-AUC") +
  theme_minimal()

library(gridExtra)
p_all <- grid.arrange(p1, p2, ncol = 2,
                      top = "Optimization Techniques: Improvement Over Baseline")
ggsave(file.path(OUTPUT_DIR, "improved_metrics_comparison.png"), p_all,
       dpi = 150, width = 16, height = 6)
cat(sprintf("\nSaved: %s/improved_metrics_comparison.png\n", OUTPUT_DIR))

cat("\nDone! Check outputs/improved_results.csv for full table.\n")
