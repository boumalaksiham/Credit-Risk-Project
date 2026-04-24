# =============================================================================
# WEEK 7: Final Evaluation and Results Consolidation
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(pROC)
library(ggplot2)
library(gridExtra)
library(grid)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD EVERYTHING
# -----------------------------------------------------------------------------

X_test_scaled <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
X_test_raw    <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
y_test        <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk
y_test_f      <- factor(y_test, levels = c(0, 1), labels = c("Bad", "Good"))

lr_model  <- readRDS(file.path(OUTPUT_DIR, "logistic_regression_model.rds"))
svm_model <- readRDS(file.path(OUTPUT_DIR, "svm_model.rds"))
rf_model  <- readRDS(file.path(OUTPUT_DIR, "random_forest_model.rds"))

lr_results  <- readRDS(file.path(OUTPUT_DIR, "lr_results.rds"))
svm_results <- readRDS(file.path(OUTPUT_DIR, "svm_results.rds"))
rf_results  <- readRDS(file.path(OUTPUT_DIR, "rf_results.rds"))

model_info <- list(
  list(model = lr_model,  name = "Logistic Regression", X_te = X_test_scaled,
       results = lr_results,  color = "steelblue"),
  list(model = svm_model, name = "SVM",                 X_te = X_test_scaled,
       results = svm_results, color = "mediumpurple"),
  list(model = rf_model,  name = "Random Forest",       X_te = X_test_raw,
       results = rf_results,  color = "forestgreen")
)

# -----------------------------------------------------------------------------
# 2. FINAL METRICS TABLE
# -----------------------------------------------------------------------------

metrics_keys <- c("accuracy", "precision", "recall", "f1", "roc_auc")

final_rows <- lapply(model_info, function(info) {
  row <- data.frame(Model = info$name)
  for (k in metrics_keys) row[[toupper(k)]] <- round(as.numeric(info$results[[k]]), 4)
  row
})
final_table <- do.call(rbind, final_rows)
rownames(final_table) <- final_table$Model
final_table$Model <- NULL

cat(strrep("=", 70), "\n")
cat("FINAL MODEL PERFORMANCE TABLE\n")
cat(strrep("=", 70), "\n")
print(final_table)

write.csv(final_table, file.path(OUTPUT_DIR, "final_results_table.csv"))
cat(sprintf("\nSaved: %s/final_results_table.csv\n", OUTPUT_DIR))

best_model_name <- rownames(final_table)[which.max(final_table$F1)]
cat(sprintf("\n>>> Best model by F1-score: %s (%.4f)\n",
            best_model_name, max(final_table$F1)))

# -----------------------------------------------------------------------------
# 3. FINAL COMPREHENSIVE VISUALIZATION
# -----------------------------------------------------------------------------

# -- Confusion matrices (row 1)
make_cm_plot <- function(model, X_te, label, fill_color) {
  preds <- predict(model, X_te)
  cm    <- confusionMatrix(preds, y_test_f, positive = "Good")
  df    <- as.data.frame(cm$table)
  ggplot(df, aes(x = Reference, y = Prediction, fill = Freq)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Freq), size = 5) +
    scale_fill_gradient(low = "white", high = fill_color) +
    labs(title = sprintf("Confusion Matrix\n%s", label),
         x = "Actual", y = "Predicted") +
    theme_minimal() + theme(legend.position = "none", plot.title = element_text(size = 9))
}

p_cm1 <- make_cm_plot(lr_model,  X_test_scaled, "Logistic Regression", "steelblue")
p_cm2 <- make_cm_plot(svm_model, X_test_scaled, "SVM",                 "mediumpurple")
p_cm3 <- make_cm_plot(rf_model,  X_test_raw,    "Random Forest",       "forestgreen")

# -- ROC curves (row 2, col 1-2)
roc_list <- list(
  roc(as.numeric(y_test_f == "Good"),
      predict(lr_model,  X_test_scaled, type = "prob")[, "Good"], quiet = TRUE),
  roc(as.numeric(y_test_f == "Good"),
      predict(svm_model, X_test_scaled, type = "prob")[, "Good"], quiet = TRUE),
  roc(as.numeric(y_test_f == "Good"),
      predict(rf_model,  X_test_raw,    type = "prob")[, "Good"], quiet = TRUE)
)
colors <- c("steelblue", "mediumpurple", "forestgreen")
names_roc <- c("LR", "SVM", "RF")

roc_df <- do.call(rbind, lapply(seq_along(roc_list), function(i) {
  data.frame(FPR = 1 - roc_list[[i]]$specificities,
             TPR = roc_list[[i]]$sensitivities,
             Model = sprintf("%s (AUC=%.3f)", names_roc[i], auc(roc_list[[i]])))
}))
p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
  geom_line(linewidth = 1) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  scale_color_manual(values = colors) +
  labs(title = "ROC Curves - All Models",
       x = "False Positive Rate", y = "True Positive Rate") +
  theme_minimal() +
  theme(legend.position = c(0.65, 0.25), legend.text = element_text(size = 7))

# -- Performance bar (row 2, col 3)
bar_df <- do.call(rbind, lapply(model_info, function(info) {
  data.frame(
    Model  = info$name,
    Metric = c("ACC", "PREC", "REC", "F1", "AUC"),
    Score  = c(info$results$accuracy, info$results$precision,
               info$results$recall,   info$results$f1,
               info$results$roc_auc)
  )
}))
p_bar <- ggplot(bar_df, aes(x = Metric, y = Score, fill = Model)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
  scale_fill_manual(values = c(
    "Logistic Regression" = "steelblue",
    "SVM"                 = "mediumpurple",
    "Random Forest"       = "forestgreen"
  )) +
  ylim(0, 1.1) +
  labs(title = "Performance Comparison", y = "Score") +
  theme_minimal() +
  theme(legend.text = element_text(size = 7), legend.key.size = unit(0.4, "cm"))

# -- LR coefficients (row 3, col 1)
coef_mat <- coef(lr_model$finalModel, s = lr_model$bestTune$lambda)
coef_vec <- as.vector(coef_mat)[-1]
names(coef_vec) <- rownames(coef_mat)[-1]
coef_df <- data.frame(Feature = names(coef_vec), Coef = coef_vec) %>%
  arrange(desc(abs(Coef))) %>% head(10)

p_lr <- ggplot(coef_df, aes(x = reorder(Feature, Coef), y = Coef, fill = Coef > 0)) +
  geom_bar(stat = "identity", color = "black") +
  coord_flip() +
  scale_fill_manual(values = c("TRUE" = "steelblue", "FALSE" = "tomato"), guide = "none") +
  geom_hline(yintercept = 0, linewidth = 0.5) +
  labs(title = "LR Coefficients (Top 10)", x = "", y = "Coefficient") +
  theme_minimal() + theme(axis.text.y = element_text(size = 7), plot.title = element_text(size = 9))

# -- RF feature importance (row 3, col 2)
fi_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  Importance = importance(rf_model, type = 2)[, 1]
) %>% arrange(desc(Importance)) %>% head(10)

p_rf <- ggplot(fi_df, aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_bar(stat = "identity", fill = "forestgreen", color = "black", alpha = 0.8) +
  coord_flip() +
  labs(title = "RF Feature Importance (Top 10)", x = "", y = "Mean Decrease Gini") +
  theme_minimal() + theme(axis.text.y = element_text(size = 7), plot.title = element_text(size = 9))

# -- Summary text (row 3, col 3)
summary_text <- sprintf(
  "SUMMARY\n%s\nDataset: German Credit Data\nSamples: 1,001 | Features: 9\n\nBest Model: %s\n  F1:      %.4f\n  ROC-AUC: %.4f\n  Recall:  %.4f\n\nAll Models Evaluated On:\n  Accuracy, Precision\n  Recall, F1, ROC-AUC\n  Fairness (gender/age)\n  Interpretability\n\nKey Finding:\n  Credit amount & duration\n  are top predictive features.",
  strrep("-", 30),
  best_model_name,
  final_table[best_model_name, "F1"],
  final_table[best_model_name, "ROC_AUC"],
  final_table[best_model_name, "RECALL"]
)

p_txt <- ggplot() +
  annotate("text", x = 0.05, y = 0.95, label = summary_text,
           hjust = 0, vjust = 1, size = 3, family = "mono") +
  annotate("rect", xmin = 0, xmax = 1, ymin = 0, ymax = 1,
           fill = "lightyellow", alpha = 0.5, color = "gray") +
  xlim(0, 1) + ylim(0, 1) + theme_void()

# Compose dashboard
top_row    <- grid.arrange(p_cm1, p_cm2, p_cm3, ncol = 3)
mid_row    <- grid.arrange(p_roc, p_bar, ncol = 2)
bottom_row <- grid.arrange(p_lr, p_rf, p_txt, ncol = 3)

dashboard <- grid.arrange(
  top_row, mid_row, bottom_row, nrow = 3,
  top = textGrob("Credit Risk Classification — Final Results Dashboard",
                 gp = gpar(fontsize = 14, fontface = "bold"))
)

ggsave(file.path(OUTPUT_DIR, "final_results_dashboard.png"), dashboard,
       dpi = 150, width = 18, height = 14)
cat(sprintf("Saved: %s/final_results_dashboard.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 4. PRINT FULL SUMMARY
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 70), "\n")
cat("PROJECT COMPLETE - SUMMARY OF OUTPUTS\n")
cat(strrep("=", 70), "\n")
for (f in sort(list.files(OUTPUT_DIR))) {
  cat(sprintf("  %s/%s\n", OUTPUT_DIR, f))
}
cat("\n>>> Week 7 complete. Project pipeline finished!\n")
