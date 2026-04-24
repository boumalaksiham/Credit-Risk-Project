# =============================================================================
# WEEK 6: Fairness and Interpretability Analysis
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(randomForest)
library(ggplot2)
library(gridExtra)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD DATA AND MODELS
# -----------------------------------------------------------------------------

X_test_scaled <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
X_test_raw    <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
y_test        <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk

y_test_f <- factor(y_test, levels = c(0, 1), labels = c("Bad", "Good"))

lr_model  <- readRDS(file.path(OUTPUT_DIR, "logistic_regression_model.rds"))
svm_model <- readRDS(file.path(OUTPUT_DIR, "svm_model.rds"))
rf_model  <- readRDS(file.path(OUTPUT_DIR, "random_forest_model.rds"))

# -----------------------------------------------------------------------------
# 2. FAIRNESS HELPER FUNCTIONS
# -----------------------------------------------------------------------------

compute_group_metrics <- function(y_true, y_pred, group_name) {
  cm   <- confusionMatrix(factor(y_pred, levels = c("Bad", "Good")),
                          factor(y_true, levels = c("Bad", "Good")),
                          positive = "Good")
  tbl  <- cm$table
  tn   <- tbl["Bad",  "Bad"]
  fp   <- tbl["Good", "Bad"]
  fn   <- tbl["Bad",  "Good"]
  tp   <- tbl["Good", "Good"]
  
  data.frame(
    Group     = group_name,
    N         = length(y_true),
    Accuracy  = cm$overall["Accuracy"],
    Precision = ifelse((tp + fp) > 0, tp / (tp + fp), 0),
    Recall    = ifelse((tp + fn) > 0, tp / (tp + fn), 0),
    F1        = cm$byClass["F1"],
    FPR       = ifelse((fp + tn) > 0, fp / (fp + tn), 0),
    FNR       = ifelse((fn + tp) > 0, fn / (fn + tp), 0),
    row.names = NULL
  )
}

fairness_analysis <- function(model, X_test, y_test_f, group_col, group_defs, model_name,
                              mask_df = NULL) {
  cat("\n", strrep("=", 60), "\n")
  cat(sprintf("FAIRNESS ANALYSIS: %s | Group: %s\n", model_name, group_col))
  cat(strrep("=", 60), "\n")
  
  df_mask  <- if (!is.null(mask_df)) mask_df else X_test
  y_pred   <- predict(model, X_test)
  
  rows <- lapply(names(group_defs), function(label) {
    mask <- group_defs[[label]](df_mask[[group_col]])
    if (sum(mask) == 0) { message("  Warning: no samples for group '", label, "'"); return(NULL) }
    
    row <- compute_group_metrics(
      as.character(y_test_f[mask]),
      as.character(y_pred[mask]),
      label
    )
    cat(sprintf("  %-25s | N=%4d | Acc=%.3f | Prec=%.3f | Rec=%.3f | F1=%.3f | FPR=%.3f | FNR=%.3f\n",
                label, row$N, row$Accuracy, row$Precision, row$Recall, row$F1, row$FPR, row$FNR))
    row
  })
  do.call(rbind, rows[!sapply(rows, is.null)])
}

plot_fairness_bars <- function(df, metric_cols, title, filename) {
  df_long <- df %>%
    select(Group, all_of(metric_cols)) %>%
    pivot_longer(-Group, names_to = "Metric", values_to = "Score")
  
  p <- ggplot(df_long, aes(x = Metric, y = Score, fill = Group)) +
    geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
    scale_fill_brewer(palette = "Set1") +
    ylim(0, 1.1) +
    labs(title = title, y = "Score") +
    theme_minimal()
  ggsave(file.path(OUTPUT_DIR, filename), p, dpi = 150, width = 10, height = 5)
  cat(sprintf("Saved: %s/%s\n", OUTPUT_DIR, filename))
}

# -----------------------------------------------------------------------------
# 3. GENDER FAIRNESS ANALYSIS
# -----------------------------------------------------------------------------

gender_groups <- list(
  "Male"   = function(col) col == 1,
  "Female" = function(col) col == 0
)

lr_gender  <- fairness_analysis(lr_model,  X_test_scaled, y_test_f, "Sex", gender_groups, "Logistic Regression")
svm_gender <- fairness_analysis(svm_model, X_test_scaled, y_test_f, "Sex", gender_groups, "SVM")
rf_gender  <- fairness_analysis(rf_model,  X_test_raw,    y_test_f, "Sex", gender_groups, "Random Forest")

plot_fairness_bars(rf_gender,
                   c("Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"),
                   "Fairness by Gender - Random Forest",
                   "fairness_gender_rf.png")

plot_fairness_bars(lr_gender,
                   c("Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"),
                   "Fairness by Gender - Logistic Regression",
                   "fairness_gender_lr.png")

# -----------------------------------------------------------------------------
# 4. AGE FAIRNESS ANALYSIS
# -----------------------------------------------------------------------------

age_group_col <- cut(
  X_test_raw$Age,
  breaks = c(0, 25, 35, 50, Inf),
  labels = c("Young (<=25)", "Adult (26-35)", "Middle-aged (36-50)", "Senior (51+)"),
  right  = TRUE
)

X_test_scaled_age <- X_test_scaled
X_test_raw_age    <- X_test_raw
X_test_scaled_age$AgeGroup <- age_group_col
X_test_raw_age$AgeGroup    <- age_group_col

age_group_defs <- list(
  "Young (<=25)"        = function(col) col == "Young (<=25)",
  "Adult (26-35)"       = function(col) col == "Adult (26-35)",
  "Middle-aged (36-50)" = function(col) col == "Middle-aged (36-50)",
  "Senior (51+)"        = function(col) col == "Senior (51+)"
)

lr_age  <- fairness_analysis(lr_model,  X_test_scaled, y_test_f, "AgeGroup", age_group_defs,
                              "Logistic Regression", mask_df = X_test_scaled_age)
svm_age <- fairness_analysis(svm_model, X_test_scaled, y_test_f, "AgeGroup", age_group_defs,
                              "SVM", mask_df = X_test_scaled_age)
rf_age  <- fairness_analysis(rf_model,  X_test_raw,    y_test_f, "AgeGroup", age_group_defs,
                              "Random Forest", mask_df = X_test_raw_age)

plot_fairness_bars(rf_age,
                   c("Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"),
                   "Fairness by Age Group - Random Forest",
                   "fairness_age_rf.png")

plot_fairness_bars(lr_age,
                   c("Accuracy", "Precision", "Recall", "F1", "FPR", "FNR"),
                   "Fairness by Age Group - Logistic Regression",
                   "fairness_age_lr.png")

# -----------------------------------------------------------------------------
# 5. DEMOGRAPHIC PARITY GAP
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("DEMOGRAPHIC PARITY GAP (FPR and FNR disparity)\n")
cat(strrep("=", 60), "\n")

for (pair in list(list("LR", lr_gender), list("SVM", svm_gender), list("RF", rf_gender))) {
  nm       <- pair[[1]]
  gdf      <- pair[[2]]
  fpr_row  <- setNames(gdf$FPR, gdf$Group)
  fnr_row  <- setNames(gdf$FNR, gdf$Group)
  cat(sprintf("\n%s:\n", nm))
  cat(sprintf("  FPR: Male=%.3f, Female=%.3f, Gap=%.3f\n",
              fpr_row["Male"], fpr_row["Female"],
              abs(fpr_row["Male"] - fpr_row["Female"])))
  cat(sprintf("  FNR: Male=%.3f, Female=%.3f, Gap=%.3f\n",
              fnr_row["Male"], fnr_row["Female"],
              abs(fnr_row["Male"] - fnr_row["Female"])))
}

# -----------------------------------------------------------------------------
# 6. INTERPRETABILITY: LR COEFFICIENTS
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("INTERPRETABILITY - LOGISTIC REGRESSION COEFFICIENTS\n")
cat(strrep("=", 60), "\n")

coef_mat <- coef(lr_model$finalModel, s = lr_model$bestTune$lambda)
coef_vec <- as.vector(coef_mat)[-1]
names(coef_vec) <- rownames(coef_mat)[-1]

coef_df <- data.frame(
  Feature     = names(coef_vec),
  Coefficient = coef_vec
) %>% arrange(desc(abs(Coefficient)))

print(head(coef_df, 10))
cat("\nPositive coefficient → more likely GOOD credit\n")
cat("Negative coefficient → more likely BAD credit\n")

# -----------------------------------------------------------------------------
# 7. INTERPRETABILITY: RF FEATURE IMPORTANCE
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("INTERPRETABILITY - RANDOM FOREST FEATURE IMPORTANCE\n")
cat(strrep("=", 60), "\n")

fi_df <- data.frame(
  Feature    = rownames(importance(rf_model)),
  Importance = importance(rf_model, type = 2)[, 1]
) %>% arrange(desc(Importance))

print(head(fi_df, 10))

# Comparative plot: RF importance vs LR |coefficient|
top_features    <- head(fi_df$Feature, 10)
shared_features <- intersect(top_features, coef_df$Feature)

comparison_df <- data.frame(
  Feature     = shared_features,
  RF          = fi_df$Importance[match(shared_features, fi_df$Feature)],
  LR_abs_norm = {
    lv <- abs(coef_df$Coefficient[match(shared_features, coef_df$Feature)])
    lv / max(lv)
  }
) %>%
  pivot_longer(-Feature, names_to = "Type", values_to = "Score")

p <- ggplot(comparison_df, aes(x = reorder(Feature, Score), y = Score, fill = Type)) +
  geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.85) +
  coord_flip() +
  scale_fill_manual(values = c("RF" = "forestgreen", "LR_abs_norm" = "steelblue"),
                    labels = c("RF Importance", "LR |Coef| (norm)")) +
  labs(title = "Feature Importance: Random Forest vs Logistic Regression",
       x = "Feature", y = "Score", fill = "") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "interpretability_comparison.png"), p, dpi = 150, width = 9, height = 5)
cat(sprintf("Saved: %s/interpretability_comparison.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 8. SAVE ALL FAIRNESS TABLES
# -----------------------------------------------------------------------------

write.csv(lr_gender,  file.path(OUTPUT_DIR, "fairness_gender_lr.csv"),  row.names = FALSE)
write.csv(svm_gender, file.path(OUTPUT_DIR, "fairness_gender_svm.csv"), row.names = FALSE)
write.csv(rf_gender,  file.path(OUTPUT_DIR, "fairness_gender_rf.csv"),  row.names = FALSE)
write.csv(lr_age,     file.path(OUTPUT_DIR, "fairness_age_lr.csv"),     row.names = FALSE)
write.csv(rf_age,     file.path(OUTPUT_DIR, "fairness_age_rf.csv"),     row.names = FALSE)

cat("\nAll fairness tables saved.\n")
cat("Week 6 complete!\n")
