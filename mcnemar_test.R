# =============================================================================
# McNemar's Test: Statistical Comparison of Models
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)

OUTPUT_DIR <- "outputs"

X_test_scaled <- read.csv(file.path(OUTPUT_DIR, "X_test_scaled.csv"))
X_test_raw    <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
y_test        <- read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk
y_test_f      <- factor(y_test, levels = c(0, 1), labels = c("Bad", "Good"))

lr  <- readRDS(file.path(OUTPUT_DIR, "logistic_regression_model.rds"))
svm <- readRDS(file.path(OUTPUT_DIR, "svm_model.rds"))
rf  <- readRDS(file.path(OUTPUT_DIR, "random_forest_model.rds"))

preds <- list(
  LR  = predict(lr,  X_test_scaled),
  SVM = predict(svm, X_test_scaled),
  RF  = predict(rf,  X_test_raw)
)

# McNemar's test for each pair
pairs <- list(c("LR", "SVM"), c("LR", "RF"), c("SVM", "RF"))

for (pair in pairs) {
  a <- pair[1]; b <- pair[2]
  
  a_wrong_b_right <- sum(preds[[a]] != y_test_f & preds[[b]] == y_test_f)
  a_right_b_wrong <- sum(preds[[a]] == y_test_f & preds[[b]] != y_test_f)
  
  # McNemar table: [[both_wrong, a_wrong_b_right], [a_right_b_wrong, both_right]]
  tbl    <- matrix(c(0, a_wrong_b_right, a_right_b_wrong, 0), nrow = 2)
  result <- mcnemar.test(tbl, correct = TRUE)
  
  cat(sprintf("%s vs %s: stat=%.2f, p=%.3f\n", a, b, result$statistic, result$p.value))
}
