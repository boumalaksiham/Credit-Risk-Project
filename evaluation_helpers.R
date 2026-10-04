# Shared metric convention: Good is positive for F1; Bad recall is also reported.
credit_summary <- function(data, lev = NULL, model = NULL) {
  cm <- caret::confusionMatrix(data$pred, data$obs, positive = "Good")
  roc <- pROC::roc(as.numeric(data$obs == "Good"), data$Good,
                   levels = c(0, 1), direction = "<", quiet = TRUE)
  tp <- sum(data$pred == "Good" & data$obs == "Good")
  fp <- sum(data$pred == "Good" & data$obs == "Bad")
  fn <- sum(data$pred == "Bad" & data$obs == "Good")
  f1 <- if (2 * tp + fp + fn == 0) 0 else 2 * tp / (2 * tp + fp + fn)
  c(ROC = as.numeric(pROC::auc(roc)),
    F = f1,
    Sens = unname(cm$byClass["Sensitivity"]),
    Spec = unname(cm$byClass["Specificity"]))
}

fit_credit_preprocessor <- function(training, columns) {
  caret::preProcess(training[, columns, drop = FALSE], method = c("center", "scale"))
}

choose_credit_threshold <- function(probabilities, labels) {
  scores <- vapply(seq(0.2, 0.8, by = 0.01), function(t) {
    pred <- factor(ifelse(probabilities >= t, "Good", "Bad"), levels = c("Bad", "Good"))
    tp <- sum(pred == "Good" & labels == "Good")
    fp <- sum(pred == "Good" & labels == "Bad")
    fn <- sum(pred == "Bad" & labels == "Good")
    if (2 * tp + fp + fn == 0) return(0)
    2 * tp / (2 * tp + fp + fn)
  }, numeric(1))
  seq(0.2, 0.8, by = 0.01)[which.max(scores)]
}
