# Select using training CV, calibrate on separate rows, then evaluate one candidate.
library(caret)
library(pROC)
source("evaluation_helpers.R")
OUTPUT_DIR <- "outputs"
X <- read.csv(file.path(OUTPUT_DIR, "X_train_raw.csv"))
y <- factor(read.csv(file.path(OUTPUT_DIR, "y_train.csv"))$Risk,
            levels = c(0, 1), labels = c("Bad", "Good"))
set.seed(42)
fit_rows <- as.vector(createDataPartition(y, p = 0.75, list = FALSE))
X_fit <- X[fit_rows, , drop = FALSE]
y_fit <- y[fit_rows]
X_cal <- X[-fit_rows, , drop = FALSE]
y_cal <- y[-fit_rows]
if (any(table(y_fit) < 5) || any(table(y_cal) == 0)) {
  stop("Both classes need sufficient training and calibration observations.")
}
folds <- createFolds(y_fit, k = 5, returnTrain = TRUE)
specifications <- list(
  LR = list(method = "glmnet", grid = expand.grid(alpha = 0,
            lambda = c(1, 0.1, 0.01, 0.001)), preprocess = c("center", "scale")),
  RF = list(method = "rf", grid = data.frame(mtry = unique(pmin(c(2, 4, 6), ncol(X)))),
            preprocess = NULL)
)
candidates <- list()
rows <- list()
for (name in names(specifications)) {
  spec <- specifications[[name]]
  for (sampling in c("none", "up")) {
    # caret repeats random oversampling inside each training fold. Avoid
    # interpolating synthetic values across one-hot categorical columns.
    control <- trainControl(method = "cv", number = 5, index = folds,
                            classProbs = TRUE, summaryFunction = credit_summary,
                            sampling = if (sampling == "up") "up" else NULL,
                            savePredictions = "final", allowParallel = FALSE)
    set.seed(42)
    fit <- train(x = X_fit, y = y_fit, method = spec$method,
                 tuneGrid = spec$grid, preProcess = spec$preprocess,
                 metric = "F", trControl = control)
    id <- paste(name, sampling, sep = "_")
    candidates[[id]] <- fit
    rows[[id]] <- data.frame(Model = id, Training_CV_F1 = max(fit$results$F, na.rm = TRUE))
  }
}
comparison <- do.call(rbind, rows)
rownames(comparison) <- NULL
winner <- comparison$Model[which.max(comparison$Training_CV_F1)]
selected <- candidates[[winner]]
calibration_prob <- predict(selected, X_cal, type = "prob")[, "Good"]
threshold <- choose_credit_threshold(calibration_prob, y_cal)
# Freeze the selected model and threshold before loading holdout labels.
saveRDS(selected, file.path(OUTPUT_DIR, "optimized_model.rds"))
write.csv(comparison, file.path(OUTPUT_DIR, "optimization_training_cv.csv"), row.names = FALSE)
write.csv(data.frame(row_index = seq_len(nrow(X)),
                    partition = ifelse(seq_len(nrow(X)) %in% fit_rows, "fit", "calibration")),
          file.path(OUTPUT_DIR, "optimization_split_manifest.csv"), row.names = FALSE)
X_test <- read.csv(file.path(OUTPUT_DIR, "X_test_raw.csv"))
y_test <- factor(read.csv(file.path(OUTPUT_DIR, "y_test.csv"))$Risk,
                 levels = c(0, 1), labels = c("Bad", "Good"))
prob <- predict(selected, X_test, type = "prob")[, "Good"]
pred <- factor(ifelse(prob >= threshold, "Good", "Bad"), levels = c("Bad", "Good"))
cm <- confusionMatrix(pred, y_test, positive = "Good")
auc <- as.numeric(pROC::auc(pROC::roc(as.numeric(y_test == "Good"), prob,
                        levels = c(0, 1), direction = "<", quiet = TRUE)))
result <- data.frame(Model = winner, Threshold = threshold,
                     Accuracy = unname(cm$overall["Accuracy"]),
                     Precision_Good = unname(cm$byClass["Precision"]),
                     Recall_Good = unname(cm$byClass["Recall"]),
                     Recall_Bad = unname(cm$byClass["Specificity"]),
                     F1_Good = unname(cm$byClass["F1"]), ROC_AUC = auc,
                     Test_N = length(y_test))
write.csv(result, file.path(OUTPUT_DIR, "improved_results.csv"), row.names = FALSE)
write.csv(data.frame(actual = y_test, predicted = pred, probability_good = prob),
          file.path(OUTPUT_DIR, "optimization_test_predictions.csv"), row.names = FALSE)
capture.output(sessionInfo(), file = file.path(OUTPUT_DIR, "optimization_session_info.txt"))
print(result)
cat("Training CV selects the candidate; calibration selects its threshold.\n")
cat("The holdout is shared with weekly analyses, not a new external cohort.\n")
