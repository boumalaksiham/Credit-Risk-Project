# =============================================================================
# WEEK 1: Data Exploration and Preprocessing Pipeline
# Credit Risk Classification - Siham Boumalak
# =============================================================================

library(tidyverse)
library(caret)
library(ggplot2)
library(reshape2)

OUTPUT_DIR <- "outputs"
dir.create(OUTPUT_DIR, showWarnings = FALSE)

# -----------------------------------------------------------------------------
# 1. LOAD AND INSPECT DATASET
# -----------------------------------------------------------------------------

DATA_PATH <- "data/german_credit_data.csv"
df <- read.csv(DATA_PATH, row.names = 1, stringsAsFactors = FALSE)

cat("=" , strrep("=", 59), "\n")
cat("DATASET OVERVIEW\n")
cat(strrep("=", 60), "\n")
cat("Shape:", nrow(df), "x", ncol(df), "\n")
cat("\nColumn names:\n"); print(names(df))
cat("\nData types:\n"); print(sapply(df, class))
cat("\nFirst 5 rows:\n"); print(head(df, 5))

# -----------------------------------------------------------------------------
# 2. CLASS DISTRIBUTION
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("CLASS DISTRIBUTION (Target: Risk)\n")
cat(strrep("=", 60), "\n")

class_counts <- table(df$Risk)
class_pcts   <- prop.table(class_counts) * 100
print(class_counts)
cat(sprintf("\nGood credit: %.1f%%\n", class_pcts["good"]))
cat(sprintf("Bad credit:  %.1f%%\n", class_pcts["bad"]))

p <- ggplot(df, aes(x = Risk, fill = Risk)) +
  geom_bar(color = "black") +
  scale_fill_manual(values = c("good" = "steelblue", "bad" = "tomato")) +
  geom_text(stat = "count", aes(label = ..count..), vjust = -0.5) +
  labs(title = "Class Distribution: Credit Risk", x = "Risk", y = "Count") +
  theme_minimal() + theme(legend.position = "none")
ggsave(file.path(OUTPUT_DIR, "class_distribution.png"), p, dpi = 150, width = 5, height = 4)
cat(sprintf("Saved: %s/class_distribution.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 3. MISSING VALUES
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("MISSING VALUES\n")
cat(strrep("=", 60), "\n")

missing_counts <- colSums(is.na(df))
missing_pct    <- colMeans(is.na(df)) * 100
missing_df     <- data.frame(
  Missing_Count = missing_counts,
  Missing_Pct   = missing_pct
)
print(missing_df[missing_df$Missing_Count > 0, ])

# -----------------------------------------------------------------------------
# 4. FEATURE DISTRIBUTIONS AND VISUALIZATIONS
# -----------------------------------------------------------------------------

numerical_cols   <- c("Age", "Credit.amount", "Duration")
categorical_cols <- c("Sex", "Housing", "Saving.accounts", "Checking.account", "Purpose", "Job")

# Numerical distributions
df_long <- df %>%
  select(all_of(numerical_cols), Risk) %>%
  pivot_longer(cols = all_of(numerical_cols), names_to = "Feature", values_to = "Value")

p <- ggplot(df_long, aes(x = Value, fill = Risk)) +
  geom_histogram(bins = 30, alpha = 0.6, position = "identity", color = NA) +
  geom_density(aes(y = ..count.. * 10), alpha = 0) +
  facet_wrap(~Feature, scales = "free") +
  scale_fill_manual(values = c("good" = "steelblue", "bad" = "tomato")) +
  labs(title = "Numerical Feature Distributions by Risk") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "numerical_distributions.png"), p, dpi = 150, width = 15, height = 4)
cat(sprintf("Saved: %s/numerical_distributions.png\n", OUTPUT_DIR))

# Categorical distributions
plots_cat <- lapply(categorical_cols, function(col) {
  ct <- df %>%
    group_by(.data[[col]], Risk) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(.data[[col]]) %>%
    mutate(pct = n / sum(n) * 100)
  
  ggplot(ct, aes(x = .data[[col]], y = pct, fill = Risk)) +
    geom_bar(stat = "identity", position = "dodge", color = "black", alpha = 0.8) +
    scale_fill_manual(values = c("good" = "steelblue", "bad" = "tomato")) +
    labs(title = paste(col, "vs Risk (%)"), x = col, y = "Percentage") +
    theme_minimal() + theme(axis.text.x = element_text(angle = 30, hjust = 1))
})

library(gridExtra)
p_cat <- do.call(grid.arrange, c(plots_cat, ncol = 3))
ggsave(file.path(OUTPUT_DIR, "categorical_distributions.png"), p_cat, dpi = 150, width = 18, height = 10)
cat(sprintf("Saved: %s/categorical_distributions.png\n", OUTPUT_DIR))

# Correlation heatmap
cor_mat  <- cor(df[, numerical_cols])
cor_melt <- melt(cor_mat)
p <- ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  geom_text(aes(label = round(value, 2))) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0) +
  labs(title = "Correlation Matrix (Numerical Features)") +
  theme_minimal()
ggsave(file.path(OUTPUT_DIR, "correlation_heatmap.png"), p, dpi = 150, width = 6, height = 4)
cat(sprintf("Saved: %s/correlation_heatmap.png\n", OUTPUT_DIR))

# -----------------------------------------------------------------------------
# 5. PREPROCESSING
# -----------------------------------------------------------------------------

cat("\n", strrep("=", 60), "\n")
cat("PREPROCESSING\n")
cat(strrep("=", 60), "\n")

source("evaluation_helpers.R")
if (anyNA(df$Risk) || !all(df$Risk %in% c("good", "bad"))) {
  stop("Risk must contain only good/bad labels, with no missing target values.")
}
# Split using a factor so caret preserves class proportions.
set.seed(42)
train_idx <- as.vector(createDataPartition(factor(df$Risk), p = 0.8, list = FALSE))
df_processed <- df

# Target encoding: good=1, bad=0
df_processed$Risk <- ifelse(df_processed$Risk == "good", 1, 0)
cat("Target encoded: good=1, bad=0\n")

# Fill missing values with 'unknown'
for (col in c("Saving.accounts", "Checking.account")) {
  df_processed[[col]][is.na(df_processed[[col]])] <- "unknown"
  cat(sprintf("Missing values in '%s' filled with 'unknown'\n", col))
}

# Sex: binary encoding
df_processed$Sex <- ifelse(df_processed$Sex == "male", 1, 0)
cat("Sex encoded: male=1, female=0\n")

# Learn categorical levels on training rows only. Reserve unknown for unseen values.
ohe_cols <- c("Housing", "Saving.accounts", "Checking.account", "Purpose")
category_levels <- lapply(df_processed[train_idx, ohe_cols, drop = FALSE], function(x) {
  unique(c(sort(unique(as.character(x[!is.na(x)]))), "unknown"))
})
for (col in ohe_cols) {
  values <- as.character(df_processed[[col]])
  values[is.na(values) | !values %in% category_levels[[col]]] <- "unknown"
  df_processed[[col]] <- factor(values, levels = category_levels[[col]])
}
dummy_formula <- as.formula(paste("~", paste(ohe_cols, collapse = " + ")))
dummies <- model.matrix(dummy_formula, data = df_processed)[, -1, drop = FALSE]
df_processed <- cbind(df_processed[, !names(df_processed) %in% ohe_cols], dummies)
y <- df_processed$Risk
X <- df_processed[, names(df_processed) != "Risk"]
if (anyNA(X)) stop("Predictors still contain missing values; inspect the source data.")
X_train_raw <- X[train_idx, , drop = FALSE]
X_test_raw <- X[-train_idx, , drop = FALSE]
y_train <- y[train_idx]
y_test <- y[-train_idx]

# Export train-fitted scaled data for inspection. Models below train on raw
# predictors and let caret fit preprocessing separately inside each CV fold.
numerical_to_scale <- c("Age", "Credit.amount", "Duration", "Job")
preproc <- fit_credit_preprocessor(X_train_raw, numerical_to_scale)
X_train <- X_train_raw
X_test <- X_test_raw
X_train[, numerical_to_scale] <- predict(preproc, X_train_raw[, numerical_to_scale])
X_test[, numerical_to_scale] <- predict(preproc, X_test_raw[, numerical_to_scale])
saveRDS(category_levels, file.path(OUTPUT_DIR, "category_levels.rds"))
write.csv(data.frame(row_index = seq_len(nrow(df)),
                     partition = ifelse(seq_len(nrow(df)) %in% train_idx, "train", "test")),
          file.path(OUTPUT_DIR, "split_manifest.csv"), row.names = FALSE)

cat(sprintf("\nTrain set size: %d samples\n", nrow(X_train)))
cat(sprintf("Test set size:  %d samples\n",  nrow(X_test)))
cat("Train class distribution:\n"); print(table(y_train))
cat("Test class distribution:\n");  print(table(y_test))

# Save processed data
write.csv(X_train,     file.path(OUTPUT_DIR, "X_train_scaled.csv"), row.names = FALSE)
write.csv(X_test,      file.path(OUTPUT_DIR, "X_test_scaled.csv"),  row.names = FALSE)
write.csv(X_train_raw, file.path(OUTPUT_DIR, "X_train_raw.csv"),    row.names = FALSE)
write.csv(X_test_raw,  file.path(OUTPUT_DIR, "X_test_raw.csv"),     row.names = FALSE)
write.csv(data.frame(Risk = y_train), file.path(OUTPUT_DIR, "y_train.csv"), row.names = FALSE)
write.csv(data.frame(Risk = y_test),  file.path(OUTPUT_DIR, "y_test.csv"),  row.names = FALSE)

saveRDS(preproc, file.path(OUTPUT_DIR, "scaler.rds"))
saveRDS(names(X), file.path(OUTPUT_DIR, "feature_names.rds"))

cat(sprintf("\nAll preprocessed data saved to '%s/'\n", OUTPUT_DIR))
cat("Week 1 complete!\n")
