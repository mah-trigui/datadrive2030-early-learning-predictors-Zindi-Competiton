# ==============================================================================
# CLEAN SCRIPT 11: H2O AUTOML MODELS
# ==============================================================================
# This script trains H2O AutoML models for automated machine learning
# ==============================================================================

library(tidyverse)
library(h2o)
library(Metrics)

# Load data splits
train_5 <- readRDS("train_age5.rds")
freeze_5 <- readRDS("freeze_age5.rds")
test_5 <- readRDS("test_age5.rds")

train_6 <- readRDS("train_age6.rds")
freeze_6 <- readRDS("freeze_age6.rds")
test_6 <- readRDS("test_age6.rds")

message("Training H2O AutoML models...")

# Initialize H2O
h2o.init(nthreads = -1, max_mem_size = "8G")

# ==============================================================================
# 1. H2O AUTOML - AGE GROUP 5
# ==============================================================================

message("\n=== H2O AutoML: Age Group 5 ===")

# Create 80/20 train/validation split
set.seed(1618)
smp_size_5 <- floor(0.8 * nrow(train_5))
train_ind_5 <- sample(seq_len(nrow(train_5)), size = smp_size_5)

train_df_5 <- train_5[train_ind_5, ]
valid_df_5 <- train_5[-train_ind_5, ]

# Convert to H2O frames
train_h2o_5 <- as.h2o(train_df_5)
valid_h2o_5 <- as.h2o(valid_df_5)
freeze_h2o_5 <- as.h2o(freeze_5)
test_h2o_5 <- as.h2o(test_5)

# Identify predictors and response
y <- "target"
x <- setdiff(names(train_h2o_5), y)

# Run AutoML
set.seed(1618)
aml_5 <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o_5,
    validation_frame = valid_h2o_5,
    max_models = 20,
    max_runtime_secs = 3600, # 1 hour max
    seed = 1618,
    stopping_metric = "RMSE",
    sort_metric = "RMSE"
)

# View leaderboard
message("\nLeaderboard (Top 5 models):")
print(h2o.get_leaderboard(aml_5, extra_columns = "ALL")[1:5, ])

# Predict on freeze set
pred_h2o_5 <- as.data.frame(h2o.predict(aml_5@leader, freeze_h2o_5))
rmse_5 <- rmse(freeze_5$target, pred_h2o_5$predict)
message(sprintf("Freeze RMSE: %.4f", rmse_5))

# Predict on test set
y_h2o_5 <- as.data.frame(h2o.predict(aml_5@leader, test_h2o_5))

# ==============================================================================
# 2. H2O AUTOML - AGE GROUP 6
# ==============================================================================

message("\n=== H2O AutoML: Age Group 6 ===")

# Create 80/20 train/validation split
set.seed(1618)
smp_size_6 <- floor(0.8 * nrow(train_6))
train_ind_6 <- sample(seq_len(nrow(train_6)), size = smp_size_6)

train_df_6 <- train_6[train_ind_6, ]
valid_df_6 <- train_6[-train_ind_6, ]

# Convert to H2O frames
train_h2o_6 <- as.h2o(train_df_6)
valid_h2o_6 <- as.h2o(valid_df_6)
freeze_h2o_6 <- as.h2o(freeze_6)
test_h2o_6 <- as.h2o(test_6)

# Run AutoML
set.seed(1618)
aml_6 <- h2o.automl(
    x = x,
    y = y,
    training_frame = train_h2o_6,
    validation_frame = valid_h2o_6,
    max_models = 20,
    max_runtime_secs = 3600,
    seed = 1618,
    stopping_metric = "RMSE",
    sort_metric = "RMSE"
)

# View leaderboard
message("\nLeaderboard (Top 5 models):")
print(h2o.get_leaderboard(aml_6, extra_columns = "ALL")[1:5, ])

# Predict on freeze set
pred_h2o_6 <- as.data.frame(h2o.predict(aml_6@leader, freeze_h2o_6))
rmse_6 <- rmse(freeze_6$target, pred_h2o_6$predict)
message(sprintf("Freeze RMSE: %.4f", rmse_6))

# Predict on test set
y_h2o_6 <- as.data.frame(h2o.predict(aml_6@leader, test_h2o_6))

# ==============================================================================
# 3. SAVE PREDICTIONS AND MODELS
# ==============================================================================

# Save predictions (freeze set)
saveRDS(data.frame(h2o = pred_h2o_5$predict), "pred_h2o_age5.rds")
saveRDS(data.frame(h2o = pred_h2o_6$predict), "pred_h2o_age6.rds")

# Save predictions (test set)
saveRDS(data.frame(h2o = y_h2o_5$predict), "test_pred_h2o_age5.rds")
saveRDS(data.frame(h2o = y_h2o_6$predict), "test_pred_h2o_age6.rds")

# Save model paths
h2o.saveModel(aml_5@leader, path = ".", force = TRUE)
h2o.saveModel(aml_6@leader, path = ".", force = TRUE)

message("\n✓ H2O AutoML training complete!")
message(sprintf("  Age Group 5 - Freeze RMSE: %.4f", rmse_5))
message(sprintf("  Age Group 6 - Freeze RMSE: %.4f", rmse_6))
message("\n  Best models saved to current directory")

# Shutdown H2O cluster
h2o.shutdown(prompt = FALSE)
