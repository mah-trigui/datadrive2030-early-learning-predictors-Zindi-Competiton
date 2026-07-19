# ==============================================================================
# CLEAN SCRIPT 10: LIGHTGBM MODELS
# ==============================================================================
# This script trains LightGBM models with K-fold cross-validation
# ==============================================================================

library(tidyverse)
library(lightgbm)
library(Metrics)

# Load data splits
train_5 <- readRDS("train_age5.rds")
freeze_5 <- readRDS("freeze_age5.rds")
test_5 <- readRDS("test_age5.rds")

train_6 <- readRDS("train_age6.rds")
freeze_6 <- readRDS("freeze_age6.rds")
test_6 <- readRDS("test_age6.rds")

message("Training LightGBM models...")

# ==============================================================================
# 1. LIGHTGBM WITH K-FOLD CV - AGE GROUP 5
# ==============================================================================

message("\n=== LightGBM Model: Age Group 5 (K-Fold CV) ===")

# Prepare train data
x_train_5 <- as.matrix(train_5[, -which(names(train_5) == "target")])
y_train_5 <- train_5$target

# Prepare freeze data
x_freeze_5 <- as.matrix(freeze_5[, -which(names(freeze_5) == "target")])
y_freeze_5 <- freeze_5$target

# Prepare test data
x_test_5 <- as.matrix(test_5[, -which(names(test_5) == "target")])

# Create LightGBM datasets
dtrain_5 <- lgb.Dataset(data = x_train_5, label = y_train_5, free_raw_data = FALSE)
dfreeze_5 <- lgb.Dataset.create.valid(dtrain_5, data = x_freeze_5, label = y_freeze_5)
valids_5 <- list(train = dtrain_5, freeze = dfreeze_5)

# Train model with early stopping
set.seed(1618)
params_lgb <- list(
    objective = "regression",
    metric = "rmse",
    learning_rate = 0.05,
    num_leaves = 31,
    min_data_in_leaf = 20,
    feature_fraction = 0.8,
    bagging_fraction = 0.8,
    bagging_freq = 5,
    verbose = -1
)

model_lgb_5 <- lgb.train(
    data = dtrain_5,
    params = params_lgb,
    nrounds = 5000,
    valids = valids_5,
    early_stopping_rounds = 100,
    eval_freq = 50
)

# Predict on freeze set
pred_light_5 <- predict(model_lgb_5, x_freeze_5)
rmse_5 <- rmse(y_freeze_5, pred_light_5)
message(sprintf("Freeze RMSE: %.4f", rmse_5))

# Predict on test set
y_light_5 <- predict(model_lgb_5, x_test_5)

# Feature importance
feature_imp_5 <- lgb.importance(model_lgb_5, percentage = TRUE)
message("\nTop 10 Important Features (Age Group 5):")
print(head(feature_imp_5, 10))

# ==============================================================================
# 2. LIGHTGBM WITH K-FOLD CV - AGE GROUP 6
# ==============================================================================

message("\n=== LightGBM Model: Age Group 6 (K-Fold CV) ===")

# Prepare train data
x_train_6 <- as.matrix(train_6[, -which(names(train_6) == "target")])
y_train_6 <- train_6$target

# Prepare freeze data
x_freeze_6 <- as.matrix(freeze_6[, -which(names(freeze_6) == "target")])
y_freeze_6 <- freeze_6$target

# Prepare test data
x_test_6 <- as.matrix(test_6[, -which(names(test_6) == "target")])

# Create LightGBM datasets
dtrain_6 <- lgb.Dataset(data = x_train_6, label = y_train_6, free_raw_data = FALSE)
dfreeze_6 <- lgb.Dataset.create.valid(dtrain_6, data = x_freeze_6, label = y_freeze_6)
valids_6 <- list(train = dtrain_6, freeze = dfreeze_6)

# Train model
set.seed(1618)
model_lgb_6 <- lgb.train(
    data = dtrain_6,
    params = params_lgb,
    nrounds = 5000,
    valids = valids_6,
    early_stopping_rounds = 100,
    eval_freq = 50
)

# Predict on freeze set
pred_light_6 <- predict(model_lgb_6, x_freeze_6)
rmse_6 <- rmse(y_freeze_6, pred_light_6)
message(sprintf("Freeze RMSE: %.4f", rmse_6))

# Predict on test set
y_light_6 <- predict(model_lgb_6, x_test_6)

# Feature importance
feature_imp_6 <- lgb.importance(model_lgb_6, percentage = TRUE)
message("\nTop 10 Important Features (Age Group 6):")
print(head(feature_imp_6, 10))

# ==============================================================================
# 3. SAVE PREDICTIONS AND MODELS
# ==============================================================================

# Save predictions (freeze set)
saveRDS(data.frame(light = pred_light_5), "pred_lightgbm_age5.rds")
saveRDS(data.frame(light = pred_light_6), "pred_lightgbm_age6.rds")

# Save predictions (test set)
saveRDS(data.frame(light = y_light_5), "test_pred_lightgbm_age5.rds")
saveRDS(data.frame(light = y_light_6), "test_pred_lightgbm_age6.rds")

# Save models
saveRDS(model_lgb_5, "model_lightgbm_age5.rds")
saveRDS(model_lgb_6, "model_lightgbm_age6.rds")

# Save feature importance
saveRDS(feature_imp_5, "feature_importance_lightgbm_age5.rds")
saveRDS(feature_imp_6, "feature_importance_lightgbm_age6.rds")

message("\n✓ LightGBM training complete!")
message(sprintf("  Age Group 5 - Freeze RMSE: %.4f", rmse_5))
message(sprintf("  Age Group 6 - Freeze RMSE: %.4f", rmse_6))
