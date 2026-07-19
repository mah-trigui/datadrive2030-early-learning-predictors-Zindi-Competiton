# ==============================================================================
# CLEAN SCRIPT 13: TIDYMODELS STACKING ENSEMBLE
# ==============================================================================
# This script creates stacked ensembles using tidymodels/stacks framework
# ==============================================================================

library(tidyverse)
library(tidymodels)
library(stacks)
library(Metrics)

# Load data splits
train_5 <- readRDS("train_age5.rds")
freeze_5 <- readRDS("freeze_age5.rds")
test_5 <- readRDS("test_age5.rds")

message("Training Tidymodels Stacking Ensemble...")

# ==============================================================================
# 1. TIDYMODELS STACK - AGE GROUP 5
# ==============================================================================

message("\n=== Tidymodels Stacking: Age Group 5 ===")

# Create train/test split
set.seed(1618)
data_split_5 <- initial_split(train_5, prop = 0.8)
train_data_5 <- training(data_split_5)
test_data_5 <- testing(data_split_5)

# Create cross-validation folds
set.seed(1618)
folds_5 <- vfold_cv(train_data_5, v = 5)

# Create recipe
recipe_5 <- recipe(target ~ ., data = train_data_5)

# Define metric
metric <- metric_set(rmse)
ctrl_grid <- control_stack_grid()
ctrl_res <- control_stack_resamples()

# ------------------------------------------------------------------------------
# Model 1: KNN
# ------------------------------------------------------------------------------

message("Training KNN model...")

knn_spec <- nearest_neighbor(
    mode = "regression",
    neighbors = tune("k")
) %>%
    set_engine("kknn")

knn_rec <- recipe_5 %>%
    step_zv(all_predictors()) %>%
    step_normalize(all_numeric_predictors())

knn_wflow <- workflow() %>%
    add_model(knn_spec) %>%
    add_recipe(knn_rec)

set.seed(1618)
knn_res <- tune_grid(
    knn_wflow,
    resamples = folds_5,
    metrics = metric,
    grid = 4,
    control = ctrl_grid
)

# ------------------------------------------------------------------------------
# Model 2: Linear Regression
# ------------------------------------------------------------------------------

message("Training Linear Regression model...")

lin_reg_spec <- linear_reg() %>%
    set_engine("lm")

lin_reg_rec <- recipe_5 %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors())

lin_reg_wflow <- workflow() %>%
    add_model(lin_reg_spec) %>%
    add_recipe(lin_reg_rec)

set.seed(1618)
lin_reg_res <- fit_resamples(
    lin_reg_wflow,
    resamples = folds_5,
    metrics = metric,
    control = ctrl_res
)

# ------------------------------------------------------------------------------
# Model 3: SVM
# ------------------------------------------------------------------------------

message("Training SVM model...")

svm_spec <- svm_rbf(
    cost = tune("cost"),
    rbf_sigma = tune("sigma")
) %>%
    set_engine("kernlab") %>%
    set_mode("regression")

svm_rec <- recipe_5 %>%
    step_dummy(all_nominal_predictors()) %>%
    step_zv(all_predictors()) %>%
    step_impute_mean(all_numeric_predictors()) %>%
    step_corr(all_predictors()) %>%
    step_normalize(all_numeric_predictors())

svm_wflow <- workflow() %>%
    add_model(svm_spec) %>%
    add_recipe(svm_rec)

set.seed(1618)
svm_res <- tune_grid(
    svm_wflow,
    resamples = folds_5,
    grid = 4,
    metrics = metric,
    control = ctrl_grid
)

# ------------------------------------------------------------------------------
# Create Stack
# ------------------------------------------------------------------------------

message("Creating stacked ensemble...")

data_st_5 <- stacks() %>%
    add_candidates(knn_res) %>%
    add_candidates(lin_reg_res) %>%
    add_candidates(svm_res)

# Blend predictions (find optimal weights)
model_st_5 <- data_st_5 %>%
    blend_predictions()

# Fit final ensemble
model_st_5 <- model_st_5 %>%
    fit_members()

# Display model weights
message("\nModel Weights:")
print(autoplot(model_st_5, type = "weights"))

# ------------------------------------------------------------------------------
# Evaluate on Freeze Set
# ------------------------------------------------------------------------------

pred_tidy_5 <- predict(model_st_5, freeze_5)
rmse_5 <- rmse(freeze_5$target, pred_tidy_5$.pred)
message(sprintf("Freeze RMSE: %.4f", rmse_5))

# Predict on test set
y_tidy_5 <- predict(model_st_5, test_5)

# ==============================================================================
# 2. SAVE PREDICTIONS AND MODELS
# ==============================================================================

# Save predictions (freeze set)
saveRDS(data.frame(tidy = pred_tidy_5$.pred), "pred_tidymodels_age5.rds")

# Save predictions (test set)
saveRDS(data.frame(tidy = y_tidy_5$.pred), "test_pred_tidymodels_age5.rds")

# Save model
saveRDS(model_st_5, "model_tidymodels_stack_age5.rds")

message("\n✓ Tidymodels stacking complete!")
message(sprintf("  Age Group 5 - Freeze RMSE: %.4f", rmse_5))
