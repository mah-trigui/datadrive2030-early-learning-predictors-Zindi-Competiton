# ==============================================================================
# CLEAN SCRIPT 09: CATBOOST MODELS
# ==============================================================================
# This script trains CatBoost models with:
# 1. Caret-based grid search
# 2. Random hyperparameter search
# ==============================================================================

library(tidyverse)
library(caret)
library(catboost)
library(Metrics)

# Load data splits
train_5 <- readRDS("train_age5.rds")
freeze_5 <- readRDS("freeze_age5.rds")
test_5 <- readRDS("test_age5.rds")

train_6 <- readRDS("train_age6.rds")
freeze_6 <- readRDS("freeze_age6.rds")
test_6 <- readRDS("test_age6.rds")

message("Training CatBoost models...")

# ==============================================================================
# 1. CATBOOST WITH CARET (RANDOM GRID SEARCH) - AGE GROUP 5
# ==============================================================================

message("\n=== CatBoost Model 1: Age Group 5 (Caret Grid Search) ===")

# Prepare data
y_train_5 <- train_5$target
X_train_5 <- train_5 %>% select(-target)

# Create catboost pools
train_pool_5 <- catboost.load_pool(data = X_train_5, label = y_train_5)
freeze_pool_5 <- catboost.load_pool(
    data = freeze_5 %>% select(-target),
    label = freeze_5$target
)

# Define parameter grid
params_grid <- expand.grid(
    depth = c(4, 6, 8),
    learning_rate = c(0.01, 0.1, 0.3),
    iterations = c(1000, 2000, 3000),
    l2_leaf_reg = c(1, 3, 5),
    rsm = c(0.7, 0.8, 0.9),
    border_count = c(32, 64, 128)
)

# Random sample from grid
set.seed(1618)
params_grid <- params_grid[sample(nrow(params_grid), min(20, nrow(params_grid))), ]

message(sprintf("Testing %d parameter combinations", nrow(params_grid)))

# Train models
best_rmse_5 <- Inf
best_params_5 <- NULL

for (i in 1:nrow(params_grid)) {
    params <- list(
        loss_function = "RMSE",
        eval_metric = "RMSE",
        depth = params_grid$depth[i],
        learning_rate = params_grid$learning_rate[i],
        iterations = params_grid$iterations[i],
        l2_leaf_reg = params_grid$l2_leaf_reg[i],
        rsm = params_grid$rsm[i],
        border_count = params_grid$border_count[i],
        random_seed = 1618,
        logging_level = "Silent"
    )

    model <- catboost.train(train_pool_5, NULL, params)

    # Predict on freeze set
    pred <- catboost.predict(model, freeze_pool_5)
    current_rmse <- rmse(freeze_5$target, pred)

    if (current_rmse < best_rmse_5) {
        best_rmse_5 <- current_rmse
        best_params_5 <- params
        best_model_5 <- model
    }

    if (i %% 5 == 0) {
        message(sprintf(
            "  Completed %d/%d combinations. Best RMSE: %.4f",
            i, nrow(params_grid), best_rmse_5
        ))
    }
}

message(sprintf("Best RMSE on freeze set: %.4f", best_rmse_5))

# Generate predictions
pred_cat_5 <- catboost.predict(best_model_5, freeze_pool_5)
y_cat_5 <- catboost.predict(
    best_model_5,
    catboost.load_pool(data = test_5 %>% select(-target))
)

# ==============================================================================
# 2. CATBOOST WITH RANDOM HYPERPARAMETER SEARCH - AGE GROUP 6
# ==============================================================================

message("\n=== CatBoost Model 2: Age Group 6 (Random Search) ===")

# Prepare data
y_train_6 <- train_6$target
X_train_6 <- train_6 %>% select(-target)

train_pool_6 <- catboost.load_pool(data = X_train_6, label = y_train_6)
freeze_pool_6 <- catboost.load_pool(
    data = freeze_6 %>% select(-target),
    label = freeze_6$target
)

# Random hyperparameter search
n_iterations <- 10
results_list <- list()

for (iter in 1:n_iterations) {
    params <- list(
        iterations = sample(500:3500, 1),
        learning_rate = sample(c(0.1, 0.3, 0.01), 1),
        depth = sample(4:8, 1),
        l2_leaf_reg = sample(1:11, 1),
        random_strength = runif(1, 0, 27),
        bagging_temperature = runif(1, 0, 10),
        border_count = sample(10:255, 1),
        rsm = runif(1, 0.5, 1),
        loss_function = "RMSE",
        eval_metric = "RMSE",
        random_seed = 1618,
        logging_level = "Silent"
    )

    model <- catboost.train(train_pool_6, freeze_pool_6, params)

    pred <- catboost.predict(model, freeze_pool_6)
    current_rmse <- rmse(freeze_6$target, pred)

    results_list[[iter]] <- list(
        params = params,
        rmse = current_rmse,
        model = model
    )

    message(sprintf("  Iteration %d/%d: RMSE = %.4f", iter, n_iterations, current_rmse))
}

# Find best model
best_idx <- which.min(sapply(results_list, function(x) x$rmse))
best_model_6 <- results_list[[best_idx]]$model
best_rmse_6 <- results_list[[best_idx]]$rmse

message(sprintf("Best RMSE on freeze set: %.4f", best_rmse_6))

# Generate predictions
pred_cat_6 <- catboost.predict(best_model_6, freeze_pool_6)
y_cat_6 <- catboost.predict(
    best_model_6,
    catboost.load_pool(data = test_6 %>% select(-target))
)

# ==============================================================================
# 3. SAVE PREDICTIONS AND MODELS
# ==============================================================================

# Save predictions (freeze set)
saveRDS(data.frame(cat = pred_cat_5), "pred_catboost_age5.rds")
saveRDS(data.frame(cat = pred_cat_6), "pred_catboost_age6.rds")

# Save predictions (test set)
saveRDS(data.frame(cat = y_cat_5), "test_pred_catboost_age5.rds")
saveRDS(data.frame(cat = y_cat_6), "test_pred_catboost_age6.rds")

# Save models
saveRDS(best_model_5, "model_catboost_age5.rds")
saveRDS(best_model_6, "model_catboost_age6.rds")

message("\n✓ CatBoost training complete!")
message(sprintf("  Age Group 5 - Freeze RMSE: %.4f", best_rmse_5))
message(sprintf("  Age Group 6 - Freeze RMSE: %.4f", best_rmse_6))
