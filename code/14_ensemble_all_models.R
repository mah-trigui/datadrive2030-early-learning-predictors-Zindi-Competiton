# ==============================================================================
# CLEAN SCRIPT 14: ENSEMBLE ALL MODELS
# ==============================================================================
# This script combines predictions from all models using different strategies:
# - Mean
# - Min
# - Max
# - Mean-Max hybrid
# - Mean-Min hybrid
# - Min-Max hybrid
# ==============================================================================

library(tidyverse)
library(Metrics)

# Load all freeze predictions for Age Group 5
pred_cat_5 <- readRDS("pred_catboost_age5.rds")
pred_light_5 <- readRDS("pred_lightgbm_age5.rds")
pred_h2o_5 <- readRDS("pred_h2o_age5.rds")
pred_sl3_5 <- readRDS("pred_sl3_age5.rds")
pred_tidy_5 <- readRDS("pred_tidymodels_age5.rds")

# Load all test predictions for Age Group 5
y_cat_5 <- readRDS("test_pred_catboost_age5.rds")
y_light_5 <- readRDS("test_pred_lightgbm_age5.rds")
y_h2o_5 <- readRDS("test_pred_h2o_age5.rds")
y_sl3_5 <- readRDS("test_pred_sl3_age5.rds")
y_tidy_5 <- readRDS("test_pred_tidymodels_age5.rds")

# Load all freeze predictions for Age Group 6
pred_cat_6 <- readRDS("pred_catboost_age6.rds")
pred_light_6 <- readRDS("pred_lightgbm_age6.rds")
pred_h2o_6 <- readRDS("pred_h2o_age6.rds")
pred_sl3_6 <- readRDS("pred_sl3_age6.rds")

# Load all test predictions for Age Group 6
y_cat_6 <- readRDS("test_pred_catboost_age6.rds")
y_light_6 <- readRDS("test_pred_lightgbm_age6.rds")
y_h2o_6 <- readRDS("test_pred_h2o_age6.rds")
y_sl3_6 <- readRDS("test_pred_sl3_age6.rds")

# Load freeze set for evaluation
freeze_5 <- readRDS("freeze_age5.rds")
freeze_6 <- readRDS("freeze_age6.rds")

# Load test IDs
id_test_5 <- readRDS("id_test_age5.rds")
id_test_6 <- readRDS("id_test_age6.rds")

message("Creating ensembles from all model predictions...")

# ==============================================================================
# 1. ENSEMBLE FOR AGE GROUP 5
# ==============================================================================

message("\n=== Age Group 5 Ensemble ===")

# Combine all freeze predictions
pred_5 <- cbind(pred_cat_5, pred_h2o_5, pred_light_5, pred_sl3_5, pred_tidy_5)
pred_5$target <- freeze_5$target

# Combine all test predictions
y_5 <- cbind(y_cat_5, y_h2o_5, y_light_5, y_sl3_5, y_tidy_5)
y_5$child_id <- id_test_5$child_id

# Strategy 1: Simple Mean
pred_5$mean <- (pred_5$cat + pred_5$h2o + pred_5$light + pred_5$sl3 + pred_5$tidy) / 5
rmse_mean_5 <- rmse(pred_5$target, pred_5$mean)
message(sprintf("Mean ensemble RMSE: %.4f", rmse_mean_5))

# Strategy 2: Min
pred_5$min <- pmin(pred_5$cat, pred_5$h2o, pred_5$light, pred_5$sl3, pred_5$tidy)
rmse_min_5 <- rmse(pred_5$target, pred_5$min)
message(sprintf("Min ensemble RMSE: %.4f", rmse_min_5))

# Strategy 3: Max
pred_5$max <- pmax(pred_5$cat, pred_5$h2o, pred_5$light, pred_5$sl3, pred_5$tidy)
rmse_max_5 <- rmse(pred_5$target, pred_5$max)
message(sprintf("Max ensemble RMSE: %.4f", rmse_max_5))

# Strategy 4: Mean-Max Hybrid
pred_5$mean_max <- (pred_5$mean + pred_5$max) / 2
rmse_mean_max_5 <- rmse(pred_5$target, pred_5$mean_max)
message(sprintf("Mean-Max ensemble RMSE: %.4f", rmse_mean_max_5))

# Strategy 5: Mean-Min Hybrid
pred_5$mean_min <- (pred_5$mean + pred_5$min) / 2
rmse_mean_min_5 <- rmse(pred_5$target, pred_5$mean_min)
message(sprintf("Mean-Min ensemble RMSE: %.4f", rmse_mean_min_5))

# Strategy 6: Min-Max Hybrid
pred_5$min_max <- (pred_5$min + pred_5$max) / 2
rmse_min_max_5 <- rmse(pred_5$target, pred_5$min_max)
message(sprintf("Min-Max ensemble RMSE: %.4f", rmse_min_max_5))

# Apply same strategies to test set
y_5$mean <- (y_5$cat + y_5$h2o + y_5$light + y_5$sl3 + y_5$tidy) / 5
y_5$min <- pmin(y_5$cat, y_5$h2o, y_5$light, y_5$sl3, y_5$tidy)
y_5$max <- pmax(y_5$cat, y_5$h2o, y_5$light, y_5$sl3, y_5$tidy)
y_5$mean_max <- (y_5$mean + y_5$max) / 2
y_5$mean_min <- (y_5$mean + y_5$min) / 2
y_5$min_max <- (y_5$min + y_5$max) / 2

# ==============================================================================
# 2. ENSEMBLE FOR AGE GROUP 6
# ==============================================================================

message("\n=== Age Group 6 Ensemble ===")

# Combine all freeze predictions (only 3 models for Age Group 6)
pred_6 <- cbind(pred_cat_6, pred_h2o_6, pred_sl3_6)
pred_6$target <- freeze_6$target

# Combine all test predictions
y_6 <- cbind(y_cat_6, y_h2o_6, y_sl3_6)
y_6$child_id <- id_test_6$child_id

# Strategy 1: Simple Mean
pred_6$mean <- (pred_6$cat + pred_6$h2o + pred_6$sl3) / 3
rmse_mean_6 <- rmse(pred_6$target, pred_6$mean)
message(sprintf("Mean ensemble RMSE: %.4f", rmse_mean_6))

# Strategy 2: Min
pred_6$min <- pmin(pred_6$cat, pred_6$h2o, pred_6$sl3)
rmse_min_6 <- rmse(pred_6$target, pred_6$min)
message(sprintf("Min ensemble RMSE: %.4f", rmse_min_6))

# Strategy 3: Max
pred_6$max <- pmax(pred_6$cat, pred_6$h2o, pred_6$sl3)
rmse_max_6 <- rmse(pred_6$target, pred_6$max)
message(sprintf("Max ensemble RMSE: %.4f", rmse_max_6))

# Strategy 4: Mean-Max Hybrid
pred_6$mean_max <- (pred_6$mean + pred_6$max) / 2
rmse_mean_max_6 <- rmse(pred_6$target, pred_6$mean_max)
message(sprintf("Mean-Max ensemble RMSE: %.4f", rmse_mean_max_6))

# Strategy 5: Mean-Min Hybrid
pred_6$mean_min <- (pred_6$mean + pred_6$min) / 2
rmse_mean_min_6 <- rmse(pred_6$target, pred_6$mean_min)
message(sprintf("Mean-Min ensemble RMSE: %.4f", rmse_mean_min_6))

# Strategy 6: Min-Max Hybrid
pred_6$min_max <- (pred_6$min + pred_6$max) / 2
rmse_min_max_6 <- rmse(pred_6$target, pred_6$min_max)
message(sprintf("Min-Max ensemble RMSE: %.4f", rmse_min_max_6))

# Apply same strategies to test set
y_6$mean <- (y_6$cat + y_6$h2o + y_6$sl3) / 3
y_6$min <- pmin(y_6$cat, y_6$h2o, y_6$sl3)
y_6$max <- pmax(y_6$cat, y_6$h2o, y_6$sl3)
y_6$mean_max <- (y_6$mean + y_6$max) / 2
y_6$mean_min <- (y_6$mean + y_6$min) / 2
y_6$min_max <- (y_6$min + y_6$max) / 2

# ==============================================================================
# 3. SAVE ENSEMBLE PREDICTIONS
# ==============================================================================

# Save freeze predictions with all ensemble strategies
saveRDS(pred_5, "ensemble_freeze_age5.rds")
saveRDS(pred_6, "ensemble_freeze_age6.rds")

# Save test predictions with all ensemble strategies
saveRDS(y_5, "ensemble_test_age5.rds")
saveRDS(y_6, "ensemble_test_age6.rds")

# ==============================================================================
# 4. SUMMARY OF BEST STRATEGIES
# ==============================================================================

message("\n=== Best Ensemble Strategies ===")

best_strategies_5 <- data.frame(
    Strategy = c("Mean", "Min", "Max", "Mean-Max", "Mean-Min", "Min-Max"),
    RMSE = c(
        rmse_mean_5, rmse_min_5, rmse_max_5,
        rmse_mean_max_5, rmse_mean_min_5, rmse_min_max_5
    )
) %>%
    arrange(RMSE)

message("\nAge Group 5:")
print(best_strategies_5)

best_strategies_6 <- data.frame(
    Strategy = c("Mean", "Min", "Max", "Mean-Max", "Mean-Min", "Min-Max"),
    RMSE = c(
        rmse_mean_6, rmse_min_6, rmse_max_6,
        rmse_mean_max_6, rmse_mean_min_6, rmse_min_max_6
    )
) %>%
    arrange(RMSE)

message("\nAge Group 6:")
print(best_strategies_6)

message("\n✓ Ensemble predictions complete!")
