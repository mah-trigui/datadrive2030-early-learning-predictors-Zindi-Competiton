# ==============================================================================
# CLEAN SCRIPT 12: SL3 (SUPERLEARNER) ENSEMBLE
# ==============================================================================
# This script trains SuperLearner (sl3) ensemble models
# ==============================================================================

library(tidyverse)
library(sl3)
library(Metrics)

# Load data splits
train_5 <- readRDS("train_age5.rds")
freeze_5 <- readRDS("freeze_age5.rds")
test_5 <- readRDS("test_age5.rds")

train_6 <- readRDS("train_age6.rds")
freeze_6 <- readRDS("freeze_age6.rds")
test_6 <- readRDS("test_age6.rds")

message("Training SuperLearner (sl3) models...")

# ==============================================================================
# 1. SL3 ENSEMBLE - AGE GROUP 5
# ==============================================================================

message("\n=== SuperLearner: Age Group 5 ===")

# Prepare data
nm_5 <- names(train_5)
nm_5 <- nm_5[-which(nm_5 == "target")]

# Create sl3 task
task_5 <- make_sl3_Task(
    data = train_5,
    outcome = "target",
    covariates = nm_5
)

# Define base learners
lrn_glm <- Lrnr_glm$new()
lrn_mean <- Lrnr_mean$new()
lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
lrn_lasso <- Lrnr_glmnet$new(alpha = 1)
lrn_ranger <- Lrnr_ranger$new()
lrn_xgb <- Lrnr_xgboost$new()
lrn_gam <- Lrnr_gam$new()
lrn_bayesglm <- Lrnr_bayesglm$new()

# Create stack of learners
stack_5 <- Stack$new(
    lrn_glm, lrn_mean, lrn_ridge, lrn_lasso,
    lrn_ranger, lrn_xgb, lrn_gam, lrn_bayesglm
)

# Create SuperLearner with NNLS metalearner
sl_5 <- Lrnr_sl$new(learners = stack_5, metalearner = Lrnr_nnls$new())

# Train SuperLearner
set.seed(1618)
message("Training base learners and metalearner...")
sl_fit_5 <- sl_5$train(task = task_5)

# Predict on freeze set
freeze_task_5 <- make_sl3_Task(
    data = freeze_5,
    covariates = nm_5
)

pred_sl3_5 <- sl_fit_5$predict(task = freeze_task_5)
rmse_5 <- rmse(freeze_5$target, pred_sl3_5)
message(sprintf("Freeze RMSE: %.4f", rmse_5))

# Predict on test set
test_task_5 <- make_sl3_Task(
    data = test_5,
    covariates = nm_5
)

y_sl3_5 <- sl_fit_5$predict(task = test_task_5)

# Display learner weights
message("\nLearner Weights (Age Group 5):")
print(sl_fit_5$coefficients)

# ==============================================================================
# 2. SL3 ENSEMBLE - AGE GROUP 6
# ==============================================================================

message("\n=== SuperLearner: Age Group 6 ===")

# Prepare data
nm_6 <- names(train_6)
nm_6 <- nm_6[-which(nm_6 == "target")]

# Create sl3 task
task_6 <- make_sl3_Task(
    data = train_6,
    outcome = "target",
    covariates = nm_6
)

# Create stack (same learners)
stack_6 <- Stack$new(
    lrn_glm, lrn_mean, lrn_ridge, lrn_lasso,
    lrn_ranger, lrn_xgb, lrn_gam, lrn_bayesglm
)

# Create SuperLearner
sl_6 <- Lrnr_sl$new(learners = stack_6, metalearner = Lrnr_nnls$new())

# Train SuperLearner
set.seed(1618)
message("Training base learners and metalearner...")
sl_fit_6 <- sl_6$train(task = task_6)

# Predict on freeze set
freeze_task_6 <- make_sl3_Task(
    data = freeze_6,
    covariates = nm_6
)

pred_sl3_6 <- sl_fit_6$predict(task = freeze_task_6)
rmse_6 <- rmse(freeze_6$target, pred_sl3_6)
message(sprintf("Freeze RMSE: %.4f", rmse_6))

# Predict on test set
test_task_6 <- make_sl3_Task(
    data = test_6,
    covariates = nm_6
)

y_sl3_6 <- sl_fit_6$predict(task = test_task_6)

# Display learner weights
message("\nLearner Weights (Age Group 6):")
print(sl_fit_6$coefficients)

# ==============================================================================
# 3. SAVE PREDICTIONS AND MODELS
# ==============================================================================

# Save predictions (freeze set)
saveRDS(data.frame(sl3 = pred_sl3_5), "pred_sl3_age5.rds")
saveRDS(data.frame(sl3 = pred_sl3_6), "pred_sl3_age6.rds")

# Save predictions (test set)
saveRDS(data.frame(sl3 = y_sl3_5), "test_pred_sl3_age5.rds")
saveRDS(data.frame(sl3 = y_sl3_6), "test_pred_sl3_age6.rds")

# Save fitted models
saveRDS(sl_fit_5, "model_sl3_age5.rds")
saveRDS(sl_fit_6, "model_sl3_age6.rds")

message("\n✓ SuperLearner training complete!")
message(sprintf("  Age Group 5 - Freeze RMSE: %.4f", rmse_5))
message(sprintf("  Age Group 6 - Freeze RMSE: %.4f", rmse_6))
