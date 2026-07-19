# ==============================================================================
# 00: MASTER PIPELINE
# ==============================================================================
# This master script runs the entire pipeline from data loading to submission
# DataDrive2030 Early Learning Predictors Challenge
# ==============================================================================

library(tidyverse)

message("========================================")
message("CHILD DEVELOPMENT ASSESSMENT PIPELINE")
message("========================================\n")

start_time <- Sys.time()

# ==============================================================================
# STEP 1: DATA PREPROCESSING
# ==============================================================================

message("STEP 1: Data Loading and Preprocessing")
message("---------------------------------------")
source("01_data_loading_and_preprocessing.R")

message("\nSTEP 2: Feature Engineering")
message("----------------------------")
source("02_feature_engineering.R")

message("\nSTEP 3: Correlation Analysis")
message("-----------------------------")
source("03_correlation_removal.R")

message("\nSTEP 4: Imputation")
message("------------------")
source("04_imputation.R")

message("\nSTEP 5: Factor Consolidation")
message("----------------------------")
source("05_factor_consolidation.R")

message("\nSTEP 6: Ordinal Encoding")
message("------------------------")
source("06_ordinal_encoding.R")

message("\nSTEP 7: Clustering Features")
message("---------------------------")
source("07_clustering.R")

message("\nSTEP 8: Train-Test Split")
message("------------------------")
source("08_train_test_split.R")

# ==============================================================================
# STEP 2: MODEL TRAINING
# ==============================================================================

message("\n\n========================================")
message("MODEL TRAINING")
message("========================================\n")

message("STEP 9: CatBoost Models")
message("-----------------------")
source("09_model_catboost.R")

message("\nSTEP 10: LightGBM Models")
message("------------------------")
source("10_model_lightgbm.R")

message("\nSTEP 11: H2O AutoML Models")
message("---------------------------")
source("11_model_h2o_automl.R")

message("\nSTEP 12: SuperLearner (SL3)")
message("---------------------------")
source("12_model_sl3_superlearner.R")

message("\nSTEP 13: Tidymodels Stacking")
message("----------------------------")
source("13_model_tidymodels_stacking.R")

# ==============================================================================
# STEP 3: ENSEMBLE AND SUBMISSION
# ==============================================================================

message("\n\n========================================")
message("ENSEMBLE & SUBMISSION")
message("========================================\n")

message("STEP 14: Ensemble All Models")
message("----------------------------")
source("14_ensemble_all_models.R")

message("\nSTEP 15: Create Submission")
message("--------------------------")
source("15_create_submission.R")

# ==============================================================================
# PIPELINE COMPLETE
# ==============================================================================

end_time <- Sys.time()
total_time <- difftime(end_time, start_time, units = "mins")

message("\n\n========================================")
message("PIPELINE COMPLETE!")
message("========================================")
message(sprintf("Total execution time: %.2f minutes", as.numeric(total_time)))
message("\nGenerated files:")
message("  - final_submission.csv (best ensemble)")
message("  - submission_mean.csv (mean ensemble)")
message("  - submission_mean_max.csv (mean-max ensemble)")
message("\nAll intermediate data saved as .rds files")
message("========================================\n")
