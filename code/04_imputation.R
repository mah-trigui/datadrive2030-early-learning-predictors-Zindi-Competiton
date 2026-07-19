# ==============================================================================
# CLEAN SCRIPT 04: ADVANCED IMPUTATION
# ==============================================================================
# This script performs missing value imputation using:
# 1. missRanger (Random Forest-based imputation)
# 2. Amelia (Multiple Imputation with averaging)
# ==============================================================================

library(tidyverse)
library(missRanger)
library(Amelia)

# Load correlation-cleaned data
df <- readRDS('correlation_cleaned_data.rds')

message(sprintf("Starting imputation for %d observations with %d variables",
                nrow(df), ncol(df)))

# ------------------------------------------------------------------------------
# 1. PREPARE DATA FOR IMPUTATION
# ------------------------------------------------------------------------------

# Extract numeric columns (exclude target for imputation)
df_numeric <- names(df[, sapply(df, is.numeric)])
aux <- df %>% select(all_of(df_numeric))
aux$target <- NULL  # Remove target to prevent data leakage

message(sprintf("Imputing %d numeric variables", ncol(aux)))
message(sprintf("Total missing values: %d (%.2f%%)",
                sum(is.na(aux)),
                100 * sum(is.na(aux)) / prod(dim(aux))))

# ------------------------------------------------------------------------------
# 2. RANDOM FOREST IMPUTATION (missRanger)
# ------------------------------------------------------------------------------

set.seed(1618)
dimp <- missRanger(
  aux,
  formula = . ~ .,
  verbose = 2,
  seed = 111,
  returnOOB = TRUE
)

message("✓ missRanger imputation complete!")

# ------------------------------------------------------------------------------
# 3. MULTIPLE IMPUTATION (Amelia)
# ------------------------------------------------------------------------------

message("Starting Amelia multiple imputation (5 imputations)...")

set.seed(1618)
fit <- amelia(aux, parallel = "multicore")

# Average the 5 imputations for pri_amount_funding_fees
ame <- fit$imputations[[1]]
ame$pri_amount_funding_fees <- (
  ame$pri_amount_funding_fees +
  fit$imputations[[2]]$pri_amount_funding_fees +
  fit$imputations[[3]]$pri_amount_funding_fees +
  fit$imputations[[4]]$pri_amount_funding_fees +
  fit$imputations[[5]]$pri_amount_funding_fees
) / 5

# Use Amelia result for funding fees (better correlation preservation)
dimp$pri_amount_funding_fees <- ame$pri_amount_funding_fees

message("✓ Amelia multiple imputation complete!")

# ------------------------------------------------------------------------------
# 4. COMBINE IMPUTED NUMERICS WITH CATEGORICALS
# ------------------------------------------------------------------------------

# Combine imputed numeric data with categorical variables and target
df_imputed <- cbind(
  dimp,
  df %>% select(-all_of(df_numeric)),
  df[, 'target']
)

names(df_imputed)[ncol(df_imputed)] <- 'target'

# Verify no missing values in numeric columns
message(sprintf("Missing values after imputation: %d", sum(is.na(df_imputed))))

message("✓ Imputation complete!")
message(sprintf("Final dimensions: %d rows x %d columns",
                nrow(df_imputed), ncol(df_imputed)))

# Save imputed data
saveRDS(df_imputed, 'imputed_data.rds')
message("✓ Imputed data saved!")
