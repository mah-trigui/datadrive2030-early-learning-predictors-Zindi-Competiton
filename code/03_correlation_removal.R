# ==============================================================================
# CLEAN SCRIPT 03: CORRELATION ANALYSIS AND MULTICOLLINEARITY REMOVAL
# ==============================================================================
# This script removes highly correlated features and low-information variables
# ==============================================================================

library(tidyverse)
library(caret)

# Load feature-engineered data
df <- readRDS("feature_engineered_data.rds")

# ------------------------------------------------------------------------------
# 1. REMOVE HIGHLY CORRELATED NUMERIC VARIABLES
# ------------------------------------------------------------------------------

# Extract numeric columns
df_numeric <- df[, sapply(df, is.numeric)]

# Compute correlation matrix (pairwise complete observations)
cor_mat <- cor(df_numeric, use = "pairwise.complete.obs")

# Replace NA correlations with 0
cor_mat[is.na(cor_mat)] <- 0

# Find highly correlated variables (correlation > 0.9)
highly_correlated <- findCorrelation(cor_mat, cutoff = 0.9)
corr_vars <- names(df_numeric[, highly_correlated])

message(sprintf("Found %d highly correlated variables to remove", length(corr_vars)))
print(corr_vars)

# Remove highly correlated variables
df <- df %>% select(-all_of(corr_vars))

# ------------------------------------------------------------------------------
# 2. DROP LOW-INFORMATION CATEGORICAL VARIABLES
# ------------------------------------------------------------------------------

to_drop_categorical <- c(
    # Location and support
    "pri_location", "pri_support_municipality", "pri_support_dbe",

    # Education and kitchen
    "pri_education", "pri_kitchen", "pri_support_ngo",

    # COVID and food
    "pri_covid_staff_salaries", "pri_food_parents_afternoon",

    # Contact and observation
    "pri_parents_contact", "obs_lighting", "obs_cooking",

    # Certificates and programs
    "certificate_registration_program", "obs_cooking_census",
    "obs_shared", "obs_material_display",

    # Phase and plan
    "phase_natemis", "plan", "certificate_register"
)

df <- df %>% select(-all_of(to_drop_categorical))

# ------------------------------------------------------------------------------
# 3. DROP ADDITIONAL LOW-PREDICTIVE-POWER VARIABLES
# ------------------------------------------------------------------------------

to_drop_additional <- c(
    # COVID and expenses
    "pri_covid_staff_retrench", "pri_expense_food", "pri_difficult_communicate",

    # Funding and fees
    "pri_amount_funding_dsd", "pri_fees_exceptions", "pri_registered_cipc",

    # Year registrations
    "count_register_year_2015", "count_register_year_grader",
    "count_register_year_2016", "count_register_year_2017",
    "count_register_year_2019", "count_register_year_2020",

    # Children counts
    "count_children_attendance", "count_children_precovid",

    # Race and staff
    "count_register_race_african", "obs_materials_12",
    "teacher_social_total", "count_staff_qual_skills",
    "count_register_race", "count_toilets_children",
    "count_staff_gender_male",

    # Urban variable (already encoded)
    "urban",

    # Observation variables
    "obs_cooking_5", "obs_lighting_3", "pri_meal_prep",

    # Staff counts
    "count_staff_paid_cooks", "sanitation_educators",
    "count_staff_paid_maintenance", "obs_heating_5", "obs_heating_4",
    "count_staff_paid_support", "count_staff_paid_managers",

    # Practitioners age 0
    "count_practitioners_age_0",

    # COVID period (already encoded)
    "pre_covid",

    # Food variables
    "pri_food_parents_breakfast", "pri_food_parents_lunch"
)

df <- df %>% select(-all_of(to_drop_additional))

# ------------------------------------------------------------------------------
# 4. REPLACE NA WITH 0 FOR SPECIFIC VARIABLES
# ------------------------------------------------------------------------------

na_to_zero <- c(
    "pri_difficult_hold", "pri_difficult_learn",
    "count_staff_paid_assistants", "pri_funding_subsidy"
)

df <- df %>%
    mutate_at(vars(all_of(na_to_zero)), ~ replace_na(., 0))

message("✓ Correlation analysis and variable removal complete!")
message(sprintf("Final dimensions: %d rows x %d columns", nrow(df), ncol(df)))
message(sprintf(
    "Removed %d variables total",
    length(corr_vars) + length(to_drop_categorical) +
        length(to_drop_additional)
))

# Save cleaned data
saveRDS(df, "correlation_cleaned_data.rds")
message("✓ Correlation-cleaned data saved!")
