# ==============================================================================
# CLEAN SCRIPT 01: DATA LOADING AND PREPROCESSING
# ==============================================================================
# Project: Child Development Assessment - Predicting developmental scores
# Dataset: Early Childhood Development (ECD) program data from South Africa
# Target: Continuous score measuring child development
# ==============================================================================

library(tidyverse)
library(DataExplorer)

# ------------------------------------------------------------------------------
# 1. LOAD DATA
# ------------------------------------------------------------------------------

train <- read.csv("Train.csv")
test <- read.csv("Test.csv")
submission <- read.csv("SampleSubmission.csv")

# Add target column to test set (as NA for binding)
test$target <- NA

# Combine train and test for unified preprocessing
df <- rbind(train, test)

message(sprintf(
    "Total observations: %d (Train: %d, Test: %d)",
    nrow(df), nrow(train), nrow(test)
))

# ------------------------------------------------------------------------------
# 2. CREATE AGE GROUPS
# ------------------------------------------------------------------------------

# Binary age group: 1 = 50-59 months OR younger, 0 = 60+ months
df$child_age_group <- ifelse(df$child_age_group == "50-59 months" |
    df$child_age_group == "Younger than 50 months", 1, 0)

# Create performance classes for EDA
df$class <- ifelse(df$child_age_group == 1,
    ifelse(df$target <= 36.01, "R", ifelse(df$target <= 46.31, "F", "A")),
    ifelse(df$target <= 43.23, "R", ifelse(df$target <= 54.37, "F", "A"))
)

# ------------------------------------------------------------------------------
# 3. DROP IRRELEVANT/HIGH-CARDINALITY COLUMNS
# ------------------------------------------------------------------------------

# Drop PQA (Program Quality Assessment) columns (indices 89-174)
df <- df[, -c(89:174)]

# Drop additional identifier columns (indices 181-186)
df <- df[, -c(181:186)]

# Drop all columns starting with "pqa_"
df <- df %>% select(-starts_with("pqa_"))

# Extensive list of columns to drop based on EDA
to_drop <- c(
    # Date and identification columns
    "child_date", "child_enrolment_date", "child_months_enrolment", "child_dob",
    "child_attends", "id_mn_best", "id_dc_best", "id_dc_n", "id_ward",
    "id_prov_n", "id_facility", "id_facility_n", "id_enumerator",

    # Free text/high cardinality columns
    "pri_mobile", "pri_school", "pri_days", "pri_date", "child_attendance",
    "pri_language", "pri_languageother", "pri_facilitiesother", "pri_landother",
    "pri_fundingother", "pri_qualification", "pri_locationother", "pri_founderother",
    "pri_qualificationother", "pri_name_network_forum", "pri_email_network_forum",
    "pri_name_network_ngo", "pri_name_network_alliance", "pri_name_network_other",
    "pri_staff_changes_reasons", "pri_staff_changes_reasonsother", "pri_money",
    "pri_moneyother", "pri_funding_salary", "pri_funding_salaryother",
    "pri_fees_exceptions_other", "pri_expenseother", "pri_reason_register_year",

    # Conditional/unregistered columns
    "pri_dsd_conditional", "pri_dsd_conditional_other", "pri_dsd_unregistered",
    "pri_dsd_unregistered_other",

    # COVID awareness columns (free text)
    "pri_covid_awareness", "pri_covid_precautions", "pri_covid_precautions_other",
    "pri_covid_awareness_other",

    # Food/clinic columns (high cardinality)
    "pri_clinic_time", "pri_clinic_travel", "pri_clinic_travelother",
    "pri_food_type", "pri_food_donor", "pri_food_donorother",

    # Records and support columns
    "pri_records", "pri_support_provider", "pri_support_providerother",
    "pri_internet_user", "pri_meal",

    # Practice/groupings columns
    "pra_groupings", "pra_plans",

    # Language match column
    "language_match",

    # Observation columns (free text or high cardinality)
    "obs_access_disability", "obs_area", "obs_materials", "obs_materialsother",
    "obs_handwashing", "obs_handwashingother", "obs_waterother",
    "obs_water_running_none", "obs_toilet", "obs_equipment", "obs_date",
    "obs_safety", "obs_hazard",

    # Count columns with gender_other
    "count_register_gender_other", "count_staff_gender_other",

    # GPS and location columns
    "gps", "latitude", "longitude",

    # Health and language columns
    "health", "healthother", "language_child", "language_assessment_w2",

    # Sanitation and position columns
    "sanitation_learners", "positionotherreason", "positionother",

    # HLE (Home Learning Environment) other
    "hle_ecd_other",

    # Other practitioner and best columns
    "other_practitioner", "mn_best", "dc_best"
)

df <- df %>% select(-all_of(to_drop))

message(sprintf("After dropping irrelevant columns: %d columns remain", ncol(df)))

# ------------------------------------------------------------------------------
# 4. HANDLE MISSING VALUES AS FACTORS
# ------------------------------------------------------------------------------

# Replace empty strings with 'not' (for factor conversion)
df[df == ""] <- "not"

# Convert all character columns to factors
df <- as.data.frame(unclass(df), stringsAsFactors = TRUE)

# Store full dataframe with child_id for later
df_id <- df

# Remove child_id from main dataframe
df <- df[, -1]

message("Data preprocessing complete!")
message(sprintf("Final dimensions: %d rows x %d columns", nrow(df), ncol(df)))
message(sprintf("Missing target values (test set): %d", sum(is.na(df$target))))

# Save preprocessing checkpoint
saveRDS(df, "preprocessed_data.rds")
saveRDS(df_id, "preprocessed_data_with_ids.rds")

message("✓ Preprocessed data saved!")
