# ==============================================================================
# CLEAN SCRIPT 02: FEATURE ENGINEERING
# ==============================================================================
# This script performs comprehensive feature engineering including:
# - Factor recoding and consolidation
# - Binary variable encoding
# - Numeric transformations
# - Outlier capping
# ==============================================================================

library(tidyverse)
library(dlookr)

# Load preprocessed data
df <- readRDS("preprocessed_data.rds")

# ------------------------------------------------------------------------------
# 1. CONVERT DATA_YEAR TO FACTOR
# ------------------------------------------------------------------------------

df$data_year <- as.factor(df$data_year)

# ------------------------------------------------------------------------------
# 2. RECODE LANGUAGE VARIABLES
# ------------------------------------------------------------------------------

# Consolidate child_languages into strength categories
df$child_languages <- fct_recode(df$child_languages,
    weak = "Afrikaans + Setswana +",
    weak = "isiZulu + isiXhosa +",
    weak = "English + Xitsonga +",
    strong = "English + isiNdebele +",
    strong = "English + Setswana +",
    strong = "English + Sesotho se Leboa (Sepedi)",
    strong = "isiXhosa + Xitsonga"
)

# ------------------------------------------------------------------------------
# 3. DROP LOW-VARIANCE/REDUNDANT NUMERIC COLUMNS
# ------------------------------------------------------------------------------

to_drop_numeric <- c(
    "child_height", "child_zha", # Height measures
    "pri_capacity", "pri_year", "pri_children_4_6_years", # Capacity
    "pri_difficult_hear", "pri_difficult_walk", "pri_difficult_see", # Disabilities
    "pri_expense_materials", "pri_expense_other", "pri_expense_rent", # Expenses
    "teacher_duration", "teacher_selfcare_total", # Teacher metrics
    "count_register_gender_female", "count_register_gender_male", # Gender counts
    "count_register_race_other", # Race counts
    "count_register_year_2013", "count_register_year_2014", "count_register_year_school", # Year counts
    "count_staff_salary_paid", "count_register_foreign", # Staff counts
    "count_staff_contract_substitute", "count_staff_contract_temporary", # Contract types
    "count_practitioners_all", "count_staff_time_part", # Practitioner counts
    "count_practitioners_age_6", "count_practitioners_age_5", # Age-specific counts
    "count_practitioners_age_4", "count_practitioners_age_3",
    "count_practitioners_age_2", "count_practitioners_age_1",
    "id_ward_n" # Ward identifier
)

df <- df %>% select(-all_of(to_drop_numeric))

# ------------------------------------------------------------------------------
# 4. BINARY NUMERIC VARIABLE ENCODING (-1, 0, 1)
# ------------------------------------------------------------------------------

# For binary numeric variables: -1 for original 0, 0 for NA, 1 for original 1
binary_vars <- c(
    "obs_heating_3", "obs_heating_4", "obs_heating_5", "obs_heating_6", "obs_heating_7",
    "obs_lighting_3", "obs_lighting_4", "obs_lighting_5", "obs_lighting_6",
    "obs_cooking_4", "obs_cooking_5"
)

df <- df %>%
    mutate_at(vars(all_of(binary_vars)), ~ replace(., . == 0, -1)) %>%
    mutate_at(vars(all_of(binary_vars)), ~ replace_na(., 0))

# ------------------------------------------------------------------------------
# 5. RECODE FACTOR VARIABLES (YES/NO ENCODING)
# ------------------------------------------------------------------------------

# Recode 'not' to 'Urban' for urban variable
df$urban <- fct_recode(df$urban, Urban = "not")

# Variables where NA/'not' should be coded as 'Yes'
yes_na <- c(
    "gps_ind", "quintile_used", "practitioner", "obs_menu_compliance",
    "obs_access_disability_0", "obs_materials_19", "obs_equipment_4",
    "obs_equipment_0", "obs_books", "obs_materials_20", "obs_materials_11",
    "obs_materials_13", "pri_clinic_travel_3", "pri_funding_salary_0",
    "pri_covid_fund_applied", "pri_holidays"
)

df[which(colnames(df) %in% yes_na)] <- as.data.frame(
    apply(
        df[which(colnames(df) %in% yes_na)], 2,
        function(x) fct_recode(x, Yes = "not")
    )
)

# Variables where NA/'not' should be coded as 'No'
no_na <- c(
    "obs_hazard_1", "obs_hazard_2", "obs_hazard_3", "obs_hazard_4", "obs_hazard_5",
    "obs_hazard_8", "obs_hazard_97", "obs_safety_1", "obs_safety_2",
    "obs_access_disability_1", "obs_access_disability_2", "obs_access_disability_4",
    "obs_access_disability_6", "pri_food_guidance", "pri_food_type_2", "pri_food_type_5",
    "pri_clinic_travel_1", "pri_funding_salary_3", "pri_money__1",
    "pri_funding_2", "pri_funding_5", "pri_funding_3", "pri_library",
    "pri_qualification_0", "pri_qualification_6", "pri_qualification_7",
    "pri_qualification_97", "pri_subsidy"
)

df[which(colnames(df) %in% no_na)] <- as.data.frame(
    apply(
        df[which(colnames(df) %in% no_na)], 2,
        function(x) fct_recode(x, No = "not")
    )
)

# ------------------------------------------------------------------------------
# 6. DROP LOW-INFORMATION FACTOR VARIABLES
# ------------------------------------------------------------------------------

to_drop_factors <- c(
    # Heating/lighting/cooking (low variance or redundant)
    "obs_heating_1", "obs_heating_2", "obs_lighting_8", "obs_lighting_1", "obs_lighting_2",
    "obs_cooking_1", "obs_cooking_2", "obs_cooking_3", "obs_cooking_6",

    # Census and observation variables
    "census", "obs_menu_same",

    # Safety variables (low variance)
    "obs_safety_5", "obs_safety_7", "obs_safety_9",

    # Access disability variables
    "obs_access_disability_0", "obs_access_disability_5",

    # Toilet variables
    "obs_toilet_paper", "obs_toilet_clean",

    # Support and records
    "pri_support", "pri_support_provider_1", "pri_garden", "pri_records_1",

    # Food type
    "pri_food_type_1", "pri_food_type_4", "pri_refrigerator",

    # COVID precautions
    "pri_covid_precautions_4", "pri_covid_precautions_5",

    # Health variables
    "pri_health_0", "pri_health_4", "pri_health_5",

    # Funding and money
    "pri_funding_4", "pri_money_1", "pri_money_2",

    # Zoning and network
    "pri_zoning", "pri_network_type_1", "pri_registered_health",

    # Qualification variables
    "pri_qualification_1", "pri_qualification_4"
)

df <- df %>% select(-all_of(to_drop_factors))

# ------------------------------------------------------------------------------
# 7. DROP ADDITIONAL LOW-VARIANCE COLUMNS
# ------------------------------------------------------------------------------

to_drop_additional <- c(
    "count_register_year_2018", "count_register_year_2021",
    "count_staff_qual_nqf4_5", "count_staff_salary_unpaid",
    "count_toilets_adults", "obs_classrooms", "pri_dsd_year"
)

df <- df %>% select(-all_of(to_drop_additional))

# ------------------------------------------------------------------------------
# 8. CONVERT SELECTED FACTORS TO NUMERIC (YES/NO ENCODING)
# ------------------------------------------------------------------------------

# Keep these as numeric (-1 for No, 0 for NA, 1 for Yes)
to_keep_numeric <- c(
    "pri_registered_npo", "pri_registered_cipc", "pri_money_97",
    "pri_fees_exceptions", "obs_materials_12", "obs_toilet_1"
)

df <- df %>%
    mutate_at(vars(all_of(to_keep_numeric)), as.numeric) %>%
    mutate_at(vars(all_of(to_keep_numeric)), ~ replace(., . == 2, -1)) %>%
    mutate_at(vars(all_of(to_keep_numeric)), ~ replace(., . == 1, 0)) %>%
    mutate_at(vars(all_of(to_keep_numeric)), ~ replace(., . == 3, 1))

# ------------------------------------------------------------------------------
# 9. CONVERT CHARACTER VARIABLES TO NUMERIC (YES/NO)
# ------------------------------------------------------------------------------

to_keep_char_numeric <- c("pri_subsidy", "obs_materials_11")

df <- df %>%
    mutate_at(vars(all_of(to_keep_char_numeric)), ~ replace(., . == "No", 0)) %>%
    mutate_at(vars(all_of(to_keep_char_numeric)), ~ replace(., . == "Yes", 1)) %>%
    mutate_at(vars(all_of(to_keep_char_numeric)), as.numeric)

# ------------------------------------------------------------------------------
# 10. ENCODE BINARY CATEGORICAL VARIABLES
# ------------------------------------------------------------------------------

# Child gender: 1 = Female, 0 = Male
df$child_gender <- ifelse(df$child_gender == "Female", 1, 0)

# Urban/Rural: 1 = Urban, 0 = Rural
df$urban <- ifelse(df$urban == "Rural", 0, 1)

# COVID period: 1 = Pre-COVID, 0 = Post-COVID
df$pre_covid <- ifelse(df$pre_covid == "Post COVID", 0, 1)

# ------------------------------------------------------------------------------
# 11. ENCODE TEACHER METRICS (-1, 0, 1)
# ------------------------------------------------------------------------------

df <- df %>%
    mutate(
        teacher_emotional_met = as.numeric(teacher_emotional_met),
        teacher_social_met = as.numeric(teacher_social_met),
        teacher_selfcare_met = as.numeric(teacher_selfcare_met)
    ) %>%
    mutate_at(
        vars("teacher_emotional_met", "teacher_social_met", "teacher_selfcare_met"),
        ~ replace(., . == 1, -1)
    ) %>%
    mutate_at(
        vars("teacher_emotional_met", "teacher_social_met", "teacher_selfcare_met"),
        ~ replace(., . == 3, 0)
    ) %>%
    mutate_at(
        vars("teacher_emotional_met", "teacher_social_met", "teacher_selfcare_met"),
        ~ replace(., . == 2, 1)
    )

# ------------------------------------------------------------------------------
# 12. OUTLIER CAPPING
# ------------------------------------------------------------------------------

# Identify numeric columns with outliers
aux <- dlookr::diagnose_outlier(df)
outlier_var <- aux[aux$outliers_cnt > 0, ]$variables

# Capping function: replace outliers with 2nd/98th percentiles
cap <- function(x) {
    quantiles <- quantile(x, c(.02, 0.25, 0.75, .98), na.rm = TRUE)
    x[x < quantiles[2] - 1.5 * IQR(x, na.rm = TRUE)] <- quantiles[1]
    x[x > quantiles[3] + 1.5 * IQR(x, na.rm = TRUE)] <- quantiles[4]
    x
}

# Apply capping to outlier variables
df[which(colnames(df) %in% outlier_var)] <- as.data.frame(
    apply(df[which(colnames(df) %in% outlier_var)], 2, cap)
)

message("✓ Feature engineering complete!")
message(sprintf("Final dimensions: %d rows x %d columns", nrow(df), ncol(df)))

# Save feature-engineered data
saveRDS(df, "feature_engineered_data.rds")
message("✓ Feature-engineered data saved!")
