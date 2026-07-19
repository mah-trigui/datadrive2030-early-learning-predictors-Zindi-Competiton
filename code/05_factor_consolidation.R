# ==============================================================================
# CLEAN SCRIPT 05: FACTOR CONSOLIDATION AND RECODING
# ==============================================================================
# This script consolidates categorical variables with rare levels
# ==============================================================================

library(tidyverse)
library(forcats)

# Load imputed data
df <- readRDS("imputed_data.rds")

# ------------------------------------------------------------------------------
# 1. CONSOLIDATE RARE FACTOR LEVELS
# ------------------------------------------------------------------------------

# Child stunted
df$child_stunted <- fct_recode(df$child_stunted,
    "not" = "Severely stunted"
)

# Child languages
df$child_languages <- fct_recode(df$child_languages,
    "not" = "Setswana",
    "Afrikaans" = "isiZulu",
    "isiXhosa" = "Sesotho se Leboa (Sepedi)"
)

# Province
df$prov_best <- fct_recode(df$prov_best,
    "NORTHERN CAPE" = "FREE STATE",
    "LIMPOPO" = "KWAZULU-NATAL"
)

# Years in programme
df$child_years_in_programme <- fct_recode(df$child_years_in_programme,
    "2nd year in programme" = "Do Not Know"
)

# Free play outdoor
df$pra_free_play_outdoor <- fct_recode(df$pra_free_play_outdoor,
    "Up to 1 hour" = "30 minutes or less"
)

# Practice agency - play
df$pra_agency_play <- fct_recode(df$pra_agency_play,
    "Both" = "Child",
    "Practitioner" = "None"
)

# Practice agency - questions
df$pra_agency_questions <- fct_recode(df$pra_agency_questions,
    "Both" = "None"
)

# Practice agency - understand
df$pra_agency_understand <- fct_recode(df$pra_agency_understand,
    "Practitioner" = "None"
)

# Practice engaged
df$pra_engaged <- fct_recode(df$pra_engaged,
    "Often" = "Sometime"
)

# Fees paid proportion
df$pri_fees_paid_proportion <- fct_recode(
    df$pri_fees_paid_proportion,
    "Most parrents, but not all" = "All parents",
    "not" = "None of the parents",
    "not" = "Only a few parents"
)

# Land ownership
df$pri_land <- fct_recode(
    df$pri_land,
    "not" = "The person in charge of the programme (e.g. principal, matron, child-minder, playgroup leader)",
    "The ECD Programme" = "Community Centre",
    "Municipality" = "School",
    "Communal land (traditional) " = "Not-for profit organisation",
    "Communal land (traditional) " = "Other",
    "Communal land (traditional) " = "Other government institution (not municipality)"
)

# Attendance
df$pri_attendance <- fct_recode(
    df$pri_attendance,
    "not" = "Three times a week",
    "not" = "Once a week",
    "not" = "Two times a week"
)

# Facilities
df$pri_facilities <- fct_recode(
    df$pri_facilities,
    "Community Centre" = "Religious institution (e.g. church, mosque)",
    "Community Centre" = "Municipality",
    "Another private individual" = "Private business",
    "Another private individual" = "Other government institution (not municipality)",
    "Another private individual" = "Not-for profit organisation",
    "not" = "The person in charge of the programme (e.g. principal, matron, child-minder, playgroup leader)"
)

# Support DSD
df$pri_support_dsd <- fct_recode(df$pri_support_dsd,
    "not" = "Twice"
)

# Registered partial
df$pri_registered_partial <- fct_recode(df$pri_registered_partial,
    "Conditionally registered" = "Lapsed registration"
)

# Staff employed
df$pri_staff_employed <- fct_recode(
    df$pri_staff_employed,
    "not" = "I have more than half of the number of staff employed compared to previous years",
    "not" = "I have less than half of the number of staff employed compared to previous years"
)

# Parents frequency
df$pri_parents_frequency <- fct_recode(df$pri_parents_frequency,
    "not" = "Quarterly"
)

# Support frequency
df$pri_support_frequency <- fct_recode(
    df$pri_support_frequency,
    "Once a term" = "Once a year",
    "not" = "Once a week"
)

# Teacher social cooperate
df$teacher_social_cooperate <- fct_recode(df$teacher_social_cooperate,
    "Most of the time" = "not"
)

# Food from parents
df$pri_food_parents_morning <- fct_recode(df$pri_food_parents_morning,
    "not" = "None"
)

message("✓ Factor consolidation complete!")

# ------------------------------------------------------------------------------
# 2. FINAL FEATURE SELECTION (DROP LOW-INFORMATION VARIABLES)
# ------------------------------------------------------------------------------

to_drop_final <- c(
    "pri_parents_frequency",
    "pri_languages",
    "pri_staff_employed",
    "pri_fees_paid_proportion",
    "pri_food_parents_morning"
)

df <- df %>% select(-all_of(to_drop_final))

message(sprintf("Final dimensions: %d rows x %d columns", nrow(df), ncol(df)))

# Remove temporary class variable if it exists
if ("class" %in% names(df)) {
    df$class <- NULL
}

# Trim whitespace from factors
df[, sapply(df, is.factor)] <- apply(df[, sapply(df, is.factor)], 2, trimws)

# Convert remaining characters to factors
df[, sapply(df, is.character)] <- lapply(df[, sapply(df, is.character)], as.factor)

# Save recoded data
saveRDS(df, "recoded_data.rds")
message("✓ Recoded data saved!")
