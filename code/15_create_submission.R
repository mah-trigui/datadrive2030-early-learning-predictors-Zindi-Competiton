# ==============================================================================
# CLEAN SCRIPT 15: CREATE FINAL SUBMISSION
# ==============================================================================
# This script creates the final submission file from ensemble predictions
# ==============================================================================

library(tidyverse)

# Load ensemble predictions
y_5 <- readRDS("ensemble_test_age5.rds")
y_6 <- readRDS("ensemble_test_age6.rds")

message("Creating final submission file...")

# ==============================================================================
# 1. COMBINE AGE GROUP PREDICTIONS
# ==============================================================================

# Select child_id and the best-performing ensemble strategy
# Based on freeze set evaluation, we'll use min_max ensemble
# (you can change this based on your freeze set results)

submission_5 <- y_5[, c("child_id", "min_max")]
submission_6 <- y_6[, c("child_id", "min_max")]

# Combine both age groups
submission <- rbind(submission_5, submission_6)

# Rename prediction column to 'target'
names(submission)[2] <- "target"

# ==============================================================================
# 2. ADD REQUIRED FEATURE COLUMNS (AS PER SAMPLE SUBMISSION)
# ==============================================================================

# The competition requires specific feature columns in the submission
# These indicate which features were most important
submission$feature_1 <- "child_observe_total"
submission$feature_2 <- "id_team"
submission$feature_3 <- "child_age"
submission$feature_4 <- "teacher_emotional_total"
submission$feature_5 <- "pri_amount_funding_fees"
submission$feature_6 <- "pri_fees_amount_4_6"
submission$feature_7 <- "child_observe_diligent"
submission$feature_8 <- "pri_expense_staff"
submission$feature_9 <- "id_mn_n"
submission$feature_10 <- "opening_hours"
submission$feature_11 <- "child_observe_attentive"
submission$feature_12 <- "pri_expense_maintenance"
submission$feature_13 <- "id_prov"
submission$feature_14 <- "prov_best"
submission$feature_15 <- "pri_expense_admin"

# ==============================================================================
# 3. SAVE SUBMISSION FILE
# ==============================================================================

write.csv(submission, "final_submission.csv", row.names = FALSE, quote = FALSE)

message(sprintf("✓ Submission file created with %d predictions", nrow(submission)))
message("  File: final_submission.csv")
message(sprintf("  Age Group 5: %d predictions", nrow(submission_5)))
message(sprintf("  Age Group 6: %d predictions", nrow(submission_6)))

# ==============================================================================
# 4. CREATE ALTERNATIVE SUBMISSIONS
# ==============================================================================

message("\nCreating alternative submissions...")

# Mean ensemble submission
submission_mean <- rbind(
    y_5[, c("child_id", "mean")],
    y_6[, c("child_id", "mean")]
)
names(submission_mean)[2] <- "target"
submission_mean$feature_1 <- "child_observe_total"
submission_mean$feature_2 <- "id_team"
submission_mean$feature_3 <- "child_age"
submission_mean$feature_4 <- "teacher_emotional_total"
submission_mean$feature_5 <- "pri_amount_funding_fees"
submission_mean$feature_6 <- "pri_fees_amount_4_6"
submission_mean$feature_7 <- "child_observe_diligent"
submission_mean$feature_8 <- "pri_expense_staff"
submission_mean$feature_9 <- "id_mn_n"
submission_mean$feature_10 <- "opening_hours"
submission_mean$feature_11 <- "child_observe_attentive"
submission_mean$feature_12 <- "pri_expense_maintenance"
submission_mean$feature_13 <- "id_prov"
submission_mean$feature_14 <- "prov_best"
submission_mean$feature_15 <- "pri_expense_admin"

write.csv(submission_mean, "submission_mean.csv", row.names = FALSE, quote = FALSE)

# Mean-Max ensemble submission
submission_mean_max <- rbind(
    y_5[, c("child_id", "mean_max")],
    y_6[, c("child_id", "mean_max")]
)
names(submission_mean_max)[2] <- "target"
submission_mean_max$feature_1 <- "child_observe_total"
submission_mean_max$feature_2 <- "id_team"
submission_mean_max$feature_3 <- "child_age"
submission_mean_max$feature_4 <- "teacher_emotional_total"
submission_mean_max$feature_5 <- "pri_amount_funding_fees"
submission_mean_max$feature_6 <- "pri_fees_amount_4_6"
submission_mean_max$feature_7 <- "child_observe_diligent"
submission_mean_max$feature_8 <- "pri_expense_staff"
submission_mean_max$feature_9 <- "id_mn_n"
submission_mean_max$feature_10 <- "opening_hours"
submission_mean_max$feature_11 <- "child_observe_attentive"
submission_mean_max$feature_12 <- "pri_expense_maintenance"
submission_mean_max$feature_13 <- "id_prov"
submission_mean_max$feature_14 <- "prov_best"
submission_mean_max$feature_15 <- "pri_expense_admin"

write.csv(submission_mean_max, "submission_mean_max.csv", row.names = FALSE, quote = FALSE)

message("\n✓ Alternative submissions created:")
message("  - submission_mean.csv (simple average)")
message("  - submission_mean_max.csv (mean-max hybrid)")

# ==============================================================================
# 5. DISPLAY SAMPLE PREDICTIONS
# ==============================================================================

message("\nSample predictions (first 10 rows):")
print(head(submission, 10))

message("\n=== ALL DONE! ===")
message("All submission files ready for upload!")
