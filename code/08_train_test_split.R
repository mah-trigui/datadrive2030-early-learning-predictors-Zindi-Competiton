# ==============================================================================
# CLEAN SCRIPT 08: TRAIN-TEST-FREEZE SPLIT
# ==============================================================================
# This script splits data by age group and creates train/freeze/test sets
# ==============================================================================

library(tidyverse)

# Load clustered data
df <- readRDS("clustered_data.rds")

# Load original data with IDs
df_id <- readRDS("preprocessed_data_with_ids.rds")

message("Creating train-test-freeze splits...")

# ------------------------------------------------------------------------------
# 1. SEPARATE BY AGE GROUP
# ------------------------------------------------------------------------------

# Age group 5 (50-59 months or younger)
df_age5 <- df[df$child_age_group == 1, ]

# Age group 6 (60+ months)
df_age6 <- df[df$child_age_group == 0, ]

# Remove age_group column as it's constant within each group
df_age5$child_age_group <- NULL
df_age6$child_age_group <- NULL

message(sprintf("Age Group 5: %d observations", nrow(df_age5)))
message(sprintf("Age Group 6: %d observations", nrow(df_age6)))

# ------------------------------------------------------------------------------
# 2. CREATE AGE GROUP 5 SPLITS
# ------------------------------------------------------------------------------

message("\n=== Age Group 5 Split ===")

# Separate train and test
test_5 <- df_age5[is.na(df_age5$target), ]
train_5 <- df_age5[!is.na(df_age5$target), ]

# Create freeze set (500 random observations from train)
set.seed(1618)
freeze_5 <- train_5[sample(nrow(train_5), 500), ]

# Remove freeze observations from train
train_5 <- train_5 %>% anti_join(freeze_5)

# Shuffle training data
train_5 <- train_5[sample(nrow(train_5), nrow(train_5)), ]

message(sprintf("  Train: %d", nrow(train_5)))
message(sprintf("  Freeze (validation): %d", nrow(freeze_5)))
message(sprintf("  Test: %d", nrow(test_5)))

# Get IDs for test set
id_test_5 <- df_id[is.na(df_id$target), c("child_id", "child_age_group")]
id_test_5 <- id_test_5[id_test_5$child_age_group == 1, ]

# ------------------------------------------------------------------------------
# 3. CREATE AGE GROUP 6 SPLITS
# ------------------------------------------------------------------------------

message("\n=== Age Group 6 Split ===")

# Separate train and test
test_6 <- df_age6[is.na(df_age6$target), ]
train_6 <- df_age6[!is.na(df_age6$target), ]

# Create freeze set (500 random observations from train)
set.seed(1618)
freeze_6 <- train_6[sample(nrow(train_6), 500), ]

# Remove freeze observations from train
train_6 <- train_6 %>% anti_join(freeze_6)

# Shuffle training data
train_6 <- train_6[sample(nrow(train_6), nrow(train_6)), ]

message(sprintf("  Train: %d", nrow(train_6)))
message(sprintf("  Freeze (validation): %d", nrow(freeze_6)))
message(sprintf("  Test: %d", nrow(test_6)))

# Get IDs for test set
id_test_6 <- df_id[is.na(df_id$target), c("child_id", "child_age_group")]
id_test_6 <- id_test_6[id_test_6$child_age_group == 0, ]

# ------------------------------------------------------------------------------
# 4. CREATE COMBINED DATASET (NO AGE GROUP SPLIT)
# ------------------------------------------------------------------------------

message("\n=== Combined Dataset (No Age Group Split) ===")

# Separate train and test
test_combined <- df[is.na(df$target), ]
train_combined <- df[!is.na(df$target), ]

# Create freeze set (500 random observations from train)
set.seed(1618)
freeze_combined <- train_combined[sample(nrow(train_combined), 500), ]

# Remove freeze observations from train
train_combined <- train_combined %>% anti_join(freeze_combined)

# Shuffle training data
train_combined <- train_combined[sample(nrow(train_combined), nrow(train_combined)), ]

message(sprintf("  Train: %d", nrow(train_combined)))
message(sprintf("  Freeze (validation): %d", nrow(freeze_combined)))
message(sprintf("  Test: %d", nrow(test_combined)))

# Get IDs for test set
id_test_combined <- df_id[is.na(df_id$target), "child_id", drop = FALSE]

# ------------------------------------------------------------------------------
# 5. SAVE ALL SPLITS
# ------------------------------------------------------------------------------

# Age Group 5
saveRDS(train_5, "train_age5.rds")
saveRDS(freeze_5, "freeze_age5.rds")
saveRDS(test_5, "test_age5.rds")
saveRDS(id_test_5, "id_test_age5.rds")

# Age Group 6
saveRDS(train_6, "train_age6.rds")
saveRDS(freeze_6, "freeze_age6.rds")
saveRDS(test_6, "test_age6.rds")
saveRDS(id_test_6, "id_test_age6.rds")

# Combined
saveRDS(train_combined, "train_combined.rds")
saveRDS(freeze_combined, "freeze_combined.rds")
saveRDS(test_combined, "test_combined.rds")
saveRDS(id_test_combined, "id_test_combined.rds")

message("\n✓ All data splits saved!")
message("\nSaved files:")
message("  Age Group 5: train_age5.rds, freeze_age5.rds, test_age5.rds")
message("  Age Group 6: train_age6.rds, freeze_age6.rds, test_age6.rds")
message("  Combined: train_combined.rds, freeze_combined.rds, test_combined.rds")
