# ==============================================================================
# CLEAN SCRIPT 06: ORDINAL ENCODING BASED ON TARGET CORRELATION
# ==============================================================================
# This script creates ordinally-encoded versions of categorical variables
# based on their relationship with the target variable
# ==============================================================================

library(tidyverse)
library(correlationfunnel)

# Load recoded data
df <- readRDS("recoded_data.rds")

message("Creating ordinal encoding based on target correlation...")

# ------------------------------------------------------------------------------
# 1. CREATE BINARIZED CORRELATION ANALYSIS
# ------------------------------------------------------------------------------

# Get numeric column names
df_numeric <- names(df[, sapply(df, is.numeric)])

# Extract non-numeric columns and filter for training data
aux <- df %>%
    filter(!is.na(target)) %>%
    select(-all_of(df_numeric))

# Binarize categorical variables
set.seed(1618)
aux <- aux %>% binarize()

# Add target variable
aux$target <- df[!is.na(df$target), ]$target

# Compute correlations
aux <- aux %>% correlate(target = target)

# ------------------------------------------------------------------------------
# 2. CREATE ORDINAL MAPPING
# ------------------------------------------------------------------------------

# Sort by feature and correlation strength, assign IDs
categ_order <- aux %>%
    arrange(feature, abs(correlation)) %>%
    group_by(feature) %>%
    mutate(id = row_number()) %>%
    mutate(bin = gsub("_", " ", bin))

# ------------------------------------------------------------------------------
# 3. MAP ORIGINAL LEVELS TO ORDINAL IDS
# ------------------------------------------------------------------------------

# Get non-numeric columns
df_numeric <- names(df[, sapply(df, is.numeric)])
aux <- df %>% select(-all_of(df_numeric))

# Create reference levels mapping
ref_levels <- aux %>%
    select(where(is.factor)) %>%
    gather(key = "factor", value = "level") %>%
    separate(level, into = c("level", "value"), sep = ":", fill = "right") %>%
    select(factor, level) %>%
    unique() %>%
    rename(feature = factor, bin = level) %>%
    mutate(bin = trimws(bin)) %>%
    left_join(categ_order, by = c("feature", "bin")) %>%
    mutate(new_bin = ifelse(is.na(id), "-OTHER", bin)) %>%
    mutate(old_bin = bin, bin = new_bin) %>%
    select(feature, bin, old_bin) %>%
    left_join(categ_order, by = c("feature", "bin")) %>%
    select(-c(bin, correlation))

# ------------------------------------------------------------------------------
# 4. CREATE ORDINAL-ENCODED DATAFRAME
# ------------------------------------------------------------------------------

df_ord <- df

# Replace factor levels with ordinal IDs
for (i in which(sapply(df_ord, is.factor))) {
    df_ord[[i]] <- ref_levels$id[match(df_ord[[i]], ref_levels$old_bin)]
}

message("✓ Ordinal encoding complete!")
message(sprintf(
    "Created ordinal-encoded dataframe: %d rows x %d columns",
    nrow(df_ord), ncol(df_ord)
))

# ------------------------------------------------------------------------------
# 5. CREATE ONE-HOT ENCODED VERSION
# ------------------------------------------------------------------------------

# Get non-numeric columns
df_numeric <- names(df[, sapply(df, is.numeric)])
aux <- df %>% select(-all_of(df_numeric))

# One-hot encode (NOT including one_hot = TRUE for correlationfunnel compatibility)
df_hot <- aux %>% binarize(one_hot = FALSE)

# Add numeric columns back
df_hot <- cbind(df_hot, df %>% select(all_of(df_numeric)))

# Reorder columns (target last)
n_cols <- ncol(df_hot)
df_hot <- df_hot[, c(1:(n_cols - 1), n_cols)]

# Make valid column names
names(df_hot) <- make.names(names(df_hot))

message("✓ One-hot encoding complete!")
message(sprintf(
    "Created one-hot encoded dataframe: %d rows x %d columns",
    nrow(df_hot), ncol(df_hot)
))

# ------------------------------------------------------------------------------
# 6. SAVE ENCODED VERSIONS
# ------------------------------------------------------------------------------

saveRDS(df, "final_data_factorized.rds")
saveRDS(df_ord, "final_data_ordinal.rds")
saveRDS(df_hot, "final_data_onehot.rds")

message("✓ All encoding versions saved!")
message("  - final_data_factorized.rds (original factors)")
message("  - final_data_ordinal.rds (ordinal encoding)")
message("  - final_data_onehot.rds (one-hot encoding)")
