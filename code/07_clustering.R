# ==============================================================================
# CLEAN SCRIPT 07: CLUSTERING FEATURE CREATION
# ==============================================================================
# This script creates cluster-based features using multiple algorithms:
# - Hierarchical Clustering (Ward's method)
# - K-means
# - Fuzzy clustering (fanny)
# - PAM (Partition Around Medoids)
# ==============================================================================

library(tidyverse)
library(correlationfunnel)
library(cluster)
library(fpc)

# Load final factorized data
df <- readRDS("final_data_factorized.rds")

message("Creating clustering features...")

# ------------------------------------------------------------------------------
# 1. CLUSTERING ON "HIGH-PERFORMING" CHILDREN FEATURES
# ------------------------------------------------------------------------------

message("\n=== Clustering 1: High-Performing Children ===")

cluster_vars_high <- c(
    "child_observe_total", "pri_registered_npo", "id_prov",
    "certificate_registration_npo", "language_assessment", "ses_cat",
    "prov_best", "child_grant", "ses_proxy", "count_register_race_white",
    "child_observe_diligent", "child_observe_interested",
    "child_observe_attentive", "child_observe_concentrated",
    "pri_amount_funding_fees", "pri_fees_amount_pv",
    "teacher_emotional_total"
)

# Binarize and select specific bins
aux_high <- df %>%
    select(all_of(cluster_vars_high)) %>%
    binarize() %>%
    select(
        child_observe_total__9.39536666666666_Inf,
        `pri_registered_npo__-1`, pri_registered_npo__1,
        teacher_emotional_total__11_Inf,
        pri_fees_amount_pv__373.65045_Inf,
        pri_amount_funding_fees__32254.3115102199_Inf,
        child_observe_interested__Almost_always,
        child_observe_attentive__Almost_always,
        child_observe_concentrated__Almost_always,
        child_observe_diligent__Almost_always,
        count_register_race_white__17.3600000000006,
        ses_proxy__3_Inf, id_prov__WC, child_grant__No,
        prov_best__WESTERN_CAPE, `ses_cat__R1751+`,
        language_assessment__English,
        certificate_registration_npo__Yes
    )

# Compute distance matrix
d_high <- dist(aux_high, method = "binary")

# Hierarchical clustering
set.seed(1618)
hc_high <- hclust(d_high, method = "ward.D2")

# K-means clustering
set.seed(1618)
kmeans_high <- kmeans(aux_high, centers = 3)

# Fuzzy clustering
fa_high <- fanny(aux_high, 3, metric = "SqEuclidean")

# PAM clustering
set.seed(1618)
pam_high <- pamk(aux_high, krange = 3)

# Create ensemble cluster (majority vote)
cluster_df_high <- data.frame(
    hc = cutree(hc_high, k = 3),
    km = kmeans_high$cluster,
    fa = fa_high$clustering,
    pam = pam_high$pamobject$clustering
)

# Simple averaging approach
cluster_df_high$cluster_high <- ceiling(
    (cluster_df_high$hc + cluster_df_high$km +
        cluster_df_high$fa + cluster_df_high$pam) / 4
)

message(sprintf(
    "Created cluster_high with %d clusters",
    length(unique(cluster_df_high$cluster_high))
))

# ------------------------------------------------------------------------------
# 2. CLUSTERING ON "LOW-PERFORMING" CHILDREN FEATURES
# ------------------------------------------------------------------------------

message("\n=== Clustering 2: Low-Performing Children ===")

cluster_vars_low <- c(
    "teacher_emotional_understand", "teacher_emotional_independent",
    "teacher_emotional_selfstarter", "child_observe_interested",
    "child_observe_attentive", "child_observe_concentrated",
    "child_observe_diligent", "teacher_emotional_met",
    "teacher_emotional_total", "child_observe_total"
)

# Binarize and select specific bins
aux_low <- df %>%
    select(all_of(cluster_vars_low)) %>%
    binarize() %>%
    select(
        teacher_emotional_understand__Not_True,
        teacher_emotional_independent__Not_True,
        teacher_emotional_selfstarter__Not_True,
        child_observe_interested__Almost_always,
        child_observe_interested__Almost_never,
        child_observe_attentive__Almost_always,
        child_observe_attentive__Sometimes,
        child_observe_concentrated__Almost_always,
        child_observe_concentrated__Sometimes,
        child_observe_concentrated__Almost_never,
        child_observe_diligent__Almost_always,
        teacher_emotional_met__1,
        `teacher_emotional_total__-Inf_6.24469166666667`,
        child_observe_total__9.39536666666666_Inf
    )

# Compute distance matrix
d_low <- dist(aux_low, method = "binary")

# Hierarchical clustering
set.seed(1618)
hc_low <- hclust(d_low, method = "ward.D2")

# K-means clustering
set.seed(1618)
kmeans_low <- kmeans(aux_low, centers = 3)

# Fuzzy clustering
fa_low <- fanny(aux_low, 3, metric = "SqEuclidean")

# PAM clustering
set.seed(1618)
pam_low <- pamk(aux_low, krange = 3)

# Create ensemble cluster
cluster_df_low <- data.frame(
    hc = cutree(hc_low, k = 3),
    km = kmeans_low$cluster,
    fa = fa_low$clustering,
    pam = pam_low$pamobject$clustering
)

# Simple averaging approach
cluster_df_low$cluster_low <- ceiling(
    (cluster_df_low$hc + cluster_df_low$km +
        cluster_df_low$fa + cluster_df_low$pam) / 4
)

message(sprintf(
    "Created cluster_low with %d clusters",
    length(unique(cluster_df_low$cluster_low))
))

# ------------------------------------------------------------------------------
# 3. ADD CLUSTER FEATURES TO DATAFRAME
# ------------------------------------------------------------------------------

df$cluster_high <- cluster_df_high$cluster_high
df$cluster_low <- cluster_df_low$cluster_low

message("\n✓ Clustering complete!")
message(sprintf("Added 2 new cluster features to dataframe"))
message(sprintf("Final dimensions: %d rows x %d columns", nrow(df), ncol(df)))

# ------------------------------------------------------------------------------
# 4. ANALYZE CLUSTER PERFORMANCE
# ------------------------------------------------------------------------------

if (!all(is.na(df$target))) {
    message("\n=== Cluster Performance Analysis ===")

    # High-performing cluster analysis
    cluster_stats_high <- df %>%
        filter(!is.na(target)) %>%
        group_by(cluster_high) %>%
        summarise(
            count = n(),
            mean_target = mean(target),
            sd_target = sd(target),
            median_target = median(target),
            min_target = min(target),
            max_target = max(target)
        )

    message("\nCluster High Performance:")
    print(cluster_stats_high)

    # Low-performing cluster analysis
    cluster_stats_low <- df %>%
        filter(!is.na(target)) %>%
        group_by(cluster_low) %>%
        summarise(
            count = n(),
            mean_target = mean(target),
            sd_target = sd(target),
            median_target = median(target),
            min_target = min(target),
            max_target = max(target)
        )

    message("\nCluster Low Performance:")
    print(cluster_stats_low)
}

# ------------------------------------------------------------------------------
# 5. SAVE CLUSTERED DATA
# ------------------------------------------------------------------------------

saveRDS(df, "clustered_data.rds")
message("\n✓ Clustered data saved!")
