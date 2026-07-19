# =============================================================================
# MAIN.R - DataDrive2030 Early Learning Predictors Challenge Pipeline
# =============================================================================
# Entry point for the full machine learning pipeline
# Predicts which features of early learning programmes predict better outcomes
# =============================================================================

cat("
================================================================================
 DATADRIVE2030 EARLY LEARNING PREDICTORS CHALLENGE
================================================================================
 Challenge: Identify features that predict better child learning outcomes
 Approach: Multi-model ensemble with feature importance analysis
================================================================================
\n")

# Set working directory (adjust if needed)
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# Run the complete pipeline
source("00_master_pipeline.R")
