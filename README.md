# DataDrive2030 Early Learning Predictors Challenge

This competition is hosted on Zindi, a machine learning platform for data science challenges.  
Here is the link to the competition: [DataDrive2030 Early Learning Predictors Challenge 🌾 - $3 000](https://zindi.africa/competitions/datadrive2030-early-learning-predictors-challenge)

Ranked in the TOP 62%
---

## Challenge Overview

**Objective:** Identify which features of early learning programmes predict better learning outcomes for children.

**Dataset:** Child development assessment data with household characteristics, programme features, and learning outcome scores.

**Models:** Multi-model ensemble combining:
- **CatBoost** (gradient boosting)
- **LightGBM** (light gradient boosting)
- **H2O AutoML** (automated machine learning)
- **SuperLearner/SL3** (meta-learner ensemble)
- **Tidymodels Stacking** (stacked generalization)

---

## Pipeline Architecture

```
MAIN.R                                    ← Entry point
└── 00_master_pipeline.R                  ← Orchestration
    ├── 01_data_loading_and_preprocessing.R    (Load, clean, basic features)
    ├── 02_feature_engineering.R               (Derived features, scaling)
    ├── 03_correlation_removal.R              (Remove multicollinearity)
    ├── 04_imputation.R                       (Handle missing values)
    ├── 05_factor_consolidation.R             (Reduce factor levels)
    ├── 06_ordinal_encoding.R                 (Encode categorical variables)
    ├── 07_clustering.R                       (Feature clustering, dimensionality reduction)
    ├── 08_train_test_split.R                 (Stratified train/validation/test)
    ├── 09_model_catboost.R                   (Train CatBoost models)
    ├── 10_model_lightgbm.R                   (Train LightGBM models)
    ├── 11_model_h2o_automl.R                 (Train H2O AutoML models)
    ├── 12_model_sl3_superlearner.R           (Train SuperLearner ensemble)
    ├── 13_model_tidymodels_stacking.R        (Train stacked ensemble)
    ├── 14_ensemble_all_models.R              (Combine all model predictions)
    └── 15_create_submission.R                (Format & save final predictions)
```

---

## Quick Start

### Run Full Pipeline:
```r
setwd("C:/path/to/DigiCow Farmer Training Adoption")
source("MAIN.R")
```

This will:
1. Load and preprocess data
2. Engineer features
3. Train 5+ model types
4. Ensemble predictions
5. Generate submission files

**Estimated time:** 30-60 minutes (depends on hardware)

---

## Pipeline Stages

### **01: Data Loading & Preprocessing**
- Load `Train.csv` and `Test.csv`
- Parse data types
- Handle missing values
- Extract basic features (age, education level, etc.)
- Output: Clean dataset per age group

### **02: Feature Engineering**
- Create interaction features
- Scale/normalize numeric variables
- Derive aggregated features
- Output: Enhanced feature set

### **03: Correlation Removal**
- Calculate feature correlations
- Remove highly correlated pairs (threshold: 0.95)
- Reduce multicollinearity
- Output: Subset of uncorrelated features

### **04: Imputation**
- Impute remaining missing values (KNN, mode, median)
- Validate imputation quality
- Output: Complete dataset

### **05: Factor Consolidation**
- Reduce categorical factor levels
- Group rare categories
- Balance factor distributions
- Output: Consolidated factors

### **06: Ordinal Encoding**
- Encode ordinal variables (e.g., education: primary → secondary → tertiary)
- One-hot encode nominal categories
- Output: Encoded features ready for models

### **07: Clustering**
- Cluster similar features
- Apply dimensionality reduction (PCA, UMAP)
- Select representative features
- Output: Reduced feature set

### **08: Train-Test Split**
- Stratified split by age group
- 70% training, 15% validation, 15% test
- Ensure balanced outcome distribution
- Output: `train_data.rds`, `val_data.rds`, `test_data.rds`

### **09-13: Model Training**
Train 5 model types with hyperparameter tuning:

| Model | Algorithm | Hyperparameters | CV |
|-------|-----------|-----------------|-----|
| **CatBoost** | Gradient Boosting | iterations=1000, depth=6-8 | 5-fold |
| **LightGBM** | Light GB | num_leaves=31, learning_rate=0.05 | 5-fold |
| **H2O AutoML** | Auto-ML | time_limit=3600s, nfolds=5 | Built-in |
| **SuperLearner** | Meta-learner | library=[xgboost, glm, rf] | CV |
| **Tidymodels Stack** | Stacking | base learners + meta-learner | Nested CV |

**Output:** Trained models + feature importance rankings

### **14: Ensemble All Models**
- Combine predictions from all models
- Three ensemble strategies:
  1. **Simple Mean:** Average of all predictions
  2. **Weighted Mean:** Weights by model performance
  3. **Stacked:** Use meta-learner to combine

**Output:** `ensemble_predictions.rds`

### **15: Create Submission**
- Format predictions for competition
- Generate 3 submission files:
  - `final_submission.csv` (best ensemble)
  - `submission_mean.csv` (simple mean)
  - `submission_mean_max.csv` (mean-max hybrid)

---

## Output Files

### Model Artifacts:
```
models/
├── catboost_age5.rds
├── catboost_age6.rds
├── lightgbm_age5.rds
├── lightgbm_age6.rds
├── h2o_model_age5
├── h2o_model_age6
├── sl3_learner_age5.rds
├── tidymodels_stack_age5.rds
└── ... (for age 6)
```

### Data Artifacts:
```
processed_data/
├── train_data.rds
├── val_data.rds
├── test_data.rds
├── feature_info.rds
└── imputation_params.rds
```

### Predictions:
```
submissions/
├── final_submission.csv          (Best ensemble)
├── submission_mean.csv           (Simple mean)
└── submission_mean_max.csv       (Mean-max)
```

### Evaluation:
```
evaluation/
├── model_performance.csv         (Per-model RMSE, MAE, R²)
├── feature_importance.csv        (Feature rankings by model)
├── confusion_matrices.rds        (Classification metrics if applicable)
└── ensemble_performance.csv      (Ensemble metrics)
```

---

## Requirements

### R Version:
- R ≥ 4.0.0

### Key Packages:
```r
install.packages(c(
  # Data manipulation
  "tidyverse", "data.table", "dplyr",

  # ML Models
  "catboost", "lightgbm", "h2o", "sl3", "tidymodels",
  "caret", "xgboost", "ranger",

  # Preprocessing
  "recipes", "themis", "embed",

  # Ensemble
  "caretEnsemble", "stacks",

  # Utilities
  "tidyselect", "forcats", "lubridate",
  "rlang", "Matrix"
))
```

### Optional (for faster computation):
```r
install.packages(c("doParallel", "parallel"))
```

---

## Feature Importance Analysis

After training, identify predictive features:

```r
# Load feature importance across models
feature_importance <- read.csv("evaluation/feature_importance.csv")

# Top 10 predictive features
top_features <- feature_importance %>%
  group_by(feature) %>%
  summarise(avg_importance = mean(importance)) %>%
  arrange(desc(avg_importance)) %>%
  slice(1:10)

print(top_features)
```

---

## Hyperparameter Tuning

Default hyperparameters are set for balance between speed and accuracy. To customize:

1. **CatBoost:** Edit `09_model_catboost.R` → `cat_params`
2. **LightGBM:** Edit `10_model_lightgbm.R` → `lgb_params`
3. **H2O AutoML:** Edit `11_model_h2o_automl.R` → `h2o_params`
4. **SuperLearner:** Edit `12_model_sl3_superlearner.R` → `sl_library`
5. **Tidymodels Stack:** Edit `13_model_tidymodels_stacking.R` → `stack_recipe`

---

## Running Individual Stages

To debug or customize, run stages individually:

```r
# Set working directory
setwd("C:/path/to/project")

# Run individual scripts
source("01_data_loading_and_preprocessing.R")
source("02_feature_engineering.R")
source("03_correlation_removal.R")
# ... and so on
```

---

## Troubleshooting

### Memory Issues:
- Reduce `train_sample_size` in configuration files
- Use `gc()` to free memory between steps

### Slow Training:
- Reduce `nfolds` in cross-validation (default: 5)
- Disable H2O AutoML: comment out `source("11_model_h2o_automl.R")`
- Use parallel processing (enable in config files)

### Missing Dependencies:
```r
# Install all required packages
source("install_requirements.R")
```

---

## Output Interpretation

### Final Submission Format:
```
id                                      | prediction
Child_Score_AgeFiveYears_row_1          | 65.3
Child_Score_AgeFiveYears_row_2          | 72.1
Child_Score_AgeSixYears_row_1           | 78.5
...
```

### Model Performance Metrics:
- **RMSE:** Root Mean Squared Error (lower is better)
- **MAE:** Mean Absolute Error
- **R²:** Coefficient of Determination (higher is better, max = 1.0)
- **MAPE:** Mean Absolute Percentage Error

---

## Notes

- **Reproducibility:** Seeds set (1618) but ensemble predictions may vary slightly across runs
- **Computation time:** ~45 minutes on 8-core machine with 16GB RAM
- **Feature scaling:** Applied in preprocessing; impacts some models (e.g., SL3)
- **Class imbalance:** Handled via stratified sampling; monitor in evaluation metrics

---

## File Organization Summary

**Production Scripts (15 clean files):**
```
00_master_pipeline.R                      ← Run via MAIN.R
01_data_loading_and_preprocessing.R
02_feature_engineering.R
03_correlation_removal.R
04_imputation.R
05_factor_consolidation.R
06_ordinal_encoding.R
07_clustering.R
08_train_test_split.R
09_model_catboost.R
10_model_lightgbm.R
11_model_h2o_automl.R
12_model_sl3_superlearner.R
13_model_tidymodels_stacking.R
14_ensemble_all_models.R
15_create_submission.R
```

**Entry Point:**
```
MAIN.R                                    ← Start here
```

**Separate Project (keep isolated):**
```
Energy_*.R                                ← Different competition
```

---

**Last Updated:** July 2026
**Challenge Platform:** DataDrive2030
**Focus Area:** Early Learning Programme Effectiveness Prediction
