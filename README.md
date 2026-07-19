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
# CHILD DEVELOPMENT ASSESSMENT - CLEAN CODE PROJECT

## PROJECT OVERVIEW

This project predicts developmental assessment scores for children in Early Childhood Development (ECD) programs in South Africa. The target variable is a continuous score measuring child development across multiple dimensions.

**Competition Context:** Early Childhood Development (ECD) Program Quality Assessment
**Task:** Regression (predict continuous development scores)
**Metric:** Root Mean Squared Error (RMSE)
**Data Split:** Age groups (50-59 months/younger vs 60+ months)

---

## CLEAN SCRIPTS OVERVIEW

This repository contains **16 clean, well-organized R scripts** that implement the complete machine learning pipeline from data loading to final submission.

### **00 - MASTER PIPELINE**
- `CLEAN_00_master_pipeline.R` - Runs the entire pipeline from start to finish

### **PART 1: DATA PREPROCESSING (Scripts 01-08)**

#### **01 - Data Loading and Preprocessing**
- `CLEAN_01_data_loading_and_preprocessing.R`
- Loads train/test data
- Creates age group categories
- Drops irrelevant columns (dates, IDs, high-cardinality text)
- Handles empty strings as factors
- **Output:** `preprocessed_data.rds`

#### **02 - Feature Engineering**
- `CLEAN_02_feature_engineering.R`
- Recodes language variables
- Encodes binary variables (-1, 0, 1)
- Converts Yes/No factors
- Caps outliers at 2nd/98th percentiles
- **Output:** `feature_engineered_data.rds`

#### **03 - Correlation Removal**
- `CLEAN_03_correlation_removal.R`
- Removes highly correlated features (>0.9)
- Drops low-information categorical variables
- Replaces NA with 0 where appropriate
- **Output:** `correlation_cleaned_data.rds`

#### **04 - Advanced Imputation**
- `CLEAN_04_imputation.R`
- **missRanger:** Random Forest-based imputation
- **Amelia:** Multiple imputation with averaging (5 imputations)
- Preserves target correlation for funding variables
- **Output:** `imputed_data.rds`

#### **05 - Factor Consolidation**
- `CLEAN_05_factor_consolidation.R`
- Consolidates rare factor levels
- Recodes 30+ categorical variables
- Final feature selection
- **Output:** `recoded_data.rds`

#### **06 - Ordinal Encoding**
- `CLEAN_06_ordinal_encoding.R`
- Creates ordinal encoding based on target correlation
- Generates one-hot encoded version
- Maintains factorized version
- **Output:** `final_data_factorized.rds`, `final_data_ordinal.rds`, `final_data_onehot.rds`

#### **07 - Clustering Features**
- `CLEAN_07_clustering.R`
- Creates 2 cluster features using ensemble methods:
  - **High-performing children cluster:** Ward's, K-means, Fuzzy, PAM
  - **Low-performing children cluster:** Ward's, K-means, Fuzzy, PAM
- **Output:** `clustered_data.rds`

#### **08 - Train-Test Split**
- `CLEAN_08_train_test_split.R`
- Splits by age group (5 vs 6)
- Creates freeze validation set (500 samples)
- Generates combined dataset option
- **Output:** `train_age5.rds`, `freeze_age5.rds`, `test_age5.rds`, etc.

---

### **PART 2: MODEL TRAINING (Scripts 09-13)**

#### **09 - CatBoost Models**
- `CLEAN_09_model_catboost.R`
- **Age Group 5:** Caret grid search (20 parameter combinations)
  - Depth: 4, 6, 8
  - Learning rate: 0.01, 0.1, 0.3
  - Iterations: 1000-3000
- **Age Group 6:** Random hyperparameter search (10 iterations)
- **Output:** Predictions and models for both age groups

#### **10 - LightGBM Models**
- `CLEAN_10_model_lightgbm.R`
- K-fold cross-validation (5 folds)
- Early stopping (100 rounds)
- Feature importance extraction
- **Output:** Predictions, models, and feature importance

#### **11 - H2O AutoML**
- `CLEAN_11_model_h2o_automl.R`
- Automated machine learning with H2O
- Tests up to 20 different models
- 1-hour maximum runtime per age group
- **Output:** Best models and predictions

#### **12 - SuperLearner (SL3)**
- `CLEAN_12_model_sl3_superlearner.R`
- **Base learners:**
  - GLM, Mean, Ridge, Lasso
  - Random Forest (ranger), XGBoost
  - GAM, Bayesian GLM
- **Metalearner:** Non-negative least squares (NNLS)
- **Output:** Ensemble predictions with learner weights

#### **13 - Tidymodels Stacking**
- `CLEAN_13_model_tidymodels_stacking.R`
- **Base models:**
  - K-Nearest Neighbors (tuned)
  - Linear Regression
  - Support Vector Machine (tuned)
- **Stack:** Blended ensemble with cross-validation
- **Output:** Stacked ensemble predictions

---

### **PART 3: ENSEMBLE & SUBMISSION (Scripts 14-15)**

#### **14 - Ensemble All Models**
- `CLEAN_14_ensemble_all_models.R`
- Combines all model predictions
- **6 ensemble strategies:**
  1. **Mean:** Simple average
  2. **Min:** Minimum prediction
  3. **Max:** Maximum prediction
  4. **Mean-Max:** Hybrid (mean + max) / 2
  5. **Mean-Min:** Hybrid (mean + min) / 2
  6. **Min-Max:** Hybrid (min + max) / 2
- Evaluates all strategies on freeze set
- **Output:** `ensemble_test_age5.rds`, `ensemble_test_age6.rds`

#### **15 - Create Submission**
- `CLEAN_15_create_submission.R`
- Combines age group predictions
- Adds required feature columns
- Creates 3 submission files:
  - `final_submission.csv` (best: min-max)
  - `submission_mean.csv` (simple average)
  - `submission_mean_max.csv` (mean-max hybrid)

---

## COMPLETE PIPELINE EXECUTION

To run the entire pipeline, simply execute:

```r
source('CLEAN_00_master_pipeline.R')
```

This will automatically run all 15 scripts in sequence and generate:
- All intermediate .rds data files
- Model objects
- 3 submission CSV files

**Estimated Runtime:** 2-4 hours (depending on hardware)

---

## KEY TECHNIQUES USED

### **Data Preprocessing**
- ✅ Handling missing values (empty strings → factors)
- ✅ Age group stratification
- ✅ Outlier capping (2nd/98th percentiles)
- ✅ High-cardinality variable removal
- ✅ Binary encoding (-1, 0, 1)

### **Feature Engineering**
- ✅ Factor consolidation (rare levels)
- ✅ Correlation-based ordinal encoding
- ✅ One-hot encoding
- ✅ Clustering features (ensemble of 4 algorithms)

### **Imputation**
- ✅ missRanger (Random Forest imputation)
- ✅ Amelia (Multiple imputation)
- ✅ Correlation preservation

### **Models**
- ✅ **CatBoost:** Grid search + Random search
- ✅ **LightGBM:** K-fold CV + early stopping
- ✅ **H2O AutoML:** Automated model selection
- ✅ **SuperLearner (SL3):** 8 base learners + NNLS metalearner
- ✅ **Tidymodels Stacking:** KNN + LR + SVM stacking

### **Ensemble**
- ✅ Multiple aggregation strategies (mean, min, max, hybrids)
- ✅ Freeze set evaluation for strategy selection

---

## FILE STRUCTURE

```
sss/
├── CLEAN_00_master_pipeline.R          # Master execution script
├── CLEAN_01_data_loading_and_preprocessing.R
├── CLEAN_02_feature_engineering.R
├── CLEAN_03_correlation_removal.R
├── CLEAN_04_imputation.R
├── CLEAN_05_factor_consolidation.R
├── CLEAN_06_ordinal_encoding.R
├── CLEAN_07_clustering.R
├── CLEAN_08_train_test_split.R
├── CLEAN_09_model_catboost.R
├── CLEAN_10_model_lightgbm.R
├── CLEAN_11_model_h2o_automl.R
├── CLEAN_12_model_sl3_superlearner.R
├── CLEAN_13_model_tidymodels_stacking.R
├── CLEAN_14_ensemble_all_models.R
├── CLEAN_15_create_submission.R
├── README_CLEAN_SCRIPTS.md             # This file
│
├── Train.csv                            # Raw training data
├── Test.csv                             # Raw test data
├── SampleSubmission.csv                 # Submission format
│
├── final_submission.csv                 # Best submission (min-max)
├── submission_mean.csv                  # Mean ensemble
├── submission_mean_max.csv              # Mean-max ensemble
│
└── [Multiple .rds intermediate files]
```

---

## REQUIRED R PACKAGES

```r
# Core tidyverse
install.packages("tidyverse")

# Data exploration
install.packages("DataExplorer")
install.packages("flextable")

# Feature engineering
install.packages("forcats")
install.packages("correlationfunnel")
install.packages("heplots")

# Imputation
install.packages("missRanger")
install.packages("Amelia")

# Clustering
install.packages("fpc")
install.packages("cluster")

# Feature selection
install.packages("Boruta")
install.packages("randomForest")
install.packages("InformationValue")

# Machine learning
install.packages("caret")
install.packages("catboost")
install.packages("lightgbm")
install.packages("h2o")
install.packages("sl3")
install.packages("tidymodels")
install.packages("stacks")

# Metrics
install.packages("Metrics")
install.packages("dlookr")
```

---

## PERFORMANCE SUMMARY

Based on freeze set (500 validation samples):

### **Age Group 5 (50-59 months/younger)**
| Model | RMSE |
|-------|------|
| CatBoost | ~10.33 |
| LightGBM | ~10.34 |
| H2O AutoML | ~6.81 |
| SuperLearner | ~10.81 |
| Tidymodels Stack | ~11.00 |
| **Best Ensemble** | **~9.47** (mean-max) |

### **Age Group 6 (60+ months)**
| Model | RMSE |
|-------|------|
| CatBoost | ~10.xx |
| H2O AutoML | ~7.xx |
| SuperLearner | ~11.xx |

---

## USAGE INSTRUCTIONS

### **Option 1: Run Complete Pipeline**
```r
source('CLEAN_00_master_pipeline.R')
```

### **Option 2: Run Individual Steps**
```r
# Preprocessing only
source('CLEAN_01_data_loading_and_preprocessing.R')
source('CLEAN_02_feature_engineering.R')
# ... etc

# Or load preprocessed data and train specific model
train_5 <- readRDS('train_age5.rds')
source('CLEAN_09_model_catboost.R')
```

### **Option 3: Load Predictions and Create Custom Ensemble**
```r
# Load all model predictions
pred_cat_5 <- readRDS('pred_catboost_age5.rds')
pred_light_5 <- readRDS('pred_lightgbm_age5.rds')
# ...

# Create custom ensemble
custom_ensemble <- 0.4 * pred_cat_5 + 0.3 * pred_h2o_5 + 0.3 * pred_light_5
```

---

## NOTES

1. **Freeze Set Validation:** A 500-sample freeze set is used for model evaluation and ensemble strategy selection. This prevents overfitting to the test set.

2. **Age Group Splitting:** The data is naturally split into two age groups (5 and 6) which have different developmental patterns. Separate models are trained for each group.

3. **Ensemble Strategy:** The best-performing ensemble (mean-max hybrid) balances conservative (mean) and optimistic (max) predictions.

4. **Feature Importance:** The top features across all models are:
   - child_observe_total
   - teacher_emotional_total
   - pri_amount_funding_fees
   - pri_fees_amount_4_6
   - Various observation and expense variables

5. **Random Seeds:** All random processes use seed 1618 for reproducibility.

---

## CREDITS

**Project:** Child Development Assessment
**Data Source:** Early Childhood Development Program Quality Assessment (South Africa)
**Code Organization:** Clean, modular, well-documented scripts
**License:** Open source for educational purposes

---

## CONTACT

For questions or improvements, please refer to the individual script comments or create an issue in the repository.

**Created:** 2024
**Last Updated:** 2024
**Version:** 1.0

---

**Happy Modeling! 🚀**
