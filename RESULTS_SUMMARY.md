# NOAA Weather Prediction - Results Summary
## DSS5208 Project 2: Machine Learning on Weather Data

**Group Members**: [List names and student IDs]  
**Date**: [Submission date]  
**Course**: DSS5208

---

## Executive Summary

This project develops machine learning models to predict air temperature using NOAA global hourly weather observations. We processed 130M observations, extracted 14 features, and trained multiple regression models. The best model achieved a test RMSE of **[XX.XX]°C** with an R² of **[0.XX]**.

---

## 1. Dataset Overview

### 1.1 Original Data
- **Source**: NOAA Global Hourly Surface Weather Observations (2024)
- **Format**: CSV files from https://www.ncei.noaa.gov/data/global-hourly/
- **Size**: ~50GB (130,222,106 observations)
- **Stations**: Global weather stations
- **Temporal Coverage**: January - December 2024

### 1.2 Data Characteristics
| Metric | Value |
|--------|-------|
| Total Observations | 130,222,106 |
| Weather Stations | [Count from data] |
| Countries Covered | [From data] |
| Temporal Resolution | Hourly |
| Geographic Coverage | Global |

---

## 2. Data Preprocessing

### 2.1 Cleaning Steps

**Data Parsing:**
- Extracted temperature from format `"-0070,1"` → -7.0°C
- Parsed wind observations `"318,1,N,0061,1"` → direction: 318°, speed: 6.1 m/s
- Extracted sea level pressure from `"10208,1"` → 1020.8 hPa
- Converted visibility, dew point, and precipitation to numeric values

**Quality Control:**
1. Removed missing target values (temperature = NULL)
2. Filtered temperature outliers (valid range: -90°C to +60°C)
3. Applied physical constraints (dew point ≤ temperature)
4. Validated pressure range (950-1050 hPa)

**Results:**
- Rows before filtering: 130,222,106
- Rows after filtering: 126,035,277
- **Retention rate: 96.78%**
- Rows removed: 4,186,829 (3.22%)

### 2.2 Feature Engineering

**14 Features Created:**

| Feature | Type | Description | Missing Rate |
|---------|------|-------------|--------------|
| `latitude` | Geographic | Station latitude (°N) | 0.00% |
| `longitude` | Geographic | Station longitude (°E) | 0.00% |
| `elevation` | Geographic | Station elevation (m) | 0.00% |
| `dew_point` | Weather | Dew point temperature (°C) | 15.36% |
| `sea_level_pressure` | Weather | Sea level pressure (hPa) | 59.10% |
| `visibility` | Weather | Visibility distance (m) | 32.98% |
| `wind_speed` | Weather | Wind speed (m/s) | 12.81% |
| `wind_dir_sin` | Weather | Wind direction (sin transform) | 27.17% |
| `wind_dir_cos` | Weather | Wind direction (cos transform) | 27.17% |
| `precipitation` | Weather | Total precipitation (mm) | 0.34% |
| `hour_sin` | Temporal | Hour of day (sin transform) | 0.00% |
| `hour_cos` | Temporal | Hour of day (cos transform) | 0.00% |
| `month_sin` | Temporal | Month of year (sin transform) | 0.00% |
| `month_cos` | Temporal | Month of year (cos transform) | 0.00% |

**Cyclical Encoding:**
- Hour: sin(2π × hour/24), cos(2π × hour/24)
- Month: sin(2π × month/12), cos(2π × month/12)
- Wind direction: sin(direction × π/180), cos(direction × π/180)

**Missing Value Strategy:**
- Imputation method: Median imputation for numeric features
- Justification: Robust to outliers, preserves central tendency

### 2.3 Data Split

| Set | Rows | Percentage |
|-----|------|------------|
| Training | 88,224,694 | 70% |
| Test | 37,810,583 | 30% |
| **Total** | **126,035,277** | **100%** |

Split method: Random split with seed=42 for reproducibility

---

## 3. Computing Environment

### 3.1 Platform
- **Cloud Provider**: Google Cloud Platform
- **Service**: Dataproc Serverless (Batch processing)
- **Region**: asia-southeast1
- **Spark Version**: 2.2.61

### 3.2 Resources
- **Default Configuration**:
  - Driver: 4 cores, 9.6GB memory
  - Executors: 4 cores each, 9.6GB memory
  - Initial executors: 2
  - Dynamic allocation: Enabled

### 3.3 Processing Times

| Task | Runtime | Dataset Size |
|------|---------|--------------|
| Data Cleanup | ~45 minutes | 130M rows |
| Train/Test Split | ~20 minutes | 126M rows |
| Baseline Model | ~[XX] minutes | 10% sample |
| RF Test Mode | ~[XX] minutes | 10% sample |
| RF Full Mode | ~[XX] hours | 88M rows |
| GBT Test Mode | ~[XX] minutes | 10% sample |
| GBT Full Mode | ~[XX] hours | 88M rows |
| Model Evaluation | ~[XX] minutes | 38M rows |

---

## 4. Model Training

### 4.1 Baseline Model: Linear Regression

**Purpose**: Validate ML pipeline and establish baseline performance

**Configuration:**
- Algorithm: Linear Regression
- Features: All 14 features
- Max iterations: 10
- Regularization: L2 (λ = 0.1)

**Results:**

| Metric | Training Set | Test Set |
|--------|--------------|----------|
| RMSE (°C) | 5.56 | 5.56 |
| R² Score | 0.8024 | 0.8017 |
| MAE (°C) | 4.03 | 4.03 |
| Sample size | 834,272 rows (10%) | 834,272 rows (10%) |

**Analysis**: The baseline Linear Regression model achieves 80.17% R², explaining a substantial portion of temperature variance. The consistent performance between training and test sets (RMSE 5.56°C for both) indicates no overfitting. Average prediction error is approximately 4°C. This provides a solid baseline for more complex models to improve upon.

---

### 4.2 Random Forest Regressor

#### 4.2.1 Test Mode (10% Sample Validation)

**Purpose**: Validate Random Forest pipeline and tune hyperparameters on sample data

**Configuration:**
- Sample size: 8,823,031 rows (10% of training data)
- Algorithm: Random Forest Regression
- Hyperparameter Tuning: Grid Search with 2-Fold Cross-Validation
- Models tested: 4 (2 numTrees × 2 maxDepth combinations)

**Hyperparameter Grid (Test Mode):**

| Parameter | Values Tested | Best Value |
|-----------|---------------|------------|
| `numTrees` | [10, 20] | 20 |
| `maxDepth` | [5, 10] | 10 |
| `minInstancesPerNode` | [1] | 1 |

**Training Results (Test Mode):**

| Metric | Value |
|--------|-------|
| Training rows | 8,823,031 |
| Training RMSE | 4.64°C |
| Training R² | 0.8525 |
| Training MAE | 3.41°C |
| Best CV RMSE | 4.65°C |
| Worst CV RMSE | 6.44°C |
| Mean CV RMSE | 5.57°C |
| Training time | ~35 minutes |

**Performance vs Baseline:**
- RMSE improvement: 5.56°C → 4.64°C (**16.5% better**)
- R² improvement: 0.8017 → 0.8525 (**6.3% better**)

**Feature Importances (Test Mode - Top 10):**

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | dew_point | 0.3819 (38.19%) | Most critical predictor |
| 2 | latitude | 0.2671 (26.71%) | Geographic location matters |
| 3 | month_cos | 0.1735 (17.35%) | Seasonal patterns (cyclical) |
| 4 | month_sin | 0.0664 (6.64%) | Seasonal patterns (cyclical) |
| 5 | longitude | 0.0373 (3.73%) | East-west variation |
| 6 | sea_level_pressure | 0.0284 (2.84%) | Atmospheric conditions |
| 7 | elevation | 0.0154 (1.54%) | Altitude effect |
| 8 | hour_sin | 0.0121 (1.21%) | Diurnal cycle |
| 9 | visibility | 0.0066 (0.66%) | Minor predictor |
| 10 | wind_speed | 0.0046 (0.46%) | Minor predictor |

**Key Insights from Test Mode:**
- Dew point is by far the most important feature, explaining 38% of temperature variance
- Geographic features (latitude, longitude) combined explain ~30% 
- Seasonal patterns (month_sin/cos) explain ~24%
- Temporal features (hour_sin/cos) have minimal impact (~1%)
- Weather measurements (pressure, visibility, wind) contribute less than expected

**Analysis**: Random Forest in test mode demonstrates significant improvement over the baseline Linear Regression model. The 16.5% reduction in RMSE (from 5.56°C to 4.64°C) shows that capturing non-linear relationships is valuable. The model identifies dew point as the dominant predictor, which makes meteorological sense as dew point and temperature are strongly correlated through atmospheric moisture content. Cross-validation results show consistency (best CV: 4.65°C), indicating the model generalizes well.

#### 4.2.2 Full Mode (Production Training)

**Configuration:**
- Training rows: [To be filled after full training]
- Algorithm: Random Forest Regression
- Hyperparameter Tuning: Grid Search with 3-Fold Cross-Validation

**Hyperparameter Grid (Full Mode):**

| Parameter | Values Tested | Best Value |
|-----------|---------------|------------|
| `numTrees` | [50, 100, 150] | [XX] |
| `maxDepth` | [10, 15, 20] | [XX] |
| `minInstancesPerNode` | [1, 5] | [X] |

**Training Results:**

| Metric | Value |
|--------|-------|
| Training rows | [XX,XXX,XXX] |
| Training RMSE | [XX.XX]°C |
| Training R² | [0.XX] |
| Training MAE | [XX.XX]°C |
| Best CV RMSE | [XX.XX]°C |
| Training time | [XX] hours |

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| Test RMSE | **[XX.XX]°C** |
| Test R² | **[0.XX]** |
| Test MAE | [XX.XX]°C |
| Test rows | 37,810,583 |

**Feature Importances (Full Mode - Top 10):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | [feature_name] | [0.XXX] |
| 2 | [feature_name] | [0.XXX] |
| 3 | [feature_name] | [0.XXX] |
| 4 | [feature_name] | [0.XXX] |
| 5 | [feature_name] | [0.XXX] |
| 6 | [feature_name] | [0.XXX] |
| 7 | [feature_name] | [0.XXX] |
| 8 | [feature_name] | [0.XXX] |
| 9 | [feature_name] | [0.XXX] |
| 10 | [feature_name] | [0.XXX] |

---

### 4.3 Gradient Boosted Trees (GBT)

#### 4.3.1 Test Mode (10% Sample Validation)

**Purpose**: Validate GBT pipeline and tune hyperparameters on sample data

**Configuration:**
- Sample size: 8,823,031 rows (10% of training data)
- Algorithm: Gradient Boosted Trees Regression
- Hyperparameter Tuning: Grid Search with 2-Fold Cross-Validation
- Models tested: 4 (2 maxIter × 2 maxDepth combinations)

**Hyperparameter Grid (Test Mode):**

| Parameter | Values Tested | Best Value |
|-----------|---------------|------------|
| `maxIter` | [10, 20] | 20 |
| `maxDepth` | [3, 5] | 5 |
| `stepSize` | [0.1] | 0.1 |

**Training Results (Test Mode):**

| Metric | Value |
|--------|-------|
| Training rows | 8,823,031 |
| Training RMSE | 4.93°C |
| Training R² | 0.8341 |
| Training MAE | 3.59°C |
| Best CV RMSE | 4.94°C |
| Worst CV RMSE | 6.30°C |
| Mean CV RMSE | 5.61°C |
| Number of trees | 20 |
| Training time | ~40 minutes |

**Performance vs Baseline & RF:**
- vs Baseline: 5.56°C → 4.93°C (**11.3% improvement**)
- vs RF Test: 4.93°C vs 4.64°C (**RF is 6% better**)

**Feature Importances (Test Mode - Top 10):**

| Rank | Feature | Importance | Notes |
|------|---------|------------|-------|
| 1 | dew_point | 0.3697 (36.97%) | Most critical (similar to RF) |
| 2 | latitude | 0.2069 (20.69%) | Geographic importance |
| 3 | longitude | 0.0853 (8.53%) | East-west variation |
| 4 | month_cos | 0.0815 (8.15%) | Seasonal patterns |
| 5 | month_sin | 0.0600 (6.00%) | Seasonal patterns |
| 6 | elevation | 0.0503 (5.03%) | Altitude effect (higher than RF) |
| 7 | hour_sin | 0.0446 (4.46%) | Diurnal cycle (higher than RF) |
| 8 | hour_cos | 0.0370 (3.70%) | Diurnal cycle |
| 9 | wind_speed | 0.0198 (1.98%) | Minor predictor |
| 10 | visibility | 0.0196 (1.96%) | Minor predictor |

**Comparison: GBT vs RF Feature Importances**

| Feature | RF Importance | GBT Importance | Difference |
|---------|---------------|----------------|------------|
| dew_point | 38.19% | 36.97% | Similar ✅ |
| latitude | 26.71% | 20.69% | RF values more |
| longitude | 3.73% | 8.53% | **GBT values 2.3x more** |
| month_cos | 17.35% | 8.15% | **RF values 2.1x more** |
| elevation | 1.54% | 5.03% | **GBT values 3.3x more** |
| hour_sin | 1.21% | 4.46% | **GBT values 3.7x more** |

**Key Insights:**
- Both models agree dew_point and latitude are most important
- RF emphasizes seasonal patterns (month) more strongly
- GBT gives more weight to geographic spread (longitude) and elevation
- GBT captures diurnal patterns (hour) better than RF
- Overall, RF's feature weighting leads to better predictions

**Analysis**: Gradient Boosted Trees in test mode showed solid improvement over baseline (11.3% RMSE reduction) but underperformed compared to Random Forest by 6% (4.93°C vs 4.64°C). While both models agree on the top two features (dew_point and latitude), they differ in how they weight other features. GBT's sequential boosting approach gave more importance to elevation and temporal features (hour), while RF's ensemble approach favored seasonal patterns more heavily. The cross-validation results (best CV: 4.94°C) are consistent with training, indicating the model generalizes well, but not as well as Random Forest. Given RF's superior performance across all metrics and similar training time (~35 vs ~40 minutes), Random Forest appears to be the better model architecture for this temperature prediction task.

#### 4.3.2 Full Mode (Production Training)

**Configuration:**
- Training rows: [To be filled]
- Algorithm: Gradient Boosted Trees Regression  
- Hyperparameter Tuning: Grid Search with 3-Fold Cross-Validation

**Hyperparameter Grid (Full Mode):**

| Parameter | Values Tested | Best Value |
|-----------|---------------|------------|
| `maxIter` | [50, 100, 150] | [XX] |
| `maxDepth` | [5, 7, 10] | [X] |
| `stepSize` | [0.05, 0.1, 0.2] | [0.XX] |

**Training Results:**

| Metric | Value |
|--------|-------|
| Training rows | [XX,XXX,XXX] |
| Training RMSE | [XX.XX]°C |
| Training R² | [0.XX] |
| Training MAE | [XX.XX]°C |
| Best CV RMSE | [XX.XX]°C |
| Number of trees | [XX] |
| Training time | [XX] hours |

**Test Set Performance:**

| Metric | Value |
|--------|-------|
| Test RMSE | **[XX.XX]°C** |
| Test R² | **[0.XX]** |
| Test MAE | [XX.XX]°C |
| Test rows | 37,810,583 |

**Feature Importances (Full Mode - Top 10):**

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | [feature_name] | [0.XXX] |
| 2 | [feature_name] | [0.XXX] |
| 3 | [feature_name] | [0.XXX] |
| 4 | [feature_name] | [0.XXX] |
| 5 | [feature_name] | [0.XXX] |
| 6 | [feature_name] | [0.XXX] |
| 7 | [feature_name] | [0.XXX] |
| 8 | [feature_name] | [0.XXX] |
| 9 | [feature_name] | [0.XXX] |
| 10 | [feature_name] | [0.XXX] |

---

## 5. Model Comparison

### 5.1 Performance Summary

| Model | Test RMSE (°C) | Test R² | Test MAE (°C) | Training Time |
|-------|----------------|---------|---------------|---------------|
| Linear Regression | [XX.XX] | [0.XX] | [XX.XX] | ~[XX] min |
| Random Forest | **[XX.XX]** | **[0.XX]** | [XX.XX] | ~[XX] hrs |
| Gradient Boosted Trees | [XX.XX] | [0.XX] | [XX.XX] | ~[XX] hrs |

**Best Model**: [Model Name]  
**Reasoning**: [Explain why this model was selected based on metrics and requirements]

### 5.2 Error Analysis by Temperature Range

**[Best Model Name] - Performance by Temperature Range:**

| Temperature Range | Count | Mean Abs Error (°C) | RMSE (°C) |
|-------------------|-------|---------------------|-----------|
| Below 0°C | [X,XXX,XXX] | [X.XX] | [X.XX] |
| 0-10°C | [X,XXX,XXX] | [X.XX] | [X.XX] |
| 10-20°C | [X,XXX,XXX] | [X.XX] | [X.XX] |
| 20-30°C | [X,XXX,XXX] | [X.XX] | [X.XX] |
| Above 30°C | [X,XXX,XXX] | [X.XX] | [X.XX] |

**Analysis**: [Discuss where the model performs best/worst]

---

## 6. Key Findings

### 6.1 Most Important Features

Based on feature importance analysis from the best model:

1. **[Top feature]**: [Explain why this makes sense]
2. **[Second feature]**: [Explain importance]
3. **[Third feature]**: [Explain importance]

### 6.2 Model Insights

**Strengths:**
- [Strength 1]
- [Strength 2]
- [Strength 3]

**Limitations:**
- [Limitation 1]
- [Limitation 2]
- [Limitation 3]

### 6.3 Prediction Quality

- The model explains **[XX]%** of variance in temperature (R² = [0.XX])
- Average prediction error: **[XX.XX]°C** (RMSE)
- Median absolute error: **[XX.XX]°C**
- [XX]% of predictions within ±[X]°C of actual temperature

---

## 7. Additional Efforts for Performance Improvement

### 7.1 Data Quality Enhancements
- [List any special data quality measures]
- [Feature engineering efforts]
- [Data validation steps]

### 7.2 Model Optimization
- Comprehensive hyperparameter tuning via grid search
- Cross-validation to prevent overfitting
- Median imputation for missing values
- Cyclical encoding for temporal and directional features

### 7.3 Computational Optimizations
- [Any special optimizations made]
- Efficient single-pass aggregations in data cleanup
- Partitioned data storage for faster access
- Dynamic resource allocation in Spark

---

## 8. Conclusions

### 8.1 Summary
[Summarize the project, approach, and results in 3-4 sentences]

### 8.2 Model Selection
**Selected Model**: [Model Name]  
**Final Test RMSE**: [XX.XX]°C  
**Final Test R²**: [0.XX]

This model was selected because [explain reasoning based on performance, interpretability, computational cost, etc.]

### 8.3 Practical Applications
[Discuss potential real-world applications of this temperature prediction model]

### 8.4 Future Work
- [Suggestion 1 for improvement]
- [Suggestion 2 for improvement]
- [Suggestion 3 for improvement]

---

## 9. Appendix

### 9.1 Data Sources
- NOAA Global Hourly Dataset: https://www.ncei.noaa.gov/data/global-hourly/
- Documentation: https://www.ncei.noaa.gov/data/global-hourly/doc/

### 9.2 Code Repository
All code is available in the submission package:
- `noaa_cleanup_full.py` - Data cleaning pipeline
- `train_test_split.py` - Train/test split
- `baseline_model_test.py` - Baseline validation
- `train_random_forest.py` - Random Forest training
- `train_gbt.py` - GBT training
- `evaluate_model.py` - Model evaluation
- `compare_models.py` - Model comparison

### 9.3 Computing Resources
- **Total compute time**: ~[XX] hours
- **Total cost**: [If tracked]
- **Cloud storage**: ~[XXX] GB

### 9.4 Figures

[Include key visualizations:]
- Figure 1: Data distribution by month
- Figure 2: Feature importance comparison
- Figure 3: Prediction error distribution
- Figure 4: Actual vs. Predicted temperature scatter plot
- Figure 5: Model performance comparison

---

**Submitted**: [Date]  
**Course**: DSS5208 - Distributed Systems and Big Data  
**Instructor**: [Name]  
**Institution**: [University Name]