# NOAA Weather Temperature Prediction

**DSS5208 Project 2: Machine Learning on Weather Data**

Predicting air temperature from NOAA global hourly weather observations using Apache Spark and machine learning.

---

## 📋 Project Overview

This project processes **130 million** hourly weather observations from NOAA's 2024 global dataset to build machine learning models for temperature prediction using Apache Spark on Google Cloud Dataproc.

**Key Achievements:**
- ✅ Processed 50GB raw CSV → 111MB compressed Parquet (99.8% reduction)
- ✅ 96.78% data retention after quality filtering
- ✅ 14 engineered features with cyclical encoding
- ✅ Trained on 88M observations, tested on 38M
- ✅ **Best test RMSE: 4.65°C, R² = 0.85** (16.4% improvement over baseline)

---

## 🗂️ Repository Structure

```
project/
├── README.md                                  # Project overview and quick start
├── FEATURE_SELECTION_COMPARISON.md            # Feature engineering analysis (V0/V1/V2)
├── DATA_CLEANUP_README.md                     # Data cleaning documentation
├── RESULTS_SUMMARY.md                         # Complete results and analysis
├── TRAINING_GUIDE.md                          # Step-by-step training instructions
├── QUICK_REFERENCE.md                         # Command reference guide
│
├── Main Pipeline Scripts (Deliverables)
│   ├── noaa_cleanup_full.py                   # Data cleaning (45 min)
│   ├── train_test_split.py                    # 70/30 split (20 min)
│   ├── baseline_model_test.py                 # Linear Regression baseline (10 min)
│   ├── train_random_forest_simplified.py      # RF training (1.2 hrs) ✓ BEST
│   ├── train_gbt_simplified.py                # GBT training (1.2 hrs)
│   ├── evaluate_model.py                      # Test set evaluation (15 min)
│   └── compare_models.py                      # Model comparison (5 min)
│
├── Alternative Implementations (Reference)
│   ├── train_random_forest.py                 # RF original version
│   └── train_gbt.py                           # GBT original version
│
├── script_for_trial/                          # Experimental scripts (not in pipeline)
│   ├── inspect_data.py
│   ├── inspect_data_v2.py
│   ├── noaa_cleanup_test.py
│   ├── train_random_forest_final.py
│   └── train_random_forest_optimized.py
│
└── Outputs (on Google Cloud Storage)
    ├── gs://.../warehouse/noaa_clean_std/     # Cleaned data (1.2 GiB)
    ├── gs://.../warehouse/noaa_train/         # Training set (70%)
    ├── gs://.../warehouse/noaa_test/          # Test set (30%)
    ├── gs://.../outputs/baseline_test/        # Baseline results
    ├── gs://.../outputs/rf_simplified/        # RF model ✓ BEST
    ├── gs://.../outputs/rf_simplified_evaluation/  # Test evaluation
    └── gs://.../outputs/gbt_simplified/       # GBT model
```

**Note:** Local folders (`data/`, `num_feature_15/`, `num_feature_7/`, `.vscode/`) contain experimental work and development files, not part of the final submission.


---

## 📊 Dataset

**Source**: NOAA Global Hourly Surface Weather Observations (2024)  
**URL**: https://www.ncei.noaa.gov/data/global-hourly/

| Metric | Value |
|--------|-------|
| Original rows | 130,222,106 |
| Original size | ~50GB (CSV) |
| After cleaning | 126,035,277 rows (96.78% retained) |
| Cleaned size | 111MB (Parquet) |
| Training set | 88,228,998 rows (70%) |
| Test set | 37,806,279 rows (30%) |


### Features (14 Total)

**Geographic** (3): latitude, longitude, elevation  
**Weather** (7): dew_point, sea_level_pressure, visibility, wind_speed, wind_dir_sin/cos, precipitation  
**Temporal** (4): hour_sin/cos, month_sin/cos (cyclical encoding)  
**Target**: temperature (°C)

### Feature Selection

These 14 features represent **all essential, available meteorological information** from the NOAA dataset. We explored alternative feature sets:

- **V1**: 32 features with temporal lag features (temp_lag_1h, etc.) - showed data leakage with random split (0.46°C RMSE, unrealistic)
- **V2**: 15 features adding ceiling_height - performed worse (6.85°C) due to 51% NULL values

**Key findings:**
- Additional available fields (cloud cover, weather observations, wind gust) have 50-100% NULL rates
- Our 14-feature baseline captures all usable information without data leakage
- More features ≠ better performance when they add noise rather than signal

**See [FEATURE_SELECTION_COMPARISON.md](FEATURE_SELECTION_COMPARISON.md) for detailed analysis of alternative feature sets and data leakage discussion.**

---

## 🤖 Models & Results

### Final Model Comparison

| Model | Train RMSE | Test RMSE | Test R² | Training Time |
|-------|------------|-----------|---------|---------------|
| **Baseline (LR)** | 5.56°C | 5.56°C | 0.8017 | ~10 min |
| **RF Test (10%)** | 4.64°C | N/A | 0.8525 | ~35 min |
| **RF Full (100%)** ⭐ | **4.65°C** | **4.65°C** | **0.8519** | **1.2 hrs** |
| **GBT Test (10%)** | 4.93°C | N/A | 0.8341 | ~40 min |

### Best Model: Random Forest (Simplified)

**Test Performance:**
- **RMSE**: 4.65°C ✓
- **R²**: 0.8519 (explains 85% of variance) ✓
- **MAE**: 3.42°C ✓
- **Improvement**: 16.4% better than baseline

**Configuration:**
- Parameters: numTrees=20, maxDepth=10
- Cross-validation: 2-fold
- Training: 88,228,998 rows
- Perfect generalization: Training RMSE = Test RMSE (no overfitting)

**Top 5 Features:**
1. **dew_point** (38.4%) - Most critical predictor
2. **latitude** (26.7%) - Geographic climate zones
3. **month_cos** (17.3%) - Seasonal patterns
4. **month_sin** (6.6%) - Seasonal patterns
5. **longitude** (3.8%) - East-west variation

---

## 🎯 Key Findings

### 1. Perfect Generalization
Training RMSE (4.65°C) = Test RMSE (4.65°C) → No overfitting ✓

### 2. Diminishing Returns
- RF Test (10% data): 4.64°C
- RF Full (100% data): 4.65°C
- **Difference**: 0.007°C (0.2% change)

Using 10× more data improved RMSE by less than 1%, showing the model already captured main patterns from the 10% sample.

### 3. RF > GBT
Random Forest outperformed Gradient Boosted Trees by 5.7% (4.64°C vs 4.93°C in test mode).

### 4. Performance by Temperature

| Temperature Range | Count | MAE | Quality |
|-------------------|-------|-----|---------|
| 10-20°C | 11.6M (31%) | 2.96°C | ⭐⭐⭐ Excellent |
| 20-30°C | 10.2M (27%) | 2.93°C | ⭐⭐⭐ Excellent |
| 0-10°C | 8.9M (24%) | 2.91°C | ⭐⭐⭐ Excellent |
| Below 0°C | 4.5M (12%) | 4.88°C | ⭐⭐ Good |
| Above 30°C | 2.5M (7%) | 6.72°C | ⭐ Challenging |

Model excels in moderate temperatures (0-30°C) covering 81% of data.

---

## 💻 Computing Environment

**Platform**: Google Cloud Dataproc Serverless  
**Region**: asia-southeast1  
**Spark Version**: 2.2.61  
**Resources**: 4-core executors, 9.6GB memory, dynamic allocation

**Total Project Time**: ~3 hours compute time

---

## 🚀 Quick Start

### Prerequisites
```powershell
# Set environment
gcloud config set project distributed-map-475111-h2
gcloud config set dataproc/region asia-southeast1
$env:BUCKET = "weather-ml-bucket-1760514177"
```

### Complete Pipeline (Recommended Order)

```powershell
# 1. Data Cleanup (45 min) ✓ DONE
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/noaa_cleanup_full.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default

# 2. Train/Test Split (20 min) ✓ DONE
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_test_split.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default

# 3. Baseline Test (10 min) ✓ DONE
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/baseline_model_test.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default

# 4. RF Simplified (1-2 hrs) ✓ DONE - BEST MODEL
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_random_forest_simplified.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/rf_simplified

# 5. Evaluate RF on Test Set (15 min) ✓ DONE
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/evaluate_model.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/outputs/rf_simplified/best_RandomForest_model `
    gs://$env:BUCKET/warehouse/noaa_test `
    gs://$env:BUCKET/outputs/rf_simplified_evaluation

# 6. Compare Models (5 min) ✓ DONE
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/compare_models.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default
```

**See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.**

---

## 📈 Technical Highlights

### Data Processing
- **Missing values**: Median imputation (robust to outliers)
- **Cyclical encoding**: sin/cos for hour, month, wind direction (preserves circular continuity)
- **Quality filters**: Physical constraints (temp: -90 to +60°C, pressure: 950-1050 hPa)
- **Compression**: Parquet format (99.8% size reduction)

### Model Optimization
- **Memory-optimized hyperparameters**: 4 models instead of 18, preventing OOM errors
- **2-fold CV**: Reduced from 3-fold for memory efficiency
- **Sequential training**: Parallelism=1 for stability
- **Conservative parameters**: maxDepth=10, numTrees=20 (proven in test mode)

### Key Learnings
- Smart sampling (10%) achieves near-optimal results with 90% less compute
- Memory optimization > aggressive hyperparameter tuning
- Conservative parameters with proper validation > exhaustive grid search
- Feature engineering (cyclical encoding) crucial for temporal patterns

---

## 📦 Submission Package

```
submission/
├── README.md                          # This file
├── TRAINING_GUIDE.md                  # Training instructions
├── RESULTS_SUMMARY.md                 # Complete analysis
├── NOAA_Weather_Prediction_Report.docx  # Main report
├── ai_communication.txt               # AI conversation log
│
├── code/
│   ├── noaa_cleanup_full.py
│   ├── train_test_split.py
│   ├── baseline_model_test.py
│   ├── train_random_forest_simplified.py
│   ├── train_gbt.py
│   ├── evaluate_model.py
│   └── compare_models.py
│
└── outputs/ (GCS links in documentation)
    ├── Cleaned data: gs://weather-ml-bucket-1760514177/warehouse/
    ├── Best model: gs://weather-ml-bucket-1760514177/outputs/rf_simplified/
    └── Test results: gs://weather-ml-bucket-1760514177/outputs/rf_simplified_evaluation/
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Large-scale data processing with Apache Spark (130M rows)
- ✅ Distributed machine learning on cloud infrastructure (GCP Dataproc)
- ✅ Feature engineering for weather data (cyclical encoding, imputation)
- ✅ Hyperparameter tuning with cross-validation
- ✅ Production ML pipeline development (data → training → evaluation)
- ✅ Memory optimization for big data workloads
- ✅ Model evaluation and comparison on held-out test sets

---

## 🔗 References

1. NOAA Global Hourly Dataset: https://www.ncei.noaa.gov/data/global-hourly/
2. Documentation: https://www.ncei.noaa.gov/data/global-hourly/doc/
3. Google Cloud Dataproc: https://cloud.google.com/dataproc/docs
4. Apache Spark MLlib: https://spark.apache.org/docs/latest/ml-guide.html

---

**Project Complete!** 🎉  
**Last Updated**: October 26, 2024  
**Version**: 2.0 - Final  
**Course**: DSS5208 - Scalable Distributed Computing for Data Science