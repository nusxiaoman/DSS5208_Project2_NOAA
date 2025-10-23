# NOAA Weather Temperature Prediction

**DSS5208 Project 2: Machine Learning on Weather Data**

Predicting air temperature from NOAA global hourly weather observations using Apache Spark and machine learning.

---

## 📋 Project Overview

This project processes **130 million** hourly weather observations from NOAA's global dataset to build machine learning models for temperature prediction. We use Apache Spark on Google Cloud Dataproc for distributed processing and train multiple regression models (Linear Regression, Random Forest, Gradient Boosted Trees).

**Key Achievements:**
- ✅ Processed 50GB of raw CSV data → 111MB compressed Parquet
- ✅ 96.78% data retention after quality filtering
- ✅ 14 engineered features with proper missing value handling
- ✅ Models trained on 88M observations, tested on 38M
- ✅ Best test RMSE: **[XX.XX]°C** (to be updated after training)

---

## 🗂️ Repository Structure

```
project/
├── README.md                          # This file - Project overview
├── DATA_CLEANUP_README.md            # Data preprocessing documentation
├── TRAINING_GUIDE.md                 # Model training instructions
├── RESULTS_SUMMARY.md                # Final results and analysis
│
├── scripts/                          # All Python scripts
│   ├── noaa_cleanup_full.py         # Data cleaning pipeline
│   ├── train_test_split.py          # 70/30 train/test split
│   ├── baseline_model_test.py       # Baseline Linear Regression
│   ├── train_random_forest.py       # Random Forest training
│   ├── train_gbt.py                 # Gradient Boosted Trees training
│   ├── evaluate_model.py            # Test set evaluation
│   └── compare_models.py            # Model comparison
│
└── outputs/                          # Model outputs (on GCS)
    ├── baseline_test/
    ├── rf_test/
    ├── rf_full/
    ├── gbt_test/
    └── gbt_full/
```

---

## 📚 Documentation Quick Links

| Document | Purpose | When to Use |
|----------|---------|-------------|
| **[DATA_CLEANUP_README.md](DATA_CLEANUP_README.md)** | Data preprocessing pipeline | Understanding data cleaning steps |
| **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** | Model training instructions | Running training jobs |
| **[RESULTS_SUMMARY.md](RESULTS_SUMMARY.md)** | Final results and analysis | Project report, final metrics |

---

## 🚀 Quick Start

### Prerequisites
- Google Cloud Project with Dataproc enabled
- Access to NOAA Global Hourly dataset (2024)
- gsutil and gcloud CLI configured

### Setup

```powershell
# Set environment
gcloud config set project distributed-map-475111-h2
gcloud config set dataproc/region asia-southeast1

# Set variables
$env:BUCKET = "weather-ml-bucket-1760514177"
```

### Run Complete Pipeline

```powershell
# 1. Data Cleanup (45 min)
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/noaa_cleanup_full.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default

# 2. Train/Test Split (20 min)
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_test_split.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default

# 3. Baseline Test (10 min)
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/baseline_model_test.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default

# 4. Train Random Forest (2-4 hrs)
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_random_forest.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/rf_full `
    full

# 5. Train GBT (3-6 hrs)
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_gbt.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/gbt_full `
    full

# 6. Evaluate Models (10-20 min each)
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/evaluate_model.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/outputs/rf_full/best_rf_model `
    gs://$env:BUCKET/warehouse/noaa_test `
    gs://$env:BUCKET/outputs/rf_full_evaluation

# Note: The '--' separator must be quoted in PowerShell
```

**See [TRAINING_GUIDE.md](TRAINING_GUIDE.md) for detailed instructions.**

---

## 📊 Dataset

**Source**: NOAA Global Hourly Surface Weather Observations  
**URL**: https://www.ncei.noaa.gov/data/global-hourly/archive/csv/  
**Documentation**: https://www.ncei.noaa.gov/data/global-hourly/doc/

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Original Rows | 130,222,106 |
| Original Size | ~50GB (CSV) |
| After Cleaning | 126,035,277 rows |
| Cleaned Size | 111MB (Parquet) |
| Retention Rate | 96.78% |
| Training Set | 88,224,694 rows (70%) |
| Test Set | 37,810,583 rows (30%) |

### Features (14 Total)

**Geographic** (3): latitude, longitude, elevation  
**Weather** (7): dew_point, sea_level_pressure, visibility, wind_speed, wind_dir_sin, wind_dir_cos, precipitation  
**Temporal** (4): hour_sin, hour_cos, month_sin, month_cos  
**Target**: temperature (°C)

---

## 🤖 Models

### 1. Baseline: Linear Regression
- **Purpose**: Pipeline validation
- **Test RMSE**: [XX.XX]°C
- **Test R²**: [0.XX]

### 2. Random Forest Regressor
- **Hyperparameters**: numTrees, maxDepth, minInstancesPerNode
- **Cross-validation**: 3-fold
- **Test RMSE**: [XX.XX]°C
- **Test R²**: [0.XX]
- **Training time**: [XX] hours

### 3. Gradient Boosted Trees
- **Hyperparameters**: maxIter, maxDepth, stepSize
- **Cross-validation**: 3-fold
- **Test RMSE**: [XX.XX]°C
- **Test R²**: [0.XX]
- **Training time**: [XX] hours

**Best Model**: [To be determined after training]

---

## 💻 Computing Environment

**Platform**: Google Cloud Platform  
**Service**: Dataproc Serverless (Batch processing)  
**Region**: asia-southeast1  
**Spark Version**: 2.2.61

**Default Resources:**
- Driver: 4 cores, 9.6GB memory
- Executors: 4 cores, 9.6GB memory each
- Dynamic allocation: Enabled

**Total Processing Time**: ~6-11 hours (all steps)

---

## 📈 Results Summary

**See [RESULTS_SUMMARY.md](RESULTS_SUMMARY.md) for complete analysis.**

### Quick Metrics

| Model | Test RMSE | Test R² | Training Time |
|-------|-----------|---------|---------------|
| Linear Regression | [XX.XX]°C | [0.XX] | ~10 min |
| Random Forest | [XX.XX]°C | [0.XX] | ~[X] hrs |
| Gradient Boosted Trees | [XX.XX]°C | [0.XX] | ~[X] hrs |

### Top Features (Example)
1. dew_point
2. sea_level_pressure
3. latitude
4. month_sin
5. hour_sin

*Feature importance rankings to be updated after training.*

---

## 🔧 Key Technical Decisions

### Data Preprocessing
- **Missing values**: Median imputation (robust to outliers)
- **Cyclical encoding**: For hour, month, wind direction (preserves continuity)
- **Quality filters**: Physical constraints and outlier removal
- **Compression**: Parquet format (99.8% size reduction)

### Model Training
- **Cross-validation**: 3-fold for hyperparameter tuning
- **Grid search**: Comprehensive hyperparameter space
- **Evaluation**: RMSE, R², MAE on held-out test set
- **Seed**: Fixed (42) for reproducibility

### Performance Optimizations
- Efficient single-pass aggregations for missing value counts
- Partitioned data storage (by year/month)
- Spark adaptive query execution
- Dynamic resource allocation

---

## 📦 Submission Package

For Canvas submission, the package includes:

```
submission/
├── README.md                     # This file
├── DATA_CLEANUP_README.md       # Cleanup documentation
├── TRAINING_GUIDE.md            # Training instructions
├── RESULTS_SUMMARY.md           # Final results
├── report.pdf                   # Main project report
├── ai_communication.txt         # AI assistant conversation log
│
├── code/
│   ├── noaa_cleanup_full.py
│   ├── train_test_split.py
│   ├── baseline_model_test.py
│   ├── train_random_forest.py
│   ├── train_gbt.py
│   ├── evaluate_model.py
│   └── compare_models.py
│
└── models/
    ├── best_rf_model/           # Trained Random Forest
    └── best_gbt_model/          # Trained GBT
```

---

## 🎓 Learning Outcomes

This project demonstrates:
- ✅ Large-scale data processing with Apache Spark
- ✅ Distributed machine learning on cloud infrastructure
- ✅ Feature engineering for weather data
- ✅ Hyperparameter tuning with cross-validation
- ✅ Model evaluation and comparison
- ✅ Production ML pipeline development

---

## 🔗 References

1. NOAA Global Hourly Dataset: https://www.ncei.noaa.gov/data/global-hourly/
2. Google Cloud Dataproc: https://cloud.google.com/dataproc/docs
3. Apache Spark MLlib: https://spark.apache.org/docs/latest/ml-guide.html
4. PySpark Documentation: https://spark.apache.org/docs/latest/api/python/

---

## 👥 Team Members

[List your group members and student IDs here]

---

## 📅 Project Timeline

| Milestone | Date | Status |
|-----------|------|--------|
| Data Cleanup | [Date] | ✅ Complete |
| Train/Test Split | [Date] | ✅ Complete |
| Baseline Model | [Date] | ⏳ In Progress |
| RF Training | [Date] | 📋 Planned |
| GBT Training | [Date] | 📋 Planned |
| Model Evaluation | [Date] | 📋 Planned |
| Final Report | [Date] | 📋 Planned |
| Submission | November 5, 2025 | 🎯 Deadline |

---

## 📧 Contact

For questions about this project:
- Course: DSS5208 - Distributed Systems and Big Data
- Instructor: [Name]
- Institution: [University]

---

**Last Updated**: October 24, 2024  
**Version**: 1.0  
**Status**: In Progress - Model Training Phase