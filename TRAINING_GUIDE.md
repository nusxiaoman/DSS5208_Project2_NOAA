# NOAA Weather Prediction - Model Training Guide

## Overview

This guide covers the complete model training pipeline after data cleanup and train/test split.

## Prerequisites

✅ Data cleanup completed  
✅ Train/test split completed  
✅ Data available at:
- Training: `gs://weather-ml-bucket-1760514177/warehouse/noaa_train/`
- Test: `gs://weather-ml-bucket-1760514177/warehouse/noaa_test/`

## Training Pipeline

### Step 1: Baseline Model Test (Quick Validation)

**Purpose**: Verify the ML pipeline works correctly with a simple Linear Regression model

**Runtime**: 5-10 minutes  
**Dataset**: 10% sample

```powershell
# Upload script
gsutil cp baseline_model_test.py gs://weather-ml-bucket-1760514177/scripts/

# Run baseline test
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/baseline_model_test.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --batch=baseline-test-$timestamp
```

**Expected Output**:
- Test RMSE: ~3-5°C (baseline performance)
- Test R²: ~0.85-0.90
- Confirms pipeline is working

---

### Step 2: Random Forest Training

#### Option A: Test Mode (Recommended First)

**Runtime**: 20-40 minutes  
**Dataset**: 10% sample  
**Purpose**: Quick validation of RF pipeline

```powershell
# Upload script
gsutil cp train_random_forest.py gs://weather-ml-bucket-1760514177/scripts/

# Run RF test mode
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/train_random_forest.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --batch=rf-test-$timestamp `
    '--' `
    gs://weather-ml-bucket-1760514177/warehouse/noaa_train `
    gs://weather-ml-bucket-1760514177/outputs/rf_test `
    test

# Note: The '--' must be in quotes for PowerShell to pass it correctly to gcloud
```

**Parameters**:
- Test mode: 2 tree configs × 2 depth configs = 4 models
- 2-fold cross-validation
- Fast validation

#### Option B: Full Production Mode

**Runtime**: 2-4 hours  
**Dataset**: 100% (~88M rows)  
**Purpose**: Final production model

```powershell
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/train_random_forest.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --ttl=2d `
    --batch=rf-full-$timestamp `
    '--' `
    gs://weather-ml-bucket-1760514177/warehouse/noaa_train `
    gs://weather-ml-bucket-1760514177/outputs/rf_full `
    full

# Note: The '--' separator must be in quotes for PowerShell
# TTL set to 2d to prevent timeout - default 4hrs is too short
```

**Parameters**:
- Full mode: 3 tree configs × 3 depth configs × 2 min instances = 18 models
- 3-fold cross-validation
- Grid search: numTrees=[50,100,150], maxDepth=[10,15,20]
- TTL: 8 hours (prevents 4-hour default timeout)

---

### Step 3: Gradient Boosted Trees Training

#### Option A: Test Mode

**Runtime**: 30-60 minutes  
**Dataset**: 10% sample

```powershell
# Upload script
gsutil cp train_gbt.py gs://weather-ml-bucket-1760514177/scripts/

# Run GBT test mode
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/train_gbt.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --batch=gbt-test-$timestamp `
    '--' `
    gs://weather-ml-bucket-1760514177/warehouse/noaa_train `
    gs://weather-ml-bucket-1760514177/outputs/gbt_test `
    test

# Note: The '--' must be in quotes for PowerShell
```

**Parameters**:
- Test mode: 2 iter × 2 depth = 4 models
- 2-fold cross-validation

#### Option B: Full Production Mode

**Runtime**: 3-6 hours  
**Dataset**: 100% (~88M rows)

```powershell
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/train_gbt.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --ttl=2d `
    --batch=gbt-full-$timestamp `
    '--' `
    gs://weather-ml-bucket-1760514177/warehouse/noaa_train `
    gs://weather-ml-bucket-1760514177/outputs/gbt_full `
    full

# Note: The '--' separator must be in quotes for PowerShell
# TTL set to 2d - GBT can take longer than RF
```

**Parameters**:
- Full mode: 3 iter × 3 depth × 3 step sizes = 27 models
- 3-fold cross-validation
- Grid search: maxIter=[50,100,150], maxDepth=[5,7,10], stepSize=[0.05,0.1,0.2]
- TTL: 12 hours (GBT typically takes longer than RF)

---

### Step 4: Model Evaluation on Test Set

After training, evaluate the best model on the held-out test set:

#### Evaluate Random Forest

```powershell
# Upload evaluation script
gsutil cp evaluate_model.py gs://weather-ml-bucket-1760514177/scripts/

# Evaluate RF model
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/evaluate_model.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --batch=eval-rf-$timestamp `
    '--' `
    gs://weather-ml-bucket-1760514177/outputs/rf_full/best_rf_model `
    gs://weather-ml-bucket-1760514177/warehouse/noaa_test `
    gs://weather-ml-bucket-1760514177/outputs/rf_full_evaluation

# Note: The '--' must be in quotes for PowerShell
```

#### Evaluate GBT

```powershell
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/evaluate_model.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --batch=eval-gbt-$timestamp `
    '--' `
    gs://weather-ml-bucket-1760514177/outputs/gbt_full/best_gbt_model `
    gs://weather-ml-bucket-1760514177/warehouse/noaa_test `
    gs://weather-ml-bucket-1760514177/outputs/gbt_full_evaluation

# Note: The '--' must be in quotes for PowerShell
```

**Runtime**: 10-20 minutes each

---

## Monitoring Jobs

### Using Console (Recommended)
https://console.cloud.google.com/dataproc/batches?project=distributed-map-475111-h2&region=asia-southeast1

### Using CLI

```powershell
# List recent jobs
gcloud dataproc batches list --region=asia-southeast1 --limit=10

# Check specific job status
$BATCH_ID = "rf-full-YYYYMMDD-HHMMSS"
gcloud dataproc batches describe $BATCH_ID --region=asia-southeast1 --format="value(state)"

# View logs
gcloud logging read "resource.type=cloud_dataproc_batch AND resource.labels.batch_id=$BATCH_ID" `
    --limit=500 `
    --format="table(textPayload)" `
    --project=distributed-map-475111-h2
```

---

## Output Structure

After training, your outputs will be organized as:

```
gs://weather-ml-bucket-1760514177/outputs/
├── baseline_test/
│   ├── metrics/
│   └── sample_predictions/
│
├── rf_test/
│   ├── best_rf_model/
│   ├── metrics/
│   ├── feature_importances/
│   └── sample_predictions/
│
├── rf_full/
│   ├── best_rf_model/
│   ├── metrics/
│   ├── feature_importances/
│   └── sample_predictions/
│
├── rf_full_evaluation/
│   ├── test_metrics/
│   ├── test_predictions/
│   └── error_analysis/
│
├── gbt_test/
│   ├── best_gbt_model/
│   ├── metrics/
│   ├── feature_importances/
│   └── sample_predictions/
│
├── gbt_full/
│   ├── best_gbt_model/
│   ├── metrics/
│   ├── feature_importances/
│   └── sample_predictions/
│
└── gbt_full_evaluation/
    ├── test_metrics/
    ├── test_predictions/
    └── error_analysis/
```

---

## Features Used

All models use the following 14 features:

### Geographic Features
- `latitude`: Station latitude
- `longitude`: Station longitude  
- `elevation`: Station elevation (m)

### Weather Features
- `dew_point`: Dew point temperature (°C)
- `sea_level_pressure`: Sea level pressure (hPa)
- `visibility`: Visibility distance (m)
- `wind_speed`: Wind speed (m/s)
- `wind_dir_sin`, `wind_dir_cos`: Cyclical wind direction
- `precipitation`: Total precipitation (mm)

### Temporal Features
- `hour_sin`, `hour_cos`: Cyclical hour of day
- `month_sin`, `month_cos`: Cyclical month of year

**Target**: `temperature` (°C)

---

## Missing Value Handling

- **Strategy**: Median imputation
- **Missing rates in training data**:
  - Dew Point: ~15%
  - Sea Level Pressure: ~59%
  - Visibility: ~33%
  - Wind Speed: ~13%
  - Others: <1%

---

## Expected Performance

### Baseline (Linear Regression)
- Test RMSE: ~3-5°C
- Test R²: ~0.85-0.90

### Random Forest
- Test RMSE: ~2-3°C
- Test R²: ~0.90-0.95

### Gradient Boosted Trees
- Test RMSE: ~2-3°C
- Test R²: ~0.90-0.95

*Actual performance may vary based on hyperparameter tuning results*

---

## Recommended Training Workflow

1. ✅ **Run Baseline Test** (5-10 min) - Verify pipeline works
2. ✅ **Run RF Test Mode** (20-40 min) - Test RF pipeline
3. ✅ **Run GBT Test Mode** (30-60 min) - Test GBT pipeline
4. ✅ **Review test results** - Compare models
5. ✅ **Run RF Full Mode** (2-4 hours) - Production RF model
6. ✅ **Run GBT Full Mode** (3-6 hours) - Production GBT model
7. ✅ **Evaluate both on test set** (10-20 min each)
8. ✅ **Compare final results** - Choose best model

**Total time**: ~6-11 hours for complete pipeline

---

## Retrieving Results

### View Metrics

```powershell
# RF metrics
gsutil cat gs://weather-ml-bucket-1760514177/outputs/rf_full/metrics/*.csv

# GBT metrics
gsutil cat gs://weather-ml-bucket-1760514177/outputs/gbt_full/metrics/*.csv

# Test evaluation metrics
gsutil cat gs://weather-ml-bucket-1760514177/outputs/rf_full_evaluation/test_metrics/*.csv
gsutil cat gs://weather-ml-bucket-1760514177/outputs/gbt_full_evaluation/test_metrics/*.csv
```

### View Feature Importances

```powershell
# RF feature importances
gsutil cat gs://weather-ml-bucket-1760514177/outputs/rf_full/feature_importances/*.csv

# GBT feature importances
gsutil cat gs://weather-ml-bucket-1760514177/outputs/gbt_full/feature_importances/*.csv
```

### Download for Local Analysis

```powershell
# Download all results
gsutil -m cp -r gs://weather-ml-bucket-1760514177/outputs/ ./local_outputs/
```

---

## Troubleshooting

### Job Takes Too Long
- Start with test mode first
- Check if Dataproc is scaling properly
- Consider reducing parameter grid

### Out of Memory Errors
```powershell
# Add more memory (if needed)
--properties="spark.executor.memory=12g,spark.driver.memory=12g"
```

### Model Performance Poor
- Check feature importances
- Review missing value rates
- Consider feature engineering
- Adjust hyperparameter ranges

---

## For Project Report

Make sure to document:
1. ✅ Model architectures (RF, GBT)
2. ✅ Hyperparameter search spaces
3. ✅ Cross-validation strategy (k-fold)
4. ✅ Training/test RMSE, R², MAE
5. ✅ Feature importances
6. ✅ Training times
7. ✅ Best model selection rationale

---

**Last Updated**: October 24, 2024  
**Version**: 1.0  
**Course**: DSS5208 Project 2