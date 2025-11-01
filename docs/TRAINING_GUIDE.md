# TRAINING GUIDE - Complete Updated Commands

## Environment Setup (Run First!)

Add this section at the beginning:

```powershell
# ==================== ENVIRONMENT SETUP ====================
# Set GCP project and region
gcloud config set project distributed-map-475111-h2
gcloud config set dataproc/region asia-southeast1

# Set bucket name as environment variable
$env:BUCKET = "weather-ml-bucket-1760514177"

# Verify settings
gcloud config list
Write-Host "Using bucket: $env:BUCKET"
```

---

## Complete Pipeline - Updated Commands

### Step 1: Data Cleanup (45 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/noaa_cleanup_full.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/data/csv `
    gs://$env:BUCKET/warehouse/noaa_clean_std
```

**Arguments:**
- Input: `gs://$env:BUCKET/data/csv`
- Output: `gs://$env:BUCKET/warehouse/noaa_clean_std`

---

### Step 2: Train/Test Split (20 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_test_split.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_clean_std `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/warehouse/noaa_test
```

**Arguments:**
- Input: `gs://$env:BUCKET/warehouse/noaa_clean_std`
- Train output: `gs://$env:BUCKET/warehouse/noaa_train`
- Test output: `gs://$env:BUCKET/warehouse/noaa_test`

---

### Step 3: Baseline Model Test (10 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/baseline_model_test.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/warehouse/noaa_test `
    gs://$env:BUCKET/outputs/baseline_test
```

**Arguments:**
- Train path: `gs://$env:BUCKET/warehouse/noaa_train`
- Test path: `gs://$env:BUCKET/warehouse/noaa_test`
- Output: `gs://$env:BUCKET/outputs/baseline_test`

---

### Step 4a: Random Forest Test Mode (35 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_random_forest.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/rf_test `
    test
```

**Arguments:**
- Train path: `gs://$env:BUCKET/warehouse/noaa_train`
- Output: `gs://$env:BUCKET/outputs/rf_test`
- Mode: `test` (10% sample)

---

### Step 4b: GBT Test Mode (40 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_gbt.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/gbt_test `
    test
```

**Arguments:**
- Train path: `gs://$env:BUCKET/warehouse/noaa_train`
- Output: `gs://$env:BUCKET/outputs/gbt_test`
- Mode: `test` (10% sample)

**Result:** RF: 4.64°C vs GBT: 4.93°C → Proceed with RF only

---

### Step 5: RF Full Mode - Simplified (1-2 hours) ⭐ BEST

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_random_forest_simplified.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/rf_simplified
```

**Arguments:**
- Train path: `gs://$env:BUCKET/warehouse/noaa_train` (full 88M rows)
- Output: `gs://$env:BUCKET/outputs/rf_simplified`

---

### Step 6: Evaluate RF on Test Set (15 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/evaluate_model.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/outputs/rf_simplified/best_RandomForest_model `
    gs://$env:BUCKET/warehouse/noaa_test `
    gs://$env:BUCKET/outputs/rf_simplified_evaluation
```

**Arguments:**
- Model path: `gs://$env:BUCKET/outputs/rf_simplified/best_RandomForest_model`
- Test data: `gs://$env:BUCKET/warehouse/noaa_test`
- Output: `gs://$env:BUCKET/outputs/rf_simplified_evaluation`

---

### Step 7: Compare All Models (5 min)

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/compare_models.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/outputs
```

**Arguments:**
- Base path: `gs://$env:BUCKET/outputs` (scans all model outputs)

---

## Optional: GBT Simplified (Not Run)

If you want to run GBT full training:

```powershell
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_gbt_simplified.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/warehouse/noaa_train `
    gs://$env:BUCKET/outputs/gbt_simplified
```

---

## Monitoring Jobs

### View All Batches
```powershell
gcloud dataproc batches list --region=asia-southeast1
```

### View Specific Batch Details
```powershell
gcloud dataproc batches describe <BATCH_ID> --region=asia-southeast1
```

### View Logs
```powershell
gcloud dataproc batches describe <BATCH_ID> --region=asia-southeast1 | Select-String "logUri"
# Then visit the logUri in browser
```

### Or use Web Console
https://console.cloud.google.com/dataproc/batches?project=distributed-map-475111-h2&region=asia-southeast1

---

## Troubleshooting

### "Bucket not found" Error

Check environment variable:
```powershell
Write-Host "Current bucket: $env:BUCKET"
```

If empty, reset:
```powershell
$env:BUCKET = "weather-ml-bucket-1760514177"
```

### "File not found" Error

Verify files exist:
```powershell
# Check scripts
gsutil ls gs://$env:BUCKET/scripts/

# Check data
gsutil ls gs://$env:BUCKET/data/csv/

# Check outputs
gsutil ls gs://$env:BUCKET/outputs/
```

### Check Output Files

```powershell
# List all warehouse data
gsutil ls -r gs://$env:BUCKET/warehouse/

# Check specific output
gsutil ls gs://$env:BUCKET/outputs/rf_simplified/
```

---

## Quick Reference - All Paths

| Purpose | Path |
|---------|------|
| **Raw Data** | `gs://$env:BUCKET/data/csv` |
| **Cleaned Data** | `gs://$env:BUCKET/warehouse/noaa_clean_std` |
| **Training Set** | `gs://$env:BUCKET/warehouse/noaa_train` |
| **Test Set** | `gs://$env:BUCKET/warehouse/noaa_test` |
| **Baseline** | `gs://$env:BUCKET/outputs/baseline_test` |
| **RF Test** | `gs://$env:BUCKET/outputs/rf_test` |
| **GBT Test** | `gs://$env:BUCKET/outputs/gbt_test` |
| **RF Full (BEST)** | `gs://$env:BUCKET/outputs/rf_simplified` |
| **Test Evaluation** | `gs://$env:BUCKET/outputs/rf_simplified_evaluation` |

---

## Complete Script List with Arguments

| Script | Arguments | Purpose |
|--------|-----------|---------|
| `noaa_cleanup_full.py` | `<input> <output>` | Clean raw CSV data |
| `train_test_split.py` | `<input> <train_out> <test_out>` | Split 70/30 |
| `baseline_model_test.py` | `<train> <test> <output>` | Linear regression baseline |
| `train_random_forest.py` | `<train> <output> [mode]` | RF training (test/full) |
| `train_gbt.py` | `<train> <output> [mode]` | GBT training (test/full) |
| `train_random_forest_simplified.py` | `<train> <output>` | RF full (memory-optimized) |
| `train_gbt_simplified.py` | `<train> <output>` | GBT full (memory-optimized) |
| `evaluate_model.py` | `<model> <test> <output>` | Test set evaluation |
| `compare_models.py` | `<outputs_base>` | Compare all models |

---

## Benefits of Parameterized Paths

### 1. Easy to Reproduce
Anyone can run with their own bucket:
```powershell
$env:BUCKET = "their-bucket-name"
# All commands work without code changes!
```

### 2. Easy to Test
Test with small sample:
```powershell
# Use test dataset
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/noaa_cleanup_full.py `
    --region=asia-southeast1 `
    --deps-bucket=$env:BUCKET `
    --subnet=default `
    '--' `
    gs://$env:BUCKET/data/test_sample `
    gs://$env:BUCKET/warehouse/test_output
```

### 3. Easy to Switch Environments
```powershell
# Development
$env:BUCKET = "weather-ml-dev"

# Production
$env:BUCKET = "weather-ml-prod"
```

---

## Full Pipeline Example (Copy-Paste Ready)

```powershell
# ==================== SETUP ====================
gcloud config set project distributed-map-475111-h2
gcloud config set dataproc/region asia-southeast1
$env:BUCKET = "weather-ml-bucket-1760514177"

# ==================== STEP 1: CLEANUP ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/noaa_cleanup_full.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/data/csv gs://$env:BUCKET/warehouse/noaa_clean_std

# ==================== STEP 2: SPLIT ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_test_split.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/warehouse/noaa_clean_std gs://$env:BUCKET/warehouse/noaa_train gs://$env:BUCKET/warehouse/noaa_test

# ==================== STEP 3: BASELINE ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/baseline_model_test.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/warehouse/noaa_train gs://$env:BUCKET/warehouse/noaa_test gs://$env:BUCKET/outputs/baseline_test

# ==================== STEP 4: RF/GBT TEST ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_random_forest.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/warehouse/noaa_train gs://$env:BUCKET/outputs/rf_test test

gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_gbt.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/warehouse/noaa_train gs://$env:BUCKET/outputs/gbt_test test

# ==================== STEP 5: RF FULL ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/train_random_forest_simplified.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/warehouse/noaa_train gs://$env:BUCKET/outputs/rf_simplified

# ==================== STEP 6: EVALUATE ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/evaluate_model.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/outputs/rf_simplified/best_RandomForest_model gs://$env:BUCKET/warehouse/noaa_test gs://$env:BUCKET/outputs/rf_simplified_evaluation

# ==================== STEP 7: COMPARE ====================
gcloud dataproc batches submit pyspark `
    gs://$env:BUCKET/scripts/compare_models.py `
    --region=asia-southeast1 --deps-bucket=$env:BUCKET --subnet=default `
    '--' gs://$env:BUCKET/outputs
```