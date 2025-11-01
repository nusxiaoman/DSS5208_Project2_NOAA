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
├── DATA_CLEANUP_README.md                     # Data cleaning documentation
├── docs                                       # Reference documentation
│   ├── AI Communication.txt
│   └── MLlib.pdf
├── experiments                                # Experimental scripts (not in pipeline)
│   ├── num_feature_15
│   ├── num_feature_7
│   └── script_for_trial
├── FEATURE_SELECTION_COMPARISON.md            # Feature engineering analysis (V0/V1/V2)
├── QUICK_REFERENCE.md                         # Command reference guide
├── README.md                                  # Project overview and quick start
├── RESULTS_SUMMARY.md                         # Complete results and analysis
├── src
│   ├── baseline_model_test.py                 # Linear Regression baseline (10 min)
│   ├── compare_models.py                      # Model comparison (5 min)
│   ├── evaluate_model.py                      # Test set evaluation (15 min)
│   ├── noaa_cleanup_full.py                   # Data cleaning (45 min)
│   ├── train_gbt_simplified.py                # GBT training (1.2 hrs)
│   ├── train_random_forest_simplified.py      # RF training (1.2 hrs) ✓ BEST
│   ├── train_test_split.py                    # 70/30 split (20 min)
│   ├── train_gbt.py                           # GBT original version
│   └── train_random_forest.py                 # RF original version
└── TRAINING_GUIDE.md                          # Step-by-step training instructions
```

# Quick start

## Prerequisite 
- User have to install google-cloud-sdk
- User have an Google Cloud account

### 1) Config parameter for your project
```shell
export DEFAULT_REGION="asia-southeast1"
export PROJECT_ID="dss5208-noaa-2025"
export PROJECT_NAME="Weather Temperature Prediction"
export MY_BUCKET="gs://temperature-ml-2025"
export BILLING_ACCOUNT_ID="01C917-3E4FE0-F7F748"
export PYSPARK_VERSION="2.2"
export PATH_RAW_DATA=$MY_BUCKET/data/csv
export PATH_CLEANED_DATA=$MY_BUCKET/warehouse/noaa_clean_std
export PATH_TRAINING_DATA=$MY_BUCKET/warehouse/noaa_train
export PATH_TEST_DATA=$MY_BUCKET/warehouse/noaa_test
export PATH_OUTPUT=$MY_BUCKET/outputs
export PATH_OUTPUT_BASELINE=$PATH_OUTPUT/baseline_test
export PATH_OUTPUT_GBT=$PATH_OUTPUT/gbt_test
export PATH_OUTPUT_RF=$PATH_OUTPUT/rf_test
export PATH_OUTPUT_RF_SIMPLIFIED=$PATH_OUTPUT/rf_simplified
export PATH_OUTPUT_RF_SIMPLIFIED_EVALUATION=$PATH_OUTPUT/rf_simplified_evaluation
```

The suggestion structure design for Bucket
```
├── data/csv/                     # Raw NOAA CSV files (~50GB)
├── scripts/                      # keep script to run batch
├── outputs/
│   ├── baseline_test/            # keep baseline training result
│   ├── rf_simplified_evaluation/ # keep eveluation data
│   ├── rf_simplified/            # keep random forest simplfied training result
│   ├── rf_test/                  # keep random forest training result
│   └── gbt_test/                 # keep GBT training result
└── warehouse/
    ├── noaa_train/               # Raw train parquet (from ETL)
    ├── noaa_test/                # Raw test parquet (from ETL)
    └── noaa_clean_std/           # Cleaned data (output)
```

### 2) config the region for your project
```shell
gcloud config set compute/region $DEFAULT_REGION
```

### 3) Create a new project
```shell
gcloud projects create $PROJECT_ID --name="$PROJECT_NAME"
gcloud config set project $PROJECT_ID
```

### 4) check your current config
```shell
gcloud config list
```

### 5) check your current billing account and link to the project
```shell
gcloud beta billing accounts list
gcloud beta billing projects link $PROJECT_ID --billing-account $BILLING_ACCOUNT_ID
```

### 6) Create a new bucket to storage data and verify the permission
```shell
gsutil mb -l $DEFAULT_REGION -p $PROJECT_ID $MY_BUCKET

gsutil iam get $MY_BUCKET
```

### 6) copy the raw data to storage (Follow the structure)
```shell
# cd /to/your/path
gsutil -m cp -r . $MY_BUCKET/data/csv
```

### 7) verify the data in storage with number of file and first 5 file by name
```shell
gsutil ls $MY_BUCKET/data/csv | wc -l

gsutil ls $MY_BUCKET/data/csv | head -n 5
```

### 8) copy script from the repository
```shell
# cd /to/your/repository
gsutil -m cp -r src/* $MY_BUCKET/scripts/

gsutil ls $MY_BUCKET/scripts/ | head -n 10
-----
.../scripts/baseline_model_test.py
.../scripts/compare_models.py
.../scripts/evaluate_model.py
.../scripts/noaa_cleanup_full.py
.../scripts/train_gbt.py
.../scripts/train_random_forest.py
.../scripts/train_random_forest_simplified.py
.../scripts/train_test_split.py
```

### 9) enable dataproc for current project and verify (it will take few minutes)
```shell
gcloud services enable dataproc.googleapis.com --project $PROJECT_ID

gcloud services list --project $PROJECT_ID --enabled

```

gcloud services list --project $PROJECT_ID --filter="config.name=dataproc.googleapis.com"

### 9) run cleanup script with ASYC mode
```shell
gcloud dataproc batches submit pyspark \
    $MY_BUCKET/scripts/noaa_cleanup_full.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=01-cleanup-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_RAW_DATA \
    $PATH_CLEANED_DATA

# Note the job id to check and verify 
Batch [01-cleanup-job-20251031-234134] submitted.
...

# you can check the output log on whe website
https://console.cloud.google.com/dataproc/batches?cloudshell=true&project=[CHANGE_TO_YOUR_PROJECT_ID]
```

### 10) run split script (make sure the previous step  is successful)
```shell
gcloud dataproc batches submit pyspark \
    $MY_BUCKET/scripts/train_test_split.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=02-split-data-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_CLEANED_DATA \
    $PATH_TRAINING_DATA \
    $PATH_TEST_DATA
```

### 11) run baseline training script (make sure the previous step  is successful)
```shell
gcloud dataproc batches submit pyspark \
    $MY_BUCKET/scripts/baseline_model_test.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=03-baseline-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_TRAINING_DATA \
    $PATH_TEST_DATA \
    $PATH_OUTPUT_BASELINE
```


### 11) run GBT training script (make sure the previous step  is successful)
```shell
# run with mode: full or test (training with 10% data)
gcloud dataproc batches submit pyspark \
     $MY_BUCKET/scripts/train_gbt.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=041-gbt-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_TRAINING_DATA \
    $PATH_OUTPUT_GBT \
    test
```


### 12) run RF training script (make sure the previous step  is successful)
```shell
# run with mode: full or test (training with 10% data)
gcloud dataproc batches submit pyspark \
     $MY_BUCKET/scripts/train_random_forest.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=051-random-forest-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_TRAINING_DATA \
    $PATH_OUTPUT_RF \
    test
```

### 13) run RF simplified training script (make sure the previous step  is successful)
```shell
# run with mode: full or test (training with 10% data)
gcloud dataproc batches submit pyspark \
     $MY_BUCKET/scripts/train_random_forest_simplified.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=052-rf-simplified-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_TRAINING_DATA \
    $PATH_OUTPUT_RF_SIMPLIFIED \
    test
```

### 14) Evaluate RF on Test Set
```shell
gcloud dataproc batches submit pyspark \
    $MY_BUCKET/scripts/evaluate_model.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=06-evaluation-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_OUTPUT_RF_SIMPLIFIED/best_RandomForest_model \
    $PATH_TEST_DATA \
    $PATH_OUTPUT_RF_SIMPLIFIED_EVALUATION
```


### 15) run baseline training script (make sure the previous step  is successful)
```shell
gcloud dataproc batches submit pyspark \
    $MY_BUCKET/scripts/compare_models.py \
    --region=$DEFAULT_REGION --deps-bucket=$MY_BUCKET \
    --subnet=default --batch=07-compare-job-$(date +"%Y%m%d-%H%M%S") \
    --version=$PYSPARK_VERSION --async \
    '--' \
    $PATH_OUTPUT
```