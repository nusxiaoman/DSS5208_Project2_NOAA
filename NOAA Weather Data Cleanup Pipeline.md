# NOAA Weather Data Cleanup Pipeline

## Overview

This cleanup pipeline processes raw NOAA weather CSV data and transforms it into a clean, ML-ready dataset for temperature prediction. The pipeline parses complex string-formatted weather measurements, handles missing values, and creates engineered features suitable for machine learning models.

## Project Structure

```
gs://weather-ml-bucket-1760514177/
├── data/csv/                      # Raw NOAA CSV files (~50GB)
├── scripts/
│   ├── noaa_cleanup_test.py      # Test script (1% sample)
│   └── noaa_cleanup_full.py      # Full dataset processing
└── warehouse/
    ├── noaa_parquet/             # Raw parquet (from ETL)
    └── noaa_clean_std/           # Cleaned data (output)
```

## Data Processing Pipeline

### Input Data Format

Raw NOAA CSV files contain weather observations with complex string-encoded measurements:

- **TMP**: `"-0070,1"` → Temperature in Celsius × 10 + quality code
- **DEW**: `"-0130,1"` → Dew point in Celsius × 10 + quality code
- **SLP**: `"10208,1"` → Sea level pressure in hPa × 10 + quality code
- **VIS**: `"025000,1,9,9"` → Visibility in meters + quality codes
- **WND**: `"318,1,N,0061,1"` → Wind direction, type, speed × 10, quality codes
- **AA1-AA3, GA1-GA3**: Precipitation observations
- Missing values encoded as: `+9999`, `99999`, `999999` depending on field

### Features Extracted

#### Geographic Features
- `latitude`: Station latitude (degrees)
- `longitude`: Station longitude (degrees)
- `elevation`: Station elevation above sea level (meters)

#### Target Variable
- `temperature`: Air temperature in Celsius

#### Weather Features
- `dew_point`: Dew point temperature in Celsius
- `sea_level_pressure`: Sea level pressure in hPa
- `visibility`: Visibility distance in meters
- `wind_speed`: Wind speed in m/s
- `wind_direction`: Wind direction in degrees (0-360)
- `precipitation`: Total precipitation in mm (sum of all precipitation observations)

#### Temporal Features
- `timestamp`: Original observation datetime
- `year`, `month`, `day`, `hour`: Extracted time components
- `hour_sin`, `hour_cos`: Cyclical encoding of hour (for 24-hour cycle)
- `month_sin`, `month_cos`: Cyclical encoding of month (for seasonal cycle)

#### Derived Features
- `wind_dir_sin`, `wind_dir_cos`: Cyclical encoding of wind direction

### Data Quality Filters

The cleanup process applies the following filters:

1. **Remove missing target values**: Rows where temperature is NULL
2. **Temperature range**: Keep only -90°C to +60°C (physically reasonable)
3. **Physical constraints**: Dew point must be ≤ temperature
4. **Pressure range**: Sea level pressure must be 950-1050 hPa
5. **Outlier removal**: Extreme values beyond reasonable bounds

### Handling Missing Values

- **Temperature (target)**: Rows with missing temperature are removed
- **Other features**: Missing values are kept as NULL for model to handle
- **Precipitation**: Missing precipitation values treated as 0.0 mm
- **Wind direction/speed**: NULL when encoded as 999 or 9999

## Usage

### Prerequisites

- Google Cloud Project: `distributed-map-475111-h2`
- GCS Bucket: `weather-ml-bucket-1760514177`
- Dataproc Serverless enabled
- Raw CSV files uploaded to `gs://weather-ml-bucket-1760514177/data/csv/`

### Environment Setup (Windows PowerShell)

```powershell
# Authenticate
gcloud auth login
gcloud auth application-default login

# Set project and region
gcloud config set project distributed-map-475111-h2
gcloud config set dataproc/region asia-southeast1

# Set environment variables
$env:PROJECT_ID = "distributed-map-475111-h2"
$env:REGION = "asia-southeast1"
$env:BUCKET = "weather-ml-bucket-1760514177"
```

### Step 1: Test with Sample Data (Recommended)

Process 1% sample to validate the pipeline:

```powershell
# Upload test script
gsutil cp noaa_cleanup_test.py gs://weather-ml-bucket-1760514177/scripts/

# Run test cleanup
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/noaa_cleanup_test.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --batch=cleanup-test-$timestamp

# Monitor job
gcloud dataproc batches list --region=asia-southeast1 --limit=5

# View logs in browser
# https://console.cloud.google.com/dataproc/batches?project=distributed-map-475111-h2&region=asia-southeast1
```

**Expected Output:**
- Sample size: ~1% of total data
- Processing time: 5-10 minutes
- Output location: `gs://weather-ml-bucket-1760514177/warehouse/noaa_clean_test/`

### Step 2: Process Full Dataset

Once test succeeds, process the complete 50GB dataset:

```powershell
# Upload full script (modify noaa_cleanup_test.py: change sample fraction from 0.01 to 1.0)
# Or create a new script without sampling

gsutil cp noaa_cleanup_full.py gs://weather-ml-bucket-1760514177/scripts/

# Run full cleanup
$timestamp = Get-Date -Format "yyyyMMdd-HHmmss"
gcloud dataproc batches submit pyspark `
    gs://weather-ml-bucket-1760514177/scripts/noaa_cleanup_full.py `
    --region=asia-southeast1 `
    --deps-bucket=weather-ml-bucket-1760514177 `
    --subnet=default `
    --properties=spark.sql.shuffle.partitions=200,spark.default.parallelism=200 `
    --batch=cleanup-full-$timestamp
```

**Expected Output:**
- Processing time: 30-60 minutes (depends on cluster size)
- Output location: `gs://weather-ml-bucket-1760514177/warehouse/noaa_clean_std/`
- Partitioned by: `year=2024/month=1/`, `year=2024/month=2/`, etc.

### Step 3: Verify Cleaned Data

```powershell
# Check output structure
gsutil ls gs://weather-ml-bucket-1760514177/warehouse/noaa_clean_std/

# View sample with PySpark inspection script
# (Similar to inspect_data_v2.py but pointing to cleaned data)
```

## Monitoring Jobs

### Using Console (Recommended)
1. Navigate to: [Dataproc Batches Console](https://console.cloud.google.com/dataproc/batches?project=distributed-map-475111-h2&region=asia-southeast1)
2. Click on your batch job
3. View status, logs, and metrics

### Using CLI

```powershell
# List recent batches
gcloud dataproc batches list --region=asia-southeast1 --limit=10

# Get batch status
$BATCH_ID = "cleanup-test-YYYYMMDD-HHMMSS"
gcloud dataproc batches describe $BATCH_ID --region=asia-southeast1

# View logs
gcloud logging read "resource.type=cloud_dataproc_batch AND resource.labels.batch_id=$BATCH_ID" `
    --limit=500 `
    --format="table(textPayload)" `
    --project=distributed-map-475111-h2
```

## Output Schema

The cleaned dataset has the following schema:

```
root
 |-- STATION: string
 |-- timestamp: timestamp
 |-- latitude: double
 |-- longitude: double
 |-- elevation: double
 |-- station_name: string
 |-- temperature: double (TARGET)
 |-- dew_point: double
 |-- sea_level_pressure: double
 |-- visibility: double
 |-- wind_direction: integer
 |-- wind_speed: double
 |-- precipitation: double
 |-- year: integer
 |-- month: integer
 |-- day: integer
 |-- hour: integer
 |-- hour_sin: double
 |-- hour_cos: double
 |-- month_sin: double
 |-- month_cos: double
 |-- wind_dir_sin: double
 |-- wind_dir_cos: double
```

## Data Statistics

After cleanup, expect the following characteristics:

### Data Volume
- **Input**: ~50GB CSV, millions of observations
- **Output**: Compressed Parquet, partitioned by year/month
- **Sample rate**: Test uses 1%, full uses 100%

### Feature Ranges
- **Temperature**: -90°C to +60°C
- **Dew Point**: -100°C to +50°C (always ≤ temperature)
- **Pressure**: 950 to 1050 hPa
- **Wind Speed**: 0 to 100 m/s
- **Visibility**: 0 to 999999 meters
- **Precipitation**: 0 to 999 mm

### Missing Value Rates (Typical)
- Temperature: 0% (filtered out)
- Dew Point: ~5-10%
- Sea Level Pressure: ~10-15%
- Visibility: ~20-30%
- Wind Speed: ~5-10%
- Precipitation: ~30-40% (treated as 0)

## Troubleshooting

### Common Issues

**Issue 1: "Path not found" error**
```
Solution: Verify input path exists
gsutil ls gs://weather-ml-bucket-1760514177/data/csv/
```

**Issue 2: "Insufficient resources" error**
```
Solution: Reduce sample size or increase cluster resources
Add properties: --properties=spark.executor.memory=4g,spark.driver.memory=4g
```

**Issue 3: Job stuck in PENDING**
```
Solution: Check Dataproc quotas and subnet configuration
gcloud compute networks subnets describe default --region=asia-southeast1
```

**Issue 4: High null rate in output**
```
Solution: Review quality code filtering in parse functions
May need to accept quality codes other than '1'
```

### Performance Tuning

For large datasets, adjust Spark properties:

```powershell
--properties=spark.sql.shuffle.partitions=400,`
  spark.default.parallelism=400,`
  spark.executor.memory=8g,`
  spark.driver.memory=8g,`
  spark.executor.cores=4
```

## Next Steps

After cleanup is complete:

1. **Split data** into train (70%) and test (30%) sets
2. **Train models** using cleaned dataset:
   - Random Forest Regression
   - Gradient Boosted Trees
3. **Evaluate** using RMSE and other metrics
4. **Feature engineering**: Consider additional derived features

## File Descriptions

### Scripts

- **noaa_cleanup_test.py**: Test version, processes 1% sample for validation
- **noaa_cleanup_full.py**: Production version, processes entire dataset
- **inspect_data_v2.py**: Data inspection utility

### Key Functions

- `parse_temperature()`: Extracts temperature from string format
- `parse_wind()`: Parses wind direction and speed
- `parse_pressure()`: Extracts sea level pressure
- `parse_visibility()`: Parses visibility distance
- `parse_precipitation()`: Sums all precipitation measurements

## References

- **NOAA Dataset Documentation**: https://www.ncei.noaa.gov/data/global-hourly/doc/
- **Google Cloud Dataproc**: https://cloud.google.com/dataproc/docs
- **PySpark Documentation**: https://spark.apache.org/docs/latest/api/python/

## Contact & Support

For issues or questions about this pipeline:
1. Check Dataproc batch logs in Cloud Console
2. Review this README for common issues
3. Verify all prerequisites are met

---

**Last Updated**: October 24, 2024  
**Version**: 1.0  
**Author**: Project Team  
**Course**: DSS5208 Project 2 - Machine Learning on Weather Data