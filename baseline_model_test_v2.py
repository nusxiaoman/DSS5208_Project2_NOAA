"""
Baseline Model Test V2 - Simple Linear Regression
Tests ML pipeline with V2 cleaned data (30 features - no station stats)
Uses 10% sample for fast validation
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when
import sys

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Baseline Model Test V2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths - V2 data
    train_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_train_v2"
    test_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_test_v2"
    output_path = "gs://weather-ml-bucket-1760514177/outputs/baseline_test_v2"
    
    print("=" * 80)
    print("NOAA Weather Prediction - Baseline Model Test V2")
    print("=" * 80)
    print("Using V2 cleaned data with 30 ML features (no station stats)")
    print("=" * 80)
    
    # Read data
    print(f"\nReading training data from: {train_path}")
    train_df = spark.read.parquet(train_path)
    
    print(f"Reading test data from: {test_path}")
    test_df = spark.read.parquet(test_path)
    
    # Sample 10% for quick baseline test
    print("\nSampling 10% of data for quick baseline test...")
    train_sample = train_df.sample(fraction=0.1, seed=42)
    test_sample = test_df.sample(fraction=0.1, seed=42)
    
    train_count = train_sample.count()
    test_count = test_sample.count()
    
    print(f"Train sample: {train_count:,} rows")
    print(f"Test sample: {test_count:,} rows")
    
    # Select features - V2 has 30 ML features (removed 3 station stats due to data leakage)
    print("\nPreparing V2 features (30 total - excluded station stats)...")
    feature_cols = [
        # Geographic (3)
        'latitude', 'longitude', 'elevation',
        # Basic weather (6)
        'dew_point', 'sea_level_pressure', 'visibility',
        'wind_speed', 'precipitation', 'wind_gust',
        # Cyclical encodings (8)
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'day_of_year_sin', 'day_of_year_cos',
        'wind_dir_sin', 'wind_dir_cos',
        # Weather conditions (4)
        'is_raining', 'is_snowing', 'is_foggy', 'is_thunderstorm',
        # Lag features (9) - CRITICAL for performance
        'temp_lag_1h', 'temp_lag_2h', 'temp_lag_3h',
        'pressure_lag_1h', 'dew_lag_1h',
        'temp_rolling_3h', 'pressure_rolling_3h',
        'temp_change_1h', 'pressure_change_1h'
    ]
    
    print(f"Total features: {len(feature_cols)}")
    print("Categories:")
    print("  Geographic: 3")
    print("  Basic weather: 6")
    print("  Cyclical encodings: 8")
    print("  Weather conditions: 4")
    print("  Lag features: 9")
    print("  ⚠ REMOVED station_avg_* (3) - data leakage!")
    
    # Prepare data - filter only null temperatures, keep other nulls
    train_data = train_sample.select(*feature_cols, col('temperature').alias('label')) \
        .filter(col('label').isNotNull())
    test_data = test_sample.select(*feature_cols, col('temperature').alias('label')) \
        .filter(col('label').isNotNull())
    
    print(f"\nAfter filtering null temperatures:")
    print(f"Train: {train_data.count():,} rows")
    print(f"Test: {test_data.count():,} rows")
    
    # Check for features that are 100% NULL (Imputer can't handle these)
    print("\nChecking for completely NULL features...")
    from pyspark.sql.functions import count as spark_count, when
    
    valid_features = []
    excluded_features = []
    
    for feat in feature_cols:
        non_null_count = train_data.select(spark_count(when(col(feat).isNotNull(), 1))).first()[0]
        if non_null_count > 0:
            valid_features.append(feat)
        else:
            excluded_features.append(feat)
            print(f"  ⚠ Excluding {feat} (100% NULL)")
    
    if excluded_features:
        print(f"\nUsing {len(valid_features)} features (excluded {len(excluded_features)} all-NULL features)")
    else:
        print(f"\nAll {len(valid_features)} features have non-NULL values")
    
    # Impute missing values with median (lag features have many nulls)
    from pyspark.ml.feature import Imputer
    from pyspark.ml import Pipeline
    
    print("\nImputing missing values with median...")
    imputer = Imputer(
        inputCols=valid_features,
        outputCols=[f"{feat}_imputed" for feat in valid_features],
        strategy='median'
    )
    
    # Assemble features
    imputed_features = [f"{feat}_imputed" for feat in valid_features]
    assembler = VectorAssembler(
        inputCols=imputed_features,
        outputCol="features"
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[imputer, assembler])
    
    # Fit and transform
    pipeline_model = pipeline.fit(train_data)
    train_assembled = pipeline_model.transform(train_data).select('features', 'label')
    test_assembled = pipeline_model.transform(test_data).select('features', 'label')
    
    print(f"After imputation:")
    print(f"Train: {train_assembled.count():,} rows")
    print(f"Test: {test_assembled.count():,} rows")
    
    # Train simple Linear Regression
    print("\n" + "=" * 80)
    print("Training Linear Regression (Baseline Model)...")
    print("=" * 80)
    
    lr = LinearRegression(
        featuresCol='features',
        labelCol='label',
        maxIter=10,
        regParam=0.1
    )
    
    print("Fitting model...")
    model = lr.fit(train_assembled)
    
    print("\nModel coefficients:")
    print(f"Intercept: {model.intercept:.4f}")
    print(f"Number of features: {len(model.coefficients)}")
    
    # Make predictions on train set
    print("\n" + "=" * 80)
    print("Evaluating on TRAINING set...")
    print("=" * 80)
    
    train_predictions = model.transform(train_assembled)
    
    train_evaluator = RegressionEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='rmse'
    )
    
    train_rmse = train_evaluator.evaluate(train_predictions)
    train_r2 = train_evaluator.setMetricName('r2').evaluate(train_predictions)
    train_mae = train_evaluator.setMetricName('mae').evaluate(train_predictions)
    
    print(f"Training RMSE: {train_rmse:.4f}°C")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}°C")
    
    # Make predictions on test set
    print("\n" + "=" * 80)
    print("Evaluating on TEST set...")
    print("=" * 80)
    
    test_predictions = model.transform(test_assembled)
    
    test_evaluator = RegressionEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='rmse'
    )
    
    test_rmse = test_evaluator.evaluate(test_predictions)
    test_r2 = test_evaluator.setMetricName('r2').evaluate(test_predictions)
    test_mae = test_evaluator.setMetricName('mae').evaluate(test_predictions)
    
    print(f"Test RMSE: {test_rmse:.4f}°C")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.4f}°C")
    
    # Show sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (10 examples)")
    print("=" * 80)
    test_predictions.select('label', 'prediction').show(10, truncate=False)
    
    # Calculate prediction errors
    from pyspark.sql.functions import abs as spark_abs
    
    test_with_error = test_predictions.withColumn(
        'error', spark_abs(col('label') - col('prediction'))
    )
    
    print("\n" + "=" * 80)
    print("PREDICTION ERROR STATISTICS")
    print("=" * 80)
    test_with_error.select('error').describe().show()
    
    # Save results summary
    print(f"\nSaving results to: {output_path}")
    
    results_data = [
        ("Linear Regression Baseline V2", 
         train_count, test_count, 
         float(train_rmse), float(test_rmse),
         float(train_r2), float(test_r2),
         float(train_mae), float(test_mae))
    ]
    
    results_df = spark.createDataFrame(
        results_data,
        ["model", "train_rows", "test_rows", 
         "train_rmse", "test_rmse", "train_r2", "test_r2", 
         "train_mae", "test_mae"]
    )
    
    results_df.write.mode('overwrite').csv(output_path + "/metrics", header=True)
    
    # Save sample predictions
    test_predictions.limit(1000).write.mode('overwrite').parquet(output_path + "/sample_predictions")
    
    print("\n" + "=" * 80)
    print("✓ BASELINE MODEL TEST V2 COMPLETED!")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Model: Linear Regression (Baseline V2)")
    print(f"  Features: {len(valid_features)} (target: 33, excluded {len(excluded_features)} all-NULL)")
    print(f"  Test RMSE: {test_rmse:.4f}°C")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}°C")
    print(f"\nOutput location: {output_path}")
    print(f"\n{'='*80}")
    print("V2 Pipeline verified! Ready for RF and GBT with enhanced features")
    print("Expected improvement from lag features!")
    print("="*80)
    
    spark.stop()

if __name__ == "__main__":
    main()