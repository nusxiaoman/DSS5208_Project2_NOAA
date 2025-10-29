"""
Baseline Model Test V2 - Simple Linear Regression with NULL Handling
Quick test to verify the ML pipeline works with V2 cleaned data (17 features)
Uses 10% sample + Median Imputation for NULLs
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Baseline Model Test V2 - with Imputer") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths - V2
    train_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_train_v2"
    test_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_test_v2"
    output_path = "gs://weather-ml-bucket-1760514177/outputs/baseline_test_v2"
    
    print("=" * 80)
    print("NOAA Weather Prediction - Baseline Model Test V2 (with Imputer)")
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
    
    # Select features (V2 adds ceiling_height)
    print("\nPreparing features...")
    
    # Features that may have NULLs and need imputation
    features_with_nulls = [
        'dew_point',
        'sea_level_pressure', 
        'visibility',
        'wind_speed',
        'wind_dir_sin',
        'wind_dir_cos',
        'ceiling_height'
    ]
    
    # Features without NULLs
    features_no_nulls = [
        'latitude',
        'longitude', 
        'elevation',
        'hour_sin',
        'hour_cos',
        'month_sin',
        'month_cos',
        'precipitation'
    ]
    
    all_features = features_no_nulls + features_with_nulls
    
    print(f"\nTotal features: {len(all_features)}")
    print(f"  Features without NULLs: {len(features_no_nulls)}")
    print(f"  Features with NULLs (will impute): {len(features_with_nulls)}")
    print(f"\nNEW in V2: ceiling_height")
    
    # Filter out rows where temperature is NULL (target variable)
    print("\nRemoving rows with NULL temperature (target)...")
    train_clean = train_sample.filter(col('temperature').isNotNull())
    test_clean = test_sample.filter(col('temperature').isNotNull())
    
    print(f"After removing NULL targets:")
    print(f"  Train: {train_clean.count():,} rows")
    print(f"  Test: {test_clean.count():,} rows")
    
    # Apply median imputation for features with NULLs
    print("\n" + "=" * 80)
    print("APPLYING MEDIAN IMPUTATION")
    print("=" * 80)
    
    imputer = Imputer(
        inputCols=features_with_nulls,
        outputCols=[f"{c}_imputed" for c in features_with_nulls],
        strategy="median"
    )
    
    print("Fitting imputer on training data...")
    imputer_model = imputer.fit(train_clean)
    
    print("Transforming training data...")
    train_imputed = imputer_model.transform(train_clean)
    
    print("Transforming test data...")
    test_imputed = imputer_model.transform(test_clean)
    
    print("DONE - NULL values imputed with median")
    
    # Build final feature list (use imputed columns)
    final_features = features_no_nulls + [f"{c}_imputed" for c in features_with_nulls]
    
    print(f"\nFinal feature list: {len(final_features)} features")
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=final_features,
        outputCol="features",
        handleInvalid='skip'  # Skip any remaining invalid values
    )
    
    train_assembled = assembler.transform(train_imputed).select('features', col('temperature').alias('label'))
    test_assembled = assembler.transform(test_imputed).select('features', col('temperature').alias('label'))
    
    final_train_count = train_assembled.count()
    final_test_count = test_assembled.count()
    
    print(f"\nAfter feature assembly:")
    print(f"  Train: {final_train_count:,} rows")
    print(f"  Test: {final_test_count:,} rows")
    
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
        ("Linear Regression Baseline V2 (with Imputer)", 
         final_train_count, final_test_count, 
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
    print(f"  Model: Linear Regression (Baseline V2 with Imputer)")
    print(f"  Features: {len(all_features)} (V0 had 14)")
    print(f"  NULL Handling: Median imputation")
    print(f"  Test RMSE: {test_rmse:.4f}°C")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}°C")
    print(f"\nOutput location: {output_path}")
    
    # Compare with V0
    print("\n" + "=" * 80)
    print("COMPARISON WITH V0")
    print("=" * 80)
    print("V0 Baseline (14 features, dropped NULLs) - provide your RMSE")
    print(f"V2 Baseline (15 features, imputed NULLs) - RMSE: {test_rmse:.4f}°C")
    print("\nKey differences:")
    print("  + NEW feature: ceiling_height")
    print("  + Better NULL handling: median imputation vs dropping rows")
    print("  + More training data retained")
    print("=" * 80)
    print("Pipeline verified! Ready for RF and GBT test models")
    print("=" * 80)
    
    spark.stop()

if __name__ == "__main__":
    main()