"""
Baseline Model Test - Simple Linear Regression
Quick test to verify the ML pipeline works with cleaned data
Uses a small sample for fast validation
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, when
import sys

def main():
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: baseline_model_test.py <train_path> <test_path> <output_path>")
        print("Example: baseline_model_test.py gs://bucket/warehouse/noaa_train gs://bucket/warehouse/noaa_test gs://bucket/outputs/baseline_test")
        sys.exit(1)
    
    train_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Baseline Model Test") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Prediction - Baseline Model Test")
    print("=" * 80)
    print(f"Train path:  {train_path}")
    print(f"Test path:   {test_path}")
    print(f"Output path: {output_path}")
    
    # Read data
    print(f"\nReading training data...")
    train_df = spark.read.parquet(train_path)
    
    print(f"Reading test data...")
    test_df = spark.read.parquet(test_path)
    
    # Sample 10% for quick baseline test
    print("\nSampling 10% of data for quick baseline test...")
    train_sample = train_df.sample(fraction=0.1, seed=42)
    test_sample = test_df.sample(fraction=0.1, seed=42)
    
    train_count = train_sample.count()
    test_count = test_sample.count()
    
    print(f"Train sample: {train_count:,} rows")
    print(f"Test sample: {test_count:,} rows")
    
    # Select features (only complete cases for baseline)
    print("\nPreparing features...")
    feature_cols = [
        'latitude', 'longitude', 'elevation',
        'dew_point', 'sea_level_pressure', 'visibility',
        'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'precipitation'
    ]
    
    # Remove rows with any missing values for baseline (simple approach)
    train_clean = train_sample.na.drop(subset=feature_cols + ['temperature'])
    test_clean = test_sample.na.drop(subset=feature_cols + ['temperature'])
    
    print(f"\nAfter removing missing values:")
    print(f"Train: {train_clean.count():,} rows")
    print(f"Test: {test_clean.count():,} rows")
    
    # Assemble features
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features"
    )
    
    train_assembled = assembler.transform(train_clean).select('features', col('temperature').alias('label'))
    test_assembled = assembler.transform(test_clean).select('features', col('temperature').alias('label'))
    
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
        ("Linear Regression Baseline", 
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
    print("✓ BASELINE MODEL TEST COMPLETED!")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Model: Linear Regression (Baseline)")
    print(f"  Test RMSE: {test_rmse:.4f}°C")
    print(f"  Test R²: {test_r2:.4f}")
    print(f"  Test MAE: {test_mae:.4f}°C")
    print(f"\nOutput location: {output_path}")
    print(f"\n{'='*80}")
    print("Pipeline verified! Ready for advanced models (Random Forest, GBT)")
    print("="*80)
    
    spark.stop()

if __name__ == "__main__":
    main()