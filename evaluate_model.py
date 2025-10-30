"""
Model Evaluation Script
Evaluates trained models on the test set and generates comprehensive metrics
"""

from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, abs as spark_abs, pow as spark_pow
import sys

def main():
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: evaluate_model.py <model_path> <test_path> <output_path>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_path = sys.argv[2]
    output_path = sys.argv[3]
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Model Evaluation") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Prediction - Model Evaluation on Test Set")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model = PipelineModel.load(model_path)
    
    # Read test data
    print(f"Reading test data from: {test_path}")
    test_df = spark.read.parquet(test_path)
    
    test_count = test_df.count()
    print(f"Test rows: {test_count:,}")
    
    # Prepare test data
    numeric_features = [
        'latitude', 'longitude', 'elevation',
        'dew_point', 'sea_level_pressure', 'visibility',
        'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'precipitation'
    ]
    
    test_data = test_df.select(
        *numeric_features,
        col('temperature').alias('label')
    ).filter(col('label').isNotNull())
    
    print(f"Test rows after filtering: {test_data.count():,}")
    
    # Make predictions
    print("\n" + "=" * 80)
    print("GENERATING PREDICTIONS ON TEST SET...")
    print("=" * 80)
    
    predictions = model.transform(test_data)
    
    # Calculate metrics
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION METRICS")
    print("=" * 80)
    
    evaluator = RegressionEvaluator(
        labelCol='label',
        predictionCol='prediction'
    )
    
    # RMSE
    rmse = evaluator.setMetricName('rmse').evaluate(predictions)
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}°C")
    
    # R²
    r2 = evaluator.setMetricName('r2').evaluate(predictions)
    print(f"R² Score: {r2:.4f}")
    
    # MAE
    mae = evaluator.setMetricName('mae').evaluate(predictions)
    print(f"Mean Absolute Error (MAE): {mae:.4f}°C")
    
    # MSE
    mse = evaluator.setMetricName('mse').evaluate(predictions)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    
    # Additional metrics
    predictions_with_error = predictions.withColumn(
        'error', col('label') - col('prediction')
    ).withColumn(
        'abs_error', spark_abs(col('error'))
    ).withColumn(
        'squared_error', spark_pow(col('error'), 2)
    )
    
    # Error statistics
    print("\n" + "=" * 80)
    print("ERROR DISTRIBUTION STATISTICS")
    print("=" * 80)
    
    error_stats = predictions_with_error.select('error', 'abs_error').describe()
    error_stats.show()
    
    # Prediction quality by temperature range
    print("\n" + "=" * 80)
    print("PERFORMANCE BY TEMPERATURE RANGE")
    print("=" * 80)
    
    from pyspark.sql.functions import when, avg, count as spark_count
    
    temp_ranges = predictions_with_error.withColumn(
        'temp_range',
        when(col('label') < 0, 'Below 0°C')
        .when((col('label') >= 0) & (col('label') < 10), '0-10°C')
        .when((col('label') >= 10) & (col('label') < 20), '10-20°C')
        .when((col('label') >= 20) & (col('label') < 30), '20-30°C')
        .otherwise('Above 30°C')
    )
    
    temp_ranges.groupBy('temp_range') \
        .agg(
            spark_count('*').alias('count'),
            avg('abs_error').alias('mean_abs_error'),
            avg('squared_error').alias('mean_squared_error')
        ) \
        .orderBy('temp_range') \
        .show()
    
    # Show sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (20 examples)")
    print("=" * 80)
    predictions.select('label', 'prediction').show(20, truncate=False)
    
    # Best and worst predictions
    print("\n" + "=" * 80)
    print("BEST PREDICTIONS (smallest error)")
    print("=" * 80)
    predictions_with_error.orderBy('abs_error').select(
        'label', 'prediction', 'abs_error'
    ).show(10, truncate=False)
    
    print("\n" + "=" * 80)
    print("WORST PREDICTIONS (largest error)")
    print("=" * 80)
    predictions_with_error.orderBy(col('abs_error').desc()).select(
        'label', 'prediction', 'abs_error'
    ).show(10, truncate=False)
    
    # Save results
    print(f"\nSaving evaluation results to: {output_path}")
    
    # Save metrics
    metrics_data = [(
        float(rmse),
        float(r2),
        float(mae),
        float(mse),
        int(test_count)
    )]
    
    metrics_df = spark.createDataFrame(
        metrics_data,
        ["test_rmse", "test_r2", "test_mae", "test_mse", "test_rows"]
    )
    
    metrics_df.write.mode('overwrite').csv(output_path + "/test_metrics", header=True)
    
    # Save all predictions
    print("Saving all predictions...")
    predictions.select('label', 'prediction').write.mode('overwrite') \
        .parquet(output_path + "/test_predictions")
    
    # Save error analysis
    print("Saving error analysis...")
    predictions_with_error.select('label', 'prediction', 'error', 'abs_error') \
        .write.mode('overwrite').parquet(output_path + "/error_analysis")
    
    print("\n" + "=" * 80)
    print("✓ MODEL EVALUATION COMPLETED!")
    print("=" * 80)
    print(f"\nTest Set Results:")
    print(f"  Test RMSE: {rmse:.4f}°C")
    print(f"  Test R²: {r2:.4f}")
    print(f"  Test MAE: {mae:.4f}°C")
    print(f"  Test rows: {test_count:,}")
    print(f"\nOutput location: {output_path}")
    print(f"  - test_metrics/")
    print(f"  - test_predictions/")
    print(f"  - error_analysis/")
    
    spark.stop()

if __name__ == "__main__":
    main()