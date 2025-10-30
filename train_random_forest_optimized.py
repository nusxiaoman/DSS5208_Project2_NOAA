"""
Random Forest Training Script - OPTIMIZED for Large Dataset
Reduced hyperparameter grid based on test results
Uses best-performing parameters from test mode as baseline
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, abs as spark_abs
import sys

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: train_random_forest_optimized.py <train_path> <output_path>")
        sys.exit(1)
    
    train_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # Initialize Spark with optimized settings
    spark = SparkSession.builder \
        .appName("NOAA Random Forest - Optimized") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.sql.shuffle.partitions", "400") \
        .config("spark.default.parallelism", "400") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Prediction - Random Forest (OPTIMIZED)")
    print("=" * 80)
    print("Optimizations:")
    print("  - Reduced hyperparameter grid (8 models vs 18)")
    print("  - Focused search around test mode best params")
    print("  - 3-fold cross-validation")
    
    # Read training data
    print(f"\nReading training data from: {train_path}")
    train_df = spark.read.parquet(train_path)
    
    train_count = train_df.count()
    print(f"Training rows: {train_count:,}")
    
    # Define features
    numeric_features = [
        'latitude', 'longitude', 'elevation',
        'dew_point', 'sea_level_pressure', 'visibility',
        'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'precipitation'
    ]
    
    print(f"\nUsing {len(numeric_features)} features")
    
    # Prepare data
    data = train_df.select(
        *numeric_features,
        col('temperature').alias('label')
    ).filter(col('label').isNotNull())
    
    print(f"Rows after filtering null labels: {data.count():,}")
    
    # Impute missing values
    print("\nImputing missing values with median...")
    imputer = Imputer(
        inputCols=numeric_features,
        outputCols=[f"{feat}_imputed" for feat in numeric_features],
        strategy='median'
    )
    
    # Assemble features
    imputed_features = [f"{feat}_imputed" for feat in numeric_features]
    assembler = VectorAssembler(
        inputCols=imputed_features,
        outputCol='features'
    )
    
    # Random Forest Regressor
    rf = RandomForestRegressor(
        featuresCol='features',
        labelCol='label',
        seed=42
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[imputer, assembler, rf])
    
    # OPTIMIZED Hyperparameter grid
    # Test mode showed: numTrees=20, maxDepth=10 was best
    # So we search around those values with reduced grid
    print("\n" + "=" * 80)
    print("OPTIMIZED Hyperparameter Grid")
    print("=" * 80)
    print("Based on test results (best: numTrees=20, maxDepth=10)")
    print("Grid: 2 numTrees × 2 maxDepth × 2 minInstances = 8 models")
    print("3-fold CV = 24 training runs (vs 54 in original)")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [50, 100]) \
        .addGrid(rf.maxDepth, [10, 15]) \
        .addGrid(rf.minInstancesPerNode, [1, 5]) \
        .build()
    
    # Cross-validator
    evaluator = RegressionEvaluator(
        labelCol='label',
        predictionCol='prediction',
        metricName='rmse'
    )
    
    cv = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=evaluator,
        numFolds=3,
        seed=42,
        parallelism=2  # Reduced to avoid memory issues
    )
    
    # Train model
    print("\nTraining Random Forest with optimized grid...")
    print("Expected time: 4-8 hours (vs 40+ hours for full grid)")
    
    cv_model = cv.fit(data)
    
    # Get best model
    best_model = cv_model.bestModel
    best_rf = best_model.stages[-1]
    
    print("\n" + "=" * 80)
    print("BEST MODEL PARAMETERS")
    print("=" * 80)
    print(f"Number of Trees: {best_rf.getNumTrees}")
    print(f"Max Depth: {best_rf.getMaxDepth()}")
    print(f"Min Instances Per Node: {best_rf.getMinInstancesPerNode()}")
    
    # Feature importances
    print("\n" + "=" * 80)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 80)
    
    importances = best_rf.featureImportances.toArray()
    feature_importance = list(zip(numeric_features, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"{i:2d}. {feature:25s} {importance:.4f}")
    
    # Make predictions on training set
    print("\n" + "=" * 80)
    print("TRAINING SET EVALUATION")
    print("=" * 80)
    
    train_predictions = best_model.transform(data)
    
    train_rmse = evaluator.evaluate(train_predictions)
    train_r2 = evaluator.setMetricName('r2').evaluate(train_predictions)
    train_mae = evaluator.setMetricName('mae').evaluate(train_predictions)
    
    print(f"Training RMSE: {train_rmse:.4f}°C")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Training MAE: {train_mae:.4f}°C")
    
    # Cross-validation metrics
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 80)
    
    avg_metrics = cv_model.avgMetrics
    print(f"Best CV RMSE: {min(avg_metrics):.4f}°C")
    print(f"Worst CV RMSE: {max(avg_metrics):.4f}°C")
    print(f"Mean CV RMSE: {sum(avg_metrics)/len(avg_metrics):.4f}°C")
    
    # Show sample predictions
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS (10 examples)")
    print("=" * 80)
    train_predictions.select('label', 'prediction').show(10, truncate=False)
    
    # Save model
    print(f"\nSaving best model to: {output_path}/best_rf_model")
    best_model.write().overwrite().save(output_path + "/best_rf_model")
    
    # Save metrics
    print(f"Saving metrics to: {output_path}/metrics")
    
    metrics_data = [(
        "RandomForest_Optimized",
        "full",
        int(train_count),
        int(best_rf.getNumTrees),
        int(best_rf.getMaxDepth()),
        int(best_rf.getMinInstancesPerNode()),
        float(train_rmse),
        float(train_r2),
        float(train_mae),
        float(min(avg_metrics))
    )]
    
    metrics_df = spark.createDataFrame(
        metrics_data,
        ["model", "mode", "train_rows", "num_trees", "max_depth", "min_instances",
         "train_rmse", "train_r2", "train_mae", "best_cv_rmse"]
    )
    
    metrics_df.write.mode('overwrite').csv(output_path + "/metrics", header=True)
    
    # Save feature importances
    print(f"Saving feature importances to: {output_path}/feature_importances")
    
    importance_data = [(feat, float(imp)) for feat, imp in feature_importance]
    importance_df = spark.createDataFrame(importance_data, ["feature", "importance"])
    importance_df.write.mode('overwrite').csv(output_path + "/feature_importances", header=True)
    
    # Save sample predictions
    train_predictions.limit(10000).write.mode('overwrite') \
        .parquet(output_path + "/sample_predictions")
    
    print("\n" + "=" * 80)
    print("✓ RANDOM FOREST TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Model: Random Forest (Optimized Grid)")
    print(f"  Training rows: {train_count:,}")
    print(f"  Models tested: 8 (vs 18 in full grid)")
    print(f"  Best num_trees: {best_rf.getNumTrees}")
    print(f"  Best max_depth: {best_rf.getMaxDepth()}")
    print(f"  Best min_instances: {best_rf.getMinInstancesPerNode()}")
    print(f"  Training RMSE: {train_rmse:.4f}°C")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Best CV RMSE: {min(avg_metrics):.4f}°C")
    print(f"\nOutput location: {output_path}")
    
    spark.stop()

if __name__ == "__main__":
    main()