"""
Gradient Boosted Trees Training Script for NOAA Weather Data
Trains GBT Regression model with hyperparameter tuning
Uses cleaned features with proper handling of missing values
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, abs as spark_abs
import sys

def main():
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("Usage: train_gbt.py <train_path> <output_path> [mode]")
        print("mode: 'test' (10% sample, quick) or 'full' (100%, production)")
        sys.exit(1)
    
    train_path = sys.argv[1]
    output_path = sys.argv[2]
    mode = sys.argv[3] if len(sys.argv) > 3 else 'full'
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA GBT Training") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Prediction - Gradient Boosted Trees")
    print("=" * 80)
    print(f"Mode: {mode.upper()}")
    
    # Read training data
    print(f"\nReading training data from: {train_path}")
    train_df = spark.read.parquet(train_path)
    
    # Sample for test mode
    if mode == 'test':
        print("\nTest mode: Using 10% sample for quick training...")
        train_df = train_df.sample(fraction=0.1, seed=42)
    
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
    
    print(f"\nUsing {len(numeric_features)} features:")
    for feat in numeric_features:
        print(f"  - {feat}")
    
    # Prepare data - select features and target
    data = train_df.select(
        *numeric_features,
        col('temperature').alias('label')
    ).filter(col('label').isNotNull())
    
    print(f"\nRows after filtering null labels: {data.count():,}")
    
    # Impute missing values with median
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
    
    # Gradient Boosted Trees Regressor
    gbt = GBTRegressor(
        featuresCol='features',
        labelCol='label',
        seed=42
    )
    
    # Create pipeline
    pipeline = Pipeline(stages=[imputer, assembler, gbt])
    
    # Hyperparameter tuning
    print("\n" + "=" * 80)
    print("Setting up hyperparameter grid...")
    print("=" * 80)
    
    if mode == 'test':
        # Quick test parameters
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxIter, [10, 20]) \
            .addGrid(gbt.maxDepth, [3, 5]) \
            .addGrid(gbt.stepSize, [0.1]) \
            .build()
        numFolds = 2
        print("Test mode: 2 x 2 = 4 models, 2-fold CV")
    else:
        # Full production parameters
        paramGrid = ParamGridBuilder() \
            .addGrid(gbt.maxIter, [50, 100, 150]) \
            .addGrid(gbt.maxDepth, [5, 7, 10]) \
            .addGrid(gbt.stepSize, [0.05, 0.1, 0.2]) \
            .build()
        numFolds = 3
        print("Full mode: 3 x 3 x 3 = 27 models, 3-fold CV")
    
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
        numFolds=numFolds,
        seed=42,
        parallelism=4
    )
    
    # Train model
    print("\nTraining Gradient Boosted Trees with cross-validation...")
    print("(This may take 30-60 minutes for test mode, 3-6 hours for full mode)")
    
    cv_model = cv.fit(data)
    
    # Get best model
    best_model = cv_model.bestModel
    best_gbt = best_model.stages[-1]
    
    print("\n" + "=" * 80)
    print("BEST MODEL PARAMETERS")
    print("=" * 80)
    print(f"Max Iterations: {best_gbt.getMaxIter()}")
    print(f"Max Depth: {best_gbt.getMaxDepth()}")
    print(f"Step Size: {best_gbt.getStepSize()}")
    print(f"Number of Trees: {best_gbt.getNumTrees}")
    
    # Feature importances
    print("\n" + "=" * 80)
    print("TOP 10 FEATURE IMPORTANCES")
    print("=" * 80)
    
    importances = best_gbt.featureImportances.toArray()
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
    print(f"\nSaving best model to: {output_path}/best_gbt_model")
    best_model.write().overwrite().save(output_path + "/best_gbt_model")
    
    # Save metrics
    print(f"Saving metrics to: {output_path}/metrics")
    
    metrics_data = [(
        "GradientBoostedTrees",
        mode,
        int(train_count),
        int(best_gbt.getMaxIter()),
        int(best_gbt.getMaxDepth()),
        float(best_gbt.getStepSize()),
        int(best_gbt.getNumTrees),
        float(train_rmse),
        float(train_r2),
        float(train_mae),
        float(min(avg_metrics))
    )]
    
    metrics_df = spark.createDataFrame(
        metrics_data,
        ["model", "mode", "train_rows", "max_iter", "max_depth", 
         "step_size", "num_trees", "train_rmse", "train_r2", "train_mae", "best_cv_rmse"]
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
    print("✓ GRADIENT BOOSTED TREES TRAINING COMPLETED!")
    print("=" * 80)
    print(f"\nResults Summary:")
    print(f"  Model: Gradient Boosted Trees")
    print(f"  Training rows: {train_count:,}")
    print(f"  Best max_iter: {best_gbt.getMaxIter()}")
    print(f"  Best max_depth: {best_gbt.getMaxDepth()}")
    print(f"  Best step_size: {best_gbt.getStepSize()}")
    print(f"  Number of trees: {best_gbt.getNumTrees}")
    print(f"  Training RMSE: {train_rmse:.4f}°C")
    print(f"  Training R²: {train_r2:.4f}")
    print(f"  Best CV RMSE: {min(avg_metrics):.4f}°C")
    print(f"\nOutput location: {output_path}")
    print(f"  - best_gbt_model/")
    print(f"  - metrics/")
    print(f"  - feature_importances/")
    print(f"  - sample_predictions/")
    
    spark.stop()

if __name__ == "__main__":
    main()