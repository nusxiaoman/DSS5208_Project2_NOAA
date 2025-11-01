#!/usr/bin/env python3
"""
Gradient Boosted Trees Training Script - Simplified Version
Memory-optimized for full 88M training dataset
Uses same proven approach as RF Simplified
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.storagelevel import StorageLevel
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml import Pipeline
import sys
import time

# Parse arguments
if len(sys.argv) < 3:
    print("Usage: train_gbt_simplified.py <train_parquet_path> <output_path>")
    sys.exit(1)

TRAIN_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2].rstrip("/")
LABEL_COL = "temperature"
SEED = 42

# Initialize Spark with optimizations
spark = SparkSession.builder \
    .appName("NOAA_GBT_Simplified") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Set helpful defaults
def set_if_absent(k, v):
    try:
        spark.conf.get(k)
    except:
        spark.conf.set(k, v)

set_if_absent("spark.sql.adaptive.enabled", "true")
set_if_absent("spark.sql.shuffle.partitions", "200")
set_if_absent("spark.default.parallelism", "200")

# Checkpoint for resilience
spark.sparkContext.setCheckpointDir(f"{OUTPUT_PATH}/_checkpoint")

print("=" * 80)
print("NOAA Gradient Boosted Trees - Simplified Version")
print("=" * 80)
print(f"Input (full training data): {TRAIN_PATH}")
print(f"Output: {OUTPUT_PATH}")

# -------------------- Load Training Data --------------------
print("\nLoading full training data (88M rows)...")
train_df = spark.read.parquet(TRAIN_PATH)

# Use correct feature set
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

# Prepare data
data = train_df.select(
    *numeric_features,
    F.col(LABEL_COL).alias('label')
).filter(F.col('label').isNotNull())

# Repartition for better parallelism
print("\nRepartitioning data for optimal processing...")
data = data.repartition(200)

# Persist for multiple passes during CV
data = data.persist(StorageLevel.MEMORY_AND_DISK)

train_count = data.count()
print(f"Training rows: {train_count:,}")

# -------------------- Build Pipeline --------------------
print("\n" + "=" * 80)
print("Building ML Pipeline")
print("=" * 80)

# Impute missing values
imputed_features = [f"{feat}_imputed" for feat in numeric_features]
imputer = Imputer(
    inputCols=numeric_features,
    outputCols=imputed_features,
    strategy='median'
)

# Assemble features
assembler = VectorAssembler(
    inputCols=imputed_features,
    outputCol='features',
    handleInvalid='keep'
)

# Gradient Boosted Trees (GBT is more memory-intensive than RF)
gbt = GBTRegressor(
    featuresCol='features',
    labelCol='label',
    seed=SEED
)

# Pipeline
pipeline = Pipeline(stages=[imputer, assembler, gbt])

# -------------------- Hyperparameter Grid --------------------
print("\nHyperparameter Grid (Memory-Optimized for GBT):")
print("  Conservative parameters based on test results:")
print("    - maxIter: [20, 50]")
print("    - maxDepth: [3, 5]")
print("    - stepSize: [0.1]")
print("  Total: 4 models × 2 folds = 8 training runs")
print("  Parallelism: 1 (sequential, memory-safe)")
print("\n  Note: GBT is more memory-intensive than RF due to:")
print("    - Sequential boosting (each tree depends on previous)")
print("    - Gradient calculations stored in memory")
print("    - Residual updates after each iteration")

param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxIter, [20, 50]) \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.stepSize, [0.1]) \
    .build()

# Evaluator
evaluator = RegressionEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='rmse'
)

# Cross-validator (conservative settings for GBT)
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=2,
    parallelism=1,  # Sequential to avoid memory pressure
    seed=SEED
)

# -------------------- Train --------------------
print("\n" + "=" * 80)
print("Training Gradient Boosted Trees with 2-Fold Cross-Validation")
print("=" * 80)
print("Expected runtime: ~2-3 hours (GBT slower than RF)")
print("Training 4 models sequentially...")

start_time = time.time()
cv_model = cv.fit(data)
training_duration = time.time() - start_time

print(f"\nTraining completed in {training_duration/60:.1f} minutes ({training_duration/3600:.2f} hours)")

# -------------------- Best Model --------------------
print("\n" + "=" * 80)
print("Best Model Parameters")
print("=" * 80)

best_model = cv_model.bestModel
best_gbt = best_model.stages[-1]

print(f"  maxIter: {best_gbt.getMaxIter()}")
print(f"  maxDepth: {best_gbt.getMaxDepth()}")
print(f"  stepSize: {best_gbt.getStepSize()}")
print(f"  numTrees: {best_gbt.getNumTrees}")

# -------------------- Training Set Evaluation --------------------
print("\n" + "=" * 80)
print("Training Set Evaluation (Internal CV)")
print("=" * 80)

# Make predictions on training data
train_predictions = cv_model.transform(data).checkpoint(eager=True)

train_rmse = evaluator.evaluate(train_predictions)
train_r2 = evaluator.setMetricName('r2').evaluate(train_predictions)
train_mae = evaluator.setMetricName('mae').evaluate(train_predictions)

print(f"Training RMSE: {train_rmse:.4f}°C")
print(f"Training R²:   {train_r2:.4f}")
print(f"Training MAE:  {train_mae:.4f}°C")

# Cross-validation results
avg_metrics = cv_model.avgMetrics
print(f"\n" + "=" * 80)
print("Cross-Validation Results (2-Fold)")
print("=" * 80)
print(f"Best CV RMSE:  {min(avg_metrics):.4f}°C")
print(f"Worst CV RMSE: {max(avg_metrics):.4f}°C")
print(f"Mean CV RMSE:  {sum(avg_metrics)/len(avg_metrics):.4f}°C")

# -------------------- Feature Importances --------------------
print("\n" + "=" * 80)
print("TOP 10 FEATURE IMPORTANCES")
print("=" * 80)

importances = best_gbt.featureImportances.toArray()
feature_importance = list(zip(numeric_features, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(feature_importance[:10], 1):
    print(f"{i:2d}. {feature:25s} {importance:.4f}")

# -------------------- Sample Predictions --------------------
print("\n" + "=" * 80)
print("SAMPLE PREDICTIONS (10 examples)")
print("=" * 80)
train_predictions.select('label', 'prediction').show(10, truncate=False)

# -------------------- Save Results --------------------
print(f"\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save metrics
print(f"Saving training metrics to {OUTPUT_PATH}/metrics/")
metrics_data = [(
    "GradientBoostedTrees",
    "full_simplified",
    int(train_count),
    int(best_gbt.getMaxIter()),
    int(best_gbt.getMaxDepth()),
    float(best_gbt.getStepSize()),
    int(best_gbt.getNumTrees),
    float(train_rmse),
    float(train_r2),
    float(train_mae),
    float(min(avg_metrics)),
    float(training_duration)
)]
spark.createDataFrame(
    metrics_data,
    ["model", "mode", "train_rows", "max_iter", "max_depth", "step_size",
     "num_trees", "train_rmse", "train_r2", "train_mae", "best_cv_rmse", "training_seconds"]
).coalesce(1).write.mode('overwrite').option('header', True).csv(f"{OUTPUT_PATH}/metrics")

# Save CV results
print(f"Saving CV results to {OUTPUT_PATH}/cv_results/")
cv_rows = []
for params, score in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    cv_rows.append((
        int(params[gbt.maxIter]),
        int(params[gbt.maxDepth]),
        float(params[gbt.stepSize]),
        float(score)
    ))
spark.createDataFrame(
    cv_rows,
    ["maxIter", "maxDepth", "stepSize", "cv_rmse"]
).coalesce(1).write.mode('overwrite').option('header', True).csv(f"{OUTPUT_PATH}/cv_results")

# Save feature importances
print(f"Saving feature importances to {OUTPUT_PATH}/feature_importances/")
importance_data = [(feat, float(imp)) for feat, imp in feature_importance]
spark.createDataFrame(
    importance_data,
    ["feature", "importance"]
).coalesce(1).write.mode('overwrite').option('header', True).csv(f"{OUTPUT_PATH}/feature_importances")

# Save sample predictions
print(f"Saving sample predictions to {OUTPUT_PATH}/sample_predictions/")
train_predictions.select('label', 'prediction') \
    .limit(10000) \
    .coalesce(1).write.mode('overwrite').parquet(f"{OUTPUT_PATH}/sample_predictions")

# Save model
print(f"Saving model to {OUTPUT_PATH}/best_GBT_model/")
best_model.write().overwrite().save(f"{OUTPUT_PATH}/best_GBT_model")

# -------------------- Summary --------------------
print("\n" + "=" * 80)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nResults Summary:")
print(f"  Training rows:   {train_count:,}")
print(f"  Training time:   {training_duration/60:.1f} min ({training_duration/3600:.2f} hrs)")
print(f"  Training RMSE:   {train_rmse:.4f}°C")
print(f"  Training R²:     {train_r2:.4f}")
print(f"  Best CV RMSE:    {min(avg_metrics):.4f}°C")
print(f"\nBest Parameters:")
print(f"  maxIter:         {best_gbt.getMaxIter()}")
print(f"  maxDepth:        {best_gbt.getMaxDepth()}")
print(f"  stepSize:        {best_gbt.getStepSize()}")
print(f"  numTrees:        {best_gbt.getNumTrees}")
print(f"\nAll artifacts saved to: {OUTPUT_PATH}")
print(f"\nNext Step: Evaluate this model on the held-out test set")
print(f"  using: gs://weather-ml-bucket-1760514177/warehouse/noaa_test")

# Clean up
data.unpersist()

spark.stop()