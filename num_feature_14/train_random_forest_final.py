#!/usr/bin/env python3
"""
NOAA Random Forest Training - Final Version
Trains on pre-split training data (88M rows)
Evaluation on test set done separately
"""
import sys, time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Parse arguments
if len(sys.argv) < 3:
    print("Usage: train_random_forest_final.py <train_parquet_path> <output_path>")
    sys.exit(1)

TRAIN_PATH = sys.argv[1]
OUTPUT_PATH = sys.argv[2].rstrip("/")
LABEL_COL = "temperature"
SEED = 42

# Initialize Spark with optimizations
spark = SparkSession.builder \
    .appName("NOAA_RF_Final") \
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
print("NOAA Random Forest Training - Final Version")
print("=" * 80)
print(f"Input (pre-split training data): {TRAIN_PATH}")
print(f"Output: {OUTPUT_PATH}")

# -------------------- Load Pre-Split Training Data --------------------
print("\nLoading pre-split training data (70% of full dataset)...")
train_df = spark.read.parquet(TRAIN_PATH)

# Use correct feature set (14 features that performed well in test)
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

# Prepare data - select features and label
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

# Random Forest
rf = RandomForestRegressor(
    featuresCol='features',
    labelCol='label',
    seed=SEED
)

# Pipeline
pipeline = Pipeline(stages=[imputer, assembler, rf])

# -------------------- Hyperparameter Grid --------------------
print("\nHyperparameter Grid (Memory-Optimized):")
print("  Based on test results: best was numTrees=20, maxDepth=10")
print("  Searching around those values with full dataset:")
print("    - numTrees: [50, 100]")
print("    - maxDepth: [10, 12]")
print("    - minInstancesPerNode: [1, 5]")
print("  Total: 8 models × 3 folds = 24 training runs")

param_grid = ParamGridBuilder() \
    .addGrid(rf.numTrees, [50, 100]) \
    .addGrid(rf.maxDepth, [10, 12]) \
    .addGrid(rf.minInstancesPerNode, [1, 5]) \
    .build()

# Evaluator
evaluator = RegressionEvaluator(
    labelCol='label',
    predictionCol='prediction',
    metricName='rmse'
)

# Cross-validator
cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=2,  # Reduced to avoid memory pressure
    seed=SEED
)

# -------------------- Train --------------------
print("\n" + "=" * 80)
print("Training Random Forest with 3-Fold Cross-Validation")
print("=" * 80)
print("This may take 2-4 hours with auto-scaling...")

start_time = time.time()
cv_model = cv.fit(data)
training_duration = time.time() - start_time

print(f"\nTraining completed in {training_duration/60:.1f} minutes ({training_duration/3600:.2f} hours)")

# -------------------- Best Model --------------------
print("\n" + "=" * 80)
print("Best Model Parameters")
print("=" * 80)

best_model = cv_model.bestModel
best_rf = best_model.stages[-1]

print(f"  numTrees: {best_rf.getNumTrees}")
print(f"  maxDepth: {best_rf.getMaxDepth()}")
print(f"  minInstancesPerNode: {best_rf.getMinInstancesPerNode()}")

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
print("Cross-Validation Results (3-Fold)")
print("=" * 80)
print(f"Best CV RMSE:  {min(avg_metrics):.4f}°C")
print(f"Worst CV RMSE: {max(avg_metrics):.4f}°C")
print(f"Mean CV RMSE:  {sum(avg_metrics)/len(avg_metrics):.4f}°C")

# -------------------- Feature Importances --------------------
print("\n" + "=" * 80)
print("TOP 10 FEATURE IMPORTANCES")
print("=" * 80)

importances = best_rf.featureImportances.toArray()
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
    "RandomForest",
    "full_training",
    int(train_count),
    int(best_rf.getNumTrees),
    int(best_rf.getMaxDepth()),
    int(best_rf.getMinInstancesPerNode()),
    float(train_rmse),
    float(train_r2),
    float(train_mae),
    float(min(avg_metrics)),
    float(training_duration)
)]
spark.createDataFrame(
    metrics_data,
    ["model", "mode", "train_rows", "num_trees", "max_depth", "min_instances",
     "train_rmse", "train_r2", "train_mae", "best_cv_rmse", "training_seconds"]
).coalesce(1).write.mode('overwrite').option('header', True).csv(f"{OUTPUT_PATH}/metrics")

# Save CV results
print(f"Saving CV results to {OUTPUT_PATH}/cv_results/")
cv_rows = []
for params, score in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    cv_rows.append((
        int(params[rf.numTrees]),
        int(params[rf.maxDepth]),
        int(params[rf.minInstancesPerNode]),
        float(score)
    ))
spark.createDataFrame(
    cv_rows,
    ["numTrees", "maxDepth", "minInstancesPerNode", "cv_rmse"]
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
print(f"Saving model to {OUTPUT_PATH}/best_RandomForest_model/")
best_model.write().overwrite().save(f"{OUTPUT_PATH}/best_RandomForest_model")

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
print(f"  numTrees:        {best_rf.getNumTrees}")
print(f"  maxDepth:        {best_rf.getMaxDepth()}")
print(f"  minInstances:    {best_rf.getMinInstancesPerNode()}")
print(f"\nAll artifacts saved to: {OUTPUT_PATH}")
print(f"\nNext Step: Evaluate this model on the held-out test set")
print(f"  using: gs://weather-ml-bucket-1760514177/warehouse/noaa_test")

# Clean up
data.unpersist()

spark.stop()