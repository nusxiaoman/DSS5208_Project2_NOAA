#!/usr/bin/env python3
"""
NOAA Random Forest Training - Final Optimized Version
Combines correct features with robust resource management
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
    print("Usage: train_random_forest_final.py <input_parquet> <output_path>")
    sys.exit(1)

INPUT_PATH = sys.argv[1]
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
print("NOAA Random Forest Training - Final Optimized Version")
print("=" * 80)
print(f"Input:  {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

# -------------------- Load Data --------------------
print("\nLoading training data...")
df = spark.read.parquet(INPUT_PATH)

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

# Prepare data
data = df.select(
    *numeric_features,
    F.col(LABEL_COL).alias('label')
).filter(F.col('label').isNotNull())

# Repartition for better parallelism
print("\nRepartitioning data for optimal processing...")
data = data.repartition(200)

# Split and persist
print("Splitting train/test (70/30)...")
train, test = data.randomSplit([0.7, 0.3], seed=SEED)
train = train.persist(StorageLevel.MEMORY_AND_DISK)
test = test.persist(StorageLevel.MEMORY_AND_DISK)

train_count = train.count()
test_count = test.count()
print(f"Train: {train_count:,} rows")
print(f"Test:  {test_count:,} rows")

# Baseline RMSE
mean_temp = train.agg(F.avg('label').alias('mean')).first()['mean']
baseline_rmse = test.select(
    F.sqrt(F.avg(F.pow(F.lit(mean_temp) - F.col('label'), 2))).alias('rmse')
).first()['rmse']
print(f"\nBaseline RMSE (predict mean): {baseline_rmse:.4f}°C")

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
print("  - numTrees: [50, 100]")
print("  - maxDepth: [10, 12]")
print("  - minInstancesPerNode: [1, 5]")
print("  - Total: 8 models × 3 folds = 24 training runs")

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
print("Training Random Forest with Cross-Validation")
print("=" * 80)
print("This may take 2-4 hours...")

start_time = time.time()
cv_model = cv.fit(train)
training_duration = time.time() - start_time

print(f"\nTraining completed in {training_duration/60:.1f} minutes")

# -------------------- Evaluate --------------------
print("\n" + "=" * 80)
print("Evaluating Best Model")
print("=" * 80)

best_model = cv_model.bestModel
best_rf = best_model.stages[-1]

print(f"\nBest Parameters:")
print(f"  numTrees: {best_rf.getNumTrees}")
print(f"  maxDepth: {best_rf.getMaxDepth()}")
print(f"  minInstancesPerNode: {best_rf.getMinInstancesPerNode()}")

# Predictions with checkpoint for resilience
print("\nMaking predictions on test set...")
predictions = cv_model.transform(test).checkpoint(eager=True)

# Metrics
test_rmse = evaluator.evaluate(predictions)
test_r2 = evaluator.setMetricName('r2').evaluate(predictions)
test_mae = evaluator.setMetricName('mae').evaluate(predictions)

print(f"\n" + "=" * 80)
print("TEST SET RESULTS")
print("=" * 80)
print(f"RMSE: {test_rmse:.4f}°C")
print(f"R²:   {test_r2:.4f}")
print(f"MAE:  {test_mae:.4f}°C")

# Cross-validation results
avg_metrics = cv_model.avgMetrics
print(f"\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS")
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

# -------------------- Save Results --------------------
print(f"\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save metrics
print(f"Saving metrics to {OUTPUT_PATH}/metrics/")
metrics_data = [(
    "RandomForest",
    float(test_rmse),
    float(test_r2),
    float(test_mae),
    float(baseline_rmse),
    int(best_rf.getNumTrees),
    int(best_rf.getMaxDepth()),
    int(best_rf.getMinInstancesPerNode()),
    float(min(avg_metrics)),
    float(training_duration)
)]
spark.createDataFrame(
    metrics_data,
    ["model", "test_rmse", "test_r2", "test_mae", "baseline_rmse",
     "num_trees", "max_depth", "min_instances", "best_cv_rmse", "training_seconds"]
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
predictions.select('label', 'prediction') \
    .limit(10000) \
    .coalesce(1).write.mode('overwrite').parquet(f"{OUTPUT_PATH}/sample_predictions")

# Save model
print(f"Saving model to {OUTPUT_PATH}/best_RandomForest_model/")
best_model.write().overwrite().save(f"{OUTPUT_PATH}/best_RandomForest_model")

# -------------------- Summary --------------------
print("\n" + "=" * 80)
print("✓ TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nFinal Results:")
print(f"  Training time:  {training_duration/60:.1f} minutes")
print(f"  Test RMSE:      {test_rmse:.4f}°C")
print(f"  Test R²:        {test_r2:.4f}")
print(f"  Best CV RMSE:   {min(avg_metrics):.4f}°C")
print(f"  vs Baseline:    {baseline_rmse:.4f}°C")
print(f"\nAll artifacts saved to: {OUTPUT_PATH}")

# Clean up
train.unpersist()
test.unpersist()

spark.stop()