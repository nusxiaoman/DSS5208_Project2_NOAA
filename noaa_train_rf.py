#!/usr/bin/env python3
import sys, time
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Args: 1) IN parquet  2) OUT base  3) LIGHT_CATS (true|false, default true)
if len(sys.argv) < 3:
    print("Usage: noaa_train_rf.py <INPUT_PARQUET> <OUTPUT_BASE> [light_cats=true|false]")
    sys.exit(1)

IN  = sys.argv[1]
OUT = sys.argv[2].rstrip("/")
LIGHT_CATS = True
if len(sys.argv) >= 4:
    LIGHT_CATS = (sys.argv[3].lower() == "true")

LABEL_COL = "label"
SEED = 42

spark = SparkSession.builder.appName("NOAA_RF_Dataproc").getOrCreate()
spark.sparkContext.setLogLevel("WARN")
spark.conf.set("spark.sql.adaptive.enabled", "true")

print(f"Training RF … LIGHT_CATS={LIGHT_CATS}")
print(f"Input  : {IN}")
print(f"Output : {OUT}")

# ---------- Load ----------
df = spark.read.parquet(IN).where(F.col(LABEL_COL).isNotNull())

# ---------- Numeric features (only if present) ----------
num_candidates = [c for c in [
    "TEMP_num","DEW_num","RH_num","WS_num","SLP_num","VIS_num",
    "LAT","LON","ELEVATION",
    "hour","dayofyear","month","hour_sin","hour_cos","doy_sin","doy_cos",
    "dew_temp_diff"
] if c in df.columns]

# ensure doubles and NaN->NULL so Imputer can work
feature_num = []
for c in num_candidates:
    df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).cast(DoubleType()))
    feature_num.append(c)

# ---------- Categoricals ----------
cat_pool = ["COUNTRY","REPORT_TYPE","STATION","NAME"]
cat_cols = [c for c in (["COUNTRY","REPORT_TYPE"] if LIGHT_CATS else cat_pool) if c in df.columns]
idx_cols = [f"{c}_idx" for c in cat_cols]
oh_cols  = [f"{c}_oh"  for c in cat_cols]

# ---------- Stages ----------
imputer   = Imputer(strategy="median", inputCols=feature_num, outputCols=[f"{c}_imp" for c in feature_num])
indexers  = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoder   = OneHotEncoder(inputCols=idx_cols, outputCols=oh_cols)
final_features = [f"{c}_imp" for c in feature_num] + oh_cols
assembler = VectorAssembler(inputCols=final_features, outputCol="features", handleInvalid="keep")

# diagnostics: feature order
print("🔎 Final feature input order (total =", len(final_features), "):")
for i, col in enumerate(final_features):
    print(f"{i}: {col}")
(spark.createDataFrame([(i, col) for i, col in enumerate(final_features)], ["index","column"])
      .coalesce(1).write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/feature_list"))

# split & cache
train, test = df.randomSplit([0.7, 0.3], seed=SEED)
train = train.persist(StorageLevel.MEMORY_AND_DISK)
test  = test.persist(StorageLevel.MEMORY_AND_DISK)

rf = RandomForestRegressor(featuresCol="features", labelCol=LABEL_COL, seed=SEED)

# modest grid; widen later if needed
rf_grid = (ParamGridBuilder()
           .addGrid(rf.numTrees, [100, 150])
           .addGrid(rf.maxDepth, [12, 15])
           .build())

evaluator = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
pipe = Pipeline(stages=[imputer] + indexers + [encoder] + [assembler, rf])

cv = CrossValidator(estimator=pipe,
                    estimatorParamMaps=rf_grid,
                    evaluator=evaluator,
                    numFolds=3,
                    parallelism=4,  # serverless can handle this
                    seed=SEED)

start = time.time()
cv_model = cv.fit(train)
dur = time.time() - start

preds = cv_model.transform(test)
rmse  = evaluator.evaluate(preds)
print(f"RF RMSE (test) = {rmse:.4f}   (took {dur/60:.1f} min)")

# save metrics
(spark.createDataFrame([("RandomForest", float(rmse))], ["model","rmse"])
      .coalesce(1).write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/metrics"))

# save cv results
rows = []
for params, score in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    rows.append((int(params[rf.numTrees]), int(params[rf.maxDepth]), float(score)))
(spark.createDataFrame(rows, ["numTrees","maxDepth","rmse"])
      .coalesce(1).write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/cv_results"))

# feature importances
best = cv_model.bestModel  # PipelineModel
stages = best.stages
rf_stage = [s for s in stages if isinstance(s, RandomForestRegressionModel)]
if rf_stage:
    imp = rf_stage[0].featureImportances.toArray().tolist()
    top = sorted(list(enumerate(imp)), key=lambda x: x[1], reverse=True)[:50]
    (spark.createDataFrame([(i, float(v)) for i, v in top], ["feature_index","importance"])
         .coalesce(1).write.mode("overwrite").option("header", True)
         .csv(f"{OUT}/feature_importances"))
    print(f"Saved top {len(top)} feature importances → {OUT}/feature_importances")
else:
    print("⚠️ No RF stage found; skipping feature importance export.")

# sample predictions
(preds.select(LABEL_COL, "prediction")
      .limit(1_000_000)
      .coalesce(1).write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/sample_predictions"))

# save model
best.write().overwrite().save(f"{OUT}/best_RandomForest_model")
print("Artifacts saved →", OUT)
