#!/usr/bin/env python3
import sys
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# --- Args ---
# 1) input parquet path (cleaned)  2) output base path  3) light_cats (true/false)
if len(sys.argv) < 3:
    print("Usage: noaa_train_rf.py <INPUT_PARQUET> <OUTPUT_BASE> [light_cats=true|false]")
    sys.exit(1)

INPUT_PARQUET = sys.argv[1]
OUTPUT_BASE   = sys.argv[2].rstrip("/")
LIGHT_CATS    = True
if len(sys.argv) >= 4:
    LIGHT_CATS = (sys.argv[3].lower() == "true")

LABEL_COL = "label"
SEED      = 42

# --- Spark session (on Dataproc) ---
spark = SparkSession.builder.appName("NOAA_RF_Dataproc").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Heuristic partitions: ~4x total cores helps throughput; Dataproc sets this too,
# but we add a floor for safety on small clusters.
spark.conf.set("spark.sql.adaptive.enabled", "true")

# --- Load cleaned data ---
dfp = spark.read.parquet(INPUT_PARQUET).where(F.col(LABEL_COL).isNotNull())

# --- Numeric & time features ---
num_candidates = [c for c in ["WS_num","DEW_num","SLP_num","VIS_num","RH_num","month_sin","month_cos"] if c in dfp.columns]

time_cols = []
if "DATE" in dfp.columns:
    dfp = (dfp.withColumn("ts", F.to_timestamp("DATE"))
               .withColumn("hour", F.hour("ts"))
               .withColumn("dayofyear", F.dayofyear("ts")))
    time_cols = ["hour","dayofyear"]
if "month" in dfp.columns:
    time_cols = sorted(set(time_cols + ["month"]))

feature_cols = num_candidates + time_cols

# --- Categoricals (light vs full) ---
cat_pool = ["COUNTRY","REPORT_TYPE","STATION","NAME"]
cat_cols = [c for c in (["COUNTRY","REPORT_TYPE"] if LIGHT_CATS else cat_pool) if c in dfp.columns]
ohe_cols = [f"{c}_oh" for c in cat_cols]

# --- Pipeline ---
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_oh"]) for c in cat_cols]
assembler = VectorAssembler(inputCols=feature_cols + ohe_cols, outputCol="features_raw", handleInvalid="keep")
scaler    = StandardScaler(inputCol="features_raw", outputCol="features", withMean=True, withStd=True)

train, test = dfp.randomSplit([0.7, 0.3], seed=SEED)

rf = RandomForestRegressor(featuresCol="features", labelCol=LABEL_COL, seed=SEED)

# Balanced “full run” grid (can enlarge later)
rf_grid = (ParamGridBuilder()
           .addGrid(rf.numTrees, [100, 150])
           .addGrid(rf.maxDepth, [12, 15])
           .build())

evaluator = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
rf_pipe   = Pipeline(stages=indexers + encoders + [assembler, scaler, rf])

cv = CrossValidator(
    estimator=rf_pipe,
    estimatorParamMaps=rf_grid,
    evaluator=evaluator,
    numFolds=3,
    parallelism=3  # try 3–6 depending on cluster size
)

print(f"Training RF on cluster… light_cats={LIGHT_CATS}  rows={dfp.count()}  feats={len(feature_cols)}  cats={len(cat_cols)}")
model = cv.fit(train)
preds = model.transform(test)
rmse  = evaluator.evaluate(preds)
print(f"RF RMSE (test) = {rmse:.4f}")

# --- Save artifacts ---
metrics_path = f"{OUTPUT_BASE}/metrics"
model_path   = f"{OUTPUT_BASE}/best_RandomForest_model"
sample_path  = f"{OUTPUT_BASE}/sample_predictions"

(spark.createDataFrame([("RandomForest", rmse)], ["model","rmse"])
     .coalesce(1).write.mode("overwrite").option("header", True).csv(metrics_path))

model.bestModel.write().overwrite().save(model_path)

(preds.select(LABEL_COL, "prediction")
      .limit(1_000_000)  # generous on cluster
      .coalesce(1).write.mode("overwrite").option("header", True).csv(sample_path))

print("Artifacts saved →", OUTPUT_BASE)
