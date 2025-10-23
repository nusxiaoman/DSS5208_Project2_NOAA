#!/usr/bin/env python3
import sys, time
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.storagelevel import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, Imputer
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit

# -----------------------------
# Args
# 1) IN  2) OUT  3) FULL(true|false)  4) REPART_N  5) STRATEGY [cv-full|cv-fast|tvs-fast]
# -----------------------------
if len(sys.argv) < 3:
    print("Usage: noaa_train_gbt.py <INPUT_PARQUET_ROOT> <OUTPUT_BASE> [full=true|false] [REPART_N=int] [STRATEGY]")
    sys.exit(1)

IN   = sys.argv[1]
OUT  = sys.argv[2].rstrip("/")
FULL = False
if len(sys.argv) >= 4:
    FULL = (str(sys.argv[3]).lower() in ("1", "true", "yes"))
REPART_N = 0
if len(sys.argv) >= 5:
    try: REPART_N = int(sys.argv[4])
    except: REPART_N = 0
STRATEGY = sys.argv[5].lower() if len(sys.argv) >= 6 else ("cv-full" if FULL else "cv-fast")

LABEL_COL = "label"
SEED = 42

spark = SparkSession.builder.appName("NOAA_GBT").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def set_if_absent(k, v):
    try: spark.conf.get(k)
    except Exception: spark.conf.set(k, v)

# sensible defaults; caller overrides via --properties
set_if_absent("spark.sql.adaptive.enabled", "true")
set_if_absent("spark.sql.shuffle.partitions", "128")
set_if_absent("spark.default.parallelism", "128")

print(f"Training GBT … mode={'full' if FULL else 'small'} | REPART_N={REPART_N} | STRATEGY={STRATEGY}")
print(f"Input  : {IN}")
print(f"Output : {OUT}")

# checkpoint & cache to be resilient to executor loss
spark.sparkContext.setCheckpointDir(f"{OUT}/_chkpt")

# ----- Load -----
df = spark.read.parquet(IN).where(F.col(LABEL_COL).isNotNull())
if REPART_N > 0:
    df = df.repartition(REPART_N)

# ---- Features (same as before) ----
num_candidates = [c for c in ["WS_num","DEW_num","SLP_num","VIS_num","RH_num"] if c in df.columns]
time_cols = []
if "DATE" in df.columns:
    df = (df.withColumn("ts", F.to_timestamp("DATE"))
            .withColumn("hour", F.hour("ts"))
            .withColumn("dayofyear", F.dayofyear("ts")))
    time_cols += ["hour","dayofyear"]
if "month" in df.columns:
    time_cols = sorted(set(time_cols + ["month"]))
feature_num = num_candidates + time_cols
for c in feature_num:
    if c in df.columns:
        df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).cast("double"))

imp_out = [f"{c}_imp" for c in feature_num]
imputer = Imputer(strategy="median", inputCols=feature_num, outputCols=imp_out)

cat_cols = [c for c in ["COUNTRY","REPORT_TYPE"] if c in df.columns]
indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoders = [OneHotEncoder(inputCols=[f"{c}_idx"], outputCols=[f"{c}_oh"]) for c in cat_cols]
ohe_cols = [f"{c}_oh" for c in cat_cols]

# --- Diagnostic: print and save feature input order ---
final_features = imp_out + ohe_cols
print("🔎 Final feature input order (total =", len(final_features), "):")
for i, col in enumerate(final_features):
    print(f"{i}: {col}")

# Optional: also save to CSV in output path for record
spark.createDataFrame([(i, col) for i, col in enumerate(final_features)], ["index", "column"]) \
    .coalesce(1).write.mode("overwrite").option("header", True) \
    .csv(f"{OUT}/feature_list")

# --- Assemble and scale ---
assembler = VectorAssembler(inputCols=final_features, outputCol="features_raw", handleInvalid="keep")
scaler = StandardScaler(inputCol="features_raw", outputCol="features", withMean=False, withStd=True)


# split, then cache train
train, test = df.randomSplit([0.7, 0.3], seed=SEED)
train = train.persist(StorageLevel.MEMORY_AND_DISK)
test  = test.persist(StorageLevel.MEMORY_AND_DISK)

# ---- Model & grids ----
gbt = GBTRegressor(featuresCol="features", labelCol=LABEL_COL, seed=SEED).setSubsamplingRate(0.8)

if STRATEGY == "cv-full":
    trees  = [100, 200]
    depth  = [6, 8]
    lrates = [0.05, 0.1]
    folds  = 3
elif STRATEGY in ("cv-fast", "tvs-fast"):
    trees  = [100, 200]
    depth  = [6, 8]
    lrates = [0.1]      # fewer LR values → 12 fits instead of 24
    folds  = 3
else:
    # fallback to cv-fast
    trees, depth, lrates, folds = [100,200], [6,8], [0.1], 3

gbt_grid = (ParamGridBuilder()
            .addGrid(gbt.maxIter, trees)
            .addGrid(gbt.maxDepth, depth)
            .addGrid(gbt.stepSize, lrates)
            .build())

evaluator = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
pipe = Pipeline(stages=[imputer] + indexers + encoders + [assembler, scaler, gbt])

# ---- Choose tuner ----
use_tvs = (STRATEGY == "tvs-fast")
if use_tvs:
    tuner = TrainValidationSplit(estimator=pipe,
                                 estimatorParamMaps=gbt_grid,
                                 evaluator=evaluator,
                                 trainRatio=0.8,
                                 parallelism=4,  # more concurrency
                                 seed=SEED)
else:
    tuner = CrossValidator(estimator=pipe,
                           estimatorParamMaps=gbt_grid,
                           evaluator=evaluator,
                           numFolds=folds,
                           parallelism=4,   # ↑ from 2 → faster wall time
                           seed=SEED)

# ---- Train ----
start = time.time()
cv_model = tuner.fit(train)
dur = time.time() - start

preds = cv_model.transform(test).checkpoint(eager=True)
rmse = evaluator.evaluate(preds)

print(f"GBT RMSE = {rmse:.4f}   (took {dur/60:.1f} min)")

# ---- Save metrics ----
(spark.createDataFrame([("GBT", float(rmse), int(FULL), dur, STRATEGY)],
                       ["model","rmse","full","seconds","strategy"])
 .coalesce(1).write.mode("overwrite").option("header", True)
 .csv(f"{OUT}/metrics"))

# ---- Save CV/TVS results ----
rows = []
# CrossValidatorModel & TrainValidationSplitModel both have avgMetrics + paramMaps
for params, score in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    rows.append((int(params[gbt.maxIter]),
                 int(params[gbt.maxDepth]),
                 float(params[gbt.stepSize]),
                 float(score)))
(spark.createDataFrame(rows, ["maxIter","maxDepth","stepSize","rmse"])
 .coalesce(1).write.mode("overwrite").option("header", True)
 .csv(f"{OUT}/cv_results"))

# ---- Feature importances ----
best = cv_model.bestModel
stages = best.stages
gbt_stage = [s for s in stages if isinstance(s, GBTRegressionModel)]
rf_stage  = [s for s in stages if isinstance(s, RandomForestRegressionModel)]
model_stage = gbt_stage[0] if gbt_stage else (rf_stage[0] if rf_stage else None)

if model_stage and hasattr(model_stage, "featureImportances"):
    importances = model_stage.featureImportances.toArray().tolist()
    fi = list(enumerate(importances))
    fi.sort(key=lambda x: x[1], reverse=True)
    top = fi[:50]
    (spark.createDataFrame([(i, float(v)) for i, v in top], ["feature_index","importance"])
        .coalesce(1).write.mode("overwrite").option("header", True)
        .csv(f"{OUT}/feature_importances"))
    print(f"Saved top {len(top)} feature importances → {OUT}/feature_importances")
else:
    print("⚠️ No tree-based model stage found; skipping feature importance export.")

# ---- Sample predictions ----
(preds.select(LABEL_COL, "prediction")
      .limit(1_000_000)
      .coalesce(1)
      .write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/sample_predictions"))

# ---- Save model ----
best.write().overwrite().save(f"{OUT}/best_GBT_model")
print("Artifacts saved →", OUT)
