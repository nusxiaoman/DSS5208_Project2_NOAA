#!/usr/bin/env python3
import sys, time
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType
from pyspark.storagelevel import StorageLevel

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, Imputer
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# -------------------------------------------------------------------
# Args:  IN  OUT  [FULL=true|false]  [REPART_N=int]  [STRATEGY]
# STRATEGY ∈ {cv-full, cv-fast}
# -------------------------------------------------------------------
if len(sys.argv) < 3:
    print("Usage: noaa_train_rf.py <INPUT_PARQUET_ROOT> <OUTPUT_BASE> [full=true|false] [REPART_N=int] [STRATEGY]")
    sys.exit(1)

IN   = sys.argv[1]
OUT  = sys.argv[2].rstrip("/")
FULL = False
if len(sys.argv) >= 4:
    FULL = (str(sys.argv[3]).lower() in ("1", "true", "yes"))
REPART_N = 0
if len(sys.argv) >= 5:
    try:
        REPART_N = int(sys.argv[4])
    except:
        REPART_N = 0
STRATEGY = sys.argv[5].lower() if len(sys.argv) >= 6 else ("cv-full" if FULL else "cv-fast")

LABEL_COL = "label"
SEED = 42

spark = SparkSession.builder.appName("NOAA_RF").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

def set_if_absent(k, v):
    try:
        spark.conf.get(k)
    except Exception:
        spark.conf.set(k, v)

# Helpful defaults (caller can override via --properties)
set_if_absent("spark.sql.adaptive.enabled", "true")
set_if_absent("spark.sql.shuffle.partitions", "160")
set_if_absent("spark.default.parallelism", "160")

print(f"Training RF … mode={'full' if FULL else 'small'} | REPART_N={REPART_N} | STRATEGY={STRATEGY}")
print(f"Input  : {IN}")
print(f"Output : {OUT}")

# Checkpoint (for resilience on serverless)
spark.sparkContext.setCheckpointDir(f"{OUT}/_chkpt")

# -------------------- Load --------------------
df = spark.read.parquet(IN).where(F.col(LABEL_COL).isNotNull())
if REPART_N > 0:
    df = df.repartition(REPART_N)

# -------------------- Feature candidates --------------------
# Works with your enriched parquet (adds only if present)
num_candidates = [c for c in [
    "TEMP_num","DEW_num","RH_num","WS_num","SLP_num","VIS_num",
    "LAT","LON","ELEVATION",
    "hour","dayofyear","month","hour_sin","hour_cos","doy_sin","doy_cos",
    "dew_temp_diff"
] if c in df.columns]

# Make doubles and turn NaN→NULL so Imputer works
feature_num = []
for c in num_candidates:
    df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).cast(DoubleType()))
    feature_num.append(c)

# Categoricals (light set by default; add more if they exist)
cat_pool = ["COUNTRY","REPORT_TYPE","STATION","NAME"]
cat_cols = [c for c in ["COUNTRY","REPORT_TYPE"] if c in df.columns]
# If you want heavier categories, swap line above with:
# cat_cols = [c for c in cat_pool if c in df.columns]

idx_cols = [f"{c}_idx" for c in cat_cols]
oh_cols  = [f"{c}_oh"  for c in cat_cols]

# -------------------- Stages (lean & robust) --------------------
# Assumes: feature_num, cat_cols, spark, OUT are defined

# 1) Numeric imputation (keep for safety even if cleanup filled most numerics)
if not feature_num:
    raise ValueError("No numeric features found; check your input schema before Imputer.")

num_imp_cols = [f"{c}_imp" for c in feature_num]
imputer = Imputer(strategy="median",
                  inputCols=feature_num,
                  outputCols=num_imp_cols)

# 2) Categorical encoding
idx_cols, oh_cols = [], []
indexers = []
encoder = None
if cat_cols:
    idx_cols = [f"{c}_idx" for c in cat_cols]
    oh_cols  = [f"{c}_oh"  for c in cat_cols]
    indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep")
                for c in cat_cols]
    encoder  = OneHotEncoder(inputCols=idx_cols, outputCols=oh_cols, handleInvalid="keep")

# 3) Assemble final features
final_features = num_imp_cols + oh_cols
if not final_features:
    raise ValueError("After preprocessing, final_features is empty. Nothing to assemble.")

print("🔎 Final feature input order (total =", len(final_features), "):")
for i, col in enumerate(final_features):
    print(f"{i}: {col}")

(spark.createDataFrame([(i, col) for i, col in enumerate(final_features)], ["index","column"])
      .coalesce(1).write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/feature_list"))

assembler = VectorAssembler(inputCols=final_features,
                            outputCol="features",
                            handleInvalid="keep")

# If you assemble pipeline later:
# stages = [imputer] + indexers + ([encoder] if encoder is not None else []) + [assembler]
# pipe = Pipeline(stages=stages + [rf_or_gbt_estimator])


# -------------------- Split & cache --------------------
train, test = df.randomSplit([0.7, 0.3], seed=SEED)
train = train.persist(StorageLevel.MEMORY_AND_DISK)
test  = test.persist(StorageLevel.MEMORY_AND_DISK)

# Baseline (predict-the-mean) for quick sanity vs ~36
mrow = train.agg(F.avg(LABEL_COL).alias("m")).first()
mean_y = mrow["m"]
baseline_rmse = (
    test.select(
        F.sqrt(F.avg(F.pow(F.lit(mean_y) - F.col(LABEL_COL), 2))).alias("rmse")
    ).first()["rmse"]
)
print(f"Baseline RMSE (predict-mean) ≈ {baseline_rmse:.4f}")


# --------------------- Model & grid ---------------------
rf = RandomForestRegressor(featuresCol="features", labelCol="label", seed=42)

if STRATEGY == "cv-full":
    ntrees = [100, 200]
    depth  = [12, 16]
    mtry   = ["sqrt", "log2"]
elif STRATEGY == "cv-fast":
    ntrees = [100, 150]
    depth  = [10, 12]
    mtry   = ["sqrt"]
else:
    ntrees = [120]
    depth  = [12]
    mtry   = ["sqrt"]

param_grid = (ParamGridBuilder()
              .addGrid(rf.numTrees, ntrees)
              .addGrid(rf.maxDepth, depth)
              .addGrid(rf.featureSubsetStrategy, mtry)
              .build())



evaluator = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
pipe = Pipeline(stages=[imputer] + indexers + [encoder] + [assembler, rf])

cv = CrossValidator(
    estimator=pipe,
    estimatorParamMaps=param_grid,   # use the grid we just built
    evaluator=evaluator,
    numFolds=3,
    parallelism=4,
    seed=SEED,
)



# -------------------- Fit --------------------
start = time.time()
cv_model = cv.fit(train)
dur = time.time() - start

preds = cv_model.transform(test).checkpoint(eager=True)
rmse  = evaluator.evaluate(preds)
print(f"RF RMSE(test) = {rmse:.4f}   (took {dur/60:.1f} min)")

# -------------------- Save metrics --------------------
(spark.createDataFrame([("RandomForest", float(rmse), float(baseline_rmse), STRATEGY, dur)],
                       ["model","rmse","baseline_rmse","strategy","seconds"])
 .coalesce(1).write.mode("overwrite").option("header", True)
 .csv(f"{OUT}/metrics"))

# -------------------- Save CV results --------------------
rows = []
for params, score in zip(cv_model.getEstimatorParamMaps(), cv_model.avgMetrics):
    rows.append((
        int(params[rf.numTrees]),
        int(params[rf.maxDepth]),
        str(params[rf.featureSubsetStrategy]),
        float(score)
    ))
(spark.createDataFrame(rows, ["numTrees","maxDepth","featureSubset","rmse"])
 .coalesce(1).write.mode("overwrite").option("header", True)
 .csv(f"{OUT}/cv_results"))

# -------------------- Feature importances --------------------
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

# -------------------- Sample predictions --------------------
(preds.select(LABEL_COL, "prediction")
      .limit(1_000_000)
      .coalesce(1).write.mode("overwrite").option("header", True)
      .csv(f"{OUT}/sample_predictions"))

# -------------------- Save model --------------------
best.write().overwrite().save(f"{OUT}/best_RandomForest_model")
print("Artifacts saved →", OUT)
