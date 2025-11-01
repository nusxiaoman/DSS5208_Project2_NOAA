#!/usr/bin/env python3
import sys
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, Imputer, OneHotEncoder, StringIndexer
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
import time

if len(sys.argv) < 3:
    print("Usage: noaa_train_lr.py <INPUT_PATH> <OUTPUT_PATH>")
    sys.exit(1)

IN  = sys.argv[1]
OUT = sys.argv[2]
LABEL_COL = "label"

spark = SparkSession.builder.appName("NOAA_LinearRegression_Test").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

# Load dataset
df = spark.read.parquet(IN)
print("Loaded data from:", IN)
df.printSchema()
print("Row count:", df.count())

# -------------------- Feature setup --------------------
# Numeric + categorical columns (adjust if missing)
feature_num = [
    "DEW_num", "SLP_num", "VIS_num", "WND_dir_num", "WND_speed_num",
    "CIG_num", "PRECIP_num", "LATITUDE", "LONGITUDE", "ELEVATION",
    "hour", "dayofyear", "month"
]
cat_cols = ["REPORT_TYPE"]

# Filter columns that exist
feature_num = [c for c in feature_num if c in df.columns]
cat_cols = [c for c in cat_cols if c in df.columns]

print(f"Numeric features: {feature_num}")
print(f"Categorical features: {cat_cols}")

# -------------------- Preprocessing pipeline --------------------
imputer = Imputer(strategy="median", inputCols=feature_num, outputCols=[f"{c}_imp" for c in feature_num])

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in cat_cols]
encoder = OneHotEncoder(inputCols=[f"{c}_idx" for c in cat_cols], outputCols=[f"{c}_oh" for c in cat_cols])

final_features = [f"{c}_imp" for c in feature_num] + [f"{c}_oh" for c in cat_cols]

assembler = VectorAssembler(inputCols=final_features, outputCol="features", handleInvalid="keep")

lr = LinearRegression(featuresCol="features", labelCol=LABEL_COL, predictionCol="prediction")

pipeline = Pipeline(stages=[imputer] + indexers + [encoder, assembler, lr])

# -------------------- Train/test split --------------------
train, test = df.randomSplit([0.8, 0.2], seed=42)

start = time.time()
model = pipeline.fit(train)
elapsed = time.time() - start

pred = model.transform(test)

# -------------------- Evaluation --------------------
evaluator = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(pred)

print(f"✅ LinearRegression RMSE = {rmse:.4f}")
print(f"⏱ Training time = {elapsed:.2f} sec")

# Save metrics
spark.createDataFrame([(rmse, elapsed)], ["rmse", "train_seconds"]) \
    .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{OUT}/metrics")

model.write().overwrite().save(f"{OUT}/model")
print("Results saved to:", OUT)
