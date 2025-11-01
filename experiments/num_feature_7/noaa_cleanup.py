#!/usr/bin/env python3
import sys
from pyspark.sql import SparkSession, functions as F

USAGE = """
Usage:
  noaa_cleanup.py <INPUT_PARQUET_BASE> <OUTPUT_PARQUET_BASE> [--chunked true|false]

Examples:
  noaa_cleanup.py gs://YOUR_BUCKET/warehouse/noaa_parquet gs://YOUR_BUCKET/warehouse/noaa_clean_std --chunked true
"""

# ---- args ----
if len(sys.argv) < 3:
    print(USAGE); sys.exit(1)

RAW   = sys.argv[1].rstrip("/")
CLEAN = sys.argv[2].rstrip("/")
CHUNKED = False
if len(sys.argv) >= 5 and sys.argv[3] == "--chunked":
    CHUNKED = (sys.argv[4].lower() == "true")

spark = (
    SparkSession.builder
    .appName("NOAA_CLEAN_Parquet")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(RAW).where(F.col("label").isNotNull())

# Ensure year/month exist for partitioning/chunking
if "year" not in df.columns or "month" not in df.columns:
    if "DATE" in df.columns:
        df = df.withColumn("ts", F.to_timestamp("DATE"))
        df = df.withColumn("year",  F.year("ts")).withColumn("month", F.month("ts"))
    else:
        df = df.withColumn("year", F.lit(0)).withColumn("month", F.lit(0))

# Convert NaN→null and cast numeric feature columns
num_cols = [c for c in ["WS_num","DEW_num","SLP_num","VIS_num","RH_num"] if c in df.columns]
for c in num_cols:
    df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).cast("double"))

# Approx median fill for numeric columns
fillmap = {}
for c in num_cols:
    q = df.approxQuantile(c, [0.5], 1e-3)
    fillmap[c] = float(q[0]) if q and q[0] is not None else 0.0
df = df.fillna(fillmap)

# Plausibility filter for label (Celsius)
df = df.where((F.col("label") >= -90) & (F.col("label") <= 60))

if CHUNKED:
    # Write month-by-month to reduce concurrent writers
    keys = (df.select("year","month").distinct().orderBy("year","month").collect())
    print("Chunked write for", len(keys), "partitions …")
    for r in keys:
        y, m = int(r["year"]), int(r["month"])
        print(f"Writing year={y} month={m} …")
        part = df.where((F.col("year")==y) & (F.col("month")==m)).coalesce(8)
        (part.write.mode("append").partitionBy("year","month").parquet(CLEAN))
else:
    # One shot write (cluster usually handles fine)
    (df.coalesce(64).write.mode("overwrite").partitionBy("year","month").parquet(CLEAN))

print(f"✅ Cleaned dataset written to: {CLEAN}")
