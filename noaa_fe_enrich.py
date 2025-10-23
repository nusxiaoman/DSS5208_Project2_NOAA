#!/usr/bin/env python3
import sys, math
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType

# Usage: noaa_fe_enrich.py <IN_PARQUET_ROOT> <OUT_PARQUET_ROOT>
if len(sys.argv) < 3:
    print("Usage: noaa_fe_enrich.py <INPUT_PARQUET_ROOT> <OUTPUT_PARQUET_ROOT>")
    sys.exit(1)

IN  = sys.argv[1]
OUT = sys.argv[2].rstrip("/")

spark = (SparkSession.builder
         .appName("NOAA_FE_ENRICH")
         .getOrCreate())
spark.sparkContext.setLogLevel("WARN")

print("Reading:", IN)
df = spark.read.parquet(IN)

# Ensure label exists
if "label" not in df.columns:
    raise RuntimeError("Expected 'label' column in the dataset")

# Make some obvious numeric candidates if present
num_candidates = [c for c in [
    "TEMP_num", "DEW_num", "RH_num", "WS_num", "SLP_num", "VIS_num",
    "LAT", "LON", "ELEVATION"
] if c in df.columns]

# Derive temporal features (hour/day-of-year/month) + cyclical encodings if DATE exists
if "DATE" in df.columns:
    df = (df
        .withColumn("ts", F.to_timestamp("DATE"))
        .withColumn("hour", F.hour("ts").cast(DoubleType()))
        .withColumn("dayofyear", F.dayofyear("ts").cast(DoubleType()))
        .withColumn("month", F.month("ts").cast(DoubleType()))
    )

    df = (df
        .withColumn("hour_sin",  F.sin(2*math.pi*F.col("hour")/24.0))
        .withColumn("hour_cos",  F.cos(2*math.pi*F.col("hour")/24.0))
        .withColumn("doy_sin",   F.sin(2*math.pi*F.col("dayofyear")/365.0))
        .withColumn("doy_cos",   F.cos(2*math.pi*F.col("dayofyear")/365.0))
    )

# Simple interaction (temperature – dew point)
if "TEMP_num" in df.columns and "DEW_num" in df.columns:
    df = df.withColumn("dew_temp_diff", F.col("TEMP_num") - F.col("DEW_num"))

# Cast obvious numerics to double and replace NaN with NULL so the trainer’s Imputer works
numeric_like = [c for c in [
    "TEMP_num","DEW_num","RH_num","WS_num","SLP_num","VIS_num",
    "LAT","LON","ELEVATION",
    "hour","dayofyear","month","hour_sin","hour_cos","doy_sin","doy_cos",
    "dew_temp_diff"
] if c in df.columns]

for c in numeric_like:
    df = df.withColumn(c, F.when(F.isnan(F.col(c)), None).otherwise(F.col(c)).cast(DoubleType()))

print("Writing enriched parquet to:", OUT)
(df.write.mode("overwrite").parquet(OUT))
print("DONE →", OUT)
