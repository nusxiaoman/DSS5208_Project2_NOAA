#!/usr/bin/env python3
import sys
from pyspark.sql import SparkSession, functions as F, types as T

USAGE = """
Usage:
  noaa_etl.py <INPUT_CSV_GLOB> <OUTPUT_PARQUET_BASE>

Examples:
  noaa_etl.py gs://YOUR_BUCKET/data/csv/*.csv gs://YOUR_BUCKET/warehouse/noaa_parquet
"""

if len(sys.argv) != 3:
    print(USAGE)
    sys.exit(1)

INPUT = sys.argv[1].rstrip("/")
OUTPUT = sys.argv[2].rstrip("/")

spark = (
    SparkSession.builder
    .appName("NOAA_ETL_CSV_to_Parquet")
    .config("spark.sql.adaptive.enabled", "true")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

# Read CSVs (header present; drop malformed rows rather than failing)
df = (spark.read
      .option("header", True)
      .option("mode", "DROPMALFORMED")
      .csv(INPUT))

# ---- helpers to parse NOAA fields ----
@F.udf(T.DoubleType())
def parse_tmp(tmp):
    # examples: "24.0,1" (value, quality); "M" for missing
    if not tmp or tmp == "M":
        return None
    try:
        return float(str(tmp).split(",")[0])
    except Exception:
        return None

@F.udf(T.DoubleType())
def parse_num(s):
    # generic "val,quality" or "M"
    if not s or s == "M":
        return None
    try:
        return float(str(s).split(",")[0])
    except Exception:
        return None

# Label and numeric weather features
df = df.withColumn("label", parse_tmp(F.col("TMP")))
for colname, outname in [
    ("WND", "WS_num"),   # wind speed
    ("DEW", "DEW_num"),  # dew point
    ("SLP", "SLP_num"),  # sea level pressure
    ("VIS", "VIS_num"),  # visibility
]:
    if colname in df.columns:
        df = df.withColumn(outname, parse_num(F.col(colname)))

# Relative humidity (approx) if DEW and TMP exist: RH ~ 100 * exp((17.625*DEW)/(243.04+DEW) - (17.625*TMP)/(243.04+TMP))
if {"DEW_num", "label"}.issubset(df.columns):
    e = F.exp
    df = df.withColumn(
        "RH_num",
        100.0 * (
            e(17.625*F.col("DEW_num")/(243.04+F.col("DEW_num"))) -
            e(17.625*F.col("label") /(243.04+F.col("label")))
        ) + 100.0
    )

# Time partitions: year/month (from DATE if present)
if "DATE" in df.columns:
    df = df.withColumn("ts", F.to_timestamp("DATE"))
    df = df.withColumn("year", F.year("ts")).withColumn("month", F.month("ts"))
else:
    df = df.withColumn("year", F.lit(0)).withColumn("month", F.lit(0))

# Write partitioned Parquet
if {"year","month"}.issubset(df.columns):
    (df.write.mode("overwrite").partitionBy("year","month").parquet(OUTPUT))
else:
    df.write.mode("overwrite").parquet(OUTPUT)

print(f"✅ ETL complete → {OUTPUT}")
