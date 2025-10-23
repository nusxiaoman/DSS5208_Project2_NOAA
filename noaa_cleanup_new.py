#!/usr/bin/env python3
import sys
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import DoubleType, TimestampType

# Usage: noaa_cleanup.py <INPUT_PATH> <OUTPUT_PATH>
# INPUT:  Parquet produced by ETL (NOAA raw columns)
# OUTPUT: Cleaned/standardized Parquet with numeric + categorical fields

if len(sys.argv) < 3:
    print("Usage: noaa_cleanup.py <INPUT_PATH> <OUTPUT_PATH>")
    sys.exit(1)

IN  = sys.argv[1]
OUT = sys.argv[2].rstrip("/")

spark = SparkSession.builder.appName("NOAA_Cleanup_Enhanced").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

df = spark.read.parquet(IN)

# -----------------------------
# Helpers to parse NOAA strings
# -----------------------------
# Grab the first number in a field (e.g., "23.0,1" -> 23.0)
def first_num(col):
    return F.regexp_extract(F.col(col).cast("string"), r'[-+]?\d*\.?\d+', 0).cast(DoubleType())

# Grab first and second numbers (e.g., "180,3" -> dir=180, speed=3)
def first_two_nums(col):
    s = F.col(col).cast("string")
    n1 = F.regexp_extract(s, r'^\s*([-+]?\d*\.?\d+)', 1).cast(DoubleType())
    n2 = F.regexp_extract(s, r'^\s*[-+]?\d*\.?\d+\s*,\s*([-+]?\d*\.?\d+)', 1).cast(DoubleType())
    return n1, n2

# Clean a numeric column from strings & NaNs
def clean_num(colname, outname):
    c = first_num(colname)
    return F.when(F.isnan(c) | F.isnull(c), None).otherwise(c).alias(outname)

# -----------------------------
# Derive/standardize core fields
# -----------------------------
# Timestamp & time parts
if "ts" not in df.columns:
    # DATE may be "yyyy-MM-dd" or full ISO. Let Spark parse it.
    df = df.withColumn("ts", F.to_timestamp("DATE"))
df = (df
      .withColumn("year", F.year("ts"))
      .withColumn("month", F.month("ts").cast("double"))
      .withColumn("dayofyear", F.dayofyear("ts").cast("double"))
      .withColumn("hour", F.hour("ts").cast("double"))
     )

# Spatial: latitude / longitude / elevation (keep originals if already numeric)
if "LATITUDE" in df.columns:
    df = df.withColumn("LATITUDE", F.col("LATITUDE").cast(DoubleType()))
if "LONGITUDE" in df.columns:
    df = df.withColumn("LONGITUDE", F.col("LONGITUDE").cast(DoubleType()))
if "ELEVATION" in df.columns:
    df = df.withColumn("ELEVATION", F.col("ELEVATION").cast(DoubleType()))

# If LAT/LON/ELEVATION came as strings in some ETL, try to parse them
if "LATITUDE" in df.columns:
    df = df.withColumn("LATITUDE",
        F.when(F.col("LATITUDE").isNull(), first_num("LATITUDE")).otherwise(F.col("LATITUDE")))
if "LONGITUDE" in df.columns:
    df = df.withColumn("LONGITUDE",
        F.when(F.col("LONGITUDE").isNull(), first_num("LONGITUDE")).otherwise(F.col("LONGITUDE")))
if "ELEVATION" in df.columns:
    df = df.withColumn("ELEVATION",
        F.when(F.col("ELEVATION").isNull(), first_num("ELEVATION")).otherwise(F.col("ELEVATION")))

# Core numeric weather vars (keep your existing ones)
if "DEW" in df.columns and "DEW_num" not in df.columns:
    df = df.withColumn("DEW_num", clean_num("DEW", "DEW_num"))
if "SLP" in df.columns and "SLP_num" not in df.columns:
    df = df.withColumn("SLP_num", clean_num("SLP", "SLP_num"))
if "VIS" in df.columns and "VIS_num" not in df.columns:
    df = df.withColumn("VIS_num", clean_num("VIS", "VIS_num"))
if "TMP" in df.columns and "TMP_num" not in df.columns:
    df = df.withColumn("TMP_num", clean_num("TMP", "TMP_num"))

# Wind: parse direction & speed if WND present
if "WND" in df.columns:
    dir_num, spd_num = first_two_nums("WND")
    df = (df
          .withColumn("WND_dir_num", F.when(F.isnan(dir_num) | F.isnull(dir_num), None).otherwise(dir_num))
          .withColumn("WND_speed_num", F.when(F.isnan(spd_num) | F.isnull(spd_num), None).otherwise(spd_num))
         )

# Ceiling: CIG -> numeric feet/m?
if "CIG" in df.columns:
    df = df.withColumn("CIG_num", clean_num("CIG", "CIG_num"))

# Precipitation: AA1 often encodes periods; take the first numeric as simple proxy
if "AA1" in df.columns:
    df = df.withColumn("PRECIP_num", clean_num("AA1", "PRECIP_num"))

# Sky condition / cloud layers: keep as categoricals if present
# (You can later StringIndex + OHE these in the training script.)
# We leave SKC/CLD as-is; they’re strings.

# -----------------------------
# Label column
# -----------------------------
# Use existing label if present; otherwise derive from TMP_num
if "label" not in df.columns:
    if "TMP_num" in df.columns:
        df = df.withColumn("label", F.col("TMP_num"))
    else:
        raise ValueError("No 'label' column and no 'TMP' available to derive label.")

# Reasonable label filter (°C)
df = df.where(F.col("label").between(-60.0, 60.0))

# -----------------------------
# Final select: keep important fields
# -----------------------------
keep_cols = []
for c in ["label", "DATE", "ts",
          "DEW_num", "SLP_num", "VIS_num", "TMP_num",
          "WND_dir_num", "WND_speed_num", "CIG_num", "PRECIP_num",
          "LATITUDE", "LONGITUDE", "ELEVATION",
          "STATION", "NAME", "REPORT_TYPE",
          "year", "month", "hour", "dayofyear"]:
    if c in df.columns:
        keep_cols.append(c)

df_out = df.select(*keep_cols)

# Write out cleaned, standardized parquet
(df_out
 .write.mode("overwrite")
 .parquet(OUT))

print("Wrote cleaned dataset to →", OUT)
