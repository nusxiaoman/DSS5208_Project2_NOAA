"""
NOAA Weather Data Cleanup V2 - Simplified
Based on working V0 structure + adds CIG (ceiling height)
NO median imputation to avoid SIGTERM issues
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, when, to_timestamp, year, month, dayofmonth, hour,
    sin, cos, lit
)
from pyspark.sql.types import DoubleType, IntegerType
import math

def parse_temperature(temp_str):
    """Parse temperature string: '-0070,1' -> -7.0°C"""
    return (
        when(col(temp_str).isNull() | (col(temp_str) == '+9999,9'), None)
        .otherwise(split(col(temp_str), ',').getItem(0).cast(DoubleType()) / 10.0)
    )

def parse_wind(wnd_col):
    """Parse wind string: '318,1,N,0061,1' -> direction, speed"""
    parts = split(col(wnd_col), ',')
    direction = when(parts.getItem(0) == '999', None).otherwise(parts.getItem(0).cast(IntegerType()))
    speed = when(parts.getItem(3) == '9999', None).otherwise(parts.getItem(3).cast(DoubleType()) / 10.0)
    return direction, speed

def parse_pressure(slp_str):
    """Parse sea level pressure: '10208,1' -> 1020.8 hPa"""
    return (
        when(col(slp_str).isNull() | (col(slp_str).contains('99999')), None)
        .otherwise(split(col(slp_str), ',').getItem(0).cast(DoubleType()) / 10.0)
    )

def parse_visibility(vis_str):
    """Parse visibility: '025000,1,9,9' -> 25000 meters"""
    return (
        when(col(vis_str).isNull() | (col(vis_str).contains('999999')), None)
        .otherwise(split(col(vis_str), ',').getItem(0).cast(DoubleType()))
    )

def parse_precipitation(precip_col):
    """Parse precipitation: '24,0002,3,1' -> 0.2 mm"""
    if precip_col not in ['AA1', 'AA2', 'AA3', 'AA4']:
        return lit(None)
    return (
        when(col(precip_col).isNull() | (col(precip_col).contains('9999')), 0.0)
        .otherwise(split(col(precip_col), ',').getItem(1).cast(DoubleType()) / 10.0)
    )

def parse_ceiling(cig_str):
    """Parse CIG: '00800,1,9,9' -> 800 meters"""
    return (
        when(col(cig_str).isNull() | (col(cig_str).contains('99999')), None)
        .otherwise(split(col(cig_str), ',').getItem(0).cast(DoubleType()))
    )

def main():
    spark = SparkSession.builder \
        .appName("NOAA Cleanup V2 - Simplified") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths
    input_path = "gs://weather-ml-bucket-1760514177/data/csv/*.csv"
    output_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
    
    print("=" * 80)
    print("NOAA WEATHER DATA CLEANUP V2 - SIMPLIFIED")
    print("=" * 80)
    print("Features: V0 (14 features) + CIG ceiling height")
    print("Total: 17 features")
    print("NO median imputation (keeps NULLs for ML to handle)")
    print("=" * 80)
    
    # Read data
    print(f"\nReading: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=False)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    available_cols = df.columns
    has_cig = 'CIG' in available_cols
    
    print(f"\nOptional fields:")
    print(f"  CIG (ceiling): {'YES' if has_cig else 'NO'}")
    
    # ==================== Basic Parsing ====================
    print("\n" + "=" * 80)
    print("PHASE 1: Parse Basic Features")
    print("=" * 80)
    
    wind_direction, wind_speed = parse_wind('WND')
    
    precip_sum = lit(0.0)
    for p in ['AA1', 'AA2', 'AA3', 'AA4']:
        if p in available_cols:
            precip_sum = precip_sum + parse_precipitation(p)
    
    # Build select list
    select_cols = [
        col('STATION'),
        to_timestamp(col('DATE')).alias('timestamp'),
        col('LATITUDE').cast(DoubleType()).alias('latitude'),
        col('LONGITUDE').cast(DoubleType()).alias('longitude'),
        col('ELEVATION').cast(DoubleType()).alias('elevation'),
        col('NAME').alias('station_name'),
        parse_temperature('TMP').alias('temperature'),
        parse_temperature('DEW').alias('dew_point'),
        parse_pressure('SLP').alias('sea_level_pressure'),
        parse_visibility('VIS').alias('visibility'),
        wind_direction.alias('wind_direction'),
        wind_speed.alias('wind_speed'),
        precip_sum.alias('precipitation')
    ]
    
    # Add ceiling if available
    if has_cig:
        select_cols.append(parse_ceiling('CIG').alias('ceiling_height'))
    
    df_clean = df.select(*select_cols)
    
    print("DONE - Basic features parsed")
    
    # ==================== Temporal Features ====================
    print("\n" + "=" * 80)
    print("PHASE 2: Add Temporal Features")
    print("=" * 80)
    
    df_clean = df_clean \
        .withColumn('year', year(col('timestamp'))) \
        .withColumn('month', month(col('timestamp'))) \
        .withColumn('day', dayofmonth(col('timestamp'))) \
        .withColumn('hour', hour(col('timestamp')))
    
    # Cyclical encoding
    df_clean = df_clean \
        .withColumn('hour_sin', sin(col('hour') * 2 * math.pi / 24)) \
        .withColumn('hour_cos', cos(col('hour') * 2 * math.pi / 24)) \
        .withColumn('month_sin', sin(col('month') * 2 * math.pi / 12)) \
        .withColumn('month_cos', cos(col('month') * 2 * math.pi / 12)) \
        .withColumn('wind_dir_sin', when(col('wind_direction').isNotNull(),
                    sin(col('wind_direction') * math.pi / 180)).otherwise(None)) \
        .withColumn('wind_dir_cos', when(col('wind_direction').isNotNull(),
                    cos(col('wind_direction') * math.pi / 180)).otherwise(None))
    
    print("DONE - Temporal encoding added")
    
    # ==================== Quality Filters ====================
    print("\n" + "=" * 80)
    print("PHASE 3: Quality Filtering")
    print("=" * 80)
    
    print("Applying filters:")
    print("  - Remove NULL temperatures")
    print("  - Temperature range: -90 to +60°C")
    print("  - Dew point <= Temperature")
    print("  - Sea level pressure: 950-1050 hPa")
    
    df_clean = df_clean.filter(col('temperature').isNotNull())
    df_clean = df_clean.filter((col('temperature') >= -90) & (col('temperature') <= 60))
    df_clean = df_clean.filter(col('dew_point').isNull() | (col('dew_point') <= col('temperature')))
    df_clean = df_clean.filter(
        col('sea_level_pressure').isNull() | 
        ((col('sea_level_pressure') >= 950) & (col('sea_level_pressure') <= 1050))
    )
    
    # Cache for counting
    df_clean.cache()
    rows_after_filter = df_clean.count()
    
    print(f"\nRows before filtering: {total_rows:,}")
    print(f"Rows after filtering: {rows_after_filter:,}")
    print(f"Retention rate: {100*rows_after_filter/total_rows:.1f}%")
    
    # ==================== Feature Summary ====================
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    
    feature_count = {
        "Geographic": 3,
        "Basic Weather": 7,
        "Temporal Encoding": 6
    }
    
    if has_cig:
        feature_count["Ceiling Height (NEW)"] = 1
    
    total_features = sum(feature_count.values())
    
    print("\nFeature breakdown:")
    for cat, count in feature_count.items():
        print(f"  {cat}: {count}")
    print(f"  " + "-"*30)
    print(f"  TOTAL: {total_features} features")
    
    print(f"\nV0 had: 14 features")
    print(f"V2 has: {total_features} features (+{total_features-14} new)")
    
    # ==================== Save ====================
    print("\n" + "=" * 80)
    print("SAVING DATA")
    print("=" * 80)
    
    print(f"Output: {output_path}")
    print("Writing partitioned parquet files...")
    print("(This should take 30-40 minutes)")
    
    df_clean.write.mode('overwrite').partitionBy('year', 'month').parquet(output_path)
    
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Output location: {output_path}")
    print(f"Total features: {total_features}")
    print(f"Total rows: {rows_after_filter:,}")
    print(f"Data retention: {100*rows_after_filter/total_rows:.1f}%")
    print("\nData is partitioned by year/month")
    print("Ready for train/test split!")
    
    df_clean.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()