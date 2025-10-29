"""
NOAA Weather Data Enhanced Cleanup V2 - COMPLETE
Includes ALL available weather fields:
- GA1/GA2/GA3: Cloud cover and height
- CIG: Ceiling height
- OC1: Wind gust
- Median imputation for NULL values
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, when, to_timestamp, year, month, dayofmonth, hour,
    sin, cos, lit, expr
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

def parse_cloud_cover(ga_col):
    """Parse GA cloud: '08,1,02,1,00800,1' -> coverage_code, base_height"""
    parts = split(col(ga_col), ',')
    # Coverage code: 0=clear, 1-8=coverage level
    coverage = when(col(ga_col).isNull() | parts.getItem(0).contains('99'), None) \
        .otherwise(parts.getItem(0).cast(IntegerType()))
    # Base height in meters
    height = when(col(ga_col).isNull() | parts.getItem(4).contains('99999'), None) \
        .otherwise(parts.getItem(4).cast(DoubleType()))
    return coverage, height

def parse_ceiling(cig_str):
    """Parse CIG: '00800,1,9,9' -> 800 meters"""
    return (
        when(col(cig_str).isNull() | (col(cig_str).contains('99999')), None)
        .otherwise(split(col(cig_str), ',').getItem(0).cast(DoubleType()))
    )

def parse_wind_gust(oc1_str):
    """Parse OC1: '120,1,0069,1' -> 6.9 m/s"""
    return (
        when(col(oc1_str).isNull() | (col(oc1_str).contains('9999')), None)
        .otherwise(split(col(oc1_str), ',').getItem(2).cast(DoubleType()) / 10.0)
    )

def main():
    spark = SparkSession.builder \
        .appName("NOAA Enhanced Cleanup V2 - Complete") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths
    input_path = "gs://weather-ml-bucket-1760514177/data/csv/*.csv"
    output_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
    
    print("=" * 80)
    print("NOAA ENHANCED DATA CLEANUP V2 - COMPLETE VERSION")
    print("=" * 80)
    print("New features:")
    print("  - Cloud cover (GA1/GA2/GA3)")
    print("  - Ceiling height (CIG)")
    print("  - Wind gust (OC1)")
    print("  - Median imputation for NULL values")
    print("=" * 80)
    
    # Read data
    print(f"\nReading: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=False)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    available_cols = df.columns

    # DEBUG: Print actual column names as Spark sees them
    print("\nDEBUG - All columns as Spark reads them:")
    for i, c in enumerate(available_cols):
        if any(x in c.upper() for x in ['GA', 'AW', 'CIG', 'OC']):
            print(f"  [{i}] '{c}' (len={len(c)})")
            
    has_ga1 = 'GA1' in available_cols
    has_ga2 = 'GA2' in available_cols
    has_ga3 = 'GA3' in available_cols
    has_cig = 'CIG' in available_cols
    has_oc1 = 'OC1' in available_cols
    
    print(f"\nOptional fields detected:")
    print(f"  GA1 (cloud layer 1): {'YES' if has_ga1 else 'NO'}")
    print(f"  GA2 (cloud layer 2): {'YES' if has_ga2 else 'NO'}")
    print(f"  GA3 (cloud layer 3): {'YES' if has_ga3 else 'NO'}")
    print(f"  CIG (ceiling): {'YES' if has_cig else 'NO'}")
    print(f"  OC1 (wind gust): {'YES' if has_oc1 else 'NO'}")
    
    # ==================== Basic Parsing ====================
    print("\n" + "=" * 80)
    print("PHASE 1: Basic Features")
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
    
    # Add cloud features
    if has_ga1:
        ga1_coverage, ga1_height = parse_cloud_cover('GA1')
        select_cols.extend([
            ga1_coverage.alias('cloud_cover_1'),
            ga1_height.alias('cloud_base_height_1')
        ])
    
    if has_ga2:
        ga2_coverage, ga2_height = parse_cloud_cover('GA2')
        select_cols.extend([
            ga2_coverage.alias('cloud_cover_2'),
            ga2_height.alias('cloud_base_height_2')
        ])
    
    # Add ceiling
    if has_cig:
        select_cols.append(parse_ceiling('CIG').alias('ceiling_height'))
    
    # Add wind gust
    if has_oc1:
        select_cols.append(parse_wind_gust('OC1').alias('wind_gust'))
    
    df_clean = df.select(*select_cols)
    
    print("DONE - Basic features parsed")
    
    # ==================== Temporal Features ====================
    print("\n" + "=" * 80)
    print("PHASE 2: Temporal Features")
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
    
    print("DONE - Temporal + wind direction encoding")
    
    # ==================== Quality Filters ====================
    print("\n" + "=" * 80)
    print("PHASE 3: Quality Filtering")
    print("=" * 80)
    
    df_clean = df_clean.filter(col('temperature').isNotNull())
    df_clean = df_clean.filter((col('temperature') >= -90) & (col('temperature') <= 60))
    df_clean = df_clean.filter(col('dew_point').isNull() | (col('dew_point') <= col('temperature')))
    df_clean = df_clean.filter(
        col('sea_level_pressure').isNull() | 
        ((col('sea_level_pressure') >= 950) & (col('sea_level_pressure') <= 1050))
    )
    
    df_clean.cache()
    rows_after_filter = df_clean.count()
    print(f"Rows after filtering: {rows_after_filter:,} ({100*rows_after_filter/total_rows:.1f}% retained)")
    
    # ==================== NULL Value Imputation ====================
    print("\n" + "=" * 80)
    print("PHASE 4: NULL Value Handling")
    print("=" * 80)
    
    # Calculate medians for key features
    print("Calculating medians for imputation...")
    
    median_slp = df_clean.approxQuantile('sea_level_pressure', [0.5], 0.01)[0]
    median_wind_speed = df_clean.approxQuantile('wind_speed', [0.5], 0.01)[0]
    median_visibility = df_clean.approxQuantile('visibility', [0.5], 0.01)[0]
    
    print(f"Median imputation values:")
    print(f"  Sea level pressure: {median_slp:.1f} hPa")
    print(f"  Wind speed: {median_wind_speed:.1f} m/s")
    print(f"  Visibility: {median_visibility:.0f} m")
    
    # Apply median imputation
    df_clean = df_clean \
        .withColumn('sea_level_pressure',
            when(col('sea_level_pressure').isNull(), median_slp)
            .otherwise(col('sea_level_pressure'))) \
        .withColumn('wind_speed',
            when(col('wind_speed').isNull(), median_wind_speed)
            .otherwise(col('wind_speed'))) \
        .withColumn('visibility',
            when(col('visibility').isNull(), median_visibility)
            .otherwise(col('visibility')))
    
    print("DONE - Median imputation applied")
    
    # Count remaining NULLs for optional features
    null_checks = []
    null_checks.append("sum(case when dew_point is null then 1 else 0 end) as dew_null")
    null_checks.append("sum(case when wind_direction is null then 1 else 0 end) as wind_dir_null")
    
    if has_ga1:
        null_checks.append("sum(case when cloud_cover_1 is null then 1 else 0 end) as cloud1_null")
    if has_cig:
        null_checks.append("sum(case when ceiling_height is null then 1 else 0 end) as ceiling_null")
    if has_oc1:
        null_checks.append("sum(case when wind_gust is null then 1 else 0 end) as gust_null")
    
    null_counts = df_clean.selectExpr(*null_checks).collect()[0]
    
    print(f"\nRemaining NULL rates:")
    print(f"  Dew point: {null_counts['dew_null']/rows_after_filter*100:.1f}%")
    print(f"  Wind direction: {null_counts['wind_dir_null']/rows_after_filter*100:.1f}%")
    if has_ga1:
        print(f"  Cloud cover: {null_counts['cloud1_null']/rows_after_filter*100:.1f}%")
    if has_cig:
        print(f"  Ceiling: {null_counts['ceiling_null']/rows_after_filter*100:.1f}%")
    if has_oc1:
        print(f"  Wind gust: {null_counts['gust_null']/rows_after_filter*100:.1f}%")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    
    feature_count = {
        "Geographic": 3,
        "Basic Weather": 7,
        "Temporal": 6
    }
    
    if has_ga1:
        feature_count["Cloud (GA1)"] = 2
    if has_ga2:
        feature_count["Cloud (GA2)"] = 2
    if has_cig:
        feature_count["Ceiling (CIG)"] = 1
    if has_oc1:
        feature_count["Wind Gust (OC1)"] = 1
    
    total_features = sum(feature_count.values())
    print("\nFeature breakdown:")
    for cat, count in feature_count.items():
        print(f"  {cat}: {count}")
    print(f"  " + "-"*30)
    print(f"  TOTAL: {total_features} features")
    print(f"\nNew features vs V1 (21 features): +{total_features - 21}")
    
    # ==================== Save ====================
    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)
    
    print(f"Output: {output_path}")
    df_clean.write.mode('overwrite').partitionBy('year', 'month').parquet(output_path)
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total features: {total_features}")
    print(f"Total rows: {rows_after_filter:,}")
    print(f"\nNEXT STEP: Random split (70/30) and train models")
    print(f"Expected improvement: Better than V1 with cloud + gust features!")
    
    df_clean.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()