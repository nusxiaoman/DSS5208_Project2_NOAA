"""
NOAA Weather Data Enhanced Cleanup V2
Based on noaa_cleanup_full.py with CRITICAL additions:
- Weather conditions (AW fields)
- Wind gust (OC1)
- Temporal lag features (POWERFUL!)
- Station statistics
- Enhanced temporal encoding (day_of_year)
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, when, to_timestamp, year, month, dayofmonth, hour, dayofyear,
    sin, cos, lit, lag, avg as spark_avg
)
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.sql.window import Window
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

def main():
    spark = SparkSession.builder \
        .appName("NOAA Enhanced Cleanup V2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths - from your original V1 script
    input_path = "gs://weather-ml-bucket-1760514177/data/csv/*.csv"
    output_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
    
    print("=" * 80)
    print("NOAA ENHANCED DATA CLEANUP V2")
    print("=" * 80)
    print("New features vs V1:")
    print("  ✓ Weather conditions (rain/snow/fog/storm) from AW fields")
    print("  ✓ Wind gust from OC1")
    print("  ✓ Temporal lag features (temp_lag_1h, etc.) - CRITICAL!")
    print("  ✓ Station statistics (avg temp, dew, pressure)")
    print("  ✓ Day of year encoding")
    print("=" * 80)
    
    # Read data
    print(f"\nReading: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=False)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    available_cols = df.columns
    has_aw = any(c.startswith('AW') for c in available_cols)
    has_oc1 = 'OC1' in available_cols
    
    print(f"\nOptional fields:")
    print(f"  AW (weather): {'✓' if has_aw else '✗'}")
    print(f"  OC1 (gust): {'✓' if has_oc1 else '✗'}")
    
    # ==================== Basic Parsing (same as V1) ====================
    print("\n" + "=" * 80)
    print("PHASE 1: Basic Features")
    print("=" * 80)
    
    wind_direction, wind_speed = parse_wind('WND')
    
    precip_sum = lit(0.0)
    for p in ['AA1', 'AA2', 'AA3', 'AA4']:
        if p in available_cols:
            precip_sum = precip_sum + parse_precipitation(p)
    
    df_clean = df.select(
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
        precip_sum.alias('precipitation'),
        *[col(c) for c in ['AW1', 'AW2', 'AW3', 'OC1'] if c in available_cols]
    )
    
    print("✓ Basic features parsed")
    
    # ==================== Temporal Features ====================
    print("\n" + "=" * 80)
    print("PHASE 2: Temporal Features (Enhanced)")
    print("=" * 80)
    
    df_clean = df_clean \
        .withColumn('year', year(col('timestamp'))) \
        .withColumn('month', month(col('timestamp'))) \
        .withColumn('day', dayofmonth(col('timestamp'))) \
        .withColumn('hour', hour(col('timestamp'))) \
        .withColumn('day_of_year', dayofyear(col('timestamp')))
    
    # Cyclical encoding
    df_clean = df_clean \
        .withColumn('hour_sin', sin(col('hour') * 2 * math.pi / 24)) \
        .withColumn('hour_cos', cos(col('hour') * 2 * math.pi / 24)) \
        .withColumn('month_sin', sin(col('month') * 2 * math.pi / 12)) \
        .withColumn('month_cos', cos(col('month') * 2 * math.pi / 12)) \
        .withColumn('day_of_year_sin', sin(col('day_of_year') * 2 * math.pi / 365)) \
        .withColumn('day_of_year_cos', cos(col('day_of_year') * 2 * math.pi / 365)) \
        .withColumn('wind_dir_sin', when(col('wind_direction').isNotNull(),
                    sin(col('wind_direction') * math.pi / 180)).otherwise(None)) \
        .withColumn('wind_dir_cos', when(col('wind_direction').isNotNull(),
                    cos(col('wind_direction') * math.pi / 180)).otherwise(None))
    
    print("✓ Temporal + wind direction encoding")
    
    # ==================== Weather Conditions ====================
    print("\n" + "=" * 80)
    print("PHASE 3: Weather Conditions")
    print("=" * 80)
    
    if has_aw:
        for col_name in ['AW1', 'AW2', 'AW3']:
            if col_name not in df_clean.columns:
                df_clean = df_clean.withColumn(col_name, lit(None))
        
        df_clean = df_clean \
            .withColumn('is_raining',
                when((col('AW1').rlike('RA|DZ|SHRA|TSRA')) |
                     (col('AW2').rlike('RA|DZ|SHRA|TSRA')) |
                     (col('AW3').rlike('RA|DZ|SHRA|TSRA')), 1).otherwise(0)) \
            .withColumn('is_snowing',
                when((col('AW1').rlike('SN|SG|IC|PE|PL')) |
                     (col('AW2').rlike('SN|SG|IC|PE|PL')) |
                     (col('AW3').rlike('SN|SG|IC|PE|PL')), 1).otherwise(0)) \
            .withColumn('is_foggy',
                when((col('AW1').rlike('FG|BR|MIFG|BCFG')) |
                     (col('AW2').rlike('FG|BR|MIFG|BCFG')) |
                     (col('AW3').rlike('FG|BR|MIFG|BCFG')), 1).otherwise(0)) \
            .withColumn('is_thunderstorm',
                when((col('AW1').rlike('TS')) |
                     (col('AW2').rlike('TS')) |
                     (col('AW3').rlike('TS')), 1).otherwise(0))
        print("✓ Weather flags created")
    else:
        df_clean = df_clean \
            .withColumn('is_raining', lit(0)) \
            .withColumn('is_snowing', lit(0)) \
            .withColumn('is_foggy', lit(0)) \
            .withColumn('is_thunderstorm', lit(0))
        print("⚠ No AW fields - placeholders added")
    
    # ==================== Wind Gust ====================
    print("\n" + "=" * 80)
    print("PHASE 4: Wind Gust")
    print("=" * 80)
    
    if has_oc1:
        df_clean = df_clean.withColumn('wind_gust',
            when(col('OC1').isNull() | col('OC1').contains('9999'), None)
            .otherwise(split(col('OC1'), ',').getItem(1).cast(DoubleType()) / 10.0))
        print("✓ Wind gust parsed")
    else:
        df_clean = df_clean.withColumn('wind_gust', lit(None).cast(DoubleType()))
        print("⚠ No OC1 - placeholder added")
    
    # Drop raw fields
    cols_to_drop = [c for c in ['AW1', 'AW2', 'AW3', 'OC1'] if c in df_clean.columns]
    df_clean = df_clean.drop(*cols_to_drop)
    
    # ==================== Quality Filters ====================
    print("\n" + "=" * 80)
    print("PHASE 5: Quality Filtering")
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
    print(f"Rows: {rows_after_filter:,} ({100*rows_after_filter/total_rows:.1f}% retained)")
    
    # ==================== Station Features ====================
    print("\n" + "=" * 80)
    print("PHASE 6: Station Statistics")
    print("=" * 80)
    
    station_stats = df_clean.groupBy('STATION').agg(
        spark_avg('temperature').alias('station_avg_temp'),
        spark_avg('dew_point').alias('station_avg_dew'),
        spark_avg('sea_level_pressure').alias('station_avg_pressure')
    )
    
    df_clean = df_clean.join(station_stats, on='STATION', how='left')
    print("✓ Station averages computed")
    
    # ==================== Lag Features ====================
    print("\n" + "=" * 80)
    print("PHASE 7: Temporal Lag Features (CRITICAL!)")
    print("=" * 80)
    
    df_sorted = df_clean.orderBy('STATION', 'timestamp')
    window_spec = Window.partitionBy('STATION').orderBy('timestamp')
    
    df_sorted = df_sorted \
        .withColumn('temp_lag_1h', lag('temperature', 1).over(window_spec)) \
        .withColumn('temp_lag_2h', lag('temperature', 2).over(window_spec)) \
        .withColumn('temp_lag_3h', lag('temperature', 3).over(window_spec)) \
        .withColumn('pressure_lag_1h', lag('sea_level_pressure', 1).over(window_spec)) \
        .withColumn('dew_lag_1h', lag('dew_point', 1).over(window_spec))
    
    window_3h = Window.partitionBy('STATION').orderBy('timestamp').rowsBetween(-3, -1)
    df_sorted = df_sorted \
        .withColumn('temp_rolling_3h', spark_avg('temperature').over(window_3h)) \
        .withColumn('pressure_rolling_3h', spark_avg('sea_level_pressure').over(window_3h))
    
    df_sorted = df_sorted \
        .withColumn('temp_change_1h',
            when(col('temp_lag_1h').isNotNull(), col('temperature') - col('temp_lag_1h')).otherwise(None)) \
        .withColumn('pressure_change_1h',
            when(col('pressure_lag_1h').isNotNull(), col('sea_level_pressure') - col('pressure_lag_1h')).otherwise(None))
    
    print("✓ Lag features: 9 (temp/pressure lags, rolling avg, changes)")
    print("  ⚠ MUST use TIME-BASED split to avoid data leakage!")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    
    features = {
        "Geographic": 3,
        "Basic Weather": 7,
        "Temporal (enhanced)": 8,
        "Weather Conditions": 4,
        "Wind Gust": 1,
        "Station Stats": 3,
        "Lag Features": 9
    }
    
    total = sum(features.values())
    print("\nFeature breakdown:")
    for cat, count in features.items():
        print(f"  {cat}: {count}")
    print(f"  {'─'*30}")
    print(f"  TOTAL: {total} features (V1 had 14)")
    print(f"  Increase: +{total-14} features (+{(total-14)/14*100:.0f}%)")
    
    # ==================== Save ====================
    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)
    
    print(f"Output: {output_path}")
    df_sorted.write.mode('overwrite').partitionBy('year', 'month').parquet(output_path)
    
    print("\n" + "=" * 80)
    print("✓ COMPLETED!")
    print("=" * 80)
    print(f"Features: {total} (was 14, +{total-14})")
    print(f"Rows: {rows_after_filter:,}")
    print(f"\nExpected improvement:")
    print(f"  V1: 4.65°C")
    print(f"  V2: 3.5-4.0°C (lag features are key!)")
    print(f"\nNEXT: Use TIME-BASED split (Jan-Sep train, Oct-Dec test)")
    
    df_sorted.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()