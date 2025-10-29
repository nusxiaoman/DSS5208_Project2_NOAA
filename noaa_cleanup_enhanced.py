"""
NOAA Weather Data Enhanced Cleanup V2 (Simplified)
Based on noaa_cleanup_full.py with focused additions:
- Weather conditions (AW fields: rain, snow, fog, thunderstorm)
- Improved NULL value handling with median imputation
- Random split compatible (no lag features needed)
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

def main():
    spark = SparkSession.builder \
        .appName("NOAA Enhanced Cleanup V2 - Simplified") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths
    input_path = "gs://weather-ml-bucket-1760514177/data/csv/*.csv"
    output_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
    
    print("=" * 80)
    print("NOAA ENHANCED DATA CLEANUP V2 (Simplified)")
    print("=" * 80)
    print("New features vs V1:")
    print("  - Weather conditions (rain/snow/fog/storm) from AW fields")
    print("  - Improved NULL handling with median imputation")
    print("  - Optimized for random train/test split")
    print("=" * 80)
    
    # Read data
    print(f"\nReading: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=False)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    available_cols = df.columns
    has_aw = any(c.startswith('AW') for c in available_cols)
    
    print(f"\nOptional fields:")
    print(f"  AW (weather): {'YES' if has_aw else 'NO'}")
    
    # ==================== Basic Parsing ====================
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
        *[col(c) for c in ['AW1', 'AW2', 'AW3'] if c in available_cols]
    )
    
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
        print("DONE - Weather flags created from AW fields")
    else:
        df_clean = df_clean \
            .withColumn('is_raining', lit(0)) \
            .withColumn('is_snowing', lit(0)) \
            .withColumn('is_foggy', lit(0)) \
            .withColumn('is_thunderstorm', lit(0))
        print("WARNING - No AW fields found, using placeholder 0 values")
    
    # Drop raw AW fields
    cols_to_drop = [c for c in ['AW1', 'AW2', 'AW3'] if c in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(*cols_to_drop)
    
    # ==================== Quality Filters ====================
    print("\n" + "=" * 80)
    print("PHASE 4: Quality Filtering")
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
    print("PHASE 5: NULL Value Handling")
    print("=" * 80)
    
    # Calculate medians for key features using approxQuantile
    print("Calculating medians for imputation...")
    
    median_slp = df_clean.approxQuantile('sea_level_pressure', [0.5], 0.01)[0]
    median_wind_speed = df_clean.approxQuantile('wind_speed', [0.5], 0.01)[0]
    median_visibility = df_clean.approxQuantile('visibility', [0.5], 0.01)[0]
    
    print(f"Median imputation values:")
    print(f"  Sea level pressure: {median_slp:.1f} hPa")
    print(f"  Wind speed: {median_wind_speed:.1f} m/s")
    print(f"  Visibility: {median_visibility:.0f} m")
    
    # Apply median imputation for critical features
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
    
    # For wind_direction, keep NULL (cyclical encoding handles it)
    # For dew_point, keep NULL (not critical, ML can handle it)
    # For precipitation, already defaulted to 0.0
    
    print("DONE - Median imputation applied")
    
    # Count remaining NULLs
    null_counts = df_clean.agg(
        (expr('sum(case when dew_point is null then 1 else 0 end)') / rows_after_filter * 100).alias('dew_point_null_pct'),
        (expr('sum(case when wind_direction is null then 1 else 0 end)') / rows_after_filter * 100).alias('wind_dir_null_pct')
    ).collect()[0]
    
    print(f"\nRemaining NULL rates:")
    print(f"  Dew point: {null_counts['dew_point_null_pct']:.1f}%")
    print(f"  Wind direction: {null_counts['wind_dir_null_pct']:.1f}%")
    
    # ==================== Summary ====================
    print("\n" + "=" * 80)
    print("FEATURE SUMMARY")
    print("=" * 80)
    
    features = {
        "Geographic": 3,
        "Basic Weather": 7,
        "Temporal": 6,
        "Weather Conditions": 4
    }
    
    total = sum(features.values())
    print("\nFeature breakdown:")
    for cat, count in features.items():
        print(f"  {cat}: {count}")
    print(f"  " + "-"*30)
    print(f"  TOTAL: {total} features")
    print(f"\nChanges from V1 (21 features):")
    print(f"  + Weather conditions (4 new features)")
    print(f"  + Median imputation for NULL values")
    print(f"  - Removed: lag features, station stats, enhanced temporal encoding")
    
    # ==================== Save ====================
    print("\n" + "=" * 80)
    print("SAVING")
    print("=" * 80)
    
    print(f"Output: {output_path}")
    df_clean.write.mode('overwrite').partitionBy('year', 'month').parquet(output_path)
    
    print("\n" + "=" * 80)
    print("COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Total features: {total}")
    print(f"Total rows: {rows_after_filter:,}")
    print(f"\nNEXT STEP: Random split (70/30) and train models")
    
    df_clean.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()