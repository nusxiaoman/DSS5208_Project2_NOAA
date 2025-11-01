"""
NOAA Weather Data Cleanup Script - Test Version
Processes raw NOAA CSV data and extracts clean features for ML training
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, split, when, regexp_extract, regexp_replace,
    to_timestamp, year, month, dayofmonth, hour,
    sin, cos, lit, expr, abs as spark_abs
)
from pyspark.sql.types import DoubleType, IntegerType
import sys
import math

def parse_temperature(temp_str):
    """Parse temperature string: '-0070,1' -> -7.0°C"""
    return (
        when(col(temp_str).isNull() | (col(temp_str) == '+9999,9'), None)
        .otherwise(
            split(col(temp_str), ',').getItem(0).cast(DoubleType()) / 10.0
        )
    )

def parse_wind(wnd_col):
    """Parse wind string: '318,1,N,0061,1' -> direction, speed"""
    parts = split(col(wnd_col), ',')
    
    # Wind direction (0-360 degrees, 999 = missing)
    direction = when(parts.getItem(0) == '999', None).otherwise(parts.getItem(0).cast(IntegerType()))
    
    # Wind speed (m/s, 9999 = missing)
    speed = when(parts.getItem(3) == '9999', None).otherwise(parts.getItem(3).cast(DoubleType()) / 10.0)
    
    return direction, speed

def parse_pressure(slp_str):
    """Parse sea level pressure: '10208,1' -> 1020.8 hPa"""
    return (
        when(col(slp_str).isNull() | (col(slp_str).contains('99999')), None)
        .otherwise(
            split(col(slp_str), ',').getItem(0).cast(DoubleType()) / 10.0
        )
    )

def parse_visibility(vis_str):
    """Parse visibility: '025000,1,9,9' -> 25000 meters (or None if missing)"""
    return (
        when(col(vis_str).isNull() | (col(vis_str).contains('999999')), None)
        .otherwise(
            split(col(vis_str), ',').getItem(0).cast(DoubleType())
        )
    )

def parse_precipitation(precip_col):
    """Parse precipitation: '24,0002,3,1' -> 0.2 mm (depth in tenths of mm)"""
    if precip_col not in ['AA1', 'AA2', 'AA3', 'GA1', 'GA2', 'GA3']:
        return lit(None)
    
    return (
        when(col(precip_col).isNull() | (col(precip_col).contains('9999')), 0.0)
        .otherwise(
            split(col(precip_col), ',').getItem(1).cast(DoubleType()) / 10.0
        )
    )

def parse_cloud_coverage(sky_cols):
    """Extract cloud coverage from sky condition columns"""
    # This is a simplified version - you may need to adjust based on actual data
    # For now, we'll just check if any cloud data exists
    return lit(None)  # Placeholder - can be enhanced based on requirements

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Data Cleanup - Test") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Input/output paths
    input_path = "gs://weather-ml-bucket-1760514177/data/csv"
    output_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_clean_test"
    
    print("=" * 80)
    print("NOAA Weather Data Cleanup - Test Mode")
    print("=" * 80)
    
    # Read raw CSV data
    print(f"\nReading data from: {input_path}")
    df = spark.read.csv(input_path, header=True, inferSchema=False, pathGlobFilter='*.csv')
    
    print(f"Total rows loaded: {df.count():,}")
    print(f"Columns: {len(df.columns)}")
    
    # Sample 1% for testing
    df_sample = df.sample(fraction=0.01, seed=42)
    print(f"\nSampled rows for testing: {df_sample.count():,}")
    
    # Parse and clean the data
    print("\nParsing weather measurements...")
    
    # Parse wind
    wind_direction, wind_speed = parse_wind('WND')
    
    # Parse precipitation - sum all available precipitation columns
    precip_sum = lit(0.0)
    for precip_col in ['AA1', 'AA2', 'AA3', 'GA1', 'GA2', 'GA3']:
        if precip_col in df_sample.columns:
            precip_sum = precip_sum + parse_precipitation(precip_col)
    
    # Create cleaned dataframe with selected features
    df_clean = df_sample.select(
        col('STATION'),
        to_timestamp(col('DATE')).alias('timestamp'),
        col('LATITUDE').cast(DoubleType()).alias('latitude'),
        col('LONGITUDE').cast(DoubleType()).alias('longitude'),
        col('ELEVATION').cast(DoubleType()).alias('elevation'),
        col('NAME').alias('station_name'),
        
        # Target variable - Temperature
        parse_temperature('TMP').alias('temperature'),
        
        # Weather features
        parse_temperature('DEW').alias('dew_point'),
        parse_pressure('SLP').alias('sea_level_pressure'),
        parse_visibility('VIS').alias('visibility'),
        wind_direction.alias('wind_direction'),
        wind_speed.alias('wind_speed'),
        precip_sum.alias('precipitation'),
    )
    
    # Add temporal features
    df_clean = df_clean.withColumn('year', year(col('timestamp'))) \
        .withColumn('month', month(col('timestamp'))) \
        .withColumn('day', dayofmonth(col('timestamp'))) \
        .withColumn('hour', hour(col('timestamp')))
    
    # Cyclical encoding for temporal features
    df_clean = df_clean \
        .withColumn('hour_sin', sin(col('hour') * 2 * math.pi / 24)) \
        .withColumn('hour_cos', cos(col('hour') * 2 * math.pi / 24)) \
        .withColumn('month_sin', sin(col('month') * 2 * math.pi / 12)) \
        .withColumn('month_cos', cos(col('month') * 2 * math.pi / 12))
    
    # Cyclical encoding for wind direction
    df_clean = df_clean \
        .withColumn('wind_dir_sin', 
                   when(col('wind_direction').isNotNull(),
                        sin(col('wind_direction') * math.pi / 180)).otherwise(None)) \
        .withColumn('wind_dir_cos',
                   when(col('wind_direction').isNotNull(),
                        cos(col('wind_direction') * math.pi / 180)).otherwise(None))
    
    # Filter out rows where temperature (target) is missing
    print("\nFiltering invalid records...")
    df_clean = df_clean.filter(col('temperature').isNotNull())
    
    # Remove extreme outliers for temperature (-90°C to +60°C)
    df_clean = df_clean.filter(
        (col('temperature') >= -90) & (col('temperature') <= 60)
    )
    
    # Remove records where dew point > temperature (physical constraint)
    df_clean = df_clean.filter(
        col('dew_point').isNull() | (col('dew_point') <= col('temperature'))
    )
    
    # Validate pressure range (950-1050 hPa)
    df_clean = df_clean.filter(
        col('sea_level_pressure').isNull() | 
        ((col('sea_level_pressure') >= 950) & (col('sea_level_pressure') <= 1050))
    )
    
    print(f"\nRows after filtering: {df_clean.count():,}")
    
    # Show sample of cleaned data
    print("\n" + "=" * 80)
    print("SAMPLE OF CLEANED DATA (10 rows)")
    print("=" * 80)
    df_clean.select(
        'temperature', 'dew_point', 'sea_level_pressure', 
        'wind_speed', 'visibility', 'precipitation',
        'latitude', 'longitude', 'hour', 'month'
    ).show(10, truncate=False)
    
    # Show statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    df_clean.select(
        'temperature', 'dew_point', 'sea_level_pressure',
        'wind_speed', 'visibility', 'precipitation'
    ).describe().show()
    
    # Check for missing values
    print("\n" + "=" * 80)
    print("MISSING VALUE COUNTS")
    print("=" * 80)
    total_rows = df_clean.count()
    for col_name in df_clean.columns:
        null_count = df_clean.filter(col(col_name).isNull()).count()
        if null_count > 0:
            print(f"{col_name}: {null_count:,} ({100*null_count/total_rows:.2f}%)")
    
    # Save cleaned data
    print(f"\n\nSaving cleaned data to: {output_path}")
    df_clean.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(output_path)
    
    print("\n" + "=" * 80)
    print("CLEANUP COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Output location: {output_path}")
    print(f"Final row count: {df_clean.count():,}")
    
    spark.stop()

if __name__ == "__main__":
    main()