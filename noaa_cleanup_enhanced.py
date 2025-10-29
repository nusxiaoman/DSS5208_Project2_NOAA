"""
Enhanced NOAA Data Cleanup - Complete Feature Set
Includes: Cloud cover, present weather, wind gust, and temporal lag features
Version 2.0 - Comprehensive Feature Engineering
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, when, lit, regexp_extract, substring, length, trim,
    sin, cos, radians, hour, month, year, dayofyear,
    to_timestamp, unix_timestamp, lag, avg as spark_avg,
    isnan, isnull, count as spark_count, sum as spark_sum,
    udf, monotonically_increasing_id
)
from pyspark.sql.types import FloatType, IntegerType, StringType
from pyspark.sql.window import Window
import math

# Initialize Spark
spark = SparkSession.builder \
    .appName("NOAA Enhanced Cleanup") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

print("=" * 80)
print("ENHANCED NOAA DATA CLEANUP - COMPLETE FEATURE SET")
print("=" * 80)
print("New features:")
print("  ✓ Cloud cover (GA fields)")
print("  ✓ Present weather (AW fields)")
print("  ✓ Wind gust (OC1)")
print("  ✓ Temporal lag features")
print("  ✓ Station-based features")
print("=" * 80)

# Input/Output paths
INPUT_PATH = "gs://weather-ml-bucket-1760514177/data/2024*.csv"
OUTPUT_PATH = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"

print(f"\nInput: {INPUT_PATH}")
print(f"Output: {OUTPUT_PATH}")

# Read raw data
print("\nReading raw CSV files...")
df = spark.read.csv(INPUT_PATH, header=True, inferSchema=False)

original_count = df.count()
print(f"Original rows: {original_count:,}")

# ==================== BASIC PARSING (Same as before) ====================

print("\n" + "=" * 80)
print("PHASE 1: Basic Feature Parsing")
print("=" * 80)

# Parse temperature
def parse_temp(temp_str):
    if temp_str and len(temp_str) >= 5:
        try:
            value = float(temp_str.split(',')[0])
            quality = temp_str.split(',')[1] if ',' in temp_str else '9'
            return value / 10.0 if quality in ['1', '5'] else None
        except:
            return None
    return None

parse_temp_udf = udf(parse_temp, FloatType())

# Parse dew point (same format as temperature)
parse_dew_udf = udf(parse_temp, FloatType())

# Parse sea level pressure
def parse_pressure(pressure_str):
    if pressure_str and len(pressure_str) >= 5:
        try:
            value = float(pressure_str.split(',')[0])
            quality = pressure_str.split(',')[1] if ',' in pressure_str else '9'
            return value / 10.0 if quality in ['1', '5'] else None
        except:
            return None
    return None

parse_pressure_udf = udf(parse_pressure, FloatType())

# Parse wind direction and speed
def parse_wind_direction(wind_str):
    if wind_str and len(wind_str) >= 5:
        try:
            direction = int(wind_str.split(',')[0])
            return direction if direction != 999 else None
        except:
            return None
    return None

def parse_wind_speed(wind_str):
    if wind_str and len(wind_str) >= 11:
        try:
            parts = wind_str.split(',')
            if len(parts) >= 4:
                speed = float(parts[3])
                quality = parts[4] if len(parts) > 4 else '9'
                return speed / 10.0 if quality in ['1', '5'] else None
        except:
            return None
    return None

parse_wind_dir_udf = udf(parse_wind_direction, FloatType())
parse_wind_speed_udf = udf(parse_wind_speed, FloatType())

# Parse visibility
def parse_visibility(vis_str):
    if vis_str and len(vis_str) >= 6:
        try:
            value = float(vis_str.split(',')[0])
            quality = vis_str.split(',')[1] if ',' in vis_str else '9'
            return value if quality in ['1', '5'] else None
        except:
            return None
    return None

parse_vis_udf = udf(parse_visibility, FloatType())

# Parse precipitation
def parse_precip(precip_str):
    if precip_str and len(precip_str) >= 4:
        try:
            value = float(precip_str.split(',')[0])
            quality = precip_str.split(',')[1] if ',' in precip_str else '9'
            return value / 10.0 if quality in ['1', '5'] and value != 9999 else 0.0
        except:
            return 0.0
    return 0.0

parse_precip_udf = udf(parse_precip, FloatType())

# Apply basic parsing
df = df.withColumn('temperature', parse_temp_udf(col('TMP'))) \
       .withColumn('dew_point', parse_dew_udf(col('DEW'))) \
       .withColumn('sea_level_pressure', parse_pressure_udf(col('SLP'))) \
       .withColumn('wind_direction', parse_wind_dir_udf(col('WND'))) \
       .withColumn('wind_speed', parse_wind_speed_udf(col('WND'))) \
       .withColumn('visibility', parse_vis_udf(col('VIS'))) \
       .withColumn('precipitation', parse_precip_udf(col('AA1')))

print("✓ Basic weather features parsed")

# ==================== NEW: CLOUD COVER PARSING ====================

print("\n" + "=" * 80)
print("PHASE 2: Cloud Cover Features (GA fields)")
print("=" * 80)

def parse_cloud_coverage(ga_str):
    """
    Parse cloud coverage code from GA field
    GA106OVC00800099 -> OVC (Overcast)
    Returns: 0=Clear, 1=Few, 2=Scattered, 3=Broken, 4=Overcast, 5=Obscured
    """
    if ga_str and len(ga_str) >= 8:
        try:
            coverage = ga_str[5:8].strip()
            coverage_map = {
                'CLR': 0, 'SKC': 0,  # Clear/Sky Clear
                'FEW': 1,             # Few clouds (1/8-2/8)
                'SCT': 2,             # Scattered (3/8-4/8)
                'BKN': 3,             # Broken (5/8-7/8)
                'OVC': 4,             # Overcast (8/8)
                'VV': 5               # Obscured/Vertical Visibility
            }
            return coverage_map.get(coverage, None)
        except:
            return None
    return None

def parse_cloud_height(ga_str):
    """
    Parse cloud base height in feet
    GA106OVC00800099 -> 800 feet
    """
    if ga_str and len(ga_str) >= 13:
        try:
            height = int(ga_str[8:13])
            return height if height != 99999 else None
        except:
            return None
    return None

parse_cloud_coverage_udf = udf(parse_cloud_coverage, IntegerType())
parse_cloud_height_udf = udf(parse_cloud_height, IntegerType())

# Parse up to 3 cloud layers
df = df.withColumn('cloud_layer1_coverage', parse_cloud_coverage_udf(col('GA1'))) \
       .withColumn('cloud_layer1_height', parse_cloud_height_udf(col('GA1'))) \
       .withColumn('cloud_layer2_coverage', parse_cloud_coverage_udf(col('GA2'))) \
       .withColumn('cloud_layer2_height', parse_cloud_height_udf(col('GA2'))) \
       .withColumn('cloud_layer3_coverage', parse_cloud_coverage_udf(col('GA3'))) \
       .withColumn('cloud_layer3_height', parse_cloud_height_udf(col('GA3')))

# Derive aggregate cloud features
df = df.withColumn('lowest_cloud_coverage', 
    when(col('cloud_layer1_coverage').isNotNull(), col('cloud_layer1_coverage'))
    .otherwise(when(col('cloud_layer2_coverage').isNotNull(), col('cloud_layer2_coverage'))
    .otherwise(col('cloud_layer3_coverage'))))

df = df.withColumn('lowest_cloud_height',
    when(col('cloud_layer1_height').isNotNull(), col('cloud_layer1_height'))
    .otherwise(when(col('cloud_layer2_height').isNotNull(), col('cloud_layer2_height'))
    .otherwise(col('cloud_layer3_height'))))

# Total cloud cover (0-4 scale)
df = df.withColumn('total_cloud_cover',
    when(col('cloud_layer1_coverage').isNotNull() | 
         col('cloud_layer2_coverage').isNotNull() | 
         col('cloud_layer3_coverage').isNotNull(),
         when(col('cloud_layer1_coverage') >= 4, lit(4))  # Overcast
         .when(col('cloud_layer1_coverage') >= 3, lit(3))  # Broken
         .when(col('cloud_layer1_coverage') >= 2, lit(2))  # Scattered
         .when(col('cloud_layer1_coverage') >= 1, lit(1))  # Few
         .otherwise(lit(0)))  # Clear
    .otherwise(None))

print("✓ Cloud cover features created")
print("  - cloud_layer1/2/3_coverage (0-5 scale)")
print("  - cloud_layer1/2/3_height (feet)")
print("  - lowest_cloud_coverage, lowest_cloud_height")
print("  - total_cloud_cover (aggregate)")

# ==================== NEW: PRESENT WEATHER PARSING ====================

print("\n" + "=" * 80)
print("PHASE 3: Present Weather Features (AW fields)")
print("=" * 80)

def parse_weather_rain(aw_str):
    """Check for rain indicators"""
    if aw_str:
        rain_codes = ['RA', 'DZ', 'SHRA', 'TSRA']  # Rain, Drizzle, Showers, Thunderstorm
        return 1 if any(code in aw_str for code in rain_codes) else 0
    return 0

def parse_weather_snow(aw_str):
    """Check for snow indicators"""
    if aw_str:
        snow_codes = ['SN', 'SG', 'IC', 'PE', 'PL']  # Snow, Snow Grains, Ice, Pellets
        return 1 if any(code in aw_str for code in snow_codes) else 0
    return 0

def parse_weather_fog(aw_str):
    """Check for fog/mist indicators"""
    if aw_str:
        fog_codes = ['FG', 'BR', 'MIFG', 'BCFG']  # Fog, Mist, Shallow Fog
        return 1 if any(code in aw_str for code in fog_codes) else 0
    return 0

def parse_weather_thunderstorm(aw_str):
    """Check for thunderstorm"""
    if aw_str:
        return 1 if 'TS' in aw_str else 0
    return 0

parse_rain_udf = udf(parse_weather_rain, IntegerType())
parse_snow_udf = udf(parse_weather_snow, IntegerType())
parse_fog_udf = udf(parse_weather_fog, IntegerType())
parse_ts_udf = udf(parse_weather_thunderstorm, IntegerType())

# Parse weather conditions from multiple AW fields
df = df.withColumn('is_raining', 
    when((parse_rain_udf(col('AW1')) == 1) | 
         (parse_rain_udf(col('AW2')) == 1) | 
         (parse_rain_udf(col('AW3')) == 1), 1).otherwise(0))

df = df.withColumn('is_snowing',
    when((parse_snow_udf(col('AW1')) == 1) | 
         (parse_snow_udf(col('AW2')) == 1) | 
         (parse_snow_udf(col('AW3')) == 1), 1).otherwise(0))

df = df.withColumn('is_foggy',
    when((parse_fog_udf(col('AW1')) == 1) | 
         (parse_fog_udf(col('AW2')) == 1) | 
         (parse_fog_udf(col('AW3')) == 1), 1).otherwise(0))

df = df.withColumn('is_thunderstorm',
    when((parse_ts_udf(col('AW1')) == 1) | 
         (parse_ts_udf(col('AW2')) == 1) | 
         (parse_ts_udf(col('AW3')) == 1), 1).otherwise(0))

print("✓ Present weather features created")
print("  - is_raining (0/1)")
print("  - is_snowing (0/1)")
print("  - is_foggy (0/1)")
print("  - is_thunderstorm (0/1)")

# ==================== NEW: WIND GUST ====================

print("\n" + "=" * 80)
print("PHASE 4: Wind Gust Feature (OC1 field)")
print("=" * 80)

def parse_wind_gust(oc1_str):
    """Parse wind gust speed"""
    if oc1_str and len(oc1_str) >= 8:
        try:
            gust = float(oc1_str[3:7])
            return gust / 10.0 if gust != 9999 else None
        except:
            return None
    return None

parse_gust_udf = udf(parse_wind_gust, FloatType())
df = df.withColumn('wind_gust', parse_gust_udf(col('OC1')))

print("✓ Wind gust feature created")

# ==================== GEOGRAPHIC & TEMPORAL (Same as before) ====================

print("\n" + "=" * 80)
print("PHASE 5: Geographic and Temporal Features")
print("=" * 80)

# Parse coordinates
df = df.withColumn('latitude', col('LATITUDE').cast(FloatType())) \
       .withColumn('longitude', col('LONGITUDE').cast(FloatType())) \
       .withColumn('elevation', col('ELEVATION').cast(FloatType()))

# Parse datetime
df = df.withColumn('datetime', to_timestamp(col('DATE'), 'yyyy-MM-dd\'T\'HH:mm:ss'))
df = df.withColumn('hour', hour(col('datetime'))) \
       .withColumn('month', month(col('datetime'))) \
       .withColumn('year', year(col('datetime'))) \
       .withColumn('day_of_year', dayofyear(col('datetime')))

# Cyclical encoding
df = df.withColumn('hour_sin', sin(2 * math.pi * col('hour') / 24)) \
       .withColumn('hour_cos', cos(2 * math.pi * col('hour') / 24)) \
       .withColumn('month_sin', sin(2 * math.pi * col('month') / 12)) \
       .withColumn('month_cos', cos(2 * math.pi * col('month') / 12)) \
       .withColumn('day_of_year_sin', sin(2 * math.pi * col('day_of_year') / 365)) \
       .withColumn('day_of_year_cos', cos(2 * math.pi * col('day_of_year') / 365))

# Wind direction cyclical
df = df.withColumn('wind_dir_sin', 
    when(col('wind_direction').isNotNull(), 
         sin(radians(col('wind_direction')))).otherwise(None))
df = df.withColumn('wind_dir_cos', 
    when(col('wind_direction').isNotNull(), 
         cos(radians(col('wind_direction')))).otherwise(None))

print("✓ Geographic features: latitude, longitude, elevation")
print("✓ Temporal features: hour, month, day_of_year (all cyclically encoded)")
print("✓ Wind direction: cyclically encoded")

# ==================== QUALITY FILTERS ====================

print("\n" + "=" * 80)
print("PHASE 6: Quality Filtering")
print("=" * 80)

# Remove rows with missing target
df_filtered = df.filter(col('temperature').isNotNull())
print(f"After removing null temperature: {df_filtered.count():,}")

# Temperature range check
df_filtered = df_filtered.filter(
    (col('temperature') >= -90) & (col('temperature') <= 60)
)
print(f"After temperature range filter (-90 to 60°C): {df_filtered.count():,}")

# Dew point <= temperature
df_filtered = df_filtered.filter(
    col('dew_point').isNull() | (col('dew_point') <= col('temperature'))
)
print(f"After dew point validation: {df_filtered.count():,}")

# Pressure range
df_filtered = df_filtered.filter(
    col('sea_level_pressure').isNull() | 
    ((col('sea_level_pressure') >= 950) & (col('sea_level_pressure') <= 1050))
)
print(f"After pressure range filter: {df_filtered.count():,}")

# ==================== STATION-BASED FEATURES ====================

print("\n" + "=" * 80)
print("PHASE 7: Station-Based Features")
print("=" * 80)

# Calculate station statistics (historical averages)
station_stats = df_filtered.groupBy('STATION').agg(
    spark_avg('temperature').alias('station_avg_temp'),
    spark_avg('dew_point').alias('station_avg_dew'),
    spark_avg('sea_level_pressure').alias('station_avg_pressure')
)

df_filtered = df_filtered.join(station_stats, on='STATION', how='left')

print("✓ Station-based features created")
print("  - station_avg_temp, station_avg_dew, station_avg_pressure")

# ==================== TEMPORAL LAG FEATURES ====================

print("\n" + "=" * 80)
print("PHASE 8: Temporal Lag Features (CAREFUL - No Data Leakage!)")
print("=" * 80)

# Sort by station and time
df_sorted = df_filtered.orderBy('STATION', 'datetime')

# Define window for lag features (by station, ordered by time)
window_spec = Window.partitionBy('STATION').orderBy('datetime')

# Create lag features (PAST data only!)
df_sorted = df_sorted.withColumn('temp_lag_1h', lag('temperature', 1).over(window_spec))
df_sorted = df_sorted.withColumn('temp_lag_2h', lag('temperature', 2).over(window_spec))
df_sorted = df_sorted.withColumn('temp_lag_3h', lag('temperature', 3).over(window_spec))

# Pressure and dew point lags
df_sorted = df_sorted.withColumn('pressure_lag_1h', lag('sea_level_pressure', 1).over(window_spec))
df_sorted = df_sorted.withColumn('dew_lag_1h', lag('dew_point', 1).over(window_spec))

# Rolling averages (past 3 hours)
window_3h = Window.partitionBy('STATION').orderBy('datetime').rowsBetween(-3, -1)
df_sorted = df_sorted.withColumn('temp_rolling_3h', spark_avg('temperature').over(window_3h))
df_sorted = df_sorted.withColumn('pressure_rolling_3h', spark_avg('sea_level_pressure').over(window_3h))

# Temperature change (rate of change)
df_sorted = df_sorted.withColumn('temp_change_1h',
    when(col('temp_lag_1h').isNotNull(),
         col('temperature') - col('temp_lag_1h')).otherwise(None))

df_sorted = df_sorted.withColumn('pressure_change_1h',
    when(col('pressure_lag_1h').isNotNull(),
         col('sea_level_pressure') - col('pressure_lag_1h')).otherwise(None))

print("✓ Temporal lag features created (PAST data only)")
print("  - temp_lag_1h, temp_lag_2h, temp_lag_3h")
print("  - pressure_lag_1h, dew_lag_1h")
print("  - temp_rolling_3h, pressure_rolling_3h")
print("  - temp_change_1h, pressure_change_1h")
print("  WARNING: Train/test split MUST be by time to avoid data leakage!")

# ==================== FINAL FEATURE SELECTION ====================

print("\n" + "=" * 80)
print("PHASE 9: Final Feature Selection")
print("=" * 80)

selected_features = [
    # Geographic (3)
    'latitude', 'longitude', 'elevation',
    
    # Basic Weather (7)
    'dew_point', 'sea_level_pressure', 'visibility',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos', 'precipitation',
    
    # NEW: Cloud Cover (5)
    'lowest_cloud_coverage', 'lowest_cloud_height', 'total_cloud_cover',
    'cloud_layer1_coverage', 'cloud_layer1_height',
    
    # NEW: Weather Conditions (4)
    'is_raining', 'is_snowing', 'is_foggy', 'is_thunderstorm',
    
    # NEW: Wind Gust (1)
    'wind_gust',
    
    # Temporal Basic (6)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'day_of_year_sin', 'day_of_year_cos',
    
    # NEW: Station Features (3)
    'station_avg_temp', 'station_avg_dew', 'station_avg_pressure',
    
    # NEW: Lag Features (8)
    'temp_lag_1h', 'temp_lag_2h', 'temp_lag_3h',
    'pressure_lag_1h', 'dew_lag_1h',
    'temp_rolling_3h', 'pressure_rolling_3h',
    'temp_change_1h',
    
    # Target
    'temperature',
    
    # Metadata
    'STATION', 'datetime', 'year', 'month'
]

df_final = df_sorted.select(*selected_features)

print(f"\nTotal features: {len(selected_features) - 5} (excluding target and metadata)")
print("\nFeature breakdown:")
print("  Geographic: 3")
print("  Basic Weather: 7")
print("  Cloud Cover: 5 (NEW!)")
print("  Weather Conditions: 4 (NEW!)")
print("  Wind Gust: 1 (NEW!)")
print("  Temporal: 6")
print("  Station Features: 3 (NEW!)")
print("  Lag Features: 8 (NEW!)")
print("  ─────────────────")
print("  TOTAL: 37 features (vs 14 before)")

# ==================== SAVE ====================

print("\n" + "=" * 80)
print("PHASE 10: Saving Cleaned Data")
print("=" * 80)

final_count = df_final.count()
retention_rate = (final_count / original_count) * 100

print(f"\nOriginal rows: {original_count:,}")
print(f"Final rows: {final_count:,}")
print(f"Retention rate: {retention_rate:.2f}%")

print(f"\nSaving to: {OUTPUT_PATH}")
df_final.write.mode('overwrite').parquet(OUTPUT_PATH)

# Get file size
file_info = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
    spark._jsc.hadoopConfiguration()
).getContentSummary(
    spark._jvm.org.apache.hadoop.fs.Path(OUTPUT_PATH)
)
size_mb = file_info.getLength() / (1024 * 1024)

print(f"Output size: {size_mb:.1f} MB")
print(f"Compression: Parquet (snappy)")

print("\n" + "=" * 80)
print("✓ ENHANCED CLEANUP COMPLETED!")
print("=" * 80)
print(f"\nFeature summary:")
print(f"  - 37 total features (163% increase from 14)")
print(f"  - Cloud cover: 5 features (CRITICAL for temperature)")
print(f"  - Weather conditions: 4 features (rain, snow, fog, storms)")
print(f"  - Temporal lags: 8 features (time-series patterns)")
print(f"  - Station features: 3 features (local climate)")
print(f"\nExpected RMSE improvement: 1.5-2.5°C (from 4.65°C to 2.2-3.2°C)")
print(f"\nNext steps:")
print(f"  1. TIME-BASED train/test split (CRITICAL to avoid leakage!)")
print(f"  2. Baseline model training")
print(f"  3. RF and XGBoost training")

spark.stop()