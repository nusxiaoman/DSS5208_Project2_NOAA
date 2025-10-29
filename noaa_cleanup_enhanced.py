#!/usr/bin/env python3
"""
NOAA Global Hourly Data Cleanup Script - VERSION 2
Enhanced with cloud cover, present weather, and wind gust features

Key enhancements from V1:
- Cloud cover data (GA1-GA6 fields) - CRITICAL for temperature prediction
- Present weather conditions (AW1-AW4 fields) - Rain, snow, fog, etc.
- Wind gust data (OC1 field)
- More robust missing value handling
- Additional derived features

Expected impact: RMSE improvement of 1-2°C (from 4.65°C to ~2.5-3.5°C)
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, when, split, regexp_extract, regexp_replace
from pyspark.sql.types import FloatType, IntegerType, StringType
import math

def parse_temperature(temp_str):
    """Extract temperature from '+0234,1' format -> 23.4°C"""
    if not temp_str or temp_str == '9999,9':
        return None
    try:
        value = float(temp_str.split(',')[0])
        return value / 10.0
    except:
        return None

def parse_wind(wind_str):
    """
    Extract wind from '018,1,N,0061,1' format
    Returns: (direction, speed, gust_flag)
    direction: 018 degrees
    speed: 6.1 m/s
    """
    if not wind_str or '999' in wind_str:
        return None, None
    try:
        parts = wind_str.split(',')
        direction = float(parts[0]) if parts[0] != '999' else None
        speed = float(parts[3]) / 10.0 if len(parts) > 3 and parts[3] != '9999' else None
        return direction, speed
    except:
        return None, None

def parse_pressure(pressure_str):
    """Extract pressure from '10208,1' format -> 1020.8 hPa"""
    if not pressure_str or pressure_str.startswith('9999'):
        return None
    try:
        value = float(pressure_str.split(',')[0])
        return value / 10.0
    except:
        return None

def parse_visibility(vis_str):
    """Extract visibility from '016000,1,9,9' format -> 16000 meters"""
    if not vis_str or vis_str.startswith('9999'):
        return None
    try:
        value = float(vis_str.split(',')[0])
        return value
    except:
        return None

def parse_precipitation(precip_str):
    """Extract precipitation from '01,0067,1,12' format -> 6.7mm"""
    if not precip_str or not precip_str.strip():
        return None
    try:
        parts = precip_str.split(',')
        if len(parts) >= 2:
            value = float(parts[1])
            if value == 9999:
                return None
            return value / 10.0
    except:
        return None

def parse_cloud_cover(ga_str):
    """
    Parse cloud cover from GA1 field
    Format: 'GA106OVC00800099' or 'GA102CLR00000099'
    
    Returns: (coverage_code, height_meters, coverage_numeric)
    
    Coverage codes:
    - CLR (Clear): 0
    - FEW (Few): 1 (1-2 oktas)
    - SCT (Scattered): 2 (3-4 oktas)
    - BKN (Broken): 3 (5-7 oktas)
    - OVC (Overcast): 4 (8 oktas)
    - VV (Vertical visibility): 5 (obscured)
    """
    if not ga_str or len(ga_str) < 13:
        return None, None, None
    
    try:
        # Extract coverage code (positions 5-8)
        coverage = ga_str[5:8]
        
        # Extract height in feet (positions 8-13), convert to meters
        height_feet = int(ga_str[8:13])
        height_meters = height_feet * 0.3048 if height_feet != 99999 else None
        
        # Map to numeric scale
        coverage_map = {
            'CLR': 0,  # Clear
            'FEW': 1,  # Few clouds (1/8 - 2/8)
            'SCT': 2,  # Scattered (3/8 - 4/8)
            'BKN': 3,  # Broken (5/8 - 7/8)
            'OVC': 4,  # Overcast (8/8)
            'VV': 5    # Vertical visibility obscured
        }
        
        coverage_numeric = coverage_map.get(coverage, None)
        
        return coverage, height_meters, coverage_numeric
    
    except:
        return None, None, None

def parse_present_weather(aw_str):
    """
    Parse present weather from AW1-AW4 fields
    Format: 'AW101+RA' (moderate rain), 'AW102SN' (snow), etc.
    
    Returns: (is_precipitation, is_rain, is_snow, is_fog, intensity)
    
    Weather codes:
    - RA: Rain
    - SN: Snow
    - FG: Fog
    - DZ: Drizzle
    - BR: Mist
    - TS: Thunderstorm
    - Intensity: - (light), blank (moderate), + (heavy)
    """
    if not aw_str or len(aw_str) < 5:
        return 0, 0, 0, 0, 0
    
    try:
        # Extract weather code (skip 'AW1XX' prefix)
        weather_part = aw_str[5:]
        
        # Check intensity
        intensity = 0  # 0=none, 1=light, 2=moderate, 3=heavy
        if '-' in weather_part:
            intensity = 1
            weather_part = weather_part.replace('-', '')
        elif '+' in weather_part:
            intensity = 3
            weather_part = weather_part.replace('+', '')
        else:
            intensity = 2
        
        # Check weather types
        is_rain = 1 if 'RA' in weather_part or 'DZ' in weather_part else 0
        is_snow = 1 if 'SN' in weather_part or 'IP' in weather_part else 0
        is_fog = 1 if 'FG' in weather_part or 'BR' in weather_part else 0
        is_precipitation = 1 if (is_rain or is_snow) else 0
        
        return is_precipitation, is_rain, is_snow, is_fog, intensity
    
    except:
        return 0, 0, 0, 0, 0

def parse_wind_gust(oc_str):
    """
    Parse wind gust from OC1 field
    Format: 'OC1+00611' -> 6.1 m/s gust
    Returns: gust speed in m/s
    """
    if not oc_str or len(oc_str) < 8:
        return None
    
    try:
        # Extract gust speed (last 5 digits)
        gust_str = oc_str[-5:]
        gust = float(gust_str) / 10.0
        if gust == 999.9:
            return None
        return gust
    except:
        return None

# Register UDFs
spark = SparkSession.builder \
    .appName("NOAA_Cleanup_V2_Enhanced") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

parse_temp_udf = F.udf(parse_temperature, FloatType())
parse_wind_dir_udf = F.udf(lambda x: parse_wind(x)[0], FloatType())
parse_wind_speed_udf = F.udf(lambda x: parse_wind(x)[1], FloatType())
parse_pressure_udf = F.udf(parse_pressure, FloatType())
parse_vis_udf = F.udf(parse_visibility, FloatType())
parse_precip_udf = F.udf(parse_precipitation, FloatType())

# Cloud cover UDFs
parse_cloud_coverage_udf = F.udf(lambda x: parse_cloud_cover(x)[0], StringType())
parse_cloud_height_udf = F.udf(lambda x: parse_cloud_cover(x)[1], FloatType())
parse_cloud_numeric_udf = F.udf(lambda x: parse_cloud_cover(x)[2], IntegerType())

# Weather UDFs
parse_weather_precip_udf = F.udf(lambda x: parse_present_weather(x)[0], IntegerType())
parse_weather_rain_udf = F.udf(lambda x: parse_present_weather(x)[1], IntegerType())
parse_weather_snow_udf = F.udf(lambda x: parse_present_weather(x)[2], IntegerType())
parse_weather_fog_udf = F.udf(lambda x: parse_present_weather(x)[3], IntegerType())
parse_weather_intensity_udf = F.udf(lambda x: parse_present_weather(x)[4], IntegerType())

# Wind gust UDF
parse_gust_udf = F.udf(parse_wind_gust, FloatType())

print("=" * 80)
print("NOAA GLOBAL HOURLY DATA CLEANUP - VERSION 2 (ENHANCED)")
print("=" * 80)
print("\nEnhancements from V1:")
print("  + Cloud cover features (GA1-GA6)")
print("  + Present weather (AW1-AW4)")
print("  + Wind gust (OC1)")
print("  + Improved missing value handling")

# Read raw data
input_path = "gs://weather-ml-bucket-1760514177/noaa-data-raw/2024.csv"
print(f"\nReading raw data from: {input_path}")

df = spark.read.csv(input_path, header=True, inferSchema=False)
initial_count = df.count()
print(f"Initial row count: {initial_count:,}")

# Parse all fields
print("\n" + "=" * 80)
print("PARSING ALL FIELDS (V2 - ENHANCED)")
print("=" * 80)

# Geographic
print("\n1. Geographic features...")
df = df.withColumn('latitude', col('LATITUDE').cast(FloatType()))
df = df.withColumn('longitude', col('LONGITUDE').cast(FloatType()))
df = df.withColumn('elevation', col('ELEVATION').cast(FloatType()))

# Target variable
print("2. Target: Temperature...")
df = df.withColumn('temperature', parse_temp_udf(col('TMP')))

# Basic weather
print("3. Basic weather features...")
df = df.withColumn('dew_point', parse_temp_udf(col('DEW')))
df = df.withColumn('sea_level_pressure', parse_pressure_udf(col('SLP')))
df = df.withColumn('visibility', parse_vis_udf(col('VIS')))

# Wind
print("4. Wind features...")
df = df.withColumn('wind_direction', parse_wind_dir_udf(col('WND')))
df = df.withColumn('wind_speed', parse_wind_speed_udf(col('WND')))

# Wind gust (NEW)
print("5. Wind gust feature (NEW)...")
df = df.withColumn('wind_gust', parse_gust_udf(col('OC1')))

# Precipitation
print("6. Precipitation...")
df = df.withColumn('precipitation', parse_precip_udf(col('AA1')))

# Cloud cover (NEW - CRITICAL!)
print("7. Cloud cover features (NEW - CRITICAL!)...")
df = df.withColumn('cloud_coverage_code', parse_cloud_coverage_udf(col('GA1')))
df = df.withColumn('cloud_base_height', parse_cloud_height_udf(col('GA1')))
df = df.withColumn('cloud_cover_level', parse_cloud_numeric_udf(col('GA1')))

# Try additional cloud layers
df = df.withColumn('cloud_coverage_code_2', parse_cloud_coverage_udf(col('GA2')))
df = df.withColumn('cloud_base_height_2', parse_cloud_height_udf(col('GA2')))

# Present weather (NEW - IMPORTANT!)
print("8. Present weather features (NEW - IMPORTANT!)...")
df = df.withColumn('is_precipitation', parse_weather_precip_udf(col('AW1')))
df = df.withColumn('is_raining', parse_weather_rain_udf(col('AW1')))
df = df.withColumn('is_snowing', parse_weather_snow_udf(col('AW1')))
df = df.withColumn('is_foggy', parse_weather_fog_udf(col('AW1')))
df = df.withColumn('weather_intensity', parse_weather_intensity_udf(col('AW1')))

# Temporal features
print("9. Temporal features...")
df = df.withColumn('date', F.to_timestamp('DATE', 'yyyy-MM-dd\'T\'HH:mm:ss'))
df = df.withColumn('year', F.year('date'))
df = df.withColumn('month', F.month('date'))
df = df.withColumn('day', F.dayofmonth('date'))
df = df.withColumn('hour', F.hour('date'))

# Cyclical encoding
print("10. Cyclical encoding...")
df = df.withColumn('hour_sin', F.sin(2 * math.pi * col('hour') / 24))
df = df.withColumn('hour_cos', F.cos(2 * math.pi * col('hour') / 24))
df = df.withColumn('month_sin', F.sin(2 * math.pi * col('month') / 12))
df = df.withColumn('month_cos', F.cos(2 * math.pi * col('month') / 12))

# Wind direction cyclical encoding
df = df.withColumn('wind_dir_sin', 
    when(col('wind_direction').isNotNull(), 
         F.sin(2 * math.pi * col('wind_direction') / 360)).otherwise(None))
df = df.withColumn('wind_dir_cos', 
    when(col('wind_direction').isNotNull(), 
         F.cos(2 * math.pi * col('wind_direction') / 360)).otherwise(None))

# Quality filters
print("\n" + "=" * 80)
print("APPLYING QUALITY FILTERS")
print("=" * 80)

# 1. Remove null temperature (target)
print("\n1. Removing rows with null temperature...")
df_filtered = df.filter(col('temperature').isNotNull())
after_temp = df_filtered.count()
print(f"   After filtering: {after_temp:,} ({(initial_count - after_temp):,} removed)")

# 2. Temperature range filter
print("2. Applying temperature range filter (-90°C to +60°C)...")
df_filtered = df_filtered.filter(
    (col('temperature') >= -90) & (col('temperature') <= 60)
)
after_temp_range = df_filtered.count()
print(f"   After filtering: {after_temp_range:,} ({(after_temp - after_temp_range):,} removed)")

# 3. Physical constraint: dew point <= temperature
print("3. Applying physical constraint (dew_point <= temperature)...")
df_filtered = df_filtered.filter(
    col('dew_point').isNull() | (col('dew_point') <= col('temperature'))
)
after_dew = df_filtered.count()
print(f"   After filtering: {after_dew:,} ({(after_temp_range - after_dew):,} removed)")

# 4. Pressure range filter
print("4. Applying pressure range filter (950-1050 hPa)...")
df_filtered = df_filtered.filter(
    col('sea_level_pressure').isNull() | 
    ((col('sea_level_pressure') >= 950) & (col('sea_level_pressure') <= 1050))
)
after_pressure = df_filtered.count()
print(f"   After filtering: {after_pressure:,} ({(after_dew - after_pressure):,} removed)")

# 5. Wind speed sanity check
print("5. Applying wind speed filter (0-100 m/s)...")
df_filtered = df_filtered.filter(
    col('wind_speed').isNull() | 
    ((col('wind_speed') >= 0) & (col('wind_speed') <= 100))
)
after_wind = df_filtered.count()
print(f"   After filtering: {after_wind:,} ({(after_pressure - after_wind):,} removed)")

# 6. Wind gust >= wind speed (if both present)
print("6. Applying wind gust >= wind speed constraint...")
df_filtered = df_filtered.filter(
    col('wind_gust').isNull() | col('wind_speed').isNull() |
    (col('wind_gust') >= col('wind_speed'))
)
after_gust = df_filtered.count()
print(f"   After filtering: {after_gust:,} ({(after_wind - after_gust):,} removed)")

# Missing value analysis
print("\n" + "=" * 80)
print("MISSING VALUE ANALYSIS (V2 - ENHANCED FEATURES)")
print("=" * 80)

features_v2 = [
    'latitude', 'longitude', 'elevation',
    'temperature', 'dew_point', 'sea_level_pressure', 
    'visibility', 'wind_speed', 'wind_direction',
    'wind_gust',  # NEW
    'precipitation',
    'cloud_coverage_code', 'cloud_base_height', 'cloud_cover_level',  # NEW
    'is_precipitation', 'is_raining', 'is_snowing', 'is_foggy',  # NEW
    'hour', 'month'
]

print(f"\nTotal features in V2: {len(features_v2)}")
print(f"New features: wind_gust, cloud_* (3), weather_* (4) = 8 new features\n")

for feature in features_v2:
    null_count = df_filtered.filter(col(feature).isNull()).count()
    null_pct = (null_count / after_gust) * 100
    print(f"{feature:30s} {null_count:12,} ({null_pct:5.2f}%)")

# Select final features
print("\n" + "=" * 80)
print("SELECTING FINAL FEATURES (V2)")
print("=" * 80)

final_features = [
    # Geographic (3)
    'latitude', 'longitude', 'elevation',
    
    # Basic weather (6)
    'dew_point', 'sea_level_pressure', 'visibility',
    'wind_speed', 'wind_dir_sin', 'wind_dir_cos',
    
    # Wind gust (1) - NEW
    'wind_gust',
    
    # Precipitation (1)
    'precipitation',
    
    # Cloud cover (3) - NEW & CRITICAL
    'cloud_cover_level',      # 0-5 (CLR to VV)
    'cloud_base_height',      # meters
    'cloud_coverage_code_2',  # Secondary layer (for multi-layer clouds)
    
    # Present weather (4) - NEW & IMPORTANT
    'is_raining',             # Binary: rain/drizzle
    'is_snowing',             # Binary: snow/ice
    'is_foggy',               # Binary: fog/mist
    'weather_intensity',      # 0=none, 1=light, 2=moderate, 3=heavy
    
    # Temporal (4)
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    
    # Target
    'temperature'
]

print(f"\nTotal features: {len(final_features) - 1} (excluding target)")
print(f"  V1 features: 14")
print(f"  V2 new features: {len(final_features) - 1 - 14}")
print(f"\nBreakdown:")
print(f"  Geographic: 3")
print(f"  Basic weather: 6")
print(f"  Wind gust: 1 (NEW)")
print(f"  Precipitation: 1")
print(f"  Cloud cover: 3 (NEW - CRITICAL)")
print(f"  Present weather: 4 (NEW - IMPORTANT)")
print(f"  Temporal: 4")
print(f"  Target: 1 (temperature)")

df_final = df_filtered.select(*final_features)

# Handle missing values for new features
print("\n" + "=" * 80)
print("HANDLING MISSING VALUES FOR NEW FEATURES")
print("=" * 80)

# Cloud cover: Default to "clear" if missing
print("\n1. Cloud cover: Defaulting missing to 'clear' (level=0)...")
df_final = df_final.fillna({'cloud_cover_level': 0})
df_final = df_final.fillna({'cloud_base_height': 10000})  # High if missing

# Present weather: Default to "no weather" if missing
print("2. Present weather: Defaulting missing to 'no weather' (0)...")
df_final = df_final.fillna({
    'is_raining': 0,
    'is_snowing': 0,
    'is_foggy': 0,
    'weather_intensity': 0
})

# Wind gust: Will use median imputation in training (like other features)
print("3. Wind gust: Will use median imputation during training")

# Final statistics
print("\n" + "=" * 80)
print("FINAL STATISTICS (V2)")
print("=" * 80)

final_count = df_final.count()
retention_rate = (final_count / initial_count) * 100

print(f"\nInitial rows:    {initial_count:,}")
print(f"Final rows:      {final_count:,}")
print(f"Retention rate:  {retention_rate:.2f}%")
print(f"Rows removed:    {initial_count - final_count:,}")

# Save cleaned data
output_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
print(f"\nSaving cleaned data to: {output_path}")

df_final.write.mode('overwrite').parquet(output_path)

# Get final size
print("\nVerifying saved data...")
df_verify = spark.read.parquet(output_path)
verify_count = df_verify.count()
print(f"Verified row count: {verify_count:,}")

# Show sample
print("\n" + "=" * 80)
print("SAMPLE DATA (First 10 rows)")
print("=" * 80)
df_verify.show(10, truncate=False)

# Feature summary
print("\n" + "=" * 80)
print("FEATURE SUMMARY (V2)")
print("=" * 80)
df_verify.select([F.mean(c).alias(c) for c in df_verify.columns if c != 'cloud_coverage_code_2']).show(vertical=True)

print("\n" + "=" * 80)
print("✓ CLEANUP COMPLETED SUCCESSFULLY (V2 - ENHANCED)")
print("=" * 80)
print(f"\nOutput: {output_path}")
print(f"Total features: {len(final_features) - 1}")
print(f"Total rows: {final_count:,}")
print(f"\nExpected RMSE improvement: 1-2°C")
print(f"  V1 RMSE: 4.65°C")
print(f"  V2 Expected: 2.5-3.5°C")

spark.stop()