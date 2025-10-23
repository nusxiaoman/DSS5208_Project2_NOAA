from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("Inspect NOAA Data") \
    .getOrCreate()

# Read one month partition
df = spark.read.parquet("gs://weather-ml-bucket-1760514177/warehouse/noaa_parquet/month=1/")

print("=" * 80)
print("TOTAL ROWS:")
print(df.count())

print("\n" + "=" * 80)
print("SCHEMA:")
df.printSchema()

print("\n" + "=" * 80)
print("ALL COLUMNS:")
print(df.columns)

print("\n" + "=" * 80)
print("SAMPLE DATA - Geographic & Time:")
df.select("STATION", "DATE", "LATITUDE", "LONGITUDE", "ELEVATION").show(5, truncate=False)

print("\n" + "=" * 80)
print("SAMPLE DATA - Temperature & Weather:")
df.select("TMP", "DEW", "SLP", "VIS").show(10, truncate=False)

print("\n" + "=" * 80)
print("SAMPLE DATA - Wind:")
df.select("WND").show(10, truncate=False)

print("\n" + "=" * 80)
print("CHECK FOR PRECIPITATION & CLOUD COLUMNS:")
precip_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['AA', 'GA', 'AJ', 'AL'])]
cloud_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in ['SKC', 'CLD', 'MW', 'MA', 'OC'])]
print(f"Precipitation columns: {precip_cols}")
print(f"Cloud columns: {cloud_cols}")

if precip_cols:
    print("\nSample precipitation data:")
    df.select(precip_cols[:3]).show(5, truncate=False)

if cloud_cols:
    print("\nSample cloud data:")
    df.select(cloud_cols[:3]).show(5, truncate=False)

print("\n" + "=" * 80)
print("DATA TYPES:")
for col_name, col_type in df.dtypes:
    print(f"{col_name}: {col_type}")

print("\n" + "=" * 80)
print("SAMPLE OF FIRST 2 ROWS (VERTICAL VIEW):")
df.show(2, truncate=False, vertical=True)

spark.stop()