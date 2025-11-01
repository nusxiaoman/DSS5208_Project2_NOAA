from pyspark.sql import SparkSession

# Initialize Spark
spark = SparkSession.builder \
    .appName("Inspect NOAA Data V2") \
    .getOrCreate()

# Read one month partition
df = spark.read.parquet("gs://weather-ml-bucket-1760514177/warehouse/noaa_parquet/year=2024/month=1/")

print("=" * 80)
print("TOTAL ROWS:")
row_count = df.count()
print(f"{row_count:,}")

print("\n" + "=" * 80)
print("ALL COLUMNS:")
print(df.columns)

print("\n" + "=" * 80)
print("SCHEMA:")
df.printSchema()

print("\n" + "=" * 80)
print("DATA TYPES:")
for col_name, col_type in df.dtypes:
    print(f"{col_name}: {col_type}")

print("\n" + "=" * 80)
print("SAMPLE DATA (5 rows, first set of columns):")
# Get first 10 columns or all if less than 10
cols_to_show = df.columns[:min(10, len(df.columns))]
df.select(cols_to_show).show(5, truncate=False)

if len(df.columns) > 10:
    print("\n" + "=" * 80)
    print("SAMPLE DATA (5 rows, next set of columns):")
    cols_to_show = df.columns[10:min(20, len(df.columns))]
    df.select(cols_to_show).show(5, truncate=False)

if len(df.columns) > 20:
    print("\n" + "=" * 80)
    print("SAMPLE DATA (5 rows, remaining columns):")
    cols_to_show = df.columns[20:]
    df.select(cols_to_show).show(5, truncate=False)

print("\n" + "=" * 80)
print("DETAILED VIEW (2 rows - vertical):")
df.show(2, truncate=False, vertical=True)

print("\n" + "=" * 80)
print("CHECK FOR NULL VALUES:")
from pyspark.sql.functions import col, count, when, isnan
null_counts = df.select([
    count(when(col(c).isNull() | isnan(col(c)), c)).alias(c) 
    if df.schema[c].dataType.typeName() in ['double', 'float']
    else count(when(col(c).isNull(), c)).alias(c)
    for c in df.columns
])
print("Null/NaN counts per column:")
for row in null_counts.collect():
    for col_name in df.columns:
        null_count = row[col_name]
        if null_count > 0:
            percentage = (null_count / row_count) * 100
            print(f"  {col_name}: {null_count:,} ({percentage:.2f}%)")

print("\n" + "=" * 80)
print("SUMMARY STATISTICS FOR NUMERIC COLUMNS:")
df.describe().show()

spark.stop()