"""
Train/Test Split Script for NOAA Weather Data
Splits cleaned data into 70% training and 30% test sets
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, rand
import sys

def main():
    # Parse command line arguments
    if len(sys.argv) < 4:
        print("Usage: train_test_split.py <input_path> <train_output> <test_output>")
        print("Example: train_test_split.py gs://bucket/warehouse/noaa_clean_std gs://bucket/warehouse/noaa_train gs://bucket/warehouse/noaa_test")
        sys.exit(1)
    
    input_path = sys.argv[1]
    train_output = sys.argv[2]
    test_output = sys.argv[3]
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Train/Test Split") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Data - Train/Test Split")
    print("=" * 80)
    print(f"Input:        {input_path}")
    print(f"Train output: {train_output}")
    print(f"Test output:  {test_output}")
    
    # Read cleaned data
    print(f"\nReading cleaned data...")
    df = spark.read.parquet(input_path)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    # Show schema
    print("\nSchema:")
    df.printSchema()
    
    # Split data: 70% train, 30% test (with fixed seed for reproducibility)
    print("\nSplitting data...")
    print("Train: 70%")
    print("Test:  30%")
    
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    
    # Cache for counting
    train_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"\nTrain set: {train_count:,} rows ({100*train_count/total_rows:.2f}%)")
    print(f"Test set:  {test_count:,} rows ({100*test_count/total_rows:.2f}%)")
    
    # Show sample from each set
    print("\n" + "=" * 80)
    print("TRAIN SET SAMPLE (5 rows)")
    print("=" * 80)
    train_df.select(
        'temperature', 'dew_point', 'sea_level_pressure',
        'wind_speed', 'latitude', 'longitude', 'hour', 'month'
    ).show(5, truncate=False)
    
    print("\n" + "=" * 80)
    print("TEST SET SAMPLE (5 rows)")
    print("=" * 80)
    test_df.select(
        'temperature', 'dew_point', 'sea_level_pressure',
        'wind_speed', 'latitude', 'longitude', 'hour', 'month'
    ).show(5, truncate=False)
    
    # Show statistics for train set
    print("\n" + "=" * 80)
    print("TRAIN SET - SUMMARY STATISTICS")
    print("=" * 80)
    train_df.select(
        'temperature', 'dew_point', 'sea_level_pressure', 'wind_speed'
    ).describe().show()
    
    # Show statistics for test set
    print("\n" + "=" * 80)
    print("TEST SET - SUMMARY STATISTICS")
    print("=" * 80)
    test_df.select(
        'temperature', 'dew_point', 'sea_level_pressure', 'wind_speed'
    ).describe().show()
    
    # Check distribution by month in both sets
    print("\n" + "=" * 80)
    print("TRAIN SET - DISTRIBUTION BY MONTH")
    print("=" * 80)
    train_df.groupBy('month').count().orderBy('month').show()
    
    print("\n" + "=" * 80)
    print("TEST SET - DISTRIBUTION BY MONTH")
    print("=" * 80)
    test_df.groupBy('month').count().orderBy('month').show()
    
    # Save train set
    print(f"\nSaving training set to: {train_output}")
    print("(This may take 10-20 minutes)")
    train_df.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(train_output)
    
    # Save test set
    print(f"\nSaving test set to: {test_output}")
    test_df.write \
        .mode('overwrite') \
        .partitionBy('year', 'month') \
        .parquet(test_output)
    
    print("\n" + "=" * 80)
    print("✓ TRAIN/TEST SPLIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Training set: {train_output}")
    print(f"  Rows: {train_count:,}")
    print(f"\nTest set: {test_output}")
    print(f"  Rows: {test_count:,}")
    print(f"\nSplit ratio: {100*train_count/total_rows:.2f}% / {100*test_count/total_rows:.2f}%")
    print("\nData is ready for model training!")
    
    # Unpersist cache
    train_df.unpersist()
    test_df.unpersist()
    
    spark.stop()

if __name__ == "__main__":
    main()