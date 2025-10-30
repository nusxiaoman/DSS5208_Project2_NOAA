"""
NOAA Weather Data - Train/Test Split
Random 70/30 split for machine learning model training
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    spark = SparkSession.builder \
        .appName("NOAA Train-Test Split") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths
    input_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
    train_output = "gs://weather-ml-bucket-1760514177/warehouse/noaa_train_v2"
    test_output = "gs://weather-ml-bucket-1760514177/warehouse/noaa_test_v2"
    
    print("=" * 80)
    print("NOAA WEATHER DATA - TRAIN/TEST SPLIT")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Train output (70%): {train_output}")
    print(f"Test output (30%): {test_output}")
    print("=" * 80)
    
    # Read cleaned data
    print("\nReading cleaned data...")
    df = spark.read.parquet(input_path)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    # Show schema
    print("\nSchema:")
    df.printSchema()
    
    # Verify target variable
    print("\nTarget variable check:")
    temp_stats = df.select('temperature').describe()
    temp_stats.show()
    
    # Check for nulls in target
    null_temps = df.filter(col('temperature').isNull()).count()
    if null_temps > 0:
        print(f"WARNING: Found {null_temps} NULL temperatures - filtering out")
        df = df.filter(col('temperature').isNotNull())
        total_rows = df.count()
        print(f"Rows after filter: {total_rows:,}")
    
    # Random split 70/30
    print("\n" + "=" * 80)
    print("SPLITTING DATA (70% train / 30% test)")
    print("=" * 80)
    
    # Set seed for reproducibility
    seed = 42
    
    # Split
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=seed)
    
    # Cache for counting
    train_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"\nSplit results:")
    print(f"  Train: {train_count:,} rows ({train_count/total_rows*100:.1f}%)")
    print(f"  Test:  {test_count:,} rows ({test_count/total_rows*100:.1f}%)")
    print(f"  Total: {total_rows:,} rows")
    
    # Verify temperature distribution in both sets
    print("\n" + "=" * 80)
    print("TEMPERATURE DISTRIBUTION CHECK")
    print("=" * 80)
    
    print("\nTrain set:")
    train_df.select('temperature').describe().show()
    
    print("\nTest set:")
    test_df.select('temperature').describe().show()
    
    # Save train set
    print("\n" + "=" * 80)
    print("SAVING TRAIN SET")
    print("=" * 80)
    print(f"Output: {train_output}")
    train_df.write.mode('overwrite').parquet(train_output)
    print("DONE - Train set saved")
    
    # Save test set
    print("\n" + "=" * 80)
    print("SAVING TEST SET")
    print("=" * 80)
    print(f"Output: {test_output}")
    test_df.write.mode('overwrite').parquet(test_output)
    print("DONE - Test set saved")
    
    # Summary
    print("\n" + "=" * 80)
    print("SPLIT COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Train set: {train_count:,} rows -> {train_output}")
    print(f"Test set:  {test_count:,} rows -> {test_output}")
    print(f"\nNext steps:")
    print(f"  1. Train models on: {train_output}")
    print(f"  2. Evaluate on: {test_output}")
    print(f"  3. Compare RMSE vs V1 baseline")
    print("=" * 80)
    
    # Cleanup
    train_df.unpersist()
    test_df.unpersist()
    spark.stop()

if __name__ == "__main__":
    main()