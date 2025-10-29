"""
NOAA Weather Data Train/Test Split V2
Splits the V2 cleaned dataset (35 features) into 70% train / 30% test
Random split with fixed seed for reproducibility
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

def main():
    spark = SparkSession.builder \
        .appName("NOAA Train/Test Split V2") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()
    
    # Paths
    input_path = "gs://weather-ml-bucket-1760514177/warehouse/noaa_cleaned_v2"
    train_output = "gs://weather-ml-bucket-1760514177/warehouse/noaa_train_v2"
    test_output = "gs://weather-ml-bucket-1760514177/warehouse/noaa_test_v2"
    
    print("=" * 80)
    print("NOAA TRAIN/TEST SPLIT V2")
    print("=" * 80)
    print(f"Input:  {input_path}")
    print(f"Train:  {train_output}")
    print(f"Test:   {test_output}")
    print(f"Split:  70% train / 30% test")
    print(f"Method: Random split (seed=42)")
    print("=" * 80)
    
    # Read cleaned V2 data
    print("\nReading cleaned V2 data...")
    df = spark.read.parquet(input_path)
    
    total_rows = df.count()
    print(f"Total rows: {total_rows:,}")
    
    # Show schema summary
    print(f"\nFeatures: {len(df.columns)}")
    print("Sample columns:")
    for col_name in df.columns[:10]:
        print(f"  - {col_name}")
    print("  ... (and more)")
    
    # Random 70/30 split with fixed seed
    print("\n" + "=" * 80)
    print("SPLITTING DATA")
    print("=" * 80)
    
    train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
    
    # Cache for counting
    train_df.cache()
    test_df.cache()
    
    train_count = train_df.count()
    test_count = test_df.count()
    
    print(f"\nTrain set: {train_count:,} rows ({100*train_count/total_rows:.1f}%)")
    print(f"Test set:  {test_count:,} rows ({100*test_count/total_rows:.1f}%)")
    
    # Verify split
    assert train_count + test_count == total_rows, "Split count mismatch!"
    print("✓ Split verified")
    
    # Check for null values in target
    train_null_temp = train_df.filter(col('temperature').isNull()).count()
    test_null_temp = test_df.filter(col('temperature').isNull()).count()
    
    print(f"\nNull temperatures:")
    print(f"  Train: {train_null_temp}")
    print(f"  Test:  {test_null_temp}")
    
    if train_null_temp > 0 or test_null_temp > 0:
        print("  ⚠ Warning: Found null temperatures (should be 0 after cleanup)")
    
    # Show temperature statistics
    print("\n" + "=" * 80)
    print("TEMPERATURE STATISTICS (Target Variable)")
    print("=" * 80)
    
    train_stats = train_df.select('temperature').summary('min', 'max', 'mean', 'stddev')
    test_stats = test_df.select('temperature').summary('min', 'max', 'mean', 'stddev')
    
    print("\nTrain set:")
    train_stats.show()
    
    print("Test set:")
    test_stats.show()
    
    # Save splits
    print("\n" + "=" * 80)
    print("SAVING SPLITS")
    print("=" * 80)
    
    print(f"\nSaving train set to: {train_output}")
    train_df.write.mode('overwrite').partitionBy('year', 'month').parquet(train_output)
    print("✓ Train set saved")
    
    print(f"\nSaving test set to: {test_output}")
    test_df.write.mode('overwrite').partitionBy('year', 'month').parquet(test_output)
    print("✓ Test set saved")
    
    # Cleanup
    train_df.unpersist()
    test_df.unpersist()
    
    print("\n" + "=" * 80)
    print("✓ SPLIT COMPLETED!")
    print("=" * 80)
    print(f"Train: {train_count:,} rows → {train_output}")
    print(f"Test:  {test_count:,} rows → {test_output}")
    print("\nNext: Train models using these datasets")
    print("=" * 80)
    
    spark.stop()

if __name__ == "__main__":
    main()