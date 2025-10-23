"""
Model Comparison Script
Compares all trained models and generates a summary report
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
import sys

def read_metrics_csv(spark, path):
    """Read metrics CSV file"""
    try:
        df = spark.read.csv(path, header=True, inferSchema=True)
        return df
    except:
        print(f"Could not read: {path}")
        return None

def main():
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Model Comparison") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Prediction - Model Comparison Report")
    print("=" * 80)
    
    bucket = "gs://weather-ml-bucket-1760514177/outputs"
    
    # Paths to check
    models = {
        "Baseline (LR)": f"{bucket}/baseline_test/metrics",
        "RF Test": f"{bucket}/rf_test/metrics",
        "RF Full": f"{bucket}/rf_full/metrics",
        "GBT Test": f"{bucket}/gbt_test/metrics",
        "GBT Full": f"{bucket}/gbt_full/metrics"
    }
    
    evaluations = {
        "RF Full": f"{bucket}/rf_full_evaluation/test_metrics",
        "GBT Full": f"{bucket}/gbt_full_evaluation/test_metrics"
    }
    
    # Collect all metrics
    print("\n" + "=" * 80)
    print("TRAINING METRICS SUMMARY")
    print("=" * 80)
    
    all_results = []
    
    for model_name, path in models.items():
        metrics_df = read_metrics_csv(spark, path + "/*.csv")
        if metrics_df:
            row = metrics_df.collect()[0]
            result = {
                "Model": model_name,
                "Train_RMSE": getattr(row, 'train_rmse', None),
                "Train_R2": getattr(row, 'train_r2', None),
                "CV_RMSE": getattr(row, 'best_cv_rmse', None),
                "Train_Rows": getattr(row, 'train_rows', None)
            }
            all_results.append(result)
            
            print(f"\n{model_name}:")
            print(f"  Training RMSE: {result['Train_RMSE']:.4f}°C" if result['Train_RMSE'] else "  N/A")
            print(f"  Training R²: {result['Train_R2']:.4f}" if result['Train_R2'] else "  N/A")
            print(f"  CV RMSE: {result['CV_RMSE']:.4f}°C" if result['CV_RMSE'] else "  N/A")
    
    # Test set evaluation
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION (Final Performance)")
    print("=" * 80)
    
    test_results = []
    
    for model_name, path in evaluations.items():
        test_df = read_metrics_csv(spark, path + "/*.csv")
        if test_df:
            row = test_df.collect()[0]
            result = {
                "Model": model_name,
                "Test_RMSE": row.test_rmse,
                "Test_R2": row.test_r2,
                "Test_MAE": row.test_mae,
                "Test_Rows": row.test_rows
            }
            test_results.append(result)
            
            print(f"\n{model_name}:")
            print(f"  Test RMSE: {result['Test_RMSE']:.4f}°C")
            print(f"  Test R²: {result['Test_R2']:.4f}")
            print(f"  Test MAE: {result['Test_MAE']:.4f}°C")
            print(f"  Test Rows: {result['Test_Rows']:,}")
    
    # Feature importances comparison
    print("\n" + "=" * 80)
    print("TOP 5 FEATURES BY MODEL")
    print("=" * 80)
    
    for model_name in ["rf_full", "gbt_full"]:
        importance_path = f"{bucket}/{model_name}/feature_importances/*.csv"
        try:
            importance_df = spark.read.csv(importance_path, header=True, inferSchema=True)
            top_features = importance_df.orderBy(col('importance').desc()).limit(5)
            
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            for row in top_features.collect():
                print(f"  {row.feature:25s} {row.importance:.4f}")
        except:
            print(f"\n{model_name}: No feature importances found")
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if test_results:
        best_model = min(test_results, key=lambda x: x['Test_RMSE'])
        print(f"\n✓ Best Model: {best_model['Model']}")
        print(f"  Test RMSE: {best_model['Test_RMSE']:.4f}°C")
        print(f"  Test R²: {best_model['Test_R2']:.4f}")
        
        print(f"\n  This model explains {best_model['Test_R2']*100:.2f}% of the variance")
        print(f"  in temperature and has an average error of {best_model['Test_RMSE']:.2f}°C")
    
    print("\n" + "=" * 80)
    
    spark.stop()

if __name__ == "__main__":
    main()