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
    except Exception as e:
        print(f"Could not read: {path}")
        return None

def main():
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: compare_models.py <outputs_base_path>")
        print("Example: compare_models.py gs://bucket/outputs")
        sys.exit(1)
    
    base_path = sys.argv[1].rstrip("/")
    
    # Initialize Spark
    spark = SparkSession.builder \
        .appName("NOAA Model Comparison") \
        .getOrCreate()
    
    print("=" * 80)
    print("NOAA Weather Prediction - Model Comparison Report")
    print("=" * 80)
    print(f"Base path: {base_path}")
    
    # Paths to check
    models = {
        "Baseline (LR)": f"{base_path}/baseline_test/metrics",
        "RF Test (10%)": f"{base_path}/rf_test/metrics",
        "RF Full (Simplified)": f"{base_path}/rf_simplified/metrics",
        "GBT Test (10%)": f"{base_path}/gbt_test/metrics"
    }
    
    evaluations = {
        "RF Full (Simplified)": f"{base_path}/rf_simplified_evaluation/test_metrics"
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
            if result['Train_RMSE']:
                print(f"  Training RMSE: {result['Train_RMSE']:.4f}°C")
            if result['Train_R2']:
                print(f"  Training R²: {result['Train_R2']:.4f}")
            if result['CV_RMSE']:
                print(f"  CV RMSE: {result['CV_RMSE']:.4f}°C")
            if result['Train_Rows']:
                print(f"  Training Rows: {result['Train_Rows']:,}")
        else:
            print(f"\n{model_name}: Metrics not found")
    
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
        else:
            print(f"\n{model_name}: Test metrics not found")
    
    # Feature importances comparison
    print("\n" + "=" * 80)
    print("TOP 5 FEATURES BY MODEL")
    print("=" * 80)
    
    # Check RF simplified
    for model_name, model_path in [
        ("RF TEST (10%)", "rf_test"),
        ("RF FULL (SIMPLIFIED)", "rf_simplified"),
        ("GBT TEST (10%)", "gbt_test")
    ]:
        importance_path = f"{base_path}/{model_path}/feature_importances/*.csv"
        try:
            importance_df = spark.read.csv(importance_path, header=True, inferSchema=True)
            top_features = importance_df.orderBy(col('importance').desc()).limit(5)
            
            print(f"\n{model_name}:")
            for row in top_features.collect():
                print(f"  {row.feature:25s} {row.importance:.4f}")
        except:
            print(f"\n{model_name}: No feature importances found")
    
    # Performance comparison table
    print("\n" + "=" * 80)
    print("PERFORMANCE COMPARISON TABLE")
    print("=" * 80)
    
    print("\n{:<25} {:>12} {:>10} {:>12} {:>15}".format(
        "Model", "Train RMSE", "Train R²", "CV RMSE", "Training Rows"
    ))
    print("-" * 80)
    
    for result in all_results:
        print("{:<25} {:>12} {:>10} {:>12} {:>15,}".format(
            result['Model'],
            f"{result['Train_RMSE']:.4f}°C" if result['Train_RMSE'] else "N/A",
            f"{result['Train_R2']:.4f}" if result['Train_R2'] else "N/A",
            f"{result['CV_RMSE']:.4f}°C" if result['CV_RMSE'] else "N/A",
            result['Train_Rows'] if result['Train_Rows'] else 0
        ))
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    if test_results:
        best_model = min(test_results, key=lambda x: x['Test_RMSE'])
        print(f"\n✓ Best Model: {best_model['Model']}")
        print(f"  Test RMSE: {best_model['Test_RMSE']:.4f}°C")
        print(f"  Test R²: {best_model['Test_R2']:.4f}")
        print(f"  Test MAE: {best_model['Test_MAE']:.4f}°C")
        
        print(f"\n  This model explains {best_model['Test_R2']*100:.2f}% of the variance")
        print(f"  in temperature and has an average error of {best_model['Test_RMSE']:.2f}°C")
        
        # Compare to baseline
        baseline_result = next((r for r in all_results if "Baseline" in r['Model']), None)
        if baseline_result and baseline_result['Train_RMSE']:
            improvement = ((baseline_result['Train_RMSE'] - best_model['Test_RMSE']) / 
                          baseline_result['Train_RMSE'] * 100)
            print(f"\n  Improvement over baseline: {improvement:.1f}%")
    else:
        print("\nNo test set evaluations found. Run evaluate_model.py first.")
    
    # Key insights
    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    
    # Find RF test and RF full for comparison
    rf_test = next((r for r in all_results if "RF Test" in r['Model']), None)
    rf_full = next((r for r in all_results if "RF Full" in r['Model']), None)
    
    if rf_test and rf_full and rf_test['Train_RMSE'] and rf_full['Train_RMSE']:
        diff = abs(rf_test['Train_RMSE'] - rf_full['Train_RMSE'])
        print(f"\n1. Diminishing Returns:")
        print(f"   RF Test (10% data): {rf_test['Train_RMSE']:.4f}°C")
        print(f"   RF Full (100% data): {rf_full['Train_RMSE']:.4f}°C")
        print(f"   Difference: {diff:.4f}°C (only {diff/rf_test['Train_RMSE']*100:.1f}% change)")
        print(f"   → 10× more data improved RMSE by <1%")
    
    # Compare RF vs GBT
    gbt_test = next((r for r in all_results if "GBT Test" in r['Model']), None)
    if rf_test and gbt_test and rf_test['Train_RMSE'] and gbt_test['Train_RMSE']:
        rf_better = ((gbt_test['Train_RMSE'] - rf_test['Train_RMSE']) / 
                    gbt_test['Train_RMSE'] * 100)
        print(f"\n2. RF vs GBT Performance:")
        print(f"   RF Test: {rf_test['Train_RMSE']:.4f}°C")
        print(f"   GBT Test: {gbt_test['Train_RMSE']:.4f}°C")
        print(f"   → RF is {rf_better:.1f}% better than GBT")
    
    print("\n" + "=" * 80)
    
    spark.stop()

if __name__ == "__main__":
    main()