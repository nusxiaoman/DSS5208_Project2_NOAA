## 🗂️ Repository Structure

```
├── DATA_CLEANUP_README.md                     # Data cleaning documentation
├── docs                                       # Reference documentation
│   ├── AI Communication.txt
│   └── MLlib.pdf
├── experiments                                # Experimental scripts (not in pipeline)
│   ├── num_feature_15
│   ├── num_feature_7
│   └── script_for_trial
├── FEATURE_SELECTION_COMPARISON.md            # Feature engineering analysis (V0/V1/V2)
├── QUICK_REFERENCE.md                         # Command reference guide
├── README.md                                  # Project overview and quick start
├── RESULTS_SUMMARY.md                         # Complete results and analysis
├── src
│   ├── baseline_model_test.py                 # Linear Regression baseline (10 min)
│   ├── compare_models.py                      # Model comparison (5 min)
│   ├── evaluate_model.py                      # Test set evaluation (15 min)
│   ├── noaa_cleanup_full.py                   # Data cleaning (45 min)
│   ├── train_gbt_simplified.py                # GBT training (1.2 hrs)
│   ├── train_random_forest_simplified.py      # RF training (1.2 hrs) ✓ BEST
│   ├── train_test_split.py                    # 70/30 split (20 min)
│   ├── train_gbt.py                           # GBT original version
│   └── train_random_forest.py                 # RF original version
└── TRAINING_GUIDE.md                          # Step-by-step training instructions
```