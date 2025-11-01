## 🗂️ Google Cloud Storage (GCS) Structure
```
├── data/csv/                     # Raw NOAA CSV files (~50GB)
├── scripts/                      # keep script to run batch
├── outputs/
│   ├── baseline_test/            # keep baseline training result
│   ├── rf_simplified_evaluation/ # keep eveluation data
│   ├── rf_simplified/            # keep random forest simplfied training result
│   ├── rf_test/                  # keep random forest training result
│   └── gbt_test/                 # keep GBT training result
└── warehouse/
    ├── noaa_train/               # Raw train parquet (from ETL)
    ├── noaa_test/                # Raw test parquet (from ETL)
    └── noaa_clean_std/           # Cleaned data (output)
```