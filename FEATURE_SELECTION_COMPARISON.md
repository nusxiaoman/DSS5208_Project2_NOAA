# NOAA Weather Prediction - Version Comparison Summary

**Project:** DSS5208 Temperature Prediction | **Dataset:** 130M rows, 2024 weather data

---

## Quick Results Comparison

| Version | Features | Baseline RMSE | RF RMSE | Status |
|---------|----------|---------------|---------|--------|
| **V0** | 14 | 5.56°C | **4.5°C** ✅ | **Use This!** |
| **V1** | 32 | **0.46°C** ⚠️ | - | Data Leakage |
| **V2** | 15 | 6.85°C | - | Worse than V0 |

---

## Version Details

### V0 - Your Working Version ✅

**Features (14):**
- Geographic: latitude, longitude, elevation (3)
- Weather: dew_point, sea_level_pressure, visibility, wind_speed, precipitation (5)
- Temporal: hour_sin/cos, month_sin/cos (4)
- Wind: wind_dir_sin/cos (2)

**Method:**
- NULL handling: Drop rows
- Split: 70/30 random

**Results:**
- Linear: 5.56°C
- **RF: 4.5°C** ✅
- **62% better than naive baseline (~12°C)**

**Status:** ✅ **LEGITIMATE and GOOD** - Use this for your report!

---

#### Why V0's 14 Features Are Comprehensive

**NOAA Dataset Has Limited Usable Fields:**

From the original NOAA Global Hourly Dataset, here's what's actually available and practical:

| Field | V0 Used? | NULL Rate | Usability |
|-------|----------|-----------|-----------|
| **TMP** (temperature) | ✅ Target | <1% | Required |
| **DEW** (dew point) | ✅ Yes | ~15% | Good ✅ |
| **SLP** (sea level pressure) | ✅ Yes | ~59% | Moderate ⚠️ |
| **WND** (wind) | ✅ Yes | ~10-27% | Good ✅ |
| **VIS** (visibility) | ✅ Yes | ~33% | Moderate ⚠️ |
| **AA1-AA4** (precipitation) | ✅ Yes | Variable | Good ✅ |
| **CIG** (ceiling height) | ❌ No | **51.5%** | Too many NULLs ❌ |
| **AW1-AW7** (weather conditions) | ❌ No | **~95%** | Not available ❌ |
| **GA1-GA6** (cloud cover) | ❌ No | **~90%** | Not available ❌ |
| **OC1** (wind gust) | ❌ No | **100%** | Not available ❌ |

**Key Insight:** V0 already uses ALL the high-quality, available fields!

**What V0 Captures:**
1. ✅ **Temperature drivers:** Dew point, pressure, wind (direct physical relationships)
2. ✅ **Location effects:** Latitude, longitude, elevation (climate zones, altitude)
3. ✅ **Temporal patterns:** Hour, month (diurnal and seasonal cycles)
4. ✅ **Weather conditions:** Precipitation, visibility (current weather state)

**What's Missing from Dataset:**
- ❌ Cloud data (90%+ NULL)
- ❌ Weather observations (95%+ NULL)
- ❌ Wind gust (100% NULL)
- ❌ Detailed sky cover (90%+ NULL)

**Conclusion:** V0's 14 features represent **all the essential, available meteorological information** in the dataset. Adding more features (V2's ceiling at 51% NULL) adds noise, not signal!
---

### V1 - With Lag Features ⚠️

**Features (32):**
- V0's 14 features PLUS:
- **Lag features (9):** temp_lag_1h, temp_lag_2h, temp_lag_3h, pressure_lag_1h, temp_rolling_3h, temp_change_1h, etc.
- Station stats (3): station_avg_temp, station_avg_dew, station_avg_pressure
- Weather conditions (4): is_raining, is_snowing, is_foggy, is_thunderstorm
- Enhanced temporal (2): day_of_year_sin/cos

**Method:**
- NULL handling: Median imputation
- Split: 70/30 **random** ❌

**Results:**
- **Baseline: 0.46°C** ⚠️
- **R² = 0.9986**

**Status:** ❌ **DATA LEAKAGE!**

**The Problem:**
```
Timeline with random split:
11am [TEST]  temp=19°C, temp_lag_1h=18°C ← from adjacent train sample!
12pm [TRAIN] temp=20°C
01pm [TEST]  temp=21°C, temp_lag_1h=20°C ← from adjacent train sample!
```

- Test samples use lag features from nearby train samples
- Model learns: "temp_now ≈ temp_lag_1h" (trivial!)
- Creates unrealistically low RMSE (0.46°C)

**How to Fix:** Use time-based split (Train: Jan-Sep, Test: Oct-Dec)
- Expected RMSE after fix: ~3.5-4.0°C

---

### V2 - With Ceiling Height

**Features (15):**
- V0's 14 features PLUS:
- **ceiling_height** (NEW) - but **51.5% NULL!**

**Method:**
- NULL handling: Median imputation (for ceiling + other features)
- Split: 70/30 random

**Results:**
- Baseline: **6.85°C** (worse than V0!)

**Status:** Valid but performed worse

**Why It Failed:**
- Ceiling height has 51% NULLs
- Median imputation set half the values to same number
- Added noise instead of signal
- Made performance worse, not better

---

## Why V0's 4.5°C is Actually Good

1. **vs Naive Baseline:**
   - Std dev: ~12°C
   - Your RF: 4.5°C
   - **Improvement: 62%** ✅

2. **vs Linear:**
   - Linear: 5.56°C
   - RF: 4.5°C
   - **Improvement: 19%** ✅

3. **Real Context:**
   - Professional weather forecasts: 2-3°C error
   - Your model with basic features: 4.5°C
   - **Very reasonable!** ✅

---

## Why V1's 0.46°C is Suspicious

1. **Better than professionals:** 0.46°C < 2-3°C (impossible!)
2. **R² = 0.9986:** Explains 99.86% of variance (unrealistic)
3. **Lag + random split = leakage:** Classic data leakage pattern
4. **Same as other group:** They got 1.5°C (also leakage)

**Evidence:** In proper time-based split, V1 would likely get 3.5-4.0°C, not 0.46°C

---


## Key Learnings

### ✅ Do This:
- Use time-based splits for temporal data
- Drop features with >50% NULLs
- Compare results to domain knowledge
- Question "too good to be true" results

### ❌ Don't Do This:
- Lag features + random split = leakage
- Median impute features with >50% NULLs
- Accept unrealistic results without investigation

---

## Recommendation for Your Report

### Use V0 Results ✅

**Why:**
- Legitimate methodology (no leakage)
- Good performance (4.5°C)
- Easy to defend
- Demonstrates proper ML practice

**Report Structure:**
1. **Methods:** V0 features and approach
2. **Results:** 4.5°C RF (62% better than naive)
3. **Discussion:** Why 4.5°C is good performance
4. **Analysis:** V1 shows data leakage example
5. **Critique:** Other group's 1.5°C likely has same issue

---

## Summary

**Your V0 RF: 4.5°C RMSE**
- ✅ Legitimate
- ✅ Good performance
- ✅ No data leakage
- ✅ **Submit this!**

**V1: 0.46°C **
- ❌ Data leakage
- ❌ Unrealistic
- ❌ Lag features + random split
- ❌ Won't work in deployment

**Bottom Line:** Suspiciously good results usually mean something's wrong. Your honest 4.5°C is what good ML looks like! 🎯

---

*Use V0 for your final submission!*
