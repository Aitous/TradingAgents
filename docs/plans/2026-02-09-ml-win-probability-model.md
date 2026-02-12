# ML Win Probability Model — TabPFN + Triple-Barrier

## Overview
Add an ML model that predicts win probability for each discovery candidate.
- **Training data**: Universe-wide historical simulation (~375K labeled samples)
- **Model**: TabPFN (foundation model for tabular data) with LightGBM fallback
- **Labels**: Triple-barrier method (+5% profit, -3% stop loss, 7-day timeout)
- **Integration**: Adds `ml_win_probability` field during enrichment

## Components

### 1. Feature Engineering (`tradingagents/ml/feature_engineering.py`)
Shared feature extraction used by both training and inference.
20 features computed locally from OHLCV via stockstats + pandas.

### 2. Dataset Builder (`scripts/build_ml_dataset.py`)
- Fetches OHLCV for ~500 stocks × 3 years
- Computes features locally (no API calls for indicators)
- Applies triple-barrier labels
- Outputs `data/ml/training_dataset.parquet`

### 3. Model Trainer (`scripts/train_ml_model.py`)
- Time-based train/validation split
- TabPFN or LightGBM training
- Walk-forward evaluation
- Outputs `data/ml/tabpfn_model.pkl` + `data/ml/metrics.json`

### 4. Pipeline Integration
- `tradingagents/ml/predictor.py` — model loading + inference
- `tradingagents/dataflows/discovery/filter.py` — call predictor during enrichment
- `tradingagents/dataflows/discovery/ranker.py` — surface in LLM prompt

## Triple-Barrier Labels
```
+1 (WIN):     Price hits +5% within 7 trading days
-1 (LOSS):    Price hits -3% within 7 trading days
 0 (TIMEOUT): Neither barrier hit
```

## Features (20)
All computed locally from OHLCV — zero API calls for indicators.
rsi_14, macd, macd_signal, macd_hist, atr_pct, bb_width_pct, bb_position,
adx, mfi, stoch_k, volume_ratio_5d, volume_ratio_20d, return_1d, return_5d,
return_20d, sma50_distance, sma200_distance, high_low_range, gap_pct, log_market_cap
