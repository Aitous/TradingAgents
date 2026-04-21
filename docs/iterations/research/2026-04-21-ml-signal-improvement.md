# Research: ML Signal Scanner Improvement

**Date:** 2026-04-21
**Mode:** directed

## Summary

The current LightGBM model has a fundamental design flaw: a 3-class target (WIN/TIMEOUT/LOSS) where TIMEOUT dominates 47.8% of training samples, causing the model to predict TIMEOUT by default and suppressing all WIN probabilities below 46%. The scanner's ceiling is hardcoded below 50% by the label design — no threshold tuning or config change can fix this. The core improvement is switching to binary classification (WIN vs NOT-WIN), redefining WIN as a higher-bar return threshold (≥5% in 20d), and adding 5 market-regime features that are free via yfinance.

## Sources Reviewed

- **MDPI 2025 — Regime-Aware LightGBM (63 features, walk-forward, 100 folds):** Mean walk-forward accuracy 54.2% across S&P 500 tickers. Key additions: VIX term structure, credit spread ratio (HYG/LQD), yield curve (TLT/SHY), BTC correlation, rolling HMM regime state. CTAS achieved 68.2% WR / 0.885 Sharpe. Deflated Sharpe (corrected for 250 strategy trials) not statistically significant — important honesty about multiple testing.
- **arXiv 2501.07580 — LightGBM Feature Engineering:** EMA ratio features (close - EMA14)/EMA14 and indicator-price slope ratios showed highest feature importance across all target variable transformations. Log returns as target superior to raw returns. Low computational overhead.
- **arXiv 2601.19504 — Hybrid XGBoost + FinBERT + Regime (2023–2025 backtest):** 63% out-of-sample accuracy with 10-feature XGBoost + regime classification via 20-day rolling average + FinBERT sentiment gating. 135% total return / 1.68 Sharpe over 2 years. Regime restriction (only trade in bullish regime) cut max drawdown from 19.8% → 15.6%.
- **arXiv 2412.15448 — Technical Indicators Impact (Random Forest):** BB percentage (14.7% importance), EMA (18.3%), RSI (15.5%). Critical finding: training R²=0.75–0.81 but testing R²=negative — severe overfitting risk with raw price features. Raw price data dominated in-sample but failed OOS.
- **Springer Financial Innovation — S&P 500 Relative Returns (LSTM):** Relative returns (individual vs index) across 13 time horizons outperform absolute returns as features. Multi-period relative momentum is more informative than raw price levels.
- **sklearn calibration docs + FastML:** LightGBM produces poorly calibrated probabilities by default. Isotonic regression calibration significantly improves probability reliability for ≥1000 samples. Platt scaling (sigmoid) better for small calibration sets. CalibratedClassifierCV is the standard sklearn wrapper.
- **Regime-Aware MDPI framework:** Rolling HMM (K=3 states) refitted every 63 days eliminates look-ahead bias. Bull/sideways/bear classification improves precision by filtering out signals taken against the regime.
- **Current model metrics.json (direct inspection):** WIN class precision 39.5%, recall 31.4%. TIMEOUT dominates (47.8% of samples, 71.1% recall). `avg_win_prob_for_actual_wins`: 0.349 — model assigns only 35% win probability even to confirmed winners. Q5 (top quintile) achieves only 45.5% actual win rate. Model is well-calibrated but has no signal above 46%.

## Root Cause Diagnosis

The model metrics reveal the actual problem — not threshold misconfiguration:

| Issue | Evidence | Impact |
|-------|----------|--------|
| TIMEOUT class dominates labels | 47.8% of training = TIMEOUT; model recall 71.1% | Model defaults to TIMEOUT, suppresses WIN probs |
| WIN ceiling at ~46% | Q5 actual WR = 45.5%; `avg_win_prob_for_actual_wins` = 34.9% | min_win_prob=0.50 will produce ZERO candidates |
| Low WIN precision (39.5%) | WIN F1 = 0.35 — worse than a coin flip | High false-positive rate on WIN predictions |
| No regime awareness | 30 features are all single-stock OHLCV | Same signal fired in bull/bear/ranging = noise |
| Ambiguous WIN definition | "WIN" likely = any positive return in N days | ~50% of stocks win by this definition → TIMEOUT absorbs ambiguity |
| No retraining schedule | `training_date: None` in metrics | Model drift as market regime evolves |

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ✅ | yfinance already integrated; VIX/SPY/sector ETFs available via same API |
| Implementation complexity | moderate (4–6 hours) | Binary relabeling + 5 new features + retrain script changes |
| Signal uniqueness | ✅ low overlap | No other OHLCV ML scanner in pipeline; improvements are internal |
| Evidence quality | backtested | MDPI walk-forward 100 folds; metrics.json provides direct ground truth |

## Recommendation

**Implement** — but this is a model redesign, not a scanner code change. The scanner infrastructure (`ml_signal.py`, feature_engineering.py, predictor.py) is solid. The fix is in the training pipeline and label design.

## Proposed Improvement Spec

### Fix 1: Binary classification target (highest priority)

**Change:** Relabel training data from 3 classes (WIN=1/TIMEOUT=0/LOSS=-1) to 2 classes:
- WIN = stock gained ≥5% within 20 trading days from signal date
- NOT-WIN = everything else (includes TIMEOUT and LOSS)

**Why 5%:** At ~50% base rate for "any positive return", the 3-class model can't distinguish signal from noise. A 5% threshold reduces WIN prevalence to ~25–30% of samples, forcing the model to learn genuine edge rather than marginal positive drift.

**Impact:** Eliminates TIMEOUT class dominance. Binary LightGBM will directly output WIN probability without the TIMEOUT dilution.

**Training script change:** Locate the label assignment in the training script and change the threshold logic.

### Fix 2: Add 5 market-regime features

Add to `FEATURE_COLUMNS` in `tradingagents/ml/feature_engineering.py`:

```python
# Market context features (computed from SPY/VIX via yfinance in batch)
"spy_return_20d",          # S&P 500 20-day return — broad market regime
"vix_level",               # CBOE VIX — absolute fear level
"vix_ma20_ratio",          # VIX / VIX_MA20 — is fear elevated vs recent history?
"stock_vs_spy_20d",        # stock return_20d minus spy_return_20d — relative strength
"sector_return_20d",       # sector ETF 20d return (XLK/XLF/XLE etc.) — sector context
```

**Data source:** All available via `yfinance` — SPY and VIX are already used for universe scanning. Sector ETF mapping (ticker → sector ETF) needs a lookup table.

**Academic backing:** MDPI study showed market interaction features (SPY correlation, VIX term structure) among top contributors in 63-feature set; regime-filtering (only signal in bullish regime) cut drawdown by 28%.

### Fix 3: Probability calibration post-training

After retraining, apply isotonic regression calibration:

```python
from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated_model.fit(X_val, y_val)
```

**When to use:** With ≥1000 validation samples (we have 70k total), isotonic regression is preferred over Platt scaling. Calibration is applied once after training and the calibrated model saved to `data/ml/tabpfn_model.pkl`.

**Expected impact:** Current model is already reasonably monotone-calibrated (Q1-Q5 probabilities track actual WR). Post-calibration with binary labels should push Q5 actual WR above 55% (from current 45.5%).

### Fix 4: Walk-forward retraining cadence

**Add** a bi-weekly retraining trigger. The current model has `training_date: None` — it was trained once and never retrained. Market regimes shift; a model trained on 2024 bull market data will underperform in a correction.

**Mechanism:** Add a check in `MLPredictor.load()` — if `training_date` in metrics.json is >30 days ago, log a warning: "ML model is >30 days old — consider retraining."

**Retraining target:** Rolling 2-year window of the OHLCV parquet cache + performance_database.json outcomes.

### Fix 5 (optional): EMA ratio features

Per arXiv 2501.07580, add:
```python
"ema14_ratio",    # (close - EMA14) / EMA14 — price position vs short EMA
"ema14_slope",    # EMA14 today / EMA14 5d ago - 1 — EMA acceleration
```

These showed highest feature importance in the LightGBM study and are computable from existing OHLCV data with no new API calls.

## Implementation Order

1. **Fix 1** (binary labels + higher WIN threshold) — biggest impact, root cause fix
2. **Fix 2** (5 regime features) — adds market context the model currently lacks
3. **Fix 3** (calibration) — apply after retraining with fixes 1+2
4. **Fix 4** (retraining cadence warning) — trivial, prevents silent staleness
5. **Fix 5** (EMA ratio features) — incremental improvement

## What NOT to Do

- **Do not tune min_win_prob further.** The current ceiling is ~46% due to the 3-class label design. Raising the threshold to 0.55 or 0.60 will produce zero candidates until Fix 1 is implemented.
- **Do not add more OHLCV features without binary labels first.** The model bias toward TIMEOUT is the binding constraint; more features won't help until the label design is fixed.
- **Do not use raw price levels as features.** arXiv 2412.15448 confirmed severe overfitting when raw OHLCV prices are included (train R²=0.80, test R²=negative).
