# Research: Options Flow Features for ML Model

**Date:** 2026-04-21
**Mode:** directed

## Summary

Options-based signals — particularly IV skew (OTM put IV minus ATM call IV) and put/call volume ratio — have strong academic evidence for predicting cross-sectional stock returns. The primary blocker for adding these as ML *training* features is the historical data gap: yfinance only provides current options chains, not historical IV snapshots. However, both signals are viable as **live inference features** injected into the predictor at scan time (alongside the existing SPY/VIX regime features), and IV skew could be added as a new standalone scanner with minimal new infrastructure.

## Sources Reviewed

- **Xing, Zhang, Zhao (2010) "What Does Individual Option Volatility Smirk Tell Us About Future Equity Returns?"** (Rice paper): SKEW = IV(OTM put, 80-95% moneyness) − IV(ATM call, 95-105%). Steepest-skew quintile underperforms by 10.9% annually. Predictability persists ≥6 months. Informed traders buy OTM puts before bad news — equity markets lag in incorporating this signal.
- **Management Science (2024) "Do Option Characteristics Predict the Underlying Stock Returns in the Cross-Section?"**: Broad sweep of option-implied features confirms IV skew remains significant after controlling for momentum, short-term reversal, and liquidity. Cross-sectional result.
- **Eksi & Roy (2025, Journal of Financial Research)**: Options prices predict reversal of nonfundamental stock shocks. IV skew increases after index inclusion events and predicts subsequent price reversal.
- **EFMA (2024) "Stock Return Predictability of Realized-Implied Volatility Spread"**: Realized minus implied vol spread (RV-IV) significant predictor of next-month returns after controlling for skew — complementary signal.
- **FlashAlpha awesome-options-analytics**: yfinance provides live options chains with IV per strike; `py_vollib` for IV calculation; gamma exposure (GEX) and dealer positioning identified as additional signals beyond put/call ratio.
- **Existing codebase `options_flow.py`**: Already computes put/call volume ratio across 3 expirations using yfinance. Produces `put_call_ratio` field. Premium filter ($25K) and vol/OI ratio (≥2×) already implemented.

## Cross-Reference with Existing Work

- **`options_flow` scanner** (`scanners/options_flow.py`): Already exists, uses yfinance, computes P/C ratio. **This research is about augmenting the ML model features, not replacing the scanner.**
- **`docs/iterations/scanners/options_flow.md`**: Pending hypothesis — "Is moneyness (ITM vs OTM) a useful signal filter?" — IV skew is directly the answer to this hypothesis.
- **No prior research** on using options features inside the ML training pipeline.

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ⚠️ partial | yfinance options chain integrated for live inference; **no historical IV data** for retraining dataset — this is the hard blocker for training features |
| Complexity | large | Training dataset: large (needs OptionMetrics/CBOE historical data, ~$$$). Live inference only: moderate (2-4h to add IV skew + P/C ratio as inference-time features) |
| Signal uniqueness | low overlap | `options_flow` scanner exists but uses volume/OI only, not IV skew. ML model has no options features at all today. |
| Evidence quality | backtested with statistics | Xing et al. (2010) is a landmark peer-reviewed paper with full backtest. 2024/2025 papers confirm the signal holds. |

## Recommendation

**Split implementation:**

1. **Skip for ML training dataset** — historical IV data requires OptionMetrics ($$$) or CBOE DataShop. Not feasible with free yfinance-only stack.

2. **Implement for live inference** (moderate, ~3h) — inject 3 options-derived features into `MLPredictor.predict()` at scan time, fetched live from yfinance, alongside the existing SPY/VIX regime injection:
   - `iv_skew`: IV(nearest OTM put, 80-95% moneyness) − IV(nearest ATM call) — directional informed-trading signal
   - `put_call_vol_ratio`: total put volume / total call volume across 3 nearest expirations — already computed in `options_flow.py`
   - `iv_rank`: current IV percentile vs 52-week range — measures whether options are expensive (event-driven premium) or cheap

3. **Augment `options_flow` scanner** (trivial, ~1h) — add IV skew as a filter/priority signal, directly resolving the pending "moneyness" hypothesis in `options_flow.md`.

## Proposed Implementation Spec

### A. Live inference feature injection (in `tradingagents/ml/predictor.py` + `feature_engineering.py`)

Add to `FEATURE_COLUMNS` (3 new features, total 40):
- `iv_skew` — float, typically 0.02–0.15; NaN if no options
- `put_call_vol_ratio` — float, typically 0.3–3.0; NaN if no options
- `iv_rank` — float 0–1; NaN if no options

Add `load_options_context(ticker)` to `MLPredictor`:
```python
def load_options_context(self, ticker: str) -> dict:
    """Fetch live IV skew, P/C ratio, IV rank for a single ticker."""
    from tradingagents.dataflows.y_finance import get_ticker_options, get_option_chain
    # 1. Get 3 nearest expirations
    # 2. For each: find ATM call (moneyness 0.95-1.05), OTM put (0.80-0.95)
    # 3. iv_skew = mean(OTM put IV) - mean(ATM call IV)
    # 4. put_call_vol_ratio = sum(put vol) / sum(call vol)
    # 5. iv_rank = (current ATM IV - 52w_low_IV) / (52w_high_IV - 52w_low_IV)
    #    approximated as ATM IV / historical_vol_ratio
```

In `_predict_ticker()` in `ml_signal.py`, call `predictor.load_options_context(ticker)` and merge into `features` dict before `predictor.predict(features)`.

### B. Augment `options_flow` scanner

Add IV skew computation to `_analyze_ticker_options()`:
- If `iv_skew > 0.05` → bearish signal → downgrade priority or skip
- If `iv_skew < 0.02` → neutral/bullish → keep MEDIUM
- Bullish unusual call flow + low IV skew → upgrade to HIGH

**Priority rules (for augmented scanner):**
- CRITICAL: unusual calls ≥3, P/C < 0.5, iv_skew < 0.02
- HIGH: unusual calls ≥2, P/C < 0.7 OR iv_skew < 0.04
- MEDIUM: unusual activity but mixed/neutral options sentiment

> **Auto-implementation note:** Training dataset path requires paid historical IV data — skipped. Live inference injection (spec A) requires model retraining with new feature columns to be useful — deferred until historical data is available. **Implementing spec B (scanner augmentation) only**, as it requires no new training data and directly resolves the pending `options_flow.md` moneyness hypothesis.
