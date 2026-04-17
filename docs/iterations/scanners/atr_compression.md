# Scanner: ATR Compression Breakout

**Module:** `tradingagents/dataflows/discovery/scanners/atr_compression.py`
**Pipeline:** momentum
**Implemented:** 2026-04-16

## Signal Logic

ATR(5) / ATR(20) ≤ 0.75 — short-term 1-week realized volatility has compressed below 75% of the 4-week baseline. Additionally, today's close must break above the 10-day prior high (directional confirmation that the squeeze is releasing upward). Uptrend filter: price above SMA(50).

The ATR ratio is the key gate: ATR uses True Range which incorporates overnight gaps, making it more robust than a simple high-low range comparison. When 5-day ATR compresses below 75% of the 20-day ATR, the market has been in an unusually tight range — a spring coiling before release.

Source: Traderslog "Volatility Breakout Systems"; VT Markets Dec 2025 cross-index backtest (+34% improvement with directional signal layered on ATR). Research: `docs/iterations/research/2026-04-16-volatility-batch.md`

## Backtest Results (walk-forward, 2y OHLCV cache, 1003 tickers)

| Metric | Value |
|--------|-------|
| Backtest period | 2025-04-15 → 2026-03-06 |
| Total picks | 738 |
| Unique tickers | 420 |
| Avg picks/day | ~3.3 (selective — does NOT hit limit daily) |
| Win rate 1d | 50.3% |
| Win rate 5d | 56.1% |
| Win rate 10d | 59.2% |
| Win rate 20d | **59.3%** |
| Avg return 20d | **+1.88%** |
| Median return 20d | +1.68% |

**Classification:** PROMOTE-MARGINAL
- WR-20d (59.3%) well above 52% threshold ✅
- Avg return (1.88%) marginally below 2.0% threshold — classified PROMOTE-MARGINAL ⚠️
- Secondary check: WR-5d = 56.1% (≥45%) — NOT a slow signal, edge appears early ✅

## Current Understanding

The ATR compression + N-day high breakout combination is the most selective volatility scanner tested: it fires ~3.3x/day vs the NR7 or BB squeeze which always filled their 10-pick limit. This selectivity is reflected in the superior win rate (59.3% at 20d). The directional confirmation (breaking above 10-day high) is the critical differentiator — without it, compression signals are ~50/50.

The 1d and 5d win rates (50.3% and 56.1%) show edge builds over the holding period, suggesting the volatility expansion takes time to play out. Optimal holding: 10–20 days.

## Pending Hypotheses

- [ ] Test `atr_ratio_max=0.65` (stricter compression) — expect fewer picks, higher WR
- [ ] Test `breakout_lookback=5` vs 10 vs 20 — shorter lookback = more setups, longer = fewer but stronger
- [ ] Check bear-market sub-period performance: filter picks.csv to 2025-Q4 (market correction)
- [ ] Confirm signal isn't concentrated in specific sectors (check unique_tickers diversity in picks.csv)

## Evidence Log

### 2026-04-16 — walk-forward backtest (738 picks, 224 trading days)
- WR improves monotonically from 1d (50.3%) → 20d (59.3%): slow-building signal
- Median 20d return +1.68% vs avg +1.88%: right-skewed distribution (big wins pull avg up)
- 420 unique tickers / 738 picks = 1.76 picks per ticker average: healthy diversity, not concentrated
- Peak pick count: Aug 2025 (10/day for several sessions) — likely post-correction vol expansion window
- Confidence: high (walk-forward, no lookahead, 738 picks across diverse universe)
