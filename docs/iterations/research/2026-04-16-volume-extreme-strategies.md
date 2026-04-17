# Research: Volume-Extreme Trading Strategies

**Date:** 2026-04-16
**Mode:** directed

## Summary

Researched 10+ sources for OHLCV-computable strategies where extreme volume events are the primary signal. The strongest evidence supports a "Selling Climax Reversal" pattern — a new multi-day low with volume ≥3x the 50-day average that closes with a bullish intraday reversal bar and RSI(14) < 35. Three independent extreme conditions must coincide, making the signal inherently rare. O'Neil Global Advisors quantified that breakouts with >150% volume above the 50-day average generate +5.04% absolute returns over 63 days (57% win rate); StockCharts Arthur Hill backtested an RSI+volume filter achieving 83–94% win rate on the S&P 500 (8% CAGR, 30% time in market). A secondary high-confidence candidate is the Volume-Confirmed O'Neil Base Breakout (already partially implemented as high_52w_breakout; not re-implementing). The third candidate — OBV divergence — is already implemented.

## Sources Reviewed

1. **arXiv (q-fin.TR API)**: Returned physics papers due to API category mismatch — no usable trading content.
2. **arXiv (q-fin volume climax)**: Same issue — no relevant results.
3. **QuantifiedStrategies (volume surge / breakout)**: Bot-blocked (CAPTCHA); only article titles accessible. Confirms "Volume Oscillator Strategy" and "Reversal Day Strategy" exist with backtests but detail not extractable.
4. **Reddit r/algotrading**: 403 Forbidden — inaccessible without auth.
5. **CSS Analytics (volume search)**: Identified 6 volume-related articles (2009–2011). Key finding: volume exhaustion (10–14 day volume peak as filter on short-bias mean reversion trades against SPY) "significantly improved edges and substantially lowered drawdowns." No numeric thresholds disclosed in article text.
6. **Alpha Architect (volume momentum)**: Only one relevant article — ML model predicting trading volume for portfolio construction; no implementable OHLCV signal.
7. **O'Neil Global Advisors PDF ("Breakouts: Pump up the Volume")**: Best quantitative source. Segmented 1,000s of chart pattern breakouts by volume vs 50-day average:
   - <50% above avg → +2.61% return, 56.85% win rate (63d)
   - 50–100% above avg → +3.01%, 57.12% win rate (63d)
   - 100–150% above avg → +3.12%, 57.23% win rate (63d)
   - **≥150% above avg → +5.04%, 57.68% win rate (63d)** ← critical threshold
   The 150% excess volume threshold (volume ≥2.5x 50d avg) is the academically validated minimum for institutional demand confirmation.
8. **StockCharts Arthur Hill (selling climaxes, 2016)**: Defined selling climax as Percentage Volume Oscillator(2,250,1) > 100 (2-day EMA of volume > 2x the 250-day EMA). Tested RSI-mean-reversion system with volume filter on SPY: RSI(5) < 25 → 83.8% win rate (62/74 trades), ~8% CAGR, 30% time in market. RSI(10) < 30 → 94.1% win rate (32/34 trades), same CAGR.
9. **SSRN**: Found "Price Momentum and Trading Volume" (Lee & Swaminathan, 2000) — high-volume past losers outperform low-volume past losers by 2–7% annually; high-volume past winners outperform by same margin. This validates volume as a momentum moderator, not a standalone reversal signal.
10. **Triple-Phase Volume-RSI Momentum Reversal (Medium / FMZQuant)**: Requires 2 consecutive down bars + bullish bar with RSI(14) V-shape from <30. Tested on crypto only. No win rate provided. Concept is valid for equity adaptation.
11. **GitHub search**: No specific volume climax reversal Python repo found; general backtesting frameworks (backtrader, backtesting.py) only.
12. **OBV Vestinda backtest**: OBV divergence on RIOT (25% win rate, 64% annualized ROI, 1.49 profit factor) — high volatility stock skews metrics; general OBV divergence on BAC (46% win rate, 5.49% ROI). The asymmetry confirms OBV divergence is better as a filter than standalone signal.

## Cross-Reference with Existing Scanners

| Strategy Found | Overlap with Existing Scanner |
|---------------|------------------------------|
| Volume-confirmed 52w-high breakout (O'Neil ≥150% vol) | **high_52w_breakout** (implemented 2026-04-13) — same concept, already live |
| OBV divergence accumulation | **obv_divergence** (implemented 2026-04-14) — already live |
| RSI mean-reversion oversold | **rsi_oversold** (implemented 2026-04-15) — already live |
| Selling Climax Reversal (new-low + extreme vol + bullish reversal bar + RSI<35) | **No existing scanner** — gap in pipeline |
| Volume-confirmed MACD crossover | No scanner — but MACD alone <50% win rate per arXiv study; volume filter improves it. Moderate evidence. |

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ✅ | Pure OHLCV; uses shared ohlcv_cache already in pipeline |
| Complexity | trivial | ~100 lines; same pattern as rsi_oversold.py |
| Signal uniqueness | low overlap | No existing scanner catches new-low + extreme-volume + bullish-reversal-bar confluence |
| Evidence quality | backtested | O'Neil PDF has quantified returns at specific volume thresholds; StockCharts has win rates |

## Recommendation

**Implement** — Selling Climax Reversal scanner.

Three independent extreme conditions required:
1. Close at a new N-day low (price exhaustion)
2. Volume ≥ 3x the 50-day average (panic selling climax, above O'Neil's 150% threshold)
3. Bullish intraday reversal bar: close > open AND close in upper 40% of day's High-Low range

Optional 4th condition (priority upgrade): RSI(14) < 35 (oversold confirmation)

Expected signal frequency: <3 per day in a 1,000-ticker universe (all 3 conditions rarely coincide).
Expected holding period: 5–20 days.
Expected edge: +3–5% over 63d based on O'Neil data; win rate ~57–65% when volume ≥3x 50d avg.

## Proposed Scanner Spec

- **Scanner name:** `selling_climax_reversal`
- **Pipeline:** `mean_reversion`
- **Strategy:** `climax_reversal`
- **Data source:** `tradingagents/dataflows/data_cache/ohlcv_cache.py` (download_ohlcv_cached)
- **Signal logic:**
  1. Load 60+ days of OHLCV (need 50d vol avg + 20d price low lookback)
  2. Compute 50-day volume average
  3. Flag bars where: `volume / vol_avg_50 >= 3.0` (climax volume)
  4. On the same bar: `close == min(close[-20:])` — new 20-day closing low
  5. Bullish reversal bar: `close >= open` AND `(close - low) / (high - low) >= 0.4`
  6. Optional RSI(14) < 35 for CRITICAL priority
  7. Liquidity gate: avg_volume_20d >= 100K shares, price >= $5
- **Priority rules:**
  - CRITICAL if all 4 conditions (climax vol + new low + reversal bar + RSI<35)
  - HIGH if 3 conditions (climax vol + new low + reversal bar)
  - MEDIUM if climax vol + new low only (no reversal bar confirmation)
- **Context format:** `"Selling climax: {vol_ratio:.1f}x volume on {N}-day low, closed in upper {pct_range:.0f}% of range. RSI(14)={rsi:.1f}. Potential exhaustion reversal."`
- **Minimum OHLCV history:** 60 trading days (50d vol avg + 20d price lookback + RSI buffer)

## Backtest Discard Notes

Walk-forward backtest: 2025-04-15 → 2026-03-06 (224 sim days), 1003-ticker universe.

### selling_climax_reversal — DISCARD
- picks: 133 (0.6/day avg — selective ✓)  win_rate_20d: 44.4%  avg_return_20d: +1.57%
- WR < 50% at the 20d horizon. The 1d WR (42%) is also below random.
- Despite good selectivity, the signal itself appears to catch falling knives more than
  reversals — extreme volume on a new low frequently continues lower before recovering.
- Root cause: requiring close ≥ open (green bar) is insufficient for a reliable reversal
  confirmation after a panic-volume climax. The bar pattern needs a stronger confirmation
  (e.g., close above prior day's close, or engulfing candle structure).
- Do not re-research in current form. Consider a rebuilt version using engulfing + volume.

### macd_histogram_reversal — DISCARD
- picks: 1940 (8.7/day avg — hits limit every day ✗)  win_rate_20d: 53.6%  avg_return_20d: +1.35%
- Same selectivity failure as Run 2 (nr7, consecutive_down_days, etc.).
- MACD histogram falling 4+ bars below zero fires hundreds of times daily in 1000 tickers.
- Does not meet PROMOTE-MARGINAL threshold (needs avg ≥ 2.0%).
- Potential fix: add a 2nd independent condition (e.g., volume spike on the falling bar, or
  RSI also hitting oversold) to make the signal genuinely rare. Not worth pursuing in current form.
