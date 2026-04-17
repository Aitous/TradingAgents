# Research: Gap-Up Continuation (Breakaway Gap Scanner)

**Date:** 2026-04-17
**Mode:** autonomous

## Summary

Opening price gaps driven by news catalysts and confirmed by above-average volume show
statistically significant same-day and multi-day continuation. A peer-reviewed PMC study
(Fakhfakh 2023) measured 54–60% winning-day rates with +0.30–0.58% avg daily gain for
positive gaps across Russell 2000, Nasdaq 100, and S&P 500 stocks. Large gaps (>0.4%)
fill less than 50% of the time vs the ~59% fill rate for all gap-ups, confirming that
selectivity (gap size + volume) sharply improves the continuation edge.

## Sources Reviewed

- **PMC / Fakhfakh 2023** (`pmc.ncbi.nlm.nih.gov/articles/PMC10017064`): Academic study
  of opening price gaps and intraday drift; 54–60% winning-day rates across major indices;
  smaller caps show more pronounced drift; full gaps > partial gaps; sequential gaps
  amplify the effect.
- **TradeThatSwing / S&P 500 gap fill stats** (`tradethatswing.com`): S&P 500 gap-ups fill
  59% of the time; gaps >0.4% drop below 50% fill rate — meaning large gaps are more likely
  to continue than reverse. Smaller gaps (0–0.19%) fill 79–89% of the time.
- **Nasdaq / Zacks gap taxonomy** (`nasdaq.com`, `zacks.com`): Classifies breakaway, runaway,
  and exhaustion gaps; breakaway gaps on high volume are confirmed price-discovery moves;
  continuation plays with catalyst have stronger follow-through.
- **Alpha Architect momentum literature** (`alphaarchitect.com`): Momentum premium strongest
  when news is "continuous" not "discrete"; gap-up on news with volume = new-information
  anchoring, not simple momentum, reducing reversal risk.
- **SSRN / Baniya 2024** (`papers.ssrn.com/abstract=4834097`): Specifically studies opening
  gap behavior for S&P 500 and NASDAQ 2019-2021; found significant correlation between gap
  size and sustained directional move (access blocked by CAPTCHA but abstract confirmed).

## Cross-Reference: Existing Scanners

- **`high_52w_breakout`**: requires a 52-week high crossing event + 1.5x volume. A gap-up
  does NOT require a 52w high — it measures the open vs prior close. Low to medium overlap:
  many gap-ups occur inside a 52w range (e.g., post-earnings gaps on stocks in mid-trend).
- **`volume_accumulation`**: detects high-volume quiet accumulation (price flat). Explicitly
  excludes the gap-up pattern (classifies it as "momentum" not "accumulation").
- **`atr_compression`**: detects volatility contraction BEFORE a move. Gap-up detects the
  move AFTER it has started. Complementary, not overlapping.
- **`volume_dry_up`**: detects very LOW volume as a coiling signal. Opposite condition.
- **`rsi_oversold`**: contrarian bounce. Gap-up is a continuation signal — opposite premise.
- **`minervini`**: catches stage-2 trending stocks via SMA structure. Gap-up catches the
  specific event of a gap opening, many of which occur in non-stage-2 stocks (e.g., a stock
  breaking out of a base for the first time). Partial overlap when gap is within a stage-2
  stock, but the *event trigger* is distinct.

**Conclusion:** Low overlap. No existing scanner detects the opening-gap event specifically.
Gap-up continuation fills a genuine blind spot in the pipeline.

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ✅ | OHLCV cache fully integrated; `download_ohlcv_cached` provides Open, High, Low, Close, Volume — zero new API sources needed |
| Complexity | moderate | Need 200d SMA, 20d volume avg, gap detection from daily open vs prior close; ~2 hours |
| Signal uniqueness | low overlap | No existing scanner targets gap-up event; complements ATR compression / volume_dry_up as a "post-squeeze trigger" detector |
| Evidence quality | backtested | PMC peer-reviewed study gives 54–60% win rate, 0.30–0.58% avg daily gain; fill-rate statistics from TradeThatSwing confirm large gaps continue >50% of the time |

## Recommendation

**Implement** — all four thresholds pass. The gap-up continuation signal is novel, uses
existing data, is selective by design (large gaps are rare), and has quantitative academic
backing with specific win rates.

## Proposed Scanner Spec

- **Scanner name:** `gap_up_continuation`
- **Data source:** `tradingagents/dataflows/data_cache/ohlcv_cache.py` (via `download_ohlcv_cached`)
- **Signal logic:**
  1. `gap_pct = (today_open − prior_close) / prior_close × 100 ≥ min_gap_pct` (default 2.0%)
  2. `intraday_hold = today_close / today_open ≥ 1 − max_reversal_pct` (default 0.97, i.e. ≤3% reversal from open)
  3. `vol_multiple = today_volume / 20d_avg_volume ≥ min_vol_multiple` (default 1.5×)
  4. Optional trend filter: `today_close > 200d_SMA` (default enabled)
- **Priority rules:**
  - CRITICAL: gap ≥ 5% AND vol ≥ 2× AND above 200d SMA
  - HIGH: gap ≥ 3% AND vol ≥ 1.5×
  - MEDIUM: gap ≥ 2% AND vol ≥ 1.5×
- **Context format:** `"Gap-up: +{gap_pct}% above prior close | vol {vol_mult}x avg | held intraday ({close_vs_open}% vs open) | {above/below} 200d SMA — breakaway continuation setup"`
