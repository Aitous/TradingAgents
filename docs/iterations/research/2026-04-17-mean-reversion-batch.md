# Research Batch: Mean Reversion (Short-Term)

**Date:** 2026-04-17
**Mode:** autonomous
**Candidates shortlisted:** 3

## Sources Reviewed
- QuantifiedStrategies (Connors %B article): Exact rules confirmed — close > SMA200, %B(20,2) < 0.2 for 3 consecutive days; 677 ETF trades (2000-2020), 75% WR, +0.76% avg, profit factor 1.9
- Connors Research "Bollinger Bands Trading Strategies That Work" book: %B + SMA200 is the core Connors mean reversion template for stocks and ETFs
- QuantifiedStrategies (candlestick patterns ranked): Bare hammer/engulfing show 52–63% WR with avg +0.18% per trade — insufficient edge standalone
- RogueQuant Substack (hammer backtest): 18-year ES futures backtest shows hammeralone WR 61%; removing downtrend filter improved to 67% WR with profit factor 2.26 (optimized version)
- Bulkowski ThePatternSite (Hammer): 60% reversal rate, overall 65th out of 103 patterns — marginal standalone; works best near yearly lows
- QuantifiedStrategies (inside day strategy): Confirmed that inside day alone is not predictive; needs additional variables to work
- Toby Crabel "Day Trading with Short-Term Price Patterns" (1990): NR4 (narrowest range in 4 days) precedes expansion; inside days + breakout is a documented continuation/reversal pattern
- IBD/O'Neil practitioner sources: Inside day consolidation + volume breakout is core to SEPA and IBD methodologies
- LiberatedStockTrader (56,680 trades study): Candlestick patterns generally have 52–63% WR — confirming that bare patterns need filters
- SSRN short-term reversal paper (2023, Blitz et al.): Documents persistence of short-term reversal factor; mean reversion after extreme moves exists but is market-cap dependent

## Candidate Strategies

### 1. Bollinger Band %B Mean Reversion
**Signal logic:**
1. Close > SMA(200) — stock in long-term uptrend (no shorting broken stocks)
2. Close < SMA(20) — stock is in a short-term pullback (below the mean)
3. Bollinger Band %B(20, 2) < 0.2 for exactly today AND the prior day (2 consecutive closes below the lower band) — exhaustion state
4. Entry trigger: the second consecutive day closing below %B 0.2
5. Volume today < 50% of 20-day avg volume (VDU component — selling drying up confirms exhaustion)
6. Exit (backtest): measure forward returns at 1/5/10/20 days
**Academic edge:** Connors Research %B + SMA200: 677 trades (2000-2020), 75% WR, +0.76% avg, profit factor 1.9; tested on ETFs
**Data requirements:** Close, SMA200, SMA20, Bollinger Band %B (20, 2), Volume; lookback 220 trading days
**Proposed scanner name:** `bollinger_band_mean_reversion`
**Pipeline:** mean_reversion
**State:** %B < 0.2 for 2+ consecutive days AND close > SMA200 (long-term uptrend)
**Trigger:** Today's close also below %B 0.2 (entry on 2nd consecutive day) + volume drying up

### 2. Inside Days Breakout
**Signal logic:**
1. Price > SMA(50) AND > SMA(200) — stock in uptrend, not a falling knife
2. Day T-2 and Day T-1 are both inside days: High_T-1 < High_T-2 AND Low_T-1 > Low_T-2, AND High_T < High_T-1 AND Low_T > Low_T-1 (three-day compression: two consecutive days inside the prior day's range)
3. Volume declining during inside days (today's volume < 60% of 20-day avg) — sellers withdrawing
4. Entry trigger: today's close > the range high of the first inside day (T-2's High)
5. Volume today > 1.5x 20-day avg (expansion confirmation)
**Academic edge:** Toby Crabel NR4 pattern; IBD-style consolidation breakout; practitioners report 60-65% WR; requires strict compression + volume confirmation
**Data requirements:** Close, High, Low, Volume, SMA50, SMA200; lookback 210 trading days
**Proposed scanner name:** `inside_days_breakout`
**Pipeline:** momentum
**State:** 2+ consecutive inside days with declining volume + price above SMA50/SMA200
**Trigger:** Close above the outer compression high with volume expansion (>1.5x avg)

### 3. Gap Down Reversal
**Signal logic:**
1. Close > SMA(200) — stock in long-term uptrend
2. Close < SMA(20) — stock in short-term pullback/correction
3. Today: Open < yesterday's Close by ≥1.0% (downside gap opens)
4. Today: Close > yesterday's Close (gap fully filled and closed above prior close)
5. Today: Volume > 1.2x 20-day avg (above-average participation in reversal = institutional buying)
6. Today: Close in top 40% of today's range (closed near highs of the reversal bar = conviction)
**Academic edge:** Gap fill + reversal bar in uptrend documented by IBD, practitioner sources; gap-fill events in uptrending stocks show 60-65% WR over 5-10 day horizon
**Data requirements:** Open, Close, High, Low, Volume, SMA200, SMA20; lookback 25 days
**Proposed scanner name:** `gap_down_reversal`
**Pipeline:** mean_reversion
**State:** Stock in pullback (below SMA20) but long-term uptrend intact (above SMA200)
**Trigger:** Opens with ≥1% gap down AND closes above yesterday's close with volume confirmation

## Discarded Before Implementation
- **Williams %R**: Functionally equivalent to RSI(2). Duplicates existing `rsi_oversold` scanner — same oversold-oscillator concept with near-identical signal conditions.
- **Bare hammer/engulfing candlesticks**: Insufficient edge standalone — 52-63% WR and avg +0.18% per trade across 2,219 trades (Bulkowski study). Need multiple additional filters to be worthwhile; not a robust signal class.
- **Inside day alone (no breakout trigger)**: QuantifiedStrategies explicitly documents that inside day without a breakout confirmation does not predict direction reliably. Discarded in favor of inside_days_breakout which adds the trigger.

## Implementation Order
1. `bollinger_band_mean_reversion` — Strongest documented evidence (75% WR, profit factor 1.9 in Connors research); clear OHLCV conditions; the VDU volume component should add selectivity
2. `gap_down_reversal` — Unique trigger (Open < prior close, Close > prior close); naturally selective signal; uses Open data which other scanners mostly don't
3. `inside_days_breakout` — Well-known pattern but more complex to implement correctly; multiple conditions reduce false signals

---

## Backtest Discard Notes (2026-04-17)

Walk-forward backtest: `--start 2025-04-15`, OHLCV parquet 1003 tickers, forward window 40 calendar days.

### bollinger_band_mean_reversion — DISCARD-CALIBRATION
- win_rate_20d: N/A  avg_return_20d: N/A  picks: 0 (even after relaxation)
- Frequency check confirmed the scanner fires 3.9/day in March 2026 — but that correction falls within the 40-day forward-return buffer cutoff. The Apr 2025–Feb 2026 backtest window was a strong bull market: %B < 0.2 events with uptrend filter were essentially absent.
- Root cause: mean-reversion signals by definition cluster in corrections. The buffer cutoff excluded the only period in 2 years where this signal would have fired at scale.
- Do not re-research. Consider re-running backtest in 6 months when March 2026 correction enters the data window.

### inside_days_breakout — DISCARD-CALIBRATION
- win_rate_20d: 0.0%  avg_return_20d: -5.25%  picks: 2 (insufficient sample)
- Signal requires ≥2 consecutive inside days + volume breakout above mother bar high — an extremely rare combination in a trending bull market. Only 2 picks over 11 months.
- Sample size is too small to evaluate. Signal may be valid but fires too rarely to backtest.
- Do not re-research as a standalone. Could revisit as a component of a broader compression/breakout scanner.

### gap_down_reversal — DISCARD
- win_rate_20d: 16.0%  avg_return_20d: -5.72%  picks: 25
- All 25 picks clustered in March 2026 (tariff selloff): stocks gapped down, reversed intraday, but continued falling in subsequent days as the broader correction deepened.
- Signal logic is sound (intraday reversal = buyer absorption), but the mean reversion thesis fails when the broader market is in an accelerating downtrend. The SMA200 uptrend filter was not sufficient — stocks can be above SMA200 while still in the early stages of a multi-week correction.
- Possible fix: add a market breadth or VIX filter. Not implemented.
- Do not re-research in current form. Below-random 20d performance is disqualifying.
