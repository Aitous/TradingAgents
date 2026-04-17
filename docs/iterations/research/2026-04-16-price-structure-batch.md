# Research Batch: Price Structure Patterns

**Date:** 2026-04-16
**Mode:** autonomous
**Candidates shortlisted:** 3

## Sources Reviewed

- **Connors / QuantifiedStrategies Substack (multiple-days-down)**: Close down 4 of last 5 days + above SMA(200) + below SMA(5). Profit factor 2.2, CAGR 3.4%, max DD 12% on SPY. Low market exposure (9%).
- **QuantifiedStrategies pullback article (via web search)**: Close above SMA(200), below SMA(20), RSI(5) < 45. Win rate 82%, CAGR 8.3%, 30% max DD during COVID. Low exposure time.
- **SetupAlpha Medium — weekly pullback backtest**: S&P 500/R1000/Nasdaq 100 2000-2025; exploits institutional weekly rebalancing. Visual charts not accessible but strategy logic clear.
- **Wilder 1978 / Fidelity / StockCharts ADX documentation**: ADX > 25 = strong trend (below 20 = weak/sideways). DI+ > DI- = bullish directional bias. Widely used threshold in systematic trend-following.
- **SSRN (multiple overnight/gap papers)**: "Price Gap Anomaly in the US Stock Market" (Plastun et al., SSRN 3461283) — gap strategy efficient but effect temporary. "What Drives Momentum and Reversal?" (Barardehi et al.) — intraday momentum distinct from overnight. Supports avoiding gap fill for daily OHLCV.
- **Quora / TheRobustTrader / EpicCtrader (gap fill statistics)**: Small gaps (<1%) fill 78-89% of time; large gaps (<5%) fill ~35-45%. Evidence that gap fill needs intraday entry/exit to capture; end-of-day gap strategies show ~2% annualized returns.
- **StocksoftResearch (gap fill backtest)**: End-of-day gap fill strategy: buy open when open >1% below prev low, exit when high reaches fill level. 80% WR for best tickers (TXN) but 2% annualized — insufficient for our pipeline.
- **QuantifiedStrategies web search results (ADX strategy)**: ADX trading strategy documented with specific backtests; CAPTCHAs blocked full access but qualitative consensus clear — ADX>25 with directional bias = strong edge filter.

## Candidate Strategies

### 1. Consecutive Down Days Reversal
**Signal logic:** Close lower than previous close for 3 or more consecutive trading days, AND current price is above SMA(200) (falling-knife guard). The "staircase down" pattern represents short-term panic selling within a longer uptrend — the same mean reversion principle as RSI(2) but triggered by count rather than indicator value.
**Academic edge:** Larry Connors "High Probability ETF Trading" (2009); SPY backtest: profit factor 2.2, CAGR 3.4%, max DD 12%, market exposure 9%. The 200-day SMA filter is critical — without it, catches downtrending stocks.
**Data requirements:** Close only — 3 days for count + 200 days for SMA(200) = 205 trading days minimum
**Proposed scanner name:** `consecutive_down_days`
**Pipeline:** mean_reversion

### 2. Pullback in Uptrend
**Signal logic:** Price above SMA(200) AND below SMA(20) — stock in confirmed long-term uptrend but short-term weakness. Entry: today's close crosses below 20-day SMA for the first time (fresh pullback entry, not already deeply extended below). Volume must be declining on the down days (institutional holders not distributing — normal profit-taking).
**Academic edge:** Multiple practitioners; CAGR 8.3%, win rate 82% documented with exact rules: close > SMA(200), close < SMA(20), RSI(5) < 45. Exit: RSI(5) > 65. Low market exposure reduces max drawdown.
**Data requirements:** Close + Volume — 20 days for SMA(20) + 200 days for SMA(200) = 205 trading days minimum
**Proposed scanner name:** `pullback_in_uptrend`
**Pipeline:** mean_reversion

### 3. ADX Trend Inception
**Signal logic:** ADX(14) rises above 25 (fresh crossing from below 25 in prior 3 days) AND DI+(14) > DI-(14) (bullish directional bias) AND price above SMA(50). ADX crossing 25 marks the transition from sideways/weak-trend to confirmed strong trend — a "trend just started" signal distinct from all other momentum scanners which track continuation.
**Academic edge:** J. Welles Wilder (1978) "New Concepts in Technical Trading Systems"; ADX>25 = strong trend threshold is the most-cited technical indicator threshold in systematic trading literature. Fidelity, StockCharts, AvaTrade all document the same threshold. Multiple TradingView community backtests show ADX-based systems outperform buy-and-hold in trending markets.
**Data requirements:** High, Low, Close — 14-period Wilder smoothing + 14-period ADX = 28+ days for ADX; SMA(50) = 55 trading days minimum
**Proposed scanner name:** `adx_trend_inception`
**Pipeline:** momentum

## Discarded Before Implementation

- **Gap Fill (gap_down_reversal)**: Requires intraday data (1-minute bars) for proper entry/exit. End-of-day backtest shows only ~2% annualized return. Gap fill rates are intraday phenomena; with daily OHLCV we can only buy at close and miss the actual fill move.
- **Gap Up Continuation**: 53% chance market closes higher after gap-up = near-random. Gap-and-go requires intraday execution in the first 30 minutes. Insufficient daily-bar edge.

## Implementation Order

1. `pullback_in_uptrend` — highest documented win rate (82%), clearest rules, strong evidence
2. `consecutive_down_days` — well-documented by Connors, distinct from existing RSI oversold (count-based vs indicator-based), fast to implement
3. `adx_trend_inception` — most computation-intensive (ADX requires Wilder smoothing), but unique signal class not covered by any existing scanner

## Backtest Discard Notes

Walk-forward backtest: 2025-04-15 → 2026-03-06 (224 sim days), 1003-ticker universe, local OHLCV parquet cache.

### consecutive_down_days — DISCARD
- picks: 2123  win_rate_20d: 53.6%  avg_return_20d: +0.83%
- Hits 10-pick limit every single day (9.5/day avg) — not selective, picks "least bad" from large pool
- Below PROMOTE-MARGINAL threshold (needs WR≥52% AND avg≥2.0%). Do not re-research.
- Root cause: "3+ consecutive down closes" in a universe of 1000 stocks fires hundreds of candidates daily;
  signal needs to be paired with a much stricter secondary filter to achieve selectivity.

### pullback_in_uptrend — DISCARD
- picks: 2107  win_rate_20d: 54.3%  avg_return_20d: +1.02%
- Hits 10-pick limit every single day (9.4/day avg) — same selectivity failure as above
- Below PROMOTE-MARGINAL threshold. Do not re-research.
- Root cause: SMA200 + RSI(5)<45 + volume decline conditions are common in a 1000-ticker universe;
  the pullback signal fires hundreds of times daily, requiring a much tighter tertiary filter.

### adx_trend_inception — DISCARD
- picks: 2191  win_rate_20d: 56.9%  avg_return_20d: +1.64%
- Hits 10-pick limit every single day (9.8/day avg) — same selectivity failure
- WR-20d of 56.9% is genuinely above random (exceeds 55% PROMOTE threshold) but avg_return_20d=1.64%
  falls short of the 3.0% PROMOTE requirement and 2.0% PROMOTE-MARGINAL requirement.
- Note: ADX computation was correct after Wilder smoothing bug was fixed (two-formula split:
  cumulative for TR/DM, average for DX→ADX). The signal direction is valid but unselective.
- Potential future path: raise adx_threshold to 30+ AND require recent all-time-high proximity
  to make this genuinely rare. Would need a complete redesign.
