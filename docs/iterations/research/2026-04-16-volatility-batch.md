# Research Batch: Volatility Contraction / Expansion

**Date:** 2026-04-16
**Mode:** directed (argument: "volatility")
**Candidates shortlisted:** 3

## Sources Reviewed

- **Bulkowski / ThePatternSite (nr7.html)**: NR7 documented with 29,021 trades, 57% win rate (bull market, up breakout), 31-day hold. Net profit $78.79/trade at $10K position. Most rigorous empirical data found.
- **FMZQuant / Medium — TTM Squeeze article**: BB(20, 2.0) inside KC(20, 1.5 ATR) = squeeze ON; 3 consecutive squeeze bars as entry filter. Signals on BTC but logic is instrument-agnostic.
- **PyQuantLab / Medium — BBKC optimization**: 243 parameter combinations tested; Sharpe >1.0 achievable on S&P 500 with 6-day period + 1.3 ATR multiplier.
- **Traderslog.com / Volatility Breakout Systems**: ATR contraction to multi-week lows reliably precedes breakouts; suggests ATR(5)/ATR(20) < 0.7 as contraction filter.
- **CSS Analytics blog**: Volatility differentials (HLV vs CCV), adaptive volatility, "Forecasting Volatility and Mean-Reversion" — qualitative support for volatility regime switching.
- **SSRN search**: V-EXPLODE paper (Manamala, SSRN 5288827) — volatility compression → directional breakout strategy; 6-year backtest 50 tickers; blocked by SSRN security but abstract confirms multi-timeframe volatility filter approach.
- **WebSearch practitioner consensus**: ATR contraction to multi-week lows → breakout is consistent finding across TradingView community, VT Markets, VolatilityBox.
- **Reddit r/algotrading + GitHub search**: TTM Squeeze widely implemented, NR7 less common but documented in quant-trading repos.

## Candidate Strategies

### 1. NR7 Breakout with Uptrend Filter
**Signal logic:** Today's High-Low range is the narrowest of the past 7 calendar days (inclusive). Add uptrend context: price above SMA(50) AND SMA(50) above SMA(200). Both moving-average conditions together ensure we're catching compression within a healthy trend, not a dying stock.
**Academic edge:** Toby Crabel (1990) "Day Trading with Short Term Price Patterns"; Bulkowski backtest: 57% win rate (bull market, up breakout), 29,021 total trades, ~31-day average hold. No lookahead — fires at end of day for next-day entry.
**Data requirements:** Close, High, Low — 7 days for NR7 + 200 days for SMA200 filter = 200 trading days minimum
**Proposed scanner name:** `nr7_breakout`
**Pipeline:** momentum

### 2. Bollinger Band Squeeze
**Signal logic:** Current Bollinger Band width is inside the Keltner Channel — i.e., `BB_upper < KC_upper AND BB_lower > KC_lower`. BB: 20-period, 2.0 std dev. KC: 20-period, 1.5 × ATR(20). Squeeze fires when this condition holds for ≥1 bar. Add uptrend context: price above SMA(50). Signal is "volatility energy coiling before release."
**Academic edge:** John Carter popularized as "TTM Squeeze"; PyQuantLab optimization study: Sharpe >1.0 on S&P 500 with 6-day BB + 1.3 ATR multiplier. The BB-inside-KC condition is also equivalent to the Keltner Channel squeeze used in institutional volatility desks.
**Data requirements:** Close, High, Low — 20-period for BB + KC + SMA(50) = 50 trading days minimum
**Proposed scanner name:** `bb_squeeze`
**Pipeline:** momentum

### 3. ATR Compression Breakout
**Signal logic:** ATR(5) / ATR(20) ≤ 0.75 (short-term volatility is compressed vs. recent baseline). Additionally, today's close is above the 10-day highest close (price trying to break out), confirming directional intent. Uptrend filter: price above SMA(50). The ratio detects when 1-week volatility has collapsed relative to 4-week baseline.
**Academic edge:** Traderslog "Volatility Breakout Systems" confirms ATR contraction → expansion as reliable cycle; VT Markets study shows combining ATR with directional signals improved profitability 34% vs directional alone (Dec 2025 cross-index backtest). Qualitative consensus across 5+ practitioner sources.
**Data requirements:** High, Low, Close — 20 days for ATR(20) + 50 days for SMA(50) = 55 trading days minimum
**Proposed scanner name:** `atr_compression`
**Pipeline:** momentum

## Discarded Before Implementation

- **Historical Volatility Ratio (HV5/HV20)**: Measures rolling std dev of log returns over 5 vs 20 days. Captures the same "compression" signal as ATR compression but on close-to-close returns. Redundant with `atr_compression`. ATR is preferred because it incorporates gaps (overnight moves) via True Range, making it more robust.
- **TTM Momentum Histogram direction confirmation**: Requires MACD-derived components layered on top of the squeeze signal. Entry direction unclear without momentum direction, risk of shorting in bull market scanner. Excluded — our pipeline only surfaces long candidates.
- **V-EXPLODE multi-timeframe approach**: SSRN paper was paywalled. Multi-timeframe (daily + weekly) adds complexity beyond a single OHLCV lookback. Deferred until paper is accessible.

## Implementation Order

1. `nr7_breakout` — strongest empirical backing (57% WR, 29K trades, Toby Crabel). Simplest signal — 7 values, one comparison. Fastest to implement and validate.
2. `bb_squeeze` — well-known, reproducible signal with practitioner-validated parameters. Moderate complexity (BB + KC calculation).
3. `atr_compression` — most general and parameter-sensitive. Implement last so threshold tuning is informed by NR7/BB results.

---

## Backtest Results (walk-forward, 2026-04-16, 1003 tickers, 224 trading days)

| Scanner | Picks | WR-1d | WR-5d | WR-10d | WR-20d | Avg-20d | Decision |
|---------|-------|-------|-------|--------|--------|---------|----------|
| atr_compression | 738 | 50.3% | 56.1% | 59.2% | 59.3% | +1.88% | **PROMOTE-MARGINAL** |
| nr7_breakout | 2158 | 49.2% | 53.4% | 55.3% | 53.8% | +1.15% | DISCARD |
| bb_squeeze | 2240 | 50.8% | 51.2% | 51.4% | 51.2% | +0.54% | DISCARD |

## Backtest Discard Notes

### NR7 Breakout — DISCARD
- win_rate_20d: 53.8% | avg_return_20d: +1.15% | picks: 2158
- Scanner hits its 10-pick limit on almost every trading day (2158 picks / 224 days = 9.6/day avg) — not selective enough. Pure NR7 in uptrend has too many occurrences across 1003 tickers to produce meaningful edge.
- Bulkowski's 57% WR applied to individual stocks in bull market; the scanner picked up NR7 patterns regardless of squeeze severity (range_ratio up to 0.99 qualifies).
- **Recommended fix before reintroduction:** Add `range_ratio_max=0.5` (only tightest compressions) + require volume below 20-day average (stocks going quiet). This should reduce picks to ~2-3/day and concentrate edge.
- Do not re-research NR7 as a concept — revisit as a hypothesis with tighter thresholds.

### BB Squeeze — DISCARD
- win_rate_20d: 51.2% | avg_return_20d: +0.54% | picks: 2240
- Near-random performance. Scanner also hits limit daily (2240 / 224 = 10/day exactly).
- The BB-inside-KC condition is frequently true for slow-moving, stable large-caps — exactly the stocks that don't break out. Without momentum confirmation (e.g., TTM histogram value > 0 = expansion starting), the squeeze state alone is ~50/50.
- **Recommended fix before reintroduction:** Require squeeze to release in the same bar (BB upper crosses above KC upper on the day), plus positive momentum (close > prior 20-day EMA). This turns it from a "still in squeeze" scanner into a "squeeze just fired" scanner.
