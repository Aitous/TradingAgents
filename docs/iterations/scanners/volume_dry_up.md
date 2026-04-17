# Scanner: Volume Dry-Up (VDU) Pocket Pivot

**Module:** `tradingagents/dataflows/discovery/scanners/volume_dry_up.py`
**Pipeline:** momentum
**Implemented:** 2026-04-16

## Signal Logic

Based on Mark Minervini's SEPA methodology and Chris Kacher / Gil Morales "Pocket Pivot"
(IBD lineage, "Trade Like an O'Neil Disciple", 2010).

Five conditions must all fire simultaneously:
1. **Full trend template**: price > SMA50 > SMA150 > SMA200 (institutionally aligned uptrend)
2. **52-week structure**: within 25% of 52-week high AND at least 25% above 52-week low
3. **Volume Dry-Up (VDU)**: today's volume < 60% of 50-day average volume (non-distribution confirmed)
4. **Up-close bar**: today's close > yesterday's close (demand present)
5. **Pocket Pivot**: today's volume > maximum volume of any down-close day in the prior 10 sessions

The Pocket Pivot is the entry trigger: it confirms that demand absorbed all recent supply without
the stock needing to decline further. The VDU (condition 3) ensures we are catching a
non-distribution consolidation, not a failing trend.

## Backtest Results (walk-forward, 2y OHLCV cache)

| Metric | Value |
|--------|-------|
| Total picks | 20 |
| Unique tickers | 13 |
| Win rate 1d / 5d / 10d / 20d | 40% / 60% / 70% / 80% |
| Avg return 20d | +3.26% |
| Median return 20d | +2.39% |

**Classification:** PROMOTE

⚠️ **Small sample caveat:** 20 picks across 224 sim days (0.09 picks/day). At n=20,
the 95% CI on 80% WR is approximately 56–94%. The signal is inherently rare — which is
what gives it precision — but the sample size warrants caution.

**Slow signal annotation:** WR-1d=40% (below-random at the 1-day horizon). The edge
only materializes at 10d+ (70% WR) and is strongest at 20d (80% WR). **Minimum hold: 10 days.**
Do not use short-term exit rules with this scanner.

## Current Understanding

The signal is self-enforcing in its selectivity: requiring all five conditions simultaneously
leaves roughly 0.09 qualifying stocks per day in a 1003-ticker universe. This precision
appears to be the source of edge — each pick is a stock where institutions are visibly
non-distributing (VDU), the trend is intact across three SMAs, the base is tight (within 25%
of highs), AND demand re-emerged on the current bar (pocket pivot). The progressive WR
improvement (40%→60%→70%→80%) across horizons is characteristic of a trend continuation
signal where the thesis takes time to play out.

## Pending Hypotheses

- [ ] Test with stricter VDU threshold: volume < 50% (currently 60%) of 50d avg
- [ ] Test extending pocket_pivot_lookback from 10 to 15 days
- [ ] Analyze whether WR improves by requiring 2+ consecutive VDU days before pocket pivot
- [ ] Sub-period analysis: check performance in 2025-Q1 bear market sub-period (tariff correction)
- [ ] Expand sample: monitor live picks until n=50, then re-classify confidence level

## Evidence Log

### 2026-04-16 — backtest (walk-forward, 20 picks, 224 sim days)
- 20 total picks across 13 unique tickers over 224 simulation days (0.09/day average)
- WR progression: 1d=40% → 5d=60% → 10d=70% → 20d=80% — clear trend-holding profile
- Avg return at 20d: +3.26%, median +2.39% — strong and consistent
- Signal fires rarely: mostly during pullback consolidations in strong bull markets (2025)
- Notably sparse in late 2025 correction period (tariffs) — confirming trend template gate works
- Confidence: moderate-high (pass PROMOTE thresholds but small sample)
