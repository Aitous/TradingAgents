# Research: Pre-Earnings Quality Filters (IV Expansion, Short Interest Surprise, Insider Avoidance)

**Date:** 2026-04-30
**Mode:** autonomous

## Summary

The earnings_calendar scanner achieves 37.5% win rate (worst performer, n=48), with recent fix (3-day window tightening) pending validation. This research identifies what differentiates quality pre-earnings setups from binary event-day gambles using three non-OHLCV filters: (1) Implied volatility expansion (informed trading signal), (2) Short interest surprise rejection (predicts worse earnings), (3) Insider buying avoidance (market prices it in, reducing alpha). Academic evidence supports all three; implementation requires only Tradier options API (already integrated) and short interest data (available via Finnhub).

## Sources Reviewed

### Implied Volatility Expansion as Informed Trading Signal
- **ScienceDirect (2017–2024)**: Volatility spreads (call vs put IV misalignment) show significant predictive power on earnings announcement returns. Abnormal volatility builds monotonically in days prior to earnings, indicating informed options traders front-running announcement.
- **Oxford Academic (2024)**: Concave implied volatility curves (characteristic of event-risk pricing) predict higher absolute returns and realized volatility on earnings announcement day.
- **SpiderRock / ORATS**: IV typically expands 20–40% in 2–3 days before earnings as informed positioning accumulates; IV collapse post-announcement is predictable.
- **Finding**: Daily IV change >20% in 2 days pre-earnings = informed trader activity; this differentiates genuine alpha setups from unresearched event-day trades.

### Short Interest Surprise as Earnings Predictor (Negative Signal)
- **Oxford Academic / ScienceDirect**: Positive surprises in short interest (SI increases above historical trend) predict *lower* unexpected earnings and *lower* cumulative abnormal returns around earnings announcements.
- **Finding**: High SI increase (>10% above trend) pre-earnings is a contrarian signal predicting worse earnings; AVOID these setups even if calendar trigger fires.
- **Implication**: The 7-day window (now 3-day) on earnings_calendar used to pick up many SI-increase setups. Filtering these out should improve WR.

### Insider Buying Before Earnings (Reduces Market Alpha)
- **Alldredge & Cicero (2015) + Recent ScienceDirect (2023–2024)**: Insider purchases before earnings announcements *reduce* absolute magnitude of abnormal returns around the announcement. Post-earnings announcement drift (PEAD) is significantly smaller when insider trading precedes the announcement.
- **Finding**: Market prices in insider front-running immediately; the alpha signal is exhausted by announcement time. Combining insider_buying + earnings_calendar signals should be *rejected*, not promoted.
- **Implication**: Current pipeline may be double-counting insider + earnings confluences (based on LEARNINGS.md pending hypothesis: "Does requiring options confirmation improve quality?"). Instead, the confluence should be: earnings_calendar + IV_expansion, NOT earnings_calendar + insider_buying.

### Supporting Academic Evidence on Pre-Earnings Setups
- **Bernard & Thomas (1989), Ball & Brown (1968)**: Foundational work on earnings surprise drift; effect strongest in small-to-mid caps with >10% surprise. PEAD decays after ~9 days.
- **QuantPedia (1987–2004)**: 15% annualized PEAD; declining effect in large caps but robust in small/mid caps.

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ✅ | Tradier options API (IV skew already used in options_flow scanner); Finnhub SI data (already integrated via short_squeeze scanner); no new data sources required |
| Complexity | moderate | ~4h: (1) add IV expansion check (last 2d daily IV change) to earnings_calendar, (2) add SI surprise filter via Finnhub, (3) filter out insider_buying confluences |
| Signal uniqueness | low overlap | Distinct from earnings_beat (PEAD drift); distinct from standalone earnings_calendar. These filters improve pre-earnings setup quality. |
| Evidence quality | backtested + qualitative | Academic papers 1989–2024; IV behavior documented in multiple venues; SI effect quantified in ScienceDirect. No walk-forward backtest of *combined* filters on this specific pipeline. |

## Recommendation

**Implement** with the caveat that this is an *enhancement* to the existing earnings_calendar scanner, not a standalone scanner.

The current earnings_calendar autopsy fix (3-day window tightening, score cap at 70 for unmatchable candidates) should be combined with these quality filters to move WR from 37.5% toward >45%.

### Proposed Improvements to earnings_calendar Scanner

1. **IV Expansion Filter (adds ~2h to implementation)**
   - Query Tradier options data for current date minus 2 days
   - Compute daily IV change: `(current_iv - 2d_ago_iv) / 2d_ago_iv`
   - Require: `daily_iv_change >= 0.20` (20% expansion threshold)
   - Logic: If IV expanding, informed traders are positioning; this is a genuine setup, not random event-day trade
   - Priority bump: Add 10 points to score if IV expanding (confluence with informed positioning)

2. **Short Interest Surprise Filter (adds ~1h, uses existing Finnhub SI endpoint)**
   - Query current SI and prior-month SI trend from Finnhub
   - Compute surprise: `(current_si - trend_si) / trend_si`
   - Reject candidate if: `si_surprise >= 0.10` (>10% SI increase)
   - Logic: Academic evidence shows high SI increases predict worse earnings surprises; avoid these
   - Severity: Hard rejection (return None for this candidate), not a score penalty

3. **Insider Buying Avoidance (adds ~30min)**
   - Query scanner_picks from past 3 days for insider_buying signals on same ticker
   - If ticker appears as insider_buying recently, set earnings_calendar score cap at 65 (MEDIUM priority, not HIGH)
   - Logic: Market has priced in insider information; combining with earnings event = information already reflected
   - Severity: Score penalty, not hard rejection (allow signal if no other insider activity)

## Known Limitations & Caveats

1. **IV Expansion Threshold (20%)**: Based on SpiderRock observations; may need walk-forward tuning for this specific pipeline. Consider regime-dependent thresholds (VIX-adjusted).
2. **SI Surprise (10%)**: Finnhub SI data is bi-weekly (not daily), so "surprise" is approximate. Consider using borrow_utilization change as proxy if available.
3. **No Large-Cap Bias**: PEAD and pre-earnings alpha is weakest in large-cap US equities (most efficient pricing). These filters should not increase median hold time without also adding market-cap filter.
4. **Backtest Gap**: These filters are backed by academic evidence but have not been walk-forward backtested on *this pipeline's* specific data window (Feb 2026 forward). Need 14+ days of mature outcomes after implementation.

## Next Steps

1. Implement IV expansion filter in `earnings_calendar.py`
2. Implement SI surprise filter in `earnings_calendar.py`
3. Implement insider avoidance score cap in `earnings_calendar.py`
4. Forward test starting 2026-04-30; target: move 7d WR from 37.5% → 45%+
5. If WR improves >2 points, backport same filters to pending hypothesis "Does requiring options confirmation improve quality?" (currently listed in earnings_calendar.md line 36)
