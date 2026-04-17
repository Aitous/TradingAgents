# Minervini Scanner

## Current Understanding
Implements Mark Minervini's SEPA (Specific Entry Point Analysis) criteria: stage 2
uptrend, price above 50/150/200 SMA in the right order, 52-week high proximity,
RS line at new highs. Historically one of the highest-conviction scanner setups.
Works best in bull market conditions; underperforms in choppy/bear markets.

## Evidence Log

### 2026-04-12 — P&L review
- 7 tracked recommendations; 3/3 1-day wins measured, avg +3.68% 1d return.
- No 7d/30d data yet (too recent), but early 1d signal is strongest of all scanners.
- Recent week (Apr 6-12): 7 candidates produced — ALB (×2), AA (×2), AVGO (×2), BAC. Consistent quality signals.
- AA reappeared Apr 8 (score=68) then Apr 12 (score=92) — second appearance coincided with Morgan Stanley upgrade catalyst, showing scanner correctly elevated conviction when confluence added.
- Confidence calibration: Good (cal_diff ≤ 0.8 across all instances).
- Confidence: medium (small sample size, market was volatile Apr 6-12 due to tariff news)

### 2026-04-12 — Fast-loop (2026-04-08 to 2026-04-12)
- minervini was top-ranked in 3 of 5 runs — highest hit-rate at #1 position of any scanner this week.
- AVGO ranked #1 on Apr 10 and Apr 11 (score 85, conf 8 both days) — persistent signal.
- Apr 2026 is risk-off (tariff volatility), yet Minervini setups are still leading. Contradicts bear-market underperformance assumption.
- Apr 12 AA thesis was highly specific: RS Rating 98, Morgan Stanley Overweight upgrade, earnings in 4 days, rising OBV. Good signal clarity.
- Confidence: high

### 2026-04-17 — P&L update (statistics.json, n=15 total picks, 3 measured)
- Statistics now show 15 total minervini picks; 3 measured with 1d/7d outcomes.
- Win rates still 100% at 1d and 7d (3/3). Avg returns: +5.16% 1d, +7.19% 7d. Outstanding.
- Newly observed: AVGO appeared Apr 10 AND Apr 11 (score=85, conf=8 both days). Same thesis. Positive persistence signal.
- AA Apr 12 scored 92/100 (highest score ever in pipeline) — RS 98/100, Morgan Stanley upgrade, earnings 4 days out. Extremely specific thesis.
- BAC Apr 11 (score=65, conf=6): lower-conviction Minervini, sector-rotation driven. Represents the tail end of the score distribution for this scanner.
- 12 picks from Apr 10-12 not yet measured (7d windows close Apr 17-19). Will be the first meaningful batch for 7d validation.
- Confidence: high (100% win rate holds, but sample inflated by market recovery week; need more diverse market context)

## Pending Hypotheses
- [ ] Does adding a market condition filter (S&P 500 above 200 SMA) improve hit rate? Early evidence (Apr 2026 volatile market, still producing top picks) suggests filtering by market condition may hurt recall.
- [ ] Does a second appearance of the same ticker (persistence across days) predict higher returns than first-time appearances? AVGO Apr 10+11 now a test case.
- [ ] Do earnings-nearby Minervini setups (within 5 days) underperform? AA Apr 12 (earnings 4 days out, score=92) is a live test case — outcome due ~Apr 17-19.
