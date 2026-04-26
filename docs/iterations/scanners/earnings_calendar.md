# Earnings Calendar Scanner

## Current Understanding
Identifies stocks with earnings announcements in the next N days. Pre-earnings
setups work best when combined with options flow (IV expansion) or insider activity.
Standalone earnings calendar signal is too broad — nearly every stock has earnings
quarterly.

## Evidence Log

### 2026-04-12 — P&L review (earnings_play strategy, 65 tracked recs)
- Note: appears in statistics.json as "earnings_play" not "earnings_calendar". The scanner feeds this strategy.
- Win rates: 38.1% 1d, 37.7% 7d, 46.2% 30d. Avg returns: -0.33% 1d, -2.05% 7d, -2.8% 30d.
- The 30d win rate (46.2%) is better than 7d (37.7%) — unusual pattern suggesting the binary earnings event resolves negatively short-term but some recover.
- Recent runs: 4 candidates (APLD, SLP, FBK, FAST) all scored 60-75 — consistently lowest-scoring scanner in recent runs. APLD (score=75, high short interest 30.6%) is the strongest type of earnings_play setup.
- Avg scores in recent runs: 67 — below the 70 average for other scanners. The ranker is appropriately skeptical of this scanner.
- Confidence: high (65 samples with clear trend)

### 2026-04-17 — P&L update (n=67, statistics.json)
- 7d win rate jumped from 37.7% → 47.6%. avg_return_7d improved from -2.05% → -0.23%. Significant uplift.
- 30d win rate 45.3%, avg_return_30d -2.73%. 30d still weak but 7d now near coin-flip.
- Newly mature picks: FBK (Apr 10, score=60) and FAST (Apr 12, score=68). Both pre-earnings setups.
- The 7d improvement likely reflects April market recovery (tariff pause Apr 9) rather than scanner improvement — picks from late March/early April that were underwater at the 7d mark later recovered when the market bounced. Context-dependent improvement.
- Earnings_play consistently scores lowest of active scanners (60-75 range) — the ranker's skepticism is well-calibrated.
- Confidence: medium (7d improvement appears market-context-driven, not structural; pattern may not persist in flat/bear market)

### 2026-04-26 — P&L autopsy (earnings_play = earnings_calendar, n=48 7d outcomes)
- 7d win rate: 37.5% (18/48), avg return: -1.22% — AUTOPSY TRIGGERED. Worst performer in pipeline alongside early_accumulation.
- Root cause analysis: max_days_until_earnings=7 pulls in earnings events too far out (4-7 days), where pre-earnings volatility setup hasn't formed yet. These are just ordinary trend trades with event-day risk premium baked in.
- Secondary issue: pure event-day plays without pre-earnings accumulation confirmation are scored too high (CRITICAL priority), leading to overconfident trading on binary event risk.
- Fix applied: (1) Reduce max_days_until_earnings from 7 → 3 days (closer to event = setup more likely intact), (2) Enrich only 2-3d candidates (line 69 guard), (3) Cap unmatchable candidates (no accumulation signal) at score 70 (HIGH/MEDIUM boundary, not CRITICAL).
- Expected impact: Eliminate immature setups (4-7d window) and reduce binary event-day trades without confirmation.
- Confidence: medium (window tightening is sound, but 7+ days needed to validate if 3-day window catches genuine setups)

## Pending Hypotheses
- [ ] Does requiring options confirmation alongside earnings improve signal quality?
- [ ] Does short interest >20% pre-earnings produce better outcomes than <10%? APLD (30.6% SI) scored highest in recent runs — worth tracking.
- [ ] **Does tightening max_days_until_earnings from 7 to 3 days move WR from 37.5% toward >45%?** — forward testing started 2026-04-26
