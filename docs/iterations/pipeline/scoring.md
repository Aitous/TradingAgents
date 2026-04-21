# Pipeline Scoring & Ranking

## Current Understanding
LLM assigns a final_score (0-100) and confidence (1-10) to each candidate.
Score and confidence are correlated but not identical — a speculative setup
can score 80 with confidence 6. The ranker uses final_score as primary sort key.
No evidence yet on whether confidence or score is a better predictor of outcomes.

## Evidence Log

### 2026-04-12 — Cross-scanner calibration analysis
- All scanners show tight calibration: avg score/10 within 0.5 of avg confidence across all scanners. No systemic miscalibration.
- The current `min_score_threshold=55` in `discovery_config.py:52` allows borderline candidates (GME social_dd score 56, TSLA options_flow 60, FRT early_accumulation 60) into final rankings.
- These low-scoring picks carry confidence 5-6 and are explicitly speculative. Raising threshold to 65 would eliminate them without losing high-conviction picks.
- insider_buying has 136 recs — only 1 below score 60 (score 50-59 bucket had 1 entry). Raising to 65 would trim ~15% of insider picks (the 20 in 60-69 range).
- Confidence: medium

### 2026-04-20 — New scanner output volume + cross-day persistence
- volatility_contraction_breakout produced 4 of 8 final picks (50%) on Apr 19. This is not a noise concern per se (all theses are specific), but it indicates the ATR ≤ 0.75 threshold may be catching many stocks simultaneously during low-volatility regimes.
- PSTG appeared in both Apr 18 and Apr 19 from volatility_contraction_breakout with the same ATR ratio (0.74). Identical thesis, same score range (76-77). This is the cross-day persistence issue — unlike short_squeeze where urgency builds, a breakout that didn't trigger overnight is either still valid (slow resolution) or a stale detection.
- Overall statistics (n=684): 39.3% 1d win rate, 44.4% 7d win rate. No change from prior iteration.
- early_accumulation confirmed poor: 43.8% 7d, -7.6% avg 30d. Code change implemented to raise bar.
- Confidence: high (statistics stable; new scanner observations noted for tracking)

## Pending Hypotheses
- [ ] Is confidence a better outcome predictor than final_score?
- [x] Does score threshold >65 improve hit rate? → Evidence supports it: low-score candidates are weak (social sentiment without data, speculative momentum). Implement threshold raise to 65.

### 2026-04-12 — P&L outcome analysis (mature recs, 2nd iteration)
- news_catalyst: 0% 7d win rate, -8.79% avg 7d return (7 samples). Worst performing strategy by far.
- social_hype: 14.3% 7d win rate, -4.84% avg 7d, -10.45% avg 30d (21-22 samples). Consistent destroyer.
- social_dd: surprisingly best long-term: 55% 30d win rate, +0.94% avg 30d return — only scanner positive at 30d.
- minervini: best short-term signal but small sample (n=3 for 1d tracking).
- **Critical gap confirmed**: `format_stats_summary()` shows only top 3 best strategies. LLM never sees news_catalyst (0% 7d) or social_hype (14.3% 7d) as poor performers.
- Confidence: high

### 2026-04-14 — P&L update (mature recs, 3rd iteration: Apr 3-9)
- news_catalyst: still 0% 7d win rate, -8.37% avg 7d (8 samples, +1). WTI appeared Apr 3 (score=72) and Apr 6 (score=78) despite 0% track record. Ranker prompt updated: news_catalyst now explicitly flagged as "AVOID by default" with 0% win rate stated in criteria section.
- social_hype: 18.2% 7d win rate (updated from 14.3%), -4.58% avg 7d (22 samples). LLY scored 82 and AI scored 80 from social_hype in Apr 3-9 — overconfident. Ranker prompt already warns "SPECULATIVE" for social_hype.
- short_squeeze: 7d 60% win rate confirmed; **30d 30%** — signal degrades sharply. Noted in short_squeeze.md.
- insider_buying staleness: 50% of insider_buying picks in Apr 3-9 were stale repeats (PAGS×4, ZBIO×4, HMH×3). Staleness suppression filter implemented in `insider_buying.py`.
- Overall pipeline: 626 tracked recs, 41.9% 7d win rate, 34.7% 30d win rate, -2.79% avg 30d return.
- Confidence: high

### 2026-04-17 — P&L update (statistics.json, n=664 total)
- Overall: 664 total recs, 39.2% 1d win rate, 44.4% 7d win rate, 36.7% 30d win rate, -0.71% avg 7d.
- Headline improvement: analyst_upgrade 7d 55.9% (was 50%), insider_buying 30d 41.6% (was 32.8%), earnings_play 7d 47.6% (was 37.7%).
- Earnings_play 7d improvement likely market-context-driven (April tariff-pause recovery), not structural.
- Insider_buying 30d improvement consistent with staleness filter (suppress_days=2, implemented Apr 14).
- Staleness filter gap found: suppress_days=2 missed FUL Apr 9 to Apr 12 (3-day gap). Fixed to suppress_days=3.
- Minervini: 100% win rate persists (n=3 measured out of 15 total). +7.19% avg 7d. Extraordinary but tiny measured sample.
- Confidence: high
