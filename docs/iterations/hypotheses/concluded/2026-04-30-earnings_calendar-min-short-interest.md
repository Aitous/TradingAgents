# Hypothesis: Require short interest >= 20% for earnings plays

**Scanner:** earnings_calendar
**Branch:** hypothesis/earnings_calendar-min-short-interest
**Period:** 2026-04-16 → 2026-04-30 (10 days)
**Outcome:** rejected ❌

## Hypothesis
Hypothesis: earnings_calendar has a 37.7% 7d win rate and -2.05% avg return — worst performer. APLD (30.6% SI) was consistently the highest-scoring earnings candidate in recent runs. Requiring short interest >= 20% forces the scanner to surface only setups where a positive earnings surprise can trigger forced short covering, converting a weak calendar signal into a squeeze catalyst.

## Results

| Metric | Baseline | Experiment | Delta |
|---|---|---|---|
| 7d win rate | —% | —% | — |
| Avg return | —% | —% | — |
| Picks | 0 | 0 | — |

## Decision
No picks were collected during the experiment period


> ⚠️ **Baseline drift detected:** `tradingagents/dataflows/discovery/scanners/earnings_calendar.py` changed 1 commit on main since 2026-04-16 (latest: b335bb8 learn(iterate): 2026-04-26 — automated iteration run (#34)). Baseline picks may reflect the updated code — interpret the delta with caution.

## Analysis
The sample size is non-existent (n=0), making the programmatic rejection a matter of insufficient data rather than a failure of the thesis. The 20% short interest threshold is likely too restrictive for a 10-day window, especially when combined with the mid-experiment tightening of the earnings window from 7 to 3 days. While high-SI "squeeze" plays like APLD are high-quality setups, this filter converts the scanner into a rare "sniper" tool that cannot be validated without a much longer observation period or a broader stock universe. 

**Follow-up Hypothesis:** Lower the short interest threshold to 10% while maintaining the 3-day earnings window to determine if moderate short pressure provides a better balance of trade frequency and signal quality.

## Action
Experiment concluded — awaiting manual review before closing.
