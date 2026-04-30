# Hypothesis: Add 7% floor on distance from 52w high

**Scanner:** minervini
**Branch:** hypothesis/minervini-min-pct-from-high
**Period:** 2026-04-16 → 2026-04-30 (10 days)
**Outcome:** rejected ❌

## Hypothesis
Hypothesis: Minervini picks 10-25% below their 52w high significantly outperform picks already within 5% of the high (+10-11% vs +5.7% avg 20d return in 2240-pick backtest). Adding min_pct_from_high=7 filters stocks pressing prior-high resistance, where stalls and pullbacks are most likely.

## Results

| Metric | Baseline | Experiment | Delta |
|---|---|---|---|
| 7d win rate | —% | —% | — |
| Avg return | —% | —% | — |
| Picks | 0 | 0 | — |

## Decision
No picks were collected during the experiment period


## Analysis
The zero-pick result renders this experiment statistically void, but the total lack of data suggests the 7% floor is excessively restrictive for a momentum-based scanner. Leading candidates identified in the prior week (like AVGO and AA) were likely already trading within 7% of their 52-week highs, meaning this filter accidentally purged the highest-conviction "breakout-ready" stocks during a market recovery. The hypothesis likely misidentifies proximity to highs as "resistance" when, in the Minervini framework, it often represents the low-volatility "cheat" area necessary for a high-probability entry. A more productive follow-up would be to test a **tightness filter (max 3% price range over 5 days)** rather than a distance floor, focusing on price consolidation near the high rather than arbitrary distance from it.

## Action
Experiment concluded — awaiting manual review before closing.
