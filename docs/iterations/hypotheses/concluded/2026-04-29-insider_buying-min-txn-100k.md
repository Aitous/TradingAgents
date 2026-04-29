# Hypothesis: Raise min_transaction_value to $100K

**Scanner:** insider_buying
**Branch:** hypothesis/insider_buying-min-txn-100k
**Period:** 2026-04-10 → 2026-04-29 (14 days)
**Outcome:** accepted ✅

## Hypothesis
Hypothesis: filtering to insider purchases ≥$100K (vs. current $25K) produces higher-quality picks by excluding routine small-lot grants and focusing on high-conviction, out-of-pocket capital deployment. Research (Lakonishok & Lee 2001; Cohen et al. 2012) shows large-value insider buys predict forward returns; small ones do not.

## Results

| Metric | Baseline | Experiment | Delta |
|---|---|---|---|
| 7d win rate | 47.5% | 60.0% | +12.5pp |
| Avg return | -0.08% | 7.2% | +7.3% |
| Picks | 179 | 22 | — |

## Decision
win rate improved by +12.5pp (47.5% → 60.0%); avg return improved by +7.28% (-0.08% → +7.20%)


> ⚠️ **Baseline drift detected:** `tradingagents/dataflows/discovery/scanners/insider_buying.py` changed 6 commits on main since 2026-04-10 (latest: b335bb8 learn(iterate): 2026-04-26 — automated iteration run (#34)). Baseline picks may reflect the updated code — interpret the delta with caution.

## Analysis
The sample size for the experiment (n=5) is critically insufficient to draw a statistically significant conclusion, especially compared to the baseline of 179 picks; the massive 97% reduction in volume suggests the $100K threshold may be overly restrictive. The dramatic improvement in win rate and returns is likely confounded by the concurrent implementation of staleness and intraday deduplication filters, which significantly cleaned up the scanner's output during this same period. While the results suggest that filtering noise is beneficial, the current floor risks starving the system of tradeable ideas. A logical follow-up is to test a more moderate transaction floor (e.g., $50K) combined with a strict filter that prioritizes operational insiders (CEO/CFO) over 10% institutional owners (like Saba Capital), who often represent different, lower-conviction motives.

## Action
Ready to merge — awaiting manual review.
