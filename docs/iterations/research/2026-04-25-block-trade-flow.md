# Research: Institutional Block Trade Flow (Lit Exchanges)

**Date:** 2026-04-25
**Mode:** autonomous

## Summary

Investigated whether unusual block trade activity on lit exchanges (NYSE/NASDAQ) could serve as a complementary signal to existing `dark_pool_flow` scanner. Finding: while block trades do exhibit information content, recent academic evidence suggests they predict **timing/duration** of price movements, not directional returns. Combined with data availability constraints (FINRA TRF requires paid vendor agreement with no free alternative), this signal does not meet implementation thresholds.

## Sources Reviewed

- **Zhu (2012, NY Fed)**: Dark pool flow predicts returns; strong-signal traders use lit exchanges, moderate-signal traders route to dark pools.
- **Buti, Rindi & Werner (2022, Financial Management)**: Dark pool retail order imbalance predicts future returns; effect is non-linear and regime-dependent.
- **NBER Working Paper w30366 (2024)**: Block trade analysis on NYSE TAQ data (S&P 100, 2019-2020). Key finding: `VolumeMax` (max shares per trade) is **not predictive of price direction**; predicts *timing* (inter-trade duration) only. Quote: "variables VolumeAll, VolumeAvg, VolumeMax are consistently not predictive [of returns]."
- **ScienceDirect (2024, high-frequency markets)**: Anomaly detection in limit order books; block trades identified via 2-5x volume threshold or Z-score >2.
- **FINRA Market Transparency**: TRF data requires vendor agreement + licensing fees; free aggregate data lacks granular block-level details. No real-time free source identified.
- **Paid alternatives**: FlowAlgo ($149-199/mo), WhaleStream ($69/mo) offer real-time block monitoring but require subscription.

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ⚠️ | FINRA TRF requires paid vendor agreement (no free alternative like meridianfin.io for dark pools). Could derive from OHLCV tape via volume anomaly detection, but loses real-time advantage and bid/ask directionality. |
| Complexity | moderate | 2-4h: Volume anomaly detector (Z-score or MA-based threshold) + scanner class + config entry. Moderate if using OHLCV; larger if integrating paid API. |
| Signal uniqueness | medium overlap | Complements `dark_pool_flow` (off-exchange) by covering on-exchange blocks, but both detect same institutional conviction signal. `options_flow` already captures institutional appetite. |
| Evidence quality | ⚠️ | Conflicting: Zhu/Buti show dark pool predictability; NBER 2024 finds block trades predict *timing only*, not returns. Academic consensus weak for directional discovery signal. |

## Cross-Reference Existing Knowledge

**Already covering institutional flow:**
- `dark_pool_flow`: Off-exchange block anomalies (meridianfin.io, free, Z-score ≥2.0)
- `options_flow`: Unusual options activity on chains
- `short_squeeze`: SI+DTC flow + price extension signal
- `insider_buying` + `insider_cluster_buying`: Form 4 transaction flow

**Confluence already detected:**
- insider_buying + momentum: 74.3% WR vs 47.7% standalone
- momentum + options_flow: 59.1% WR vs 46.8% standalone

**Saturation note:** 2026-04-23 research concluded pipeline has "exhausted accessible free non-OHLCV signals"; this research confirms that block trade flow is either (a) data-constrained (paid API required) or (b) evidence-weak (timing-only prediction per NBER 2024).

## Recommendation

**Skip — data constraint + weak evidence combination.**

**Primary blocker:** FINRA TRF block trade data requires paid vendor agreement with no free scrapable alternative (unlike meridianfin.io for dark pools). Could attempt to derive from OHLCV tape using volume anomalies, but NBER 2024 paper shows block size predicts *timing*, not *direction* — reducing utility for a discovery scanner focused on return prediction.

**Secondary concern:** Signal is redundant with existing `dark_pool_flow` (both detect institutional conviction). Confluence with dark_pool_flow would require implementation first, adding complexity.

## Alternative Paths (If Pursuing Paid Data ROI)

If business model supports paid data tiers:
1. **Integrate FlowAlgo API** ($149-199/mo): Real-time block alerts + dark pool + options flow consolidated.
2. **Backtest block-trade-as-timing-filter**: Rather than standalone scanner, use blocks as a "confirm timing" filter on existing signals (e.g., "buy momentum candidate if block trade spike in last 2 days" = convergence of fundamental edge + institutional timing).
3. **Volume anomaly detector (free OHLCV path)**: Derive from `volume_dry_up` logic (inverted); flag 5x volume spikes. Less precise than real-time blocks but free and already integrated.

