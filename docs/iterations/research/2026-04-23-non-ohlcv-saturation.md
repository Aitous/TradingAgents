# Research: Non-OHLCV Signal Saturation Analysis

**Date:** 2026-04-23
**Mode:** autonomous

## Summary

Surveyed research opportunities across non-OHLCV data sources (options sentiment, institutional flows, market breadth, borrow costs, earnings catalysts) to identify gaps. Finding: the pipeline has already researched and implemented the high-confidence signals accessible with free data. Remaining opportunities are either data-constrained (paid APIs), operationally stale (45+ day lags), or already confluenced in the system. Recommendation: shift focus from new signal research to evaluating implemented-but-unevaluated scanners (`dark_pool_flow`, `insider_cluster_buying`) and exploring paid data sources.

## Sources Reviewed

- **IV Skew / Term Structure**: Bedendo & Hodges (2024, FDIC WP); shows multi-factor IV dynamics (level, slope, curvature, skewness) but requires paid historical IV data (OptionMetrics, etc.)
- **Borrow Cost Spikes**: ScienceDirect (2025) on short squeeze likelihood; shows 80%+ accuracy but requires ORTEX paid API ($$$) or FINRA scraping (high friction)
- **13F Institutional Accumulation**: Cohen et al. (2010), Di Mascio & Lines (2017); alpha from institutional position increases decays over 135-day lag (too stale for discovery) and has eroded as retail adoption grew
- **Put-Call Ratio Extremes**: Cboe/Britannica; mixed academic evidence — high PCR (extreme fear) statistically significant but low PCR underperformance not significant
- **Earnings + Options Activity Confluence**: ORATS/SpiderRock; IV spikes 5-10 pts before earnings, unusual volume clusters around earnings dates; requires live options chain access (not integrated in pipeline)
- **Market Breadth Regime Filters**: AmberData; VIX term structure and A/D lines modulate signal timing; existing gap_down_reversal research noted "add market-breadth filter" but this is enhancement to existing scanners, not new research
- **Reddit/Social Sentiment**: r/algotrading search yielded no dedicated IV strategy threads; semantic_news, reddit_dd, reddit_trending stubs exist but require continuous monitoring (more operational challenge than research)

## Fit Evaluation

| Signal | Data Source | Lag | Complexity | Uniqueness | Evidence | Verdict |
|--------|-------------|-----|-----------|-----------|----------|---------|
| IV Skew (surface changes) | OptionMetrics/IVRank | Real-time | Moderate | Low | Academic (2024) | ❌ Paid ($) |
| Borrow Cost Spikes | ORTEX/FINRA | Real-time | Moderate | Low (SI+DTC exists) | Backtested (2025) | ❌ Paid ($$$) |
| 13F Institutional | SEC EDGAR | 45-135 days | Moderate | Low | Academic | ❌ Too Stale |
| Put-Call Ratio Extremes | Cboe free | Real-time | Trivial | Low | Mixed (only high PCR sig.) | ❌ Weak Evidence |
| Earnings+Options Confluence | Options chains | Real-time | Moderate | Low | Qualitative | ❌ No Chain Access |
| Market Breadth Regimes | Yahoo/CBOE | Real-time | Moderate | Low | Qualitative | ⚠️ Enhancement Only |

## Cross-Reference Existing Knowledge

**Already successfully implemented & performing:**
- `insider_buying` (47.7% 7d WR; 100K+ min txn filter applied)
- `insider_cluster_buying` (implemented 2026-04-21; **zero P&L data yet**)
- `short_squeeze` (60% 7d, +2.15% avg; SI+DTC combination optimal)
- `options_flow` (45.1% 7d, 94 recs; premium filter confirmed applied)
- `dark_pool_flow` (implemented 2026-04-21; **zero P&L data yet**)

**Confluence already detected:**
- insider_buying + momentum: 74.3% WR vs 47.7% standalone (+26.6 pts lift)
- momentum + options_flow: 59.1% WR vs 46.8% standalone (+12.3 pts lift)

**Known gaps that are NOT actionable:**
- 13F + insider combo (too slow; 135d lag >> discovery horizon)
- IV skew features on options_flow (requires $10K+/year vendor access)
- Borrow cost filter on short_squeeze (requires ORTEX subscription or FINRA web scraping; Zhu/Buti academic backing exists but data access is barrier)

## Recommendation

**Skip new scanner research.** Instead, pursue three redirects:

### 1. **Evaluate Implemented-But-Unevaluated** (Highest Priority)

Two scanners are fully coded and registered but have zero P&L history:
- `dark_pool_flow` (meridianfin.io scraper functional; should be producing 1-2 picks/day)
- `insider_cluster_buying` (Form 4 parser functional; should be producing 3-10 picks/day)

**Action:** Instrument live P&L tracking now. Within 1-2 weeks of 5-10 day lookback data, we can determine if either merits promotion or deprecation.

### 2. **Confluence Testing** (Medium Priority)

Existing confluence shows insider_buying+momentum works (74.3% vs 47.7%). Systematically test:
- options_flow + momentum (already +12.3 pts)
- short_squeeze + breadth (do squeezes fail in negative breadth?)
- earnings_beat + insider_buying (do PEAD picks align with insider conviction?)

**Action:** Write a confluence-scanner scaffold to test N-way signal combinations automatically.

### 3. **Consider Paid Data Sources** (Long-term)

If business model allows (e.g., subscription tiers), three sources have clear ROI potential:
- **ORTEX borrow costs**: $250-500/mo; enables borrow-cost-acceleration squeeze filter (2025 paper shows 80%+ accuracy)
- **OptionMetrics historical IV**: $10K+/year; enables IV skew/term-structure features on existing options_flow scanner
- **Premium Bloomberg/Reuters terminals**: Full ecosystem but >$2K/mo

**Action:** Document ROI calculation for each. If options_flow + IV skew can hit 55%+ WR (vs current 45%), that's worth exploration.

## Backtest Expectation

No new scanner implementation recommended. Focus on evaluation/enhancement.

