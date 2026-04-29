# Research: Analyst Recommendation Revision Breadth

**Date:** 2026-04-28
**Mode:** autonomous

## Summary

Analyst estimate revision momentum is one of the strongest documented short-term alpha factors in academic finance (Blitz et al. 2022: >6% gross alpha in isolation, >12% combined). The signal is most robustly captured as **breadth** — net count of upward vs. downward analyst revisions over 30 days — rather than EPS magnitude, which requires paid historical consensus snapshots. Finnhub's `recommendation_trends` endpoint (already integrated) returns monthly buy/hold/sell counts going back 4 months, enabling a direct breadth delta without cold-start. This is structurally distinct from the existing `analyst_upgrades` scanner, which parses news sentiment text (noisy, unreliable) rather than actual structured analyst count data.

## Sources Reviewed

- **Blitz et al. 2022, SSRN 4115411** (via Alpha Architect summary): Analyst revision breadth (upward revisions minus downward revisions / total analysts, 30-day window) generates >6% gross alpha individually and >12% combined with 4 other short-term factors — one of the strongest individual signals studied
- **Bond University paper (ScienceDirect 2017)**: Analyst forecast revisions have incremental explanatory power for future returns above and beyond price momentum; effect is asymmetric — upward revisions are consistently significant, downward revisions show mixed results
- **Quant-Investing 150-year backtest**: Multi-dimensional momentum including analyst revision breadth (REV6 = 6-month change in analyst forecasts) adds +0.41%/year to price momentum alone; combined 9.65% vs 9.24%
- **FMP Education article**: EPS revision pressure signal methodology — revision delta + revision rate + dispersion penalty; requires daily consensus snapshots (paid tier for historical)
- **Tavily / Brave search synthesis**: Practitioner consensus aligns with academic evidence; revision breadth (buy/sell count change) is more accessible than EPS magnitude tracking

## Cross-Reference Existing Knowledge

- **`analyst_upgrades` scanner** (`scanners/analyst_upgrades.py`): Uses Alpha Vantage news sentiment scraping to detect upgrades — parses article text for keywords. Unreliable, indirect signal. This scanner uses structured Finnhub numeric count data. **Distinct, not redundant.**
- **`options_flow` scanner**: Institutional conviction via options premium size. Orthogonal signal.
- **Non-OHLCV saturation analysis (2026-04-23)**: Did not evaluate recommendation trend breadth as a standalone signal; focused on EPS magnitude which requires paid data. This approach sidesteps that barrier.

## Fit Evaluation

| Dimension | Score | Notes |
|-----------|-------|-------|
| Data availability | ✅ | Finnhub `recommendation_trends` already integrated in `finnhub_api.py` and used in filter stage; returns 4 months of monthly buy/sell/hold counts |
| Complexity | moderate | Scanner reads trends per-ticker from Finnhub; batching needed to stay under 60 req/min rate limit |
| Signal uniqueness | low overlap | `analyst_upgrades` uses news NLP (unreliable); this uses structured numeric counts — meaningfully different data source and computation |
| Evidence quality | qualitative + academic | Blitz et al. (2022) strongest reference; academic papers confirm incremental alpha above price momentum |

## Recommendation

**Implement** — all four auto-implement thresholds pass.

## Proposed Scanner Spec

- **Scanner name:** `analyst_revision_breadth`
- **Data source:** `tradingagents/dataflows/finnhub_api.py` → `get_finnhub_client().recommendation_trends(ticker)`
- **Pipeline:** `edge`
- **Strategy:** `analyst_revision_momentum`

### Signal Logic

1. Call `finnhub_client.recommendation_trends(ticker)` — returns list of monthly dicts: `[{period, strongBuy, buy, hold, sell, strongSell}, ...]`, most recent first
2. Need ≥ 2 months of data; compare month[0] (latest) vs month[1] (prior month) or month[2] (2 months ago)
3. Compute **net bullish** for each period: `net = strongBuy + buy - sell - strongSell`
4. Compute **delta**: `delta = net[0] - net[1_or_2]`
5. Require minimum analyst coverage: `total_analysts[0] >= 5`
6. Signal fires if `delta >= min_delta` (default 2 net analyst upgrades vs prior month)

### Priority Rules
- **CRITICAL**: `delta >= 5` AND `total_analysts >= 8`
- **HIGH**: `delta >= 3` AND `total_analysts >= 5`
- **MEDIUM**: `delta >= 2` AND `total_analysts >= 5`

### Context Format
`"Analyst revision breadth: +{delta} net upgrades vs prior month ({buy}B/{hold}H/{sell}S, {n} analysts) — estimate revision momentum signal"`

### Rate Limiting
Finnhub free tier: 60 calls/min. Scanner should process universe tickers with a 1-second sleep every 50 tickers, or use ThreadPoolExecutor with max_workers=3.
