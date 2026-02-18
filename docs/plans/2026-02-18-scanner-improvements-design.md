# Scanner Improvements Design

**Date:** 2026-02-18
**Status:** Approved

## Problem

Most scanners produce weak or broken signals:
- Insider buying drops transaction details (name, title, value)
- Options flow ignores its own premium filter, only checks nearest expiration
- Volume accumulation can't distinguish buying from selling
- Reddit DD scores posts with an LLM then ignores the score
- Semantic news is just regex extraction, not semantic
- Market movers finds stocks after they moved
- ML signal threshold (35%) is worse than a coin flip

Three useful scanner types are missing entirely: analyst upgrades, technical breakouts, sector rotation.

## Phase 1: Fix Existing Scanners

### 1. Insider Buying
- Preserve `insider_name`, `title`, `transaction_value`, `shares` from scraper
- Priority by significance: CEO/CFO >$100K = CRITICAL, director >$50K = HIGH, other = MEDIUM
- Cluster detection: 2+ insiders buying same stock within 14 days = CRITICAL
- Rich context: "CFO Jane Smith purchased $250K of shares"

### 2. Options Flow
- Apply the existing `min_premium` threshold ($25K) — currently configured but never checked
- Scan up to 3 nearest expirations instead of 1
- Classify moneyness: ITM call buying (conviction) > OTM (speculative)
- Weight by expiration: 30+ DTE scored higher than weeklies

### 3. Volume Accumulation
- Price-change filter: volume >2x AND absolute price change <3% (quiet accumulation only)
- Multi-day mode: 3 of last 5 days >1.5x average = sustained accumulation
- Exclude distribution: high volume + big price drop = skip

### 4. Reddit DD
- Use LLM quality score for priority: 80+ = HIGH, 60-79 = MEDIUM, <60 = skip
- Subreddit weighting: r/investing bonus, r/pennystocks penalty
- Include post title and LLM score in context

### 5. Reddit Trending
- Add mention count to context: "47 mentions in 6hrs"
- Priority by volume: 50+ = HIGH, 20-49 = MEDIUM
- Basic sentiment check from available data

### 6. Semantic News
- Include actual headline text in context (not just "Mentioned in recent market news")
- Catalyst classification from headline keywords: upgrade/FDA/acquisition/earnings
- Priority based on catalyst type

### 7. Earnings Calendar
- Add historical earnings reaction via `get_pre_earnings_accumulation_signal()`
- Include EPS/revenue estimates from `get_ticker_earnings_estimate()`
- Priority: proximity + accumulation signal = CRITICAL

### 8. Market Movers
- Market cap filter: exclude <$300M
- Volume validation: require avg volume >500K
- Include change percentage in context
- Cross-reference with news for catalyst attribution

### 9. ML Signal
- Raise `min_win_prob` default from 0.35 to 0.50
- Log model metadata (version, training date) if available
- Add feature importances to context when model exposes them

## Phase 2: New Scanners

### 10. Analyst Upgrades Scanner
- Uses existing `get_analyst_rating_changes()` from `alpha_vantage_analysts.py`
- Filters for upgrades, initiations, price target increases from last 3 days
- Priority: upgrade with >20% target increase = HIGH, initiation = MEDIUM
- Strategy: `analyst_upgrade`

### 11. Technical Breakout Scanner
- Uses yfinance OHLCV data (no new APIs)
- Detects: volume-confirmed breakout above 20-day high, or 52-week high on 2x+ volume
- Priority: 3x+ volume at breakout = HIGH, 2x+ = MEDIUM
- Strategy: `momentum` (reuses existing enum)

### 12. Sector Rotation Scanner
- Compares sector ETF relative strength: 5-day vs 20-day periods
- Flags individual stocks in accelerating sectors that haven't moved yet
- Uses yfinance sector ETFs (XLK, XLF, XLE, etc.)
- Strategy: new `sector_rotation` enum value

## Data Sources

All improvements use existing APIs:
- yfinance (free, no key)
- Alpha Vantage (existing key)
- Finnhub (existing key)
- OpenInsider scraping (existing)
- Reddit PRAW (existing)

No new API subscriptions required.
