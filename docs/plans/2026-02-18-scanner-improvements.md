# Scanner Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix signal quality issues in all 9 existing discovery scanners and add 3 new scanners (analyst upgrades, technical breakout, sector rotation).

**Architecture:** Each scanner is a subclass of `BaseScanner` in `tradingagents/dataflows/discovery/scanners/`. Scanners register via `SCANNER_REGISTRY.register()` at import time. They return `List[Dict]` of candidate dicts with `ticker`, `source`, `context`, `priority`, `strategy` fields. The filter and ranker downstream consume these candidates.

**Tech Stack:** Python, yfinance, Alpha Vantage API, Finnhub API, OpenInsider scraping, PRAW (Reddit)

---

## Phase 1: Fix Existing Scanners

### Task 1: Fix Insider Buying — Preserve Transaction Details

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/insider_buying.py`

**Context:** The scraper (`finviz_scraper.py:get_finviz_insider_buying`) returns structured dicts with `insider`, `title`, `value_num`, `qty`, `price`, `trade_type` when called with `return_structured=True`. But the scanner calls it with `return_structured=False` (markdown string) and then parses only the ticker from markdown rows, losing all transaction details.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/insider_buying.py` fully to understand current logic.

**Step 2: Rewrite the scan() method**

Replace the scan method. Key changes:
- Call `get_finviz_insider_buying(lookback_days, min_transaction_value, return_structured=True)` to get structured data
- Preserve `insider_name`, `title`, `transaction_value`, `shares` in candidate output
- Priority by significance: CEO/CFO title + value >$100K = CRITICAL, director + >$50K = HIGH, other = MEDIUM
- Cluster detection: if 2+ unique insiders bought same ticker, boost to CRITICAL
- Rich context string: `"CEO John Smith purchased $250K of AAPL shares"`

```python
def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not self.is_enabled():
        return []

    logger.info("🔍 Scanning insider buying (OpenInsider)...")

    try:
        from tradingagents.dataflows.finviz_scraper import get_finviz_insider_buying

        transactions = get_finviz_insider_buying(
            lookback_days=self.lookback_days,
            min_transaction_value=self.min_transaction_value,
            return_structured=True,
        )

        if not transactions:
            logger.info("No insider buying transactions found")
            return []

        logger.info(f"Found {len(transactions)} insider transactions")

        # Group by ticker for cluster detection
        by_ticker: Dict[str, list] = {}
        for txn in transactions:
            ticker = txn.get("ticker", "").upper().strip()
            if not ticker:
                continue
            by_ticker.setdefault(ticker, []).append(txn)

        candidates = []
        for ticker, txns in by_ticker.items():
            # Use the largest transaction as primary
            txns.sort(key=lambda t: t.get("value_num", 0), reverse=True)
            primary = txns[0]

            insider_name = primary.get("insider", "Unknown")
            title = primary.get("title", "")
            value = primary.get("value_num", 0)
            value_str = primary.get("value_str", f"${value:,.0f}")
            num_insiders = len(txns)

            # Priority by significance
            title_lower = title.lower()
            is_c_suite = any(t in title_lower for t in ["ceo", "cfo", "coo", "cto", "president", "chairman"])
            is_director = "director" in title_lower

            if num_insiders >= 2:
                priority = Priority.CRITICAL.value
            elif is_c_suite and value >= 100_000:
                priority = Priority.CRITICAL.value
            elif is_c_suite or (is_director and value >= 50_000):
                priority = Priority.HIGH.value
            elif value >= 50_000:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

            # Build context
            if num_insiders > 1:
                context = f"Cluster: {num_insiders} insiders buying {ticker}. Largest: {title} {insider_name} purchased {value_str}"
            else:
                context = f"{title} {insider_name} purchased {value_str} of {ticker}"

            candidates.append({
                "ticker": ticker,
                "source": self.name,
                "context": context,
                "priority": priority,
                "strategy": self.strategy,
                "insider_name": insider_name,
                "insider_title": title,
                "transaction_value": value,
                "num_insiders_buying": num_insiders,
            })

            if len(candidates) >= self.limit:
                break

        logger.info(f"Insider buying: {len(candidates)} candidates")
        return candidates

    except Exception as e:
        logger.error(f"Insider buying scan failed: {e}", exc_info=True)
        return []
```

**Step 3: Run verification**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners.insider_buying
cls = SCANNER_REGISTRY.scanners['insider_buying']
print(f'name={cls.name}, strategy={cls.strategy}, pipeline={cls.pipeline}')
print('Has scan method:', hasattr(cls, 'scan'))
"
```

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/insider_buying.py
git commit -m "fix(insider-buying): preserve transaction details, add cluster detection and smart priority"
```

---

### Task 2: Fix Options Flow — Apply Premium Filter, Multi-Expiration

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/options_flow.py`

**Context:** `self.min_premium` is loaded at line 50 but never used. Only `expirations[0]` is scanned (line 104). Need to apply premium filter and scan up to 3 expirations.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/options_flow.py` fully.

**Step 2: Fix the `_scan_ticker` method**

Key changes to `_scan_ticker()`:
- Loop through up to 3 expirations instead of just `expirations[0]`
- Add premium filter: skip strikes where `volume * lastPrice * 100 < self.min_premium`
- Track which expiration had the most unusual activity
- Add `days_to_expiry` classification in output

Replace the inner scanning logic (the `_scan_ticker` method). The core change is:

```python
def _scan_ticker(self, ticker: str) -> Optional[Dict[str, Any]]:
    """Scan a single ticker for unusual options activity."""
    try:
        expirations = get_ticker_options(ticker)
        if not expirations:
            return None

        # Scan up to 3 nearest expirations
        max_expirations = min(3, len(expirations))
        total_unusual_calls = 0
        total_unusual_puts = 0
        total_call_vol = 0
        total_put_vol = 0
        best_expiration = None
        best_unusual_count = 0

        for exp in expirations[:max_expirations]:
            try:
                options = get_option_chain(ticker, exp)
            except Exception:
                continue

            if options is None:
                continue

            calls_df, puts_df = (None, None)
            if isinstance(options, tuple) and len(options) == 2:
                calls_df, puts_df = options
            elif hasattr(options, "calls") and hasattr(options, "puts"):
                calls_df, puts_df = options.calls, options.puts
            else:
                continue

            exp_unusual_calls = 0
            exp_unusual_puts = 0

            # Analyze calls
            if calls_df is not None and not calls_df.empty:
                for _, opt in calls_df.iterrows():
                    vol = opt.get("volume", 0) or 0
                    oi = opt.get("openInterest", 0) or 0
                    price = opt.get("lastPrice", 0) or 0

                    if vol < self.min_volume:
                        continue
                    # Premium filter (volume * price * 100 shares per contract)
                    if (vol * price * 100) < self.min_premium:
                        continue
                    if oi > 0 and (vol / oi) >= self.min_volume_oi_ratio:
                        exp_unusual_calls += 1

                    total_call_vol += vol

            # Analyze puts
            if puts_df is not None and not puts_df.empty:
                for _, opt in puts_df.iterrows():
                    vol = opt.get("volume", 0) or 0
                    oi = opt.get("openInterest", 0) or 0
                    price = opt.get("lastPrice", 0) or 0

                    if vol < self.min_volume:
                        continue
                    if (vol * price * 100) < self.min_premium:
                        continue
                    if oi > 0 and (vol / oi) >= self.min_volume_oi_ratio:
                        exp_unusual_puts += 1

                    total_put_vol += vol

            total_unusual_calls += exp_unusual_calls
            total_unusual_puts += exp_unusual_puts

            exp_total = exp_unusual_calls + exp_unusual_puts
            if exp_total > best_unusual_count:
                best_unusual_count = exp_total
                best_expiration = exp

        total_unusual = total_unusual_calls + total_unusual_puts
        if total_unusual == 0:
            return None

        # Calculate put/call ratio
        pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 999

        if pc_ratio < 0.7:
            sentiment = "bullish"
        elif pc_ratio > 1.3:
            sentiment = "bearish"
        else:
            sentiment = "neutral"

        priority = Priority.HIGH.value if sentiment == "bullish" else Priority.MEDIUM.value

        context = (
            f"Unusual options: {total_unusual} strikes across {max_expirations} exp, "
            f"P/C={pc_ratio:.2f} ({sentiment}), "
            f"{total_unusual_calls} unusual calls / {total_unusual_puts} unusual puts"
        )

        return {
            "ticker": ticker,
            "source": self.name,
            "context": context,
            "priority": priority,
            "strategy": self.strategy,
            "put_call_ratio": round(pc_ratio, 2),
            "unusual_calls": total_unusual_calls,
            "unusual_puts": total_unusual_puts,
            "best_expiration": best_expiration,
        }

    except Exception as e:
        logger.debug(f"Error scanning {ticker}: {e}")
        return None
```

**Step 3: Verify**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners.options_flow
cls = SCANNER_REGISTRY.scanners['options_flow']
print(f'name={cls.name}, strategy={cls.strategy}')
"
```

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/options_flow.py
git commit -m "fix(options-flow): apply premium filter, scan multiple expirations"
```

---

### Task 3: Fix Volume Accumulation — Distinguish Accumulation from Distribution

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/volume_accumulation.py`

**Context:** Currently flags any unusual volume. Need to add price-change context and multi-day accumulation detection.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/volume_accumulation.py` fully.

**Step 2: Add price-change and multi-day enrichment**

After the existing volume parsing, add enrichment using yfinance data. The key addition is a helper that checks whether the volume spike is accumulation (flat price) vs distribution (big drop):

```python
def _enrich_volume_candidate(self, ticker: str, cand: Dict[str, Any]) -> Dict[str, Any]:
    """Add price-change context to distinguish accumulation from distribution."""
    try:
        from tradingagents.dataflows.y_finance import download_history

        hist = download_history(ticker, period="10d", interval="1d", auto_adjust=True, progress=False)
        if hist.empty or len(hist) < 2:
            return cand

        # Today's price change
        latest_close = float(hist["Close"].iloc[-1])
        prev_close = float(hist["Close"].iloc[-2])
        day_change_pct = ((latest_close - prev_close) / prev_close) * 100

        cand["day_change_pct"] = round(day_change_pct, 2)

        # Multi-day volume pattern: count days with >1.5x avg volume in last 5 days
        if len(hist) >= 6:
            avg_vol = float(hist["Volume"].iloc[:-5].mean()) if len(hist) > 5 else float(hist["Volume"].mean())
            if avg_vol > 0:
                recent_high_vol_days = sum(
                    1 for v in hist["Volume"].iloc[-5:] if float(v) > avg_vol * 1.5
                )
                cand["high_vol_days_5d"] = recent_high_vol_days
                if recent_high_vol_days >= 3:
                    cand["context"] += f" | Sustained: {recent_high_vol_days}/5 days above 1.5x avg"

        # Classify signal
        if abs(day_change_pct) < 3:
            # Quiet accumulation — the best signal
            cand["volume_signal"] = "accumulation"
            cand["context"] += f" | Price flat ({day_change_pct:+.1f}%) — quiet accumulation"
        elif day_change_pct < -5:
            # Distribution / panic selling
            cand["volume_signal"] = "distribution"
            cand["priority"] = Priority.LOW.value
            cand["context"] += f" | Price dropped {day_change_pct:+.1f}% — possible distribution"
        else:
            cand["volume_signal"] = "momentum"

    except Exception as e:
        logger.debug(f"Volume enrichment failed for {ticker}: {e}")

    return cand
```

Call this method for each candidate after the existing parsing loop, before appending to the final list. Skip (don't append) candidates with `volume_signal == "distribution"`.

**Step 3: Verify**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners.volume_accumulation
print('volume_accumulation registered:', 'volume_accumulation' in SCANNER_REGISTRY.scanners)
"
```

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/volume_accumulation.py
git commit -m "fix(volume): distinguish accumulation from distribution, add multi-day pattern"
```

---

### Task 4: Fix Reddit DD — Use LLM Quality Score

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/reddit_dd.py`

**Context:** The LLM evaluates each DD post with a 0-100 quality score, but the scanner stores it as `dd_score` and uses Reddit upvotes for priority instead. Additionally, the tool `"scan_reddit_dd"` may not exist in the registry, causing the scanner to always fall back.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/reddit_dd.py` fully, and check if `"scan_reddit_dd"` exists in `tradingagents/tools/registry.py`.

**Step 2: Fix priority logic to use quality score**

In the structured result parsing section (where dd posts are iterated), change the priority assignment:

```python
# Replace the existing priority logic with:
dd_score = post.get("quality_score", post.get("score", 0))

if dd_score >= 80:
    priority = Priority.HIGH.value
elif dd_score >= 60:
    priority = Priority.MEDIUM.value
else:
    # Skip low-quality posts
    continue
```

Also preserve the score and post title in context:

```python
title = post.get("title", "")[:100]
context = f"Reddit DD (score: {dd_score}/100): {title}"
```

And in the candidate dict, include:
```python
"dd_quality_score": dd_score,
"dd_title": title,
```

If the `"scan_reddit_dd"` tool doesn't exist in the registry, add a fallback that calls `get_reddit_undiscovered_dd()` directly (imported from `tradingagents.dataflows.reddit_api`).

**Step 3: Verify**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners.reddit_dd
print('reddit_dd registered:', 'reddit_dd' in SCANNER_REGISTRY.scanners)
"
```

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/reddit_dd.py
git commit -m "fix(reddit-dd): use LLM quality score for priority, preserve post details"
```

---

### Task 5: Fix Reddit Trending — Add Mention Count and Sentiment

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/reddit_trending.py`

**Context:** Currently all candidates get MEDIUM priority with a generic "Reddit trending discussion" context. No mention counts or sentiment info.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/reddit_trending.py` fully.

**Step 2: Enrich with mention counts**

If the tool returns structured data (list of dicts), extract mention counts. If it returns text, count ticker occurrences. Use counts for priority:

```python
# After extracting tickers, count mentions
from collections import Counter
ticker_counts = Counter()
# ... count each ticker mention in result text/data

for ticker in unique_tickers:
    count = ticker_counts.get(ticker, 1)

    if count >= 50:
        priority = Priority.HIGH.value
    elif count >= 20:
        priority = Priority.MEDIUM.value
    else:
        priority = Priority.LOW.value

    context = f"Trending on Reddit: ~{count} mentions"
```

**Step 3: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/reddit_trending.py
git commit -m "fix(reddit-trending): add mention counts, scale priority by volume"
```

---

### Task 6: Fix Semantic News — Include Headlines, Add Catalyst Classification

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/semantic_news.py`

**Context:** `self.min_importance` is loaded (line 23) but never used. Context is generic "Mentioned in recent market news" with no headline text. Scanner just regex-extracts uppercase words.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/semantic_news.py` fully.

**Step 2: Improve context and add catalyst classification**

When creating candidates, include the actual headline text. Add simple keyword-based catalyst classification for priority:

```python
CATALYST_KEYWORDS = {
    Priority.CRITICAL.value: ["fda approval", "acquisition", "merger", "buyout", "takeover"],
    Priority.HIGH.value: ["upgrade", "initiated", "beat", "surprise", "contract win", "patent"],
    Priority.MEDIUM.value: ["downgrade", "miss", "lawsuit", "investigation", "recall"],
}

def _classify_catalyst(self, headline: str) -> str:
    """Classify news headline by catalyst type and return priority."""
    headline_lower = headline.lower()
    for priority, keywords in CATALYST_KEYWORDS.items():
        if any(kw in headline_lower for kw in keywords):
            return priority
    return Priority.MEDIUM.value
```

For each news item, preserve the headline and set priority by catalyst type:

```python
headline = news_item.get("title", "")[:150]
priority = self._classify_catalyst(headline)
context = f"News catalyst: {headline}" if headline else "Mentioned in recent market news"
```

Also store `news_context` as a list of headline dicts for the downstream ranker:

```python
"news_context": [{"news_title": headline, "news_summary": summary, "published_at": timestamp}]
```

**Step 3: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/semantic_news.py
git commit -m "fix(semantic-news): include headlines, add catalyst classification"
```

---

### Task 7: Fix Earnings Calendar — Add Accumulation Signal and Estimates

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/earnings_calendar.py`

**Context:** Currently a pure calendar. `get_pre_earnings_accumulation_signal()` and `get_ticker_earnings_estimate()` already exist in the codebase but aren't used.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/earnings_calendar.py` fully.

**Step 2: Add accumulation signal enrichment**

After the existing candidate creation, add a post-processing step. For each candidate with days_until between 2 and 7, check for volume accumulation:

```python
def _enrich_earnings_candidate(self, cand: Dict[str, Any]) -> Dict[str, Any]:
    """Enrich earnings candidate with accumulation signal and estimates."""
    ticker = cand["ticker"]

    # Check pre-earnings volume accumulation
    try:
        from tradingagents.dataflows.y_finance import get_pre_earnings_accumulation_signal

        signal = get_pre_earnings_accumulation_signal(ticker)
        if signal and signal.get("signal"):
            vol_ratio = signal.get("volume_ratio", 0)
            cand["has_accumulation"] = True
            cand["accumulation_volume_ratio"] = vol_ratio
            cand["context"] += f" | Pre-earnings accumulation: {vol_ratio:.1f}x volume"
            # Boost priority if accumulation detected
            cand["priority"] = Priority.CRITICAL.value
    except Exception:
        pass

    # Add earnings estimates
    try:
        from tradingagents.dataflows.finnhub_api import get_ticker_earnings_estimate

        est = get_ticker_earnings_estimate(ticker)
        if est and est.get("has_upcoming_earnings"):
            eps = est.get("eps_estimate")
            if eps is not None:
                cand["eps_estimate"] = eps
                cand["context"] += f" | EPS est: ${eps:.2f}"
    except Exception:
        pass

    return cand
```

Call this for each candidate before appending to the final list. Limit enrichment to avoid API rate limits (only enrich top 10 by proximity).

**Step 3: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/earnings_calendar.py
git commit -m "fix(earnings): add pre-earnings accumulation signal and EPS estimates"
```

---

### Task 8: Fix Market Movers — Add Market Cap and Volume Filters

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/market_movers.py`

**Context:** Takes whatever Alpha Vantage returns with no filtering. Penny stocks with 400% gains on 100 shares get included.

**Step 1: Read the current scanner**

Read `tradingagents/dataflows/discovery/scanners/market_movers.py` fully.

**Step 2: Add filtering configuration and validation**

Add configurable filters in `__init__`:

```python
self.min_price = self.scanner_config.get("min_price", 5.0)
self.min_volume = self.scanner_config.get("min_volume", 500_000)
```

After parsing candidates from the tool result, validate each one:

```python
def _validate_mover(self, ticker: str) -> bool:
    """Quick validation: price and volume check."""
    try:
        from tradingagents.dataflows.y_finance import get_stock_price, get_ticker_info

        price = get_stock_price(ticker)
        if price is not None and price < self.min_price:
            return False

        info = get_ticker_info(ticker)
        avg_vol = info.get("averageVolume", 0) if info else 0
        if avg_vol and avg_vol < self.min_volume:
            return False

        return True
    except Exception:
        return True  # Don't filter on errors
```

Call `_validate_mover()` before appending each candidate. This removes penny stocks and illiquid names.

**Step 3: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/market_movers.py
git commit -m "fix(market-movers): add price and volume validation filters"
```

---

### Task 9: Fix ML Signal — Raise Threshold

**Files:**
- Modify: `tradingagents/dataflows/discovery/scanners/ml_signal.py`

**Context:** Default `min_win_prob` is 0.35 (35%). This is barely better than random.

**Step 1: Change default threshold**

In `__init__`, change the default:

```python
# Change from:
self.min_win_prob = self.scanner_config.get("min_win_prob", 0.35)
# To:
self.min_win_prob = self.scanner_config.get("min_win_prob", 0.50)
```

Also adjust priority thresholds to match:

```python
# Change from:
if win_prob >= 0.50:
    priority = Priority.CRITICAL.value
elif win_prob >= 0.40:
    priority = Priority.HIGH.value
else:
    priority = Priority.MEDIUM.value
# To:
if win_prob >= 0.65:
    priority = Priority.CRITICAL.value
elif win_prob >= 0.55:
    priority = Priority.HIGH.value
else:
    priority = Priority.MEDIUM.value
```

**Step 2: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/ml_signal.py
git commit -m "fix(ml-signal): raise min win probability to 50%, adjust priority tiers"
```

---

## Phase 2: New Scanners

### Task 10: Add Strategy Enum Values for New Scanners

**Files:**
- Modify: `tradingagents/dataflows/discovery/utils.py`

**Step 1: Add new enum values**

Add after the existing `SOCIAL_DD` entry:

```python
SECTOR_ROTATION = "sector_rotation"
TECHNICAL_BREAKOUT = "technical_breakout"
```

`ANALYST_UPGRADE` already exists in the enum.

**Step 2: Commit**

```bash
git add tradingagents/dataflows/discovery/utils.py
git commit -m "feat: add sector_rotation and technical_breakout strategy enum values"
```

---

### Task 11: Add Analyst Upgrades Scanner

**Files:**
- Create: `tradingagents/dataflows/discovery/scanners/analyst_upgrades.py`
- Modify: `tradingagents/dataflows/discovery/scanners/__init__.py`

**Context:** `get_analyst_rating_changes(return_structured=True)` already exists in `alpha_vantage_analysts.py`. Returns list of dicts with `ticker`, `action`, `date`, `hours_old`, `headline`, `source`, `url`.

**Step 1: Create the scanner**

```python
"""Analyst upgrade and initiation scanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class AnalystUpgradeScanner(BaseScanner):
    """Scan for recent analyst upgrades and coverage initiations."""

    name = "analyst_upgrades"
    pipeline = "edge"
    strategy = "analyst_upgrade"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_days = self.scanner_config.get("lookback_days", 3)
        self.max_hours_old = self.scanner_config.get("max_hours_old", 72)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📊 Scanning analyst upgrades and initiations...")

        try:
            from tradingagents.dataflows.alpha_vantage_analysts import (
                get_analyst_rating_changes,
            )

            changes = get_analyst_rating_changes(
                lookback_days=self.lookback_days,
                change_types=["upgrade", "initiated"],
                top_n=self.limit * 2,
                return_structured=True,
            )

            if not changes:
                logger.info("No analyst upgrades found")
                return []

            candidates = []
            for change in changes:
                ticker = change.get("ticker", "").upper().strip()
                if not ticker:
                    continue

                action = change.get("action", "unknown")
                hours_old = change.get("hours_old", 999)
                headline = change.get("headline", "")
                source = change.get("source", "")

                if hours_old > self.max_hours_old:
                    continue

                # Priority by freshness and action type
                if action == "upgrade" and hours_old <= 24:
                    priority = Priority.HIGH.value
                elif action == "initiated" and hours_old <= 24:
                    priority = Priority.HIGH.value
                elif hours_old <= 48:
                    priority = Priority.MEDIUM.value
                else:
                    priority = Priority.LOW.value

                context = f"Analyst {action}: {headline}" if headline else f"Analyst {action} ({source})"

                candidates.append({
                    "ticker": ticker,
                    "source": self.name,
                    "context": context,
                    "priority": priority,
                    "strategy": self.strategy,
                    "analyst_action": action,
                    "hours_old": hours_old,
                })

                if len(candidates) >= self.limit:
                    break

            logger.info(f"Analyst upgrades: {len(candidates)} candidates")
            return candidates

        except Exception as e:
            logger.error(f"Analyst upgrades scan failed: {e}", exc_info=True)
            return []


SCANNER_REGISTRY.register(AnalystUpgradeScanner)
```

**Step 2: Register in `__init__.py`**

Add to the import block:

```python
analyst_upgrades,  # noqa: F401
```

**Step 3: Verify**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners
print('analyst_upgrades' in SCANNER_REGISTRY.scanners)
cls = SCANNER_REGISTRY.scanners['analyst_upgrades']
print(f'name={cls.name}, strategy={cls.strategy}, pipeline={cls.pipeline}')
"
```

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/analyst_upgrades.py tradingagents/dataflows/discovery/scanners/__init__.py
git commit -m "feat: add analyst upgrades scanner"
```

---

### Task 12: Add Technical Breakout Scanner

**Files:**
- Create: `tradingagents/dataflows/discovery/scanners/technical_breakout.py`
- Modify: `tradingagents/dataflows/discovery/scanners/__init__.py`

**Context:** Uses yfinance OHLCV data. Detects volume-confirmed breakouts above recent resistance or 52-week highs. Scans same ticker universe as ML/options scanners.

**Step 1: Create the scanner**

```python
"""Technical breakout scanner — volume-confirmed price breakouts."""

from typing import Any, Dict, List, Optional

import pandas as pd

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TICKER_FILE = "data/tickers.txt"


def _load_tickers_from_file(path: str) -> List[str]:
    """Load ticker symbols from a text file."""
    try:
        with open(path) as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        if tickers:
            logger.info(f"Breakout scanner: loaded {len(tickers)} tickers from {path}")
            return tickers
    except FileNotFoundError:
        logger.warning(f"Ticker file not found: {path}")
    except Exception as e:
        logger.warning(f"Failed to load ticker file {path}: {e}")
    return []


class TechnicalBreakoutScanner(BaseScanner):
    """Scan for volume-confirmed technical breakouts."""

    name = "technical_breakout"
    pipeline = "momentum"
    strategy = "technical_breakout"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ticker_file = self.scanner_config.get("ticker_file", DEFAULT_TICKER_FILE)
        self.max_tickers = self.scanner_config.get("max_tickers", 150)
        self.min_volume_multiple = self.scanner_config.get("min_volume_multiple", 2.0)
        self.lookback_days = self.scanner_config.get("lookback_days", 20)
        self.max_workers = self.scanner_config.get("max_workers", 8)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📈 Scanning for technical breakouts...")

        tickers = _load_tickers_from_file(self.ticker_file)
        if not tickers:
            logger.warning("No tickers loaded for breakout scan")
            return []

        tickers = tickers[: self.max_tickers]

        # Batch download OHLCV
        from tradingagents.dataflows.y_finance import download_history

        try:
            data = download_history(
                tickers,
                period="3mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
        except Exception as e:
            logger.error(f"Batch download failed: {e}")
            return []

        if data.empty:
            return []

        candidates = []
        for ticker in tickers:
            result = self._check_breakout(ticker, data)
            if result:
                candidates.append(result)
            if len(candidates) >= self.limit:
                break

        candidates.sort(key=lambda c: c.get("volume_multiple", 0), reverse=True)
        logger.info(f"Technical breakouts: {len(candidates)} candidates")
        return candidates[: self.limit]

    def _check_breakout(self, ticker: str, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Check if ticker has a volume-confirmed breakout."""
        try:
            # Extract single-ticker data from multi-ticker download
            if isinstance(data.columns, pd.MultiIndex):
                if ticker not in data.columns.get_level_values(1):
                    return None
                df = data.xs(ticker, axis=1, level=1).dropna()
            else:
                df = data.dropna()

            if len(df) < self.lookback_days + 5:
                return None

            close = df["Close"]
            volume = df["Volume"]
            high = df["High"]

            latest_close = float(close.iloc[-1])
            latest_vol = float(volume.iloc[-1])

            # 20-day lookback resistance (excluding last day)
            lookback_high = float(high.iloc[-(self.lookback_days + 1) : -1].max())

            # Average volume over lookback period
            avg_vol = float(volume.iloc[-(self.lookback_days + 1) : -1].mean())

            if avg_vol <= 0:
                return None

            vol_multiple = latest_vol / avg_vol

            # Breakout conditions:
            # 1. Price closed above the lookback-period high
            # 2. Volume is at least min_volume_multiple times average
            is_breakout = latest_close > lookback_high and vol_multiple >= self.min_volume_multiple

            if not is_breakout:
                return None

            # Check if near 52-week high for bonus
            if len(df) >= 252:
                high_52w = float(high.iloc[-252:].max())
                near_52w_high = latest_close >= high_52w * 0.95
            else:
                high_52w = float(high.max())
                near_52w_high = latest_close >= high_52w * 0.95

            # Priority
            if vol_multiple >= 3.0 and near_52w_high:
                priority = Priority.CRITICAL.value
            elif vol_multiple >= 3.0 or near_52w_high:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

            breakout_pct = ((latest_close - lookback_high) / lookback_high) * 100

            context = (
                f"Breakout: closed {breakout_pct:+.1f}% above {self.lookback_days}d high "
                f"on {vol_multiple:.1f}x volume"
            )
            if near_52w_high:
                context += " | Near 52-week high"

            return {
                "ticker": ticker,
                "source": self.name,
                "context": context,
                "priority": priority,
                "strategy": self.strategy,
                "volume_multiple": round(vol_multiple, 2),
                "breakout_pct": round(breakout_pct, 2),
                "near_52w_high": near_52w_high,
            }

        except Exception as e:
            logger.debug(f"Breakout check failed for {ticker}: {e}")
            return None


SCANNER_REGISTRY.register(TechnicalBreakoutScanner)
```

**Step 2: Register in `__init__.py`**

Add `technical_breakout` to imports.

**Step 3: Verify**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners
print('technical_breakout' in SCANNER_REGISTRY.scanners)
"
```

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/technical_breakout.py tradingagents/dataflows/discovery/scanners/__init__.py
git commit -m "feat: add technical breakout scanner"
```

---

### Task 13: Add Sector Rotation Scanner

**Files:**
- Create: `tradingagents/dataflows/discovery/scanners/sector_rotation.py`
- Modify: `tradingagents/dataflows/discovery/scanners/__init__.py`

**Context:** Compares sector ETF relative strength (5-day vs 20-day). Flags stocks in accelerating sectors that haven't moved yet. Uses yfinance — no new APIs.

**Step 1: Create the scanner**

```python
"""Sector rotation scanner — finds laggards in accelerating sectors."""

from typing import Any, Dict, List, Optional

import pandas as pd

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# SPDR Select Sector ETFs
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

DEFAULT_TICKER_FILE = "data/tickers.txt"


def _load_tickers_from_file(path: str) -> List[str]:
    """Load ticker symbols from a text file."""
    try:
        with open(path) as f:
            return [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except Exception:
        return []


class SectorRotationScanner(BaseScanner):
    """Detect sector momentum shifts and find laggards in accelerating sectors."""

    name = "sector_rotation"
    pipeline = "momentum"
    strategy = "sector_rotation"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ticker_file = self.scanner_config.get("ticker_file", DEFAULT_TICKER_FILE)
        self.max_tickers = self.scanner_config.get("max_tickers", 100)
        self.min_sector_accel = self.scanner_config.get("min_sector_acceleration", 2.0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("🔄 Scanning sector rotation...")

        from tradingagents.dataflows.y_finance import download_history, get_ticker_info

        # Step 1: Identify accelerating sectors
        try:
            etf_symbols = list(SECTOR_ETFS.keys())
            etf_data = download_history(
                etf_symbols, period="2mo", interval="1d", auto_adjust=True, progress=False
            )
        except Exception as e:
            logger.error(f"Failed to download sector ETF data: {e}")
            return []

        if etf_data.empty:
            return []

        accelerating_sectors = self._find_accelerating_sectors(etf_data)
        if not accelerating_sectors:
            logger.info("No accelerating sectors detected")
            return []

        sector_names = [SECTOR_ETFS.get(etf, etf) for etf in accelerating_sectors]
        logger.info(f"Accelerating sectors: {', '.join(sector_names)}")

        # Step 2: Find laggard stocks in those sectors
        tickers = _load_tickers_from_file(self.ticker_file)
        if not tickers:
            return []

        tickers = tickers[: self.max_tickers]

        candidates = []
        for ticker in tickers:
            result = self._check_sector_laggard(ticker, accelerating_sectors, get_ticker_info)
            if result:
                candidates.append(result)
            if len(candidates) >= self.limit:
                break

        logger.info(f"Sector rotation: {len(candidates)} candidates")
        return candidates

    def _find_accelerating_sectors(self, data: pd.DataFrame) -> List[str]:
        """Find sectors where 5-day return is accelerating vs 20-day trend."""
        accelerating = []

        for etf in SECTOR_ETFS:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if etf not in data.columns.get_level_values(1):
                        continue
                    close = data.xs(etf, axis=1, level=1)["Close"].dropna()
                else:
                    close = data["Close"].dropna()

                if len(close) < 21:
                    continue

                ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100
                ret_20d = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100

                # Acceleration: 5-day annualized return significantly beats 20-day
                # i.e., the sector is moving faster recently
                daily_rate_5d = ret_5d / 5
                daily_rate_20d = ret_20d / 20

                if daily_rate_20d != 0:
                    acceleration = daily_rate_5d / daily_rate_20d
                elif daily_rate_5d > 0:
                    acceleration = 10.0  # Strong acceleration from flat
                else:
                    acceleration = 0

                if acceleration >= self.min_sector_accel and ret_5d > 0:
                    accelerating.append(etf)
                    logger.debug(
                        f"{etf} ({SECTOR_ETFS[etf]}): 5d={ret_5d:+.1f}%, "
                        f"20d={ret_20d:+.1f}%, accel={acceleration:.1f}x"
                    )
            except Exception as e:
                logger.debug(f"Error analyzing {etf}: {e}")

        return accelerating

    def _check_sector_laggard(
        self, ticker: str, accelerating_sectors: List[str], get_info_fn
    ) -> Optional[Dict[str, Any]]:
        """Check if stock is in an accelerating sector but hasn't moved yet."""
        try:
            info = get_info_fn(ticker)
            if not info:
                return None

            stock_sector = info.get("sector", "")

            # Map stock sector to ETF
            sector_to_etf = {v: k for k, v in SECTOR_ETFS.items()}
            sector_etf = sector_to_etf.get(stock_sector)

            if not sector_etf or sector_etf not in accelerating_sectors:
                return None

            # Check if stock is lagging its sector (hasn't caught up yet)
            from tradingagents.dataflows.y_finance import download_history

            hist = download_history(ticker, period="1mo", interval="1d", auto_adjust=True, progress=False)
            if hist.empty or len(hist) < 6:
                return None

            close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
            ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100

            # Stock is a laggard if it moved less than 1% while sector is accelerating
            if ret_5d > 2.0:
                return None  # Already moved, not a laggard

            context = (
                f"Sector rotation: {stock_sector} sector accelerating, "
                f"{ticker} lagging at {ret_5d:+.1f}% (5d)"
            )

            return {
                "ticker": ticker,
                "source": self.name,
                "context": context,
                "priority": Priority.MEDIUM.value,
                "strategy": self.strategy,
                "sector": stock_sector,
                "sector_etf": sector_etf,
                "stock_5d_return": round(ret_5d, 2),
            }

        except Exception as e:
            logger.debug(f"Sector check failed for {ticker}: {e}")
            return None


SCANNER_REGISTRY.register(SectorRotationScanner)
```

**Step 2: Register in `__init__.py`**

Add `sector_rotation` to imports.

**Step 3: Verify**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
import tradingagents.dataflows.discovery.scanners
for name in sorted(SCANNER_REGISTRY.scanners):
    cls = SCANNER_REGISTRY.scanners[name]
    print(f'{name:25s} pipeline={cls.pipeline:12s} strategy={cls.strategy}')
print(f'Total: {len(SCANNER_REGISTRY.scanners)} scanners')
"
```

Expected: 12 scanners total.

**Step 4: Commit**

```bash
git add tradingagents/dataflows/discovery/scanners/sector_rotation.py tradingagents/dataflows/discovery/scanners/__init__.py tradingagents/dataflows/discovery/utils.py
git commit -m "feat: add sector rotation scanner"
```

---

### Task 14: Final Verification

**Step 1: Run all scanner registration**

```bash
python -c "
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import Strategy
import tradingagents.dataflows.discovery.scanners

valid_strategies = {s.value for s in Strategy}
errors = []
for name, cls in SCANNER_REGISTRY.scanners.items():
    if cls.strategy not in valid_strategies:
        errors.append(f'{name}: strategy {cls.strategy!r} not in Strategy enum')
if errors:
    print('ERRORS:')
    for e in errors: print(f'  {e}')
else:
    print(f'All {len(SCANNER_REGISTRY.scanners)} scanners have valid strategies')
"
```

**Step 2: Run existing tests**

```bash
pytest tests/ -x -q
```

**Step 3: Final commit if any cleanup needed**

```bash
git add -A && git commit -m "chore: scanner improvements cleanup"
```
