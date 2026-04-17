# /research-and-backtest

Research multiple OHLCV-based trading strategies in a single run, implement each as a real scanner, walk-forward backtest them against the 2-year local parquet cache, promote winners to `main`, and cleanly discard losers.

**Usage:**
- `/research-and-backtest` — autonomous mode: Claude picks the strategy class
- `/research-and-backtest "momentum"` — directed mode: research a specific class (e.g. "momentum", "mean reversion", "volatility breakout", "volume divergence")

**What this produces per run:** 3–5 research candidates → full scanner implementations → walk-forward backtest (no API calls) → 0–N promotions to `main` → brief discard notes for failures.

In CI (`CI=true`), stop before git operations — the workflow handles them.

---

## Step 1: Set Research Agenda

**Directed mode** (`$ARGUMENTS` is not empty): use the argument as the strategy class, skip to Step 2.

**Autonomous mode** (no argument):
- Read `docs/iterations/LEARNINGS.md` and list every file in `docs/iterations/research/` to build a do-not-repeat list.
- Note which pipelines already have OHLCV-capable scanners:
  - `mean_reversion`: `rsi_oversold`
  - `momentum`: `high_52w_breakout`, `minervini`, `technical_breakout`, `obv_divergence`
- Pick the highest-leverage uncovered OHLCV strategy class. Priority order:
  1. A well-known academic class with no current scanner (e.g. volatility regime, Bollinger squeeze, ATR expansion, calendar seasonality)
  2. A class mentioned in `LEARNINGS.md` as a gap
  3. A class complementary to the weakest-performing scanner
- Print: `"Strategy class: <class> — Reason: <why>"`

---

## Step 2: Deep Research — Find Hidden Gem Strategies

The goal is to find strategies with **documented statistical edge** that aren't widely known or implemented. Cast a wide net — search at least **8 distinct sources** before shortlisting. Do not stop after finding 3 obvious candidates; push into less-trafficked corners of the internet where the real edge lives.

**Always use Jina Reader** — prepend `https://r.jina.ai/` to every URL for clean extraction:

### 2a. Academic / Quantitative Research

```
# arXiv — search for systematic strategy papers
WebFetch("https://export.arxiv.org/api/query?search_query=ti:<topic>+cat:q-fin.TR&sortBy=relevance&max_results=8")
WebFetch("https://export.arxiv.org/api/query?search_query=abs:<topic>+cat:q-fin&sortBy=submittedDate&max_results=8")
# For each paper abstract that looks promising, fetch full text:
WebFetch("https://r.jina.ai/https://arxiv.org/abs/<id>")

# SSRN — working papers often predate published research by years
WebSearch("site:ssrn.com <topic> trading strategy backtest OHLCV")
WebFetch("https://r.jina.ai/https://papers.ssrn.com/sol3/papers.cfm?abstract_id=<id>")

# QuantPedia — curated academic strategy database
WebFetch("https://r.jina.ai/https://quantpedia.com/strategies/<topic>/")
WebFetch("https://r.jina.ai/https://quantpedia.com/category/<related-category>/")
```

### 2b. Practitioner Research Sites

```
# QuantifiedStrategies — systematic backtested strategies with exact rules
WebFetch("https://r.jina.ai/https://www.quantifiedstrategies.com/?s=<topic>")
# Read the full article for any result that mentions a win rate or return:
WebFetch("https://r.jina.ai/https://www.quantifiedstrategies.com/<article-slug>/")

# Alpha Architect — factor research with academic rigor
WebFetch("https://r.jina.ai/https://alphaarchitect.com/?s=<topic>")
WebFetch("https://r.jina.ai/https://alphaarchitect.com/<article-slug>/")

# CSS Analytics — lesser-known, deep quantitative analysis
WebFetch("https://r.jina.ai/https://cssanalytics.wordpress.com/?s=<topic>")

# Philosophical Economics — macro + equity research
WebFetch("https://r.jina.ai/https://www.philosophicaleconomics.com/?s=<topic>")

# Quant Dare / Hudson Thames — implementation-focused
WebFetch("https://r.jina.ai/https://hudsonthames.org/blog/")
WebSearch("site:hudsonthames.org <topic> strategy")
```

### 2c. Community Sources (Hidden Gems Live Here)

```
# Reddit r/algotrading — sort by top all-time for validated community finds
WebFetch("https://r.jina.ai/https://www.reddit.com/r/algotrading/search/?q=<topic>&sort=top&t=all")
# For any promising post, fetch the full thread:
WebFetch("https://r.jina.ai/https://www.reddit.com/r/algotrading/comments/<id>/")

# Reddit r/quant — more academic lean
WebFetch("https://r.jina.ai/https://www.reddit.com/r/quant/search/?q=<topic>&sort=top&t=all")

# Hacker News — quant finance discussions often surface niche papers
WebFetch("https://r.jina.ai/https://hn.algolia.com/api/v1/search?query=<topic>+trading&tags=story&hitsPerPage=10")
# For any HN post with >50 points, fetch the link it points to

# GitHub — open-source backtests reveal what actually works
WebSearch("site:github.com <topic> strategy backtest python stars:>50")
# Fetch any promising README:
WebFetch("https://r.jina.ai/https://github.com/<user>/<repo>")
```

### 2d. Books and Classic Sources

```
# Connors Research — RSI, short-term mean reversion
WebFetch("https://r.jina.ai/https://www.connorsresearch.com/trading-strategies/")

# Systematic Investor — R-based but strategy logic is portable
WebSearch("site:systematicinvestor.wordpress.com <topic>")

# Following the Trend (Clenow) / Stocks on the Move — trend-following
WebSearch("<topic> strategy \"annualized return\" \"win rate\" site:clenow.com OR site:following-the-trend.com")
```

### 2e. Strategy Discovery Rules

After searching 8+ sources, apply these filters before shortlisting:

**OHLCV-only gate** — only strategies computable from Open/High/Low/Close/Volume + derived indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, ADX, Stochastic, VWAP approximation). Skip anything requiring earnings, short interest, options, fundamental data, or external APIs.

**Evidence gate** — must have at least *qualitative* evidence of edge (a backtest result, a paper with statistics, or a practitioner reporting consistent real-money results). Pure theory or "this makes intuitive sense" is not enough.

**Duplication gate** — skip anything already covered by an existing scanner in `tradingagents/dataflows/discovery/scanners/` or previously researched in `docs/iterations/research/`.

**Novelty preference** — when two candidates have similar expected edge, prefer the one that's less obvious (e.g. a volume-price pattern from a 2019 arXiv paper over a basic moving average crossover). The pipeline already has basic momentum covered.

Shortlist **3–5 candidates**. For each, extract:
- Exact entry signal (thresholds, indicator values, lookback period)
- The source's reported win rate / avg return / Sharpe (even if approximate)
- Minimum OHLCV history required (in trading days)

Print: `"Shortlisted: [A, B, C] — Sources: [list] — Pre-implementation discards: [X] (reason)"`

---

## Step 3: Write Research Batch File

Save to `docs/iterations/research/YYYY-MM-DD-<class-slug>-batch.md` (e.g. `2026-04-20-volatility-breakout-batch.md`).

Template:
```markdown
# Research Batch: <Strategy Class>

**Date:** YYYY-MM-DD
**Mode:** directed | autonomous
**Candidates shortlisted:** N

## Sources Reviewed
- <source>: <key finding>

## Candidate Strategies

### 1. <Strategy Name>
**Signal logic:** <precise entry condition with thresholds>
**Academic edge:** <win rate / avg return / holding period / source>
**Data requirements:** <OHLCV fields needed, lookback in trading days>
**Proposed scanner name:** `<snake_case_name>`
**Pipeline:** momentum | mean_reversion | edge | events

### 2. <Strategy Name>
...

## Discarded Before Implementation
- **<Name>**: <one-line reason>

## Implementation Order
1. `<scanner_name>` — <reason for ranking>
2. `<scanner_name>` — ...
```

Append a row to `docs/iterations/LEARNINGS.md` under `## Research`:
```
| <Class> Batch | research/YYYY-MM-DD-<class-slug>-batch.md | YYYY-MM-DD | Researching N candidates in <class> class |
```

---

## Step 4: Implement All Scanners (before running any backtest)

Read `tradingagents/dataflows/discovery/scanners/rsi_oversold.py` as the canonical template. For each shortlisted candidate, do all four sub-steps:

### 4a. Create scanner file

`tradingagents/dataflows/discovery/scanners/<name>.py`

Required structure:
```python
"""<Strategy name> scanner.

<Academic citation / source>

Signal: <precise logic in one sentence>.
Expected holding period: <N>–<M> days.
Research: docs/iterations/research/YYYY-MM-DD-<class-slug>-batch.md
"""
from typing import Any, Dict, List, Optional
import pandas as pd
from tradingagents.dataflows.data_cache.ohlcv_cache import download_ohlcv_cached
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.dataflows.universe import load_universe
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

class <Name>Scanner(BaseScanner):
    name = "<snake_case>"
    pipeline = "<pipeline>"
    strategy = "<snake_case_strategy>"

    def __init__(self, config):
        super().__init__(config)
        # ALL params via scanner_config.get() — no hardcoded thresholds
        self.<param> = self.scanner_config.get("<param>", <default>)
        self.min_price = self.scanner_config.get("min_price", 5.0)
        self.min_avg_volume = self.scanner_config.get("min_avg_volume", 100_000)
        self.vol_avg_days = self.scanner_config.get("vol_avg_days", 20)
        self.max_tickers = self.scanner_config.get("max_tickers", 0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []
        tickers = load_universe(self.config)
        if not tickers:
            return []
        if self.max_tickers:
            tickers = tickers[:self.max_tickers]
        cache_dir = self.config.get("discovery", {}).get("ohlcv_cache_dir", "data/ohlcv_cache")
        data = download_ohlcv_cached(tickers, period="1y", cache_dir=cache_dir)
        if not data:
            return []

        candidates = []
        for ticker, df in data.items():
            result = self._check_signal(df)
            if result:
                result["ticker"] = ticker
                candidates.append(result)

        candidates.sort(key=lambda c: c.pop("_sort_key", 0), reverse=True)
        candidates = candidates[:self.limit]
        logger.info(f"{self.name}: {len(candidates)} candidates")
        return candidates

    def _check_signal(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        try:
            df = df.dropna(subset=["Close", "Volume"])
            if len(df) < <min_rows>:
                return None
            close = df["Close"]
            volume = df["Volume"]
            price = float(close.iloc[-1])
            avg_vol = float(volume.iloc[-(self.vol_avg_days+1):-1].mean())
            if price < self.min_price or avg_vol < self.min_avg_volume:
                return None

            # --- signal computation ---
            if <signal_not_triggered>:
                return None

            priority = Priority.HIGH.value  # adjust per signal strength
            return {
                "source": self.name,
                "context": f"<readable signal description>",
                "priority": priority,
                "strategy": self.strategy,
                "_sort_key": <numeric_rank>,
            }
        except Exception as e:
            logger.debug(f"{self.name} check failed: {e}")
            return None

SCANNER_REGISTRY.register(<Name>Scanner)
```

### 4b. Register in `__init__.py`

Add in alphabetical order to `tradingagents/dataflows/discovery/scanners/__init__.py`:
```python
from . import <name>  # noqa: F401
```

### 4c. Add config to `default_config.py`

Add under `discovery.scanners` in `tradingagents/default_config.py`:
```python
"<name>": {
    "enabled": True,
    "pipeline": "<pipeline>",
    "limit": 10,
    "max_tickers": 0,
    "<param>": <default>,  # one entry per scanner_config.get() call
},
```

---

## Step 5: Run Walk-Forward Backtest

Check if `scripts/backtest_scanners.py` exists:
```bash
ls scripts/backtest_scanners.py
```

**If it does not exist**, create it. The script must:
- Accept `--scanners <name1> <name2>` (space-separated), `--start YYYY-MM-DD`, `--out <dir>` CLI args
- Load the largest parquet from `data/ohlcv_cache/` (the 1003-ticker full-universe file)
- Walk every trading day from `--start` to `last_day - 40 calendar days` (forward-return buffer)
- For each simulation date: slice parquet to rows ≤ `sim_date`, import the scanner class dynamically, patch `download_ohlcv_cached` via `unittest.mock.patch` to return the slice, call `scanner.scan({})`
- For each pick: look up 1d/5d/10d/20d forward close prices from the same parquet to compute returns
- Write `results/backtest/<YYYY-MM-DD>/picks.csv` (one row per pick) and `summary.json` (one object per scanner with `picks`, `win_rate_1d/5d/10d/20d`, `avg_return_20d`, `median_return_20d`)
- To discover scanner classes: find `BaseScanner` subclasses in the imported module (do not use a hard-coded SCANNER_MODULES dict — import the module and inspect its members)

Then run:
```bash
python scripts/backtest_scanners.py \
  --scanners <name1> <name2> <name3> \
  --start 2025-04-15 \
  --out results/backtest
```

Zero API calls — uses only the local parquet cache.

**If a scanner produces 0 picks:** relax its primary threshold by 20% in `default_config.py`, re-run for that scanner only, then restore the original value. If still 0 picks, classify as DISCARD-CALIBRATION.

---

## Step 6: Classify Each Scanner

Read `results/backtest/<date>/summary.json`. Apply this decision matrix:

| Condition | Decision |
|-----------|----------|
| `win_rate_20d ≥ 55%` AND `avg_return_20d ≥ 3.0%` | **PROMOTE** |
| `win_rate_20d ≥ 52%` AND `avg_return_20d ≥ 2.0%` | **PROMOTE-MARGINAL** |
| `win_rate_20d ≥ 50%` AND `picks < 30` | **INCONCLUSIVE** |
| `win_rate_20d < 50%` | **DISCARD** |
| `picks == 0` after threshold relaxation | **DISCARD-CALIBRATION** |

Secondary check: if 20d passes but `win_rate_5d < 45%`, annotate "slow signal — hold ≥10d" but still promote.

---

## Step 7: Promote Winners (PROMOTE / PROMOTE-MARGINAL)

Scanner file, `__init__.py` import, and `default_config.py` entry already exist from Step 4 — no changes needed there.

Write `docs/iterations/scanners/<name>.md`:
```markdown
# Scanner: <Name>

**Module:** `tradingagents/dataflows/discovery/scanners/<name>.py`
**Pipeline:** <pipeline>
**Implemented:** YYYY-MM-DD

## Signal Logic
<copy from research file>

## Backtest Results (walk-forward, 2y OHLCV cache)
| Metric | Value |
|--------|-------|
| Total picks | N |
| Unique tickers | N |
| Win rate 1d / 5d / 10d / 20d | X% / X% / X% / X% |
| Avg return 20d | X% |
| Median return 20d | X% |

**Classification:** PROMOTE | PROMOTE-MARGINAL

## Current Understanding
<2–3 sentences on what the backtest reveals>

## Pending Hypotheses
- [ ] Test stricter/looser primary threshold
- [ ] Confirm signal in bear-market sub-period (check 2024-Q4 picks in picks.csv)

## Evidence Log
### YYYY-MM-DD — backtest (walk-forward, N picks)
- <key observations from picks.csv>
- Confidence: high (walk-forward)
```

Add row to `LEARNINGS.md` scanner table:
```
| <name> | scanners/<name>.md | YYYY-MM-DD | Backtest: WR-20d=X%, avg-20d=Y% — PROMOTED |
```

---

## Step 8: Discard Losers (DISCARD / DISCARD-CALIBRATION)

- Delete: `tradingagents/dataflows/discovery/scanners/<name>.py`
- Remove import from `tradingagents/dataflows/discovery/scanners/__init__.py`
- Remove config block from `tradingagents/default_config.py`
- Do NOT write a domain file. Append to the batch research file instead:

```markdown
## Backtest Discard Notes

### <Scanner Name> — DISCARD
- win_rate_20d: X%  avg_return_20d: Y%  picks: N
- Below-random performance at 20d horizon. Do not re-research.

### <Scanner Name> — DISCARD-CALIBRATION
- Zero picks after threshold relaxation.
- Likely cause: <threshold too strict / lookback too long>
```

---

## Step 9: INCONCLUSIVE Scanners

Keep scanner enabled in production (accumulate live picks). Write minimal domain file:
```markdown
# Scanner: <Name>

**Status:** Live monitoring — inconclusive backtest (N picks, below 30-pick threshold)
**Backtest:** win_rate_20d=X%, picks=N

## Pending Hypotheses
- [ ] Reclassify after 30 live picks
```

Add to `LEARNINGS.md`: `"Inconclusive backtest (N picks); live monitoring"`

---

## Step 10: Print Final Report

```
=================================================================
  /research-and-backtest — Results Summary
  Strategy class: <class>    Date: YYYY-MM-DD
=================================================================

RESEARCH
  Candidates shortlisted : N
  Discarded pre-impl     : N (OHLCV-incompatible / duplicate)

BACKTEST RESULTS
  Scanner            Picks  WR-1d  WR-5d  WR-10d  WR-20d  Avg-20d  Decision
  ─────────────────────────────────────────────────────────────────────────
  <name1>            NNN    XX%    XX%    XX%     XX%     +X.X%    PROMOTE
  <name2>            NNN    XX%    XX%    XX%     XX%     +X.X%    PROMOTE-MARGINAL
  <name3>            NNN    XX%    XX%    XX%     XX%     -X.X%    DISCARD
  <name4>            14     XX%    XX%    XX%     XX%     +X.X%    INCONCLUSIVE

ACTIONS TAKEN
  Promoted        : <name1>, <name2>
  Monitoring      : <name4>
  Discarded       : <name3>

FILES WRITTEN
  docs/iterations/research/YYYY-MM-DD-<class>-batch.md
  docs/iterations/scanners/<name1>.md  [promoted]
  docs/iterations/scanners/<name2>.md  [promoted, marginal]
  docs/iterations/scanners/<name4>.md  [monitoring]

NEXT STEPS
  - Monitor <name4> until 30 live picks, then re-classify
  - Run /backtest-hypothesis on <name2> to test threshold tuning
  - Run /research-and-backtest "<next class>" to continue coverage
=================================================================
```

---

## Step 11: Commit (skip if `CI=true`)

```bash
git add \
  tradingagents/dataflows/discovery/scanners/ \
  tradingagents/default_config.py \
  scripts/backtest_scanners.py \
  docs/iterations/research/ \
  docs/iterations/scanners/ \
  docs/iterations/LEARNINGS.md
git commit -m "research-and-backtest(<class>): YYYY-MM-DD — promoted N, discarded M, monitoring K"
```

Then push to `research/current` (create if needed), same pattern as `/research-strategy` Step 7.
