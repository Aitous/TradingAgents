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
- Read `docs/iterations/LEARNINGS.md` in full. The `## Discarded Signals` table is the authoritative do-not-repeat list — extract every signal name from it and treat these as hard exclusions for the entire run. A signal in that table must not be proposed, shortlisted, or implemented unless the "Re-research Condition" column explicitly permits it and that condition is now met.
- Note which pipelines already have OHLCV-capable scanners:
  - `mean_reversion`: `rsi_oversold`
  - `momentum`: `high_52w_breakout`, `minervini`, `technical_breakout`, `obv_divergence`
- **Check the QuantPedia backlog first:** If `docs/iterations/research/quantpedia_backlog.md` exists, read it and note the top strategies not yet researched or discarded. These are pre-vetted OHLCV-compatible candidates with documented Sharpe ratios — treat them as the primary candidate pool before doing any web searches. If the backlog file does not exist, run it: `python scripts/fetch_quantpedia_strategies.py --min-sharpe 0.3 --top 30`
- Pick the highest-leverage uncovered OHLCV strategy class. Priority order:
  1. Top unresearched strategy from the QuantPedia backlog (already filtered for OHLCV compatibility and Sharpe ≥ 0.3)
  2. A well-known academic class with no current scanner (e.g. volatility regime, Bollinger squeeze, ATR expansion, calendar seasonality)
  3. A class mentioned in `LEARNINGS.md` as a gap
  4. A class complementary to the weakest-performing scanner
- Print: `"Strategy class: <class> — Reason: <why>"`

---

## Step 2: Deep Research — Find Hidden Gem Strategies

The goal is to find strategies with **documented statistical edge** that aren't widely known or implemented. Cast a wide net — search at least **8 distinct sources** before shortlisting. Do not stop after finding 3 obvious candidates; push into less-trafficked corners of the internet where the real edge lives.

**Always use Jina Reader** — prepend `https://r.jina.ai/` to every URL for clean extraction:

### 2a. QuantPedia Backlog (Start Here)

If `docs/iterations/research/quantpedia_backlog.md` exists, read the top 10 entries. For each strategy in the backlog that matches the target class and hasn't been researched yet, this counts as one confirmed source — you still need 7 more from below, but the QuantPedia entry gives you a precise signal spec and Sharpe to anchor the search.

### 2b. Academic / Quantitative Research

```
# arXiv — search for systematic strategy papers
# IMPORTANT: Prefer papers published after 2019 — pre-2015 strategies are often arbitraged away
WebFetch("https://export.arxiv.org/api/query?search_query=ti:<topic>+cat:q-fin.TR&sortBy=submittedDate&max_results=8")
WebFetch("https://export.arxiv.org/api/query?search_query=abs:<topic>+cat:q-fin&sortBy=submittedDate&max_results=8")
# For each paper abstract that looks promising AND was submitted after 2019, fetch full text:
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

### 2e. Under-Trafficked High-Signal Sources (Check These Before Shortlisting)

These sources are less frequently cited but produce higher-quality, less-arbitraged signals:

```
# EPChan — practitioner-level mean reversion and stat arb, specific parameters
WebFetch("https://r.jina.ai/https://epchan.blogspot.com/search?q=<topic>")
WebSearch("site:epchan.com <topic> strategy")

# Newfound Research — regime-aware strategies with full backtest statistics
WebFetch("https://r.jina.ai/https://www.thinknewfound.com/?s=<topic>")

# Flirting with Models (Corey Hoffstein) — factor investing, momentum, trend
WebFetch("https://r.jina.ai/https://blog.thinknewfound.com/?s=<topic>")
WebSearch("site:blog.thinknewfound.com <topic>")

# Robot Wealth — ML + systematic trading with reproducible code
WebFetch("https://r.jina.ai/https://robotwealth.com/?s=<topic>")

# Composer / QuantConnect leaderboard — live-traded strategies filtered by Sharpe
WebSearch("site:quantconnect.com <topic> strategy sharpe")
# For QuantConnect results with Sharpe > 0.5 and >1 year live:
WebFetch("https://r.jina.ai/https://www.quantconnect.com/terminal/#open/<id>")
```

### 2f. Strategy Discovery Rules

After searching 8+ sources (including at least 2 from Section 2e), apply these filters before shortlisting:

**OHLCV-only gate** — only strategies computable from Open/High/Low/Close/Volume + derived indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ATR, OBV, ADX, Stochastic, VWAP approximation). Skip anything requiring earnings, short interest, options, fundamental data, or external APIs.

**Evidence gate** — must have at least *qualitative* evidence of edge (a backtest result, a paper with statistics, or a practitioner reporting consistent real-money results). Pure theory or "this makes intuitive sense" is not enough.

**Literature decay filter** — for academic papers, prefer evidence published after 2019. Pre-2015 documented anomalies have often been partially or fully arbitraged away. If only pre-2019 evidence exists, note this in the research file and apply a lower expected-edge estimate.

**Duplication gate** — skip anything:
- Already covered by an existing scanner in `tradingagents/dataflows/discovery/scanners/`
- Listed in the `## Discarded Signals` table in `docs/iterations/LEARNINGS.md` (check by exact signal name), unless its "Re-research Condition" is met
- Previously researched in `docs/iterations/research/` batch files

**Novelty preference** — when two candidates have similar expected edge, prefer the one that's less obvious (e.g. a volume-price pattern from a 2019 arXiv paper over a basic moving average crossover). The pipeline already has basic momentum covered.

**Independent cross-source confirmation** — a candidate qualifies for shortlisting only if the edge is reported by ≥2 *independent* sources (different authors, different venues). Two blog posts that both cite the same underlying paper count as ONE source. Two papers from the same research group count as ONE source. Explicitly note the distinct sources for each shortlisted candidate.

**Regime-conditional dimension** — every shortlisted candidate must specify in which market regime the signal is expected to work: trending / ranging / high-volatility / low-volatility / any. If the source doesn't specify, make an educated assessment and note it. Strategies with no regime specification tend to fail when the regime shifts; this annotation forces awareness before implementation.

**State + trigger structure check** — before shortlisting, ask: does this strategy have TWO independent components: (1) a persistent market state (e.g. uptrend, low volatility, compressed range) AND (2) a rare confirmation trigger (e.g. breakout bar, volume surge, pocket pivot)? Strategies with this dual-condition structure have empirically outperformed single-condition signals in this pipeline (`atr_compression`: ATR regime + price breakout; `volume_dry_up`: VDU state + pocket pivot bar). Single-condition signals tend to fire too frequently and fail the selectivity gate. If a candidate lacks this structure, either design one in during implementation or downgrade its priority.

Shortlist **3–5 candidates**. For each, extract:
- Exact entry signal (thresholds, indicator values, lookback period)
- The source's reported win rate / avg return / Sharpe (even if approximate)
- Minimum OHLCV history required (in trading days)
- **State component** (what persistent condition must be true?)
- **Trigger component** (what rare event fires the signal?)

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

**Model selection:** Spawn one Sonnet subagent per scanner (Agent tool,
`model="sonnet"`) to write the scanner file, `__init__.py` import, and
`default_config.py` entry. Pass each subagent: the full signal spec from Step
3, the content of `rsi_oversold.py` as the canonical template, and the
relevant `default_config.py` section. All other steps (research, classification,
markdown, commit) run at the current model level.

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

## Step 4.5: Signal Frequency Gate (Pre-Backtest Selectivity Check)

Before running the full walk-forward backtest, check that each scanner is selective enough to be worth testing. Full backtests take 5–15 minutes; catching unselective signals here saves that time.

For each new scanner, run:
```bash
python scripts/estimate_signal_frequency.py --scanner <name> --days 20
```

**Decision rules based on avg picks/day:**

| avg picks/day | Rating | Action |
|---------------|--------|--------|
| > 8 | 🔴 UNSELECTIVE | **STOP** — do not run full backtest. The signal fires daily and will almost certainly fail promotion. Fix the scanner first. |
| 5–8 | 🟡 BORDERLINE | Add one more tightening condition or raise the primary threshold before proceeding. Re-run frequency check. |
| 2–5 | 🟢 SELECTIVE | Proceed to full backtest. |
| < 2 | 🟢🟢 VERY SELECTIVE | Proceed. Small-sample caveat will likely apply — note in classification. |

**How to fix an unselective scanner (avg > 5/day):**
1. Raise the primary threshold (e.g. tighter RSI, higher ATR ratio, larger price move required)
2. Add a second independent rare condition (a "trigger" component on top of a "state" component)
3. Add a trend filter (e.g. `price > SMA200` — this alone often halves pick count)
4. Increase minimum lookback window (e.g. require state to persist for N days, not just today)

**Empirical benchmark from this project:**
- `volume_dry_up`: 0.09/day → WR-20d=80% ✅ PROMOTED
- `atr_compression`: 3.3/day → WR-20d=59% ✅ PROMOTED
- `macd_histogram_reversal`: 8.7/day → WR-20d=54% ❌ DISCARDED
- `consecutive_down_days`: 9.5/day → WR-20d=54% ❌ DISCARDED

Do not proceed to Step 5 for any scanner with avg > 8 picks/day. Fix it first or discard it at this stage (mark as DISCARD-CALIBRATION in the batch research file).

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

Read `results/backtest/<date>/summary.json`.

**Pre-classification: cost adjustment**

Before applying the decision matrix, subtract realistic transaction costs from `avg_return_20d`:
- Round-trip commission + spread: **10 bps** (0.10%)
- Slippage (entry + exit): **5 bps** (0.05%)
- Total cost deduction: **15 bps** (0.15%)

`adj_return_20d = avg_return_20d - 0.15%`

Use `adj_return_20d` (not raw `avg_return_20d`) in the decision matrix below.

**Pre-classification: regime-split check**

Split the walk-forward period into 4 equal sub-periods. Count how many sub-periods have `win_rate_20d ≥ 50%`:
- ≥ 3 of 4 sub-periods pass → **regime-stable** (no annotation needed)
- 2 of 4 sub-periods pass → annotate "**regime-dependent** — monitor for regime changes"
- ≤ 1 of 4 sub-periods pass → **DISCARD** regardless of overall WR (strategy only worked in one market regime)

This check runs before the decision matrix. A scanner that fails the regime-split goes straight to DISCARD even if its overall WR passes.

**Decision matrix** (apply after both pre-classification checks):

| Condition | Decision |
|-----------|----------|
| `win_rate_20d ≥ 55%` AND `adj_return_20d ≥ 2.85%` | **PROMOTE** |
| `win_rate_20d ≥ 52%` AND `adj_return_20d ≥ 1.85%` | **PROMOTE-MARGINAL** |
| `win_rate_20d ≥ 50%` AND `picks < 30` | **INCONCLUSIVE** |
| `win_rate_20d < 50%` OR regime-split ≤ 1/4 | **DISCARD** |
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
| Avg return 20d (raw) | X% |
| Avg return 20d (cost-adj, -15bps) | X% |
| Median return 20d | X% |
| Regime stability | X/4 sub-periods ≥ 50% WR |
| Top sector concentration | X% in <sector> |
| Avg picks/day | X |
| Est. concurrent positions (picks/day × 20d hold) | X |

**Classification:** PROMOTE | PROMOTE-MARGINAL
**Regime:** trending / ranging / high-vol / low-vol / any

> **Capacity note:** At X picks/day with 20d avg hold = ~X concurrent positions. At 1% position sizing this strategy saturates a portfolio at ~$Xk AUM. Monitor if pick rate increases significantly.

> **Sector note:** <mention if top sector >40% of picks — flag as sector-risk or confirm sector-agnostic>

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
- Remove from `scripts/backtest_scanners.py` `SCANNER_MODULES`
- **Append a row to the `## Discarded Signals` table in `docs/iterations/LEARNINGS.md`:**
  ```
  | `<name>` | DISCARD / DISCARD-CALIBRATION | YYYY-MM-DD | <one-line why> | <condition under which re-research is permitted, or "Never"> |
  ```
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

## Step 10: Post-Backtest Analysis

For any promoted scanner with ≥30 picks, run the feature importance analysis to identify which numeric fields most predict success — this surfaces threshold tuning opportunities immediately:

```bash
python scripts/analyze_backtest_picks.py --scanner <name> --horizon 20
```

If 2+ scanners were promoted this run, check for confluence lift across the full picks dataset:

```bash
python scripts/confluence_analysis.py --horizon 20 --min-picks 5
```

Confluence pairs with positive WR lift indicate the two scanners are orthogonal signals — note in the domain files and `LEARNINGS.md` as a confirmed combination worth monitoring in the live pipeline.

---

## Step 11: Print Final Report

```
=================================================================
  /research-and-backtest — Results Summary
  Strategy class: <class>    Date: YYYY-MM-DD
=================================================================

RESEARCH
  Candidates shortlisted : N
  Discarded pre-impl     : N (OHLCV-incompatible / duplicate)
  Frequency-gated (>8/day): N

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
  - (If confluence analysis found positive lift pairs, note them in LEARNINGS.md)
=================================================================
```

---

## Step 12: Commit (skip if `CI=true`)

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
