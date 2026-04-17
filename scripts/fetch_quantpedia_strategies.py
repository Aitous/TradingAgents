#!/usr/bin/env python3
"""QuantPedia strategy backlog scraper.

Fetches the QuantPedia strategy list and filters for OHLCV-compatible strategies
that are not already covered by existing scanners. Outputs a ranked backlog of
candidates to research, sorted by expected edge (Sharpe, returns).

Produces a deterministic, pre-vetted research backlog — replacing ad-hoc keyword
searches that return different results each run.

Usage:
    python scripts/fetch_quantpedia_strategies.py
    python scripts/fetch_quantpedia_strategies.py --output docs/iterations/research/quantpedia_backlog.md
    python scripts/fetch_quantpedia_strategies.py --min-sharpe 0.5 --top 20

Output:
    - Filtered, ranked list of OHLCV-compatible strategies
    - Saves to docs/iterations/research/quantpedia_backlog.md by default
"""

import argparse
import json
import re
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import requests
except ImportError:
    print("pip install requests  (needed for this script)")
    sys.exit(1)


# ── Keywords that indicate a strategy needs non-OHLCV data ──────────────────
NON_OHLCV_KEYWORDS = [
    "earnings", "eps", "fundamental", "balance sheet", "p/e", "pe ratio",
    "short interest", "short selling", "short ratio",
    "options", "implied volatility", "iv rank", "put/call",
    "insider", "13f", "institutional",
    "sentiment", "news", "twitter", "reddit", "social",
    "macro", "economic", "gdp", "inflation", "fed",
    "dividend", "yield", "payout",
    "analyst", "rating", "upgrade",
    "dark pool", "tape reading",
    "crypto", "bitcoin", "forex",
]

# ── Already implemented scanners (do-not-duplicate) ─────────────────────────
EXISTING_SCANNERS = {
    "rsi_oversold", "high_52w_breakout", "minervini", "technical_breakout",
    "atr_compression", "obv_divergence", "volume_dry_up",
    "earnings_beat", "earnings_calendar", "insider_buying", "options_flow",
    "short_squeeze", "dark_pool_flow", "analyst_upgrades",
    "volume_accumulation", "market_movers", "sector_rotation",
}

# ── Strategies known to be already researched/discarded ─────────────────────
DISCARDED = {
    "nr7 breakout", "bollinger squeeze", "consecutive down days",
    "pullback in uptrend", "adx trend inception",
    "selling climax reversal", "macd histogram reversal",
    "gap fill", "gap up continuation",
}


def fetch_quantpedia_strategies(min_sharpe: float) -> list[dict]:
    """Fetch strategies from QuantPedia public API."""
    url = "https://quantpedia.com/api/strategies/"
    params = {"format": "json", "page_size": 200}

    strategies = []
    page = 1
    while True:
        params["page"] = page
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"  ⚠️  QuantPedia API error on page {page}: {e}")
            break

        results = data.get("results", [])
        if not results:
            break
        strategies.extend(results)

        if not data.get("next"):
            break
        page += 1
        time.sleep(0.5)

    return strategies


def is_ohlcv_compatible(strategy: dict) -> tuple[bool, str]:
    """Return (is_compatible, reason) for a strategy."""
    name = (strategy.get("name") or "").lower()
    desc = (strategy.get("description") or "").lower()
    instruments = [i.lower() for i in (strategy.get("instruments") or [])]
    factors = [f.lower() for f in (strategy.get("factors") or [])]

    text = f"{name} {desc} {' '.join(instruments)} {' '.join(factors)}"

    # Must be equity-based
    if any(kw in text for kw in ["forex", "currency", "fx", "bitcoin", "crypto", "commodity", "futures", "bond"]):
        if "equity" not in text and "stock" not in text:
            return False, "non-equity instrument"

    # Check for non-OHLCV data requirements
    for kw in NON_OHLCV_KEYWORDS:
        if kw in text:
            return False, f"requires non-OHLCV data: '{kw}'"

    # Must be in the right instrument class
    instrument_list = " ".join(instruments)
    if instrument_list and not any(
        kw in instrument_list for kw in ["equit", "stock", "share", "us stock", "equity"]
    ):
        # If instruments are specified and none are equity, skip
        if any(kw in instrument_list for kw in ["forex", "futures", "bond", "commodity", "crypto"]):
            return False, "non-equity instruments specified"

    return True, ""


def already_covered(strategy: dict) -> bool:
    """Return True if this strategy is already in our pipeline or discarded."""
    name = (strategy.get("name") or "").lower()
    desc = (strategy.get("description") or "").lower()
    text = f"{name} {desc}"

    for scanner in EXISTING_SCANNERS:
        scanner_kw = scanner.replace("_", " ")
        if scanner_kw in text:
            return True

    for discarded in DISCARDED:
        if discarded in text:
            return True

    return False


def extract_sharpe(strategy: dict) -> float | None:
    """Extract Sharpe ratio from strategy metadata."""
    perf = strategy.get("performance") or {}
    sharpe = perf.get("sharpe_ratio") or strategy.get("sharpe_ratio")
    if sharpe is not None:
        try:
            return float(sharpe)
        except (ValueError, TypeError):
            pass
    return None


def format_markdown(strategies: list[dict], min_sharpe: float, top: int) -> str:
    lines = [
        f"# QuantPedia Strategy Backlog",
        f"",
        f"**Generated:** {date.today().isoformat()}  ",
        f"**Filters:** OHLCV-compatible, equity, not already covered, Sharpe ≥ {min_sharpe}  ",
        f"**Source:** quantpedia.com/api/strategies/",
        f"",
        f"Use this as the research backlog for `/research-and-backtest` runs.  ",
        f"Strategies are sorted by Sharpe ratio (highest first).",
        f"",
        f"---",
        f"",
    ]

    if not strategies:
        lines.append("No qualifying strategies found. Try lowering `--min-sharpe`.")
        return "\n".join(lines)

    for i, s in enumerate(strategies[:top], 1):
        sharpe = extract_sharpe(s)
        sharpe_str = f"{sharpe:.2f}" if sharpe else "N/A"
        perf = s.get("performance") or {}
        annual_return = perf.get("annual_return") or s.get("annual_return")
        ar_str = f"{annual_return:.1f}%" if annual_return else "N/A"
        holding = s.get("holding_period") or "N/A"
        url = s.get("url") or f"https://quantpedia.com/strategies/{s.get('id', '')}/"

        lines += [
            f"### {i}. {s.get('name', 'Unknown')}",
            f"",
            f"**Sharpe:** {sharpe_str}  **Annual Return:** {ar_str}  **Holding Period:** {holding}",
            f"",
            f"{(s.get('description') or '')[:300].strip()}{'...' if len(s.get('description') or '') > 300 else ''}",
            f"",
            f"**Source:** {url}",
            f"",
        ]

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Fetch QuantPedia OHLCV strategy backlog")
    parser.add_argument(
        "--output", default="docs/iterations/research/quantpedia_backlog.md",
        help="Output markdown file (default: docs/iterations/research/quantpedia_backlog.md)"
    )
    parser.add_argument(
        "--min-sharpe", type=float, default=0.3,
        help="Minimum Sharpe ratio filter (default: 0.3)"
    )
    parser.add_argument(
        "--top", type=int, default=30,
        help="Number of strategies to include in output (default: 30)"
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Also dump raw filtered strategies to a .json file"
    )
    args = parser.parse_args()

    print("Fetching QuantPedia strategies...")
    raw = fetch_quantpedia_strategies(args.min_sharpe)
    print(f"  Fetched {len(raw)} total strategies")

    # Filter
    compatible = []
    skip_counts = {}
    for s in raw:
        ok, reason = is_ohlcv_compatible(s)
        if not ok:
            skip_counts[reason] = skip_counts.get(reason, 0) + 1
            continue
        if already_covered(s):
            skip_counts["already covered"] = skip_counts.get("already covered", 0) + 1
            continue
        sharpe = extract_sharpe(s)
        if sharpe is not None and sharpe < args.min_sharpe:
            skip_counts[f"sharpe < {args.min_sharpe}"] = skip_counts.get(f"sharpe < {args.min_sharpe}", 0) + 1
            continue
        compatible.append(s)

    # Sort by Sharpe descending (None last)
    compatible.sort(key=lambda s: extract_sharpe(s) or -999, reverse=True)

    print(f"  After filtering: {len(compatible)} OHLCV-compatible, uncovered strategies")
    if skip_counts:
        print("  Skip reasons:")
        for reason, count in sorted(skip_counts.items(), key=lambda x: -x[1]):
            print(f"    {count:4d}  {reason}")

    # Write markdown
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = format_markdown(compatible, args.min_sharpe, args.top)
    out_path.write_text(md)
    print(f"\n✅ Written {min(len(compatible), args.top)} strategies to {out_path}")

    # Optional JSON dump
    if args.json:
        json_path = out_path.with_suffix(".json")
        json_path.write_text(json.dumps(compatible[: args.top], indent=2))
        print(f"✅ JSON dump: {json_path}")

    # Preview top 5
    if compatible:
        print("\nTop 5 candidates:")
        for s in compatible[:5]:
            sharpe = extract_sharpe(s)
            print(f"  {s.get('name', '?')}  (Sharpe={sharpe:.2f if sharpe else 'N/A'})")


if __name__ == "__main__":
    main()
