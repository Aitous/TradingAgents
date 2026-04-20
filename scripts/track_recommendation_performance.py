#!/usr/bin/env python3
"""
Daily Performance Tracker

Tracks the performance of historical recommendations and updates the database.
Run this daily (via cron or manually) to monitor how recommendations perform over time.

Usage:
    python scripts/track_recommendation_performance.py

Cron example (runs daily at 5pm after market close):
    0 17 * * 1-5 cd /path/to/TradingAgents && python scripts/track_recommendation_performance.py
"""

import glob
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.dataflows.y_finance import get_stock_price
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def load_recommendations() -> List[Dict[str, Any]]:
    """Load all historical recommendations, preferring the performance database.

    The performance database preserves accumulated return data (return_1d,
    return_7d, win_1d, etc.) across runs.  Raw date files are only used to
    pick up new recommendations not yet in the database.
    """
    recommendations_dir = "data/recommendations"
    if not os.path.exists(recommendations_dir):
        logger.warning(f"No recommendations directory found at {recommendations_dir}")
        return []

    # Step 1: Load existing accumulated data from the performance database
    existing: Dict[str, Dict[str, Any]] = {}
    db_path = os.path.join(recommendations_dir, "performance_database.json")
    if os.path.exists(db_path):
        try:
            with open(db_path, "r") as f:
                db = json.load(f)
            for recs in db.get("recommendations_by_date", {}).values():
                if isinstance(recs, list):
                    for rec in recs:
                        key = f"{rec.get('ticker')}|{rec.get('discovery_date')}"
                        existing[key] = rec
            logger.info(f"Loaded {len(existing)} records from performance database")
        except Exception as e:
            logger.error(f"Error loading performance database: {e}")

    # Step 2: Scan raw date files for any new recommendations
    new_count = 0
    for filepath in glob.glob(os.path.join(recommendations_dir, "*.json")):
        basename = os.path.basename(filepath)
        if basename in ("performance_database.json", "statistics.json"):
            continue
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
            recs = data.get("recommendations", [])
            run_date = data.get("date", basename.replace(".json", ""))
            for rec in recs:
                rec["discovery_date"] = run_date
                key = f"{rec.get('ticker')}|{run_date}"
                if key not in existing:
                    existing[key] = rec
                    new_count += 1
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    if new_count:
        logger.info(f"Merged {new_count} new recommendations from raw files")

    return list(existing.values())


def _parse_price(raw) -> float | None:
    """Extract a numeric price from get_stock_price output.

    The function may return a float directly or a markdown string like
    "**Current Price**: $123.45".  Handle both cases.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return float(raw)
    import re

    m = re.search(r"\$([0-9,.]+)", str(raw))
    return float(m.group(1).replace(",", "")) if m else None


def update_performance(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Update performance metrics for all recommendations."""
    today = datetime.now().strftime("%Y-%m-%d")

    for rec in recommendations:
        ticker = rec.get("ticker")
        discovery_date = rec.get("discovery_date")
        entry_price = rec.get("entry_price")

        if not all([ticker, discovery_date, entry_price]):
            continue

        if rec.get("status") == "closed":
            continue

        try:
            current_price = _parse_price(get_stock_price(ticker, curr_date=today))
            if current_price is None:
                logger.warning(f"Could not get price for {ticker}")
                continue

            rec_date = datetime.strptime(discovery_date, "%Y-%m-%d")
            days_held = (datetime.now() - rec_date).days
            return_pct = ((current_price - entry_price) / entry_price) * 100

            rec["current_price"] = current_price
            rec["return_pct"] = round(return_pct, 2)
            rec["days_held"] = days_held
            rec["last_updated"] = today

            # Capture milestone returns (only once per milestone)
            if days_held >= 1 and "return_1d" not in rec:
                rec["return_1d"] = round(return_pct, 2)
                rec["win_1d"] = return_pct > 0

            if days_held >= 7 and "return_7d" not in rec:
                rec["return_7d"] = round(return_pct, 2)
                rec["win_7d"] = return_pct > 0

            if days_held >= 30 and "return_30d" not in rec:
                rec["return_30d"] = round(return_pct, 2)
                rec["win_30d"] = return_pct > 0
                rec["status"] = "closed"

            logger.info(
                f"✓ {ticker}: Entry ${entry_price:.2f} → Current ${current_price:.2f} ({return_pct:+.1f}%) [{days_held}d]"
            )

        except Exception as e:
            logger.error(f"✗ Error tracking {ticker}: {e}")

    return recommendations


def save_performance_database(recommendations: List[Dict[str, Any]]):
    """Save the updated performance database."""
    db_path = "data/recommendations/performance_database.json"

    # Group by discovery date for organized storage
    by_date = {}
    for rec in recommendations:
        date = rec.get("discovery_date", "unknown")
        if date not in by_date:
            by_date[date] = []
        by_date[date].append(rec)

    database = {
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_recommendations": len(recommendations),
        "recommendations_by_date": by_date,
    }

    with open(db_path, "w") as f:
        json.dump(database, f, indent=2)

    logger.info(f"\n💾 Saved performance database to {db_path}")


def calculate_statistics(recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate aggregate statistics from historical performance.

    Delegates to DiscoveryAnalytics.calculate_statistics so there is a single
    source of truth for strategy normalization and metric calculation.
    """
    from tradingagents.dataflows.discovery.analytics import DiscoveryAnalytics

    analytics = DiscoveryAnalytics()
    return analytics.calculate_statistics(recommendations)


def print_statistics(stats: Dict[str, Any]):
    """Print formatted statistics report."""
    logger.info("\n" + "=" * 60)
    logger.info("RECOMMENDATION PERFORMANCE STATISTICS")
    logger.info("=" * 60)

    logger.info(f"\nTotal Recommendations Tracked: {stats['total_recommendations']}")

    # Overall stats
    logger.info("\n📊 OVERALL PERFORMANCE")
    logger.info("-" * 60)

    if stats["overall_7d"]["count"] > 0:
        logger.info("7-Day Performance:")
        logger.info(f"  • Tracked: {stats['overall_7d']['count']} recommendations")
        logger.info(f"  • Win Rate: {stats['overall_7d']['win_rate']}%")
        logger.info(f"  • Avg Return: {stats['overall_7d']['avg_return']:+.2f}%")

    if stats["overall_30d"]["count"] > 0:
        logger.info("\n30-Day Performance:")
        logger.info(f"  • Tracked: {stats['overall_30d']['count']} recommendations")
        logger.info(f"  • Win Rate: {stats['overall_30d']['win_rate']}%")
        logger.info(f"  • Avg Return: {stats['overall_30d']['avg_return']:+.2f}%")

    # By strategy
    if stats["by_strategy"]:
        logger.info("\n📈 PERFORMANCE BY STRATEGY")
        logger.info("-" * 60)

        # Sort by win rate (if available)
        sorted_strategies = sorted(
            stats["by_strategy"].items(), key=lambda x: x[1].get("win_rate_7d", 0), reverse=True
        )

        for strategy, data in sorted_strategies:
            logger.info(f"\n{strategy}:")
            logger.info(f"  • Total: {data['count']} recommendations")

            if data.get("win_rate_7d"):
                logger.info(
                    f"  • 7-Day Win Rate: {data['win_rate_7d']}% ({data['wins_7d']}W/{data['losses_7d']}L)"
                )

            if data.get("win_rate_30d"):
                logger.info(
                    f"  • 30-Day Win Rate: {data['win_rate_30d']}% ({data['wins_30d']}W/{data['losses_30d']}L)"
                )


def _get_price_on_date(ticker: str, date_str: str) -> float | None:
    """Fetch closing price for ticker on a given date using yfinance."""
    try:
        return _parse_price(get_stock_price(ticker, curr_date=date_str))
    except Exception:
        return None


def _compute_forward_returns(entry_price: float, discovery_date: str, ticker: str, today: str) -> dict:
    """Compute 1d/7d/30d return milestones that are now reachable."""
    from datetime import datetime as dt
    results = {}
    discovery_dt = dt.strptime(discovery_date, "%Y-%m-%d")
    today_dt = dt.strptime(today, "%Y-%m-%d")
    days_held = (today_dt - discovery_dt).days

    for horizon, key in [(1, "1d"), (7, "7d"), (30, "30d")]:
        if days_held >= horizon:
            target = (discovery_dt + __import__("datetime").timedelta(days=horizon)).strftime("%Y-%m-%d")
            price = _get_price_on_date(ticker, target)
            if price:
                ret = round((price - entry_price) / entry_price * 100, 2)
                results[f"return_{key}"] = ret
                results[f"win_{key}"] = ret > 0
    return results


def _load_json_file(path: str) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_json_atomic(path: str, data: dict):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.rename(tmp, path)


def update_scanner_picks():
    """Backfill entry prices and forward returns for raw scanner picks."""
    today = datetime.now().strftime("%Y-%m-%d")
    picks_dir = "data/scanner_picks"
    if not os.path.exists(picks_dir):
        return

    for filepath in sorted(glob.glob(os.path.join(picks_dir, "*.json"))):
        data = _load_json_file(filepath)
        picks = data.get("picks", [])
        changed = False

        for pick in picks:
            if pick.get("status") == "closed":
                continue
            ticker = pick.get("ticker")
            discovery_date = pick.get("discovery_date")
            if not ticker or not discovery_date:
                continue

            # Backfill entry price (T+1 open approximated as T+1 close)
            if not pick.get("entry_price"):
                from datetime import datetime as dt, timedelta
                t1 = (dt.strptime(discovery_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                price = _get_price_on_date(ticker, t1)
                if price:
                    pick["entry_price"] = price
                    changed = True

            entry_price = pick.get("entry_price")
            if not entry_price:
                continue

            milestones = _compute_forward_returns(entry_price, discovery_date, ticker, today)
            for k, v in milestones.items():
                if k not in pick:
                    pick[k] = v
                    changed = True

            if pick.get("return_30d") is not None and pick.get("status") != "closed":
                pick["status"] = "closed"
                changed = True

        if changed:
            data["picks"] = picks
            _save_json_atomic(filepath, data)
            logger.info(f"Updated scanner picks: {filepath}")


def update_discovery_events():
    """Backfill entry prices and forward returns for discovery events (ranker input set)."""
    today = datetime.now().strftime("%Y-%m-%d")
    events_dir = "data/discovery_events"
    if not os.path.exists(events_dir):
        return

    for filepath in sorted(glob.glob(os.path.join(events_dir, "*.json"))):
        data = _load_json_file(filepath)
        events = data.get("events", [])
        changed = False

        for event in events:
            if event.get("status") == "closed":
                continue
            ticker = event.get("ticker")
            discovery_date = event.get("discovery_date")
            if not ticker or not discovery_date:
                continue

            if not event.get("entry_price"):
                from datetime import datetime as dt, timedelta
                t1 = (dt.strptime(discovery_date, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
                price = _get_price_on_date(ticker, t1)
                if price:
                    event["entry_price"] = price
                    changed = True

            entry_price = event.get("entry_price")
            if not entry_price:
                continue

            milestones = _compute_forward_returns(entry_price, discovery_date, ticker, today)
            for k, v in milestones.items():
                if k not in event:
                    event[k] = v
                    changed = True

            if event.get("return_30d") is not None and event.get("status") != "closed":
                event["status"] = "closed"
                changed = True

        if changed:
            data["events"] = events
            _save_json_atomic(filepath, data)
            logger.info(f"Updated discovery events: {filepath}")


def main():
    """Main execution function."""
    logger.info("🔍 Loading historical recommendations...")
    recommendations = load_recommendations()

    if not recommendations:
        logger.warning("No recommendations found to track.")
        return

    logger.info(f"Found {len(recommendations)} total recommendations")

    # Filter to only track open positions (not closed after 30 days)
    open_recs = [r for r in recommendations if r.get("status") != "closed"]
    logger.info(f"Tracking {len(open_recs)} open positions...")

    logger.info("\n📊 Updating performance metrics...\n")
    updated_recs = update_performance(recommendations)

    logger.info("\n📈 Calculating statistics...")
    stats = calculate_statistics(updated_recs)

    print_statistics(stats)

    save_performance_database(updated_recs)

    # Also save stats separately
    stats_path = "data/recommendations/statistics.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"💾 Saved statistics to {stats_path}")

    logger.info("\n📋 Updating raw scanner picks...")
    update_scanner_picks()

    logger.info("\n📋 Updating discovery events (ranker input set)...")
    update_discovery_events()

    logger.info("\n✅ Performance tracking complete!")


if __name__ == "__main__":
    main()
