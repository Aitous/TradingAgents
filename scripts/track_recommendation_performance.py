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
from pathlib import Path
from typing import Any, Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tradingagents.dataflows.y_finance import get_stock_price
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def load_recommendations() -> List[Dict[str, Any]]:
    """Load all historical recommendations from the recommendations directory."""
    recommendations_dir = "data/recommendations"
    if not os.path.exists(recommendations_dir):
        logger.warning(f"No recommendations directory found at {recommendations_dir}")
        return []

    all_recs = []
    pattern = os.path.join(recommendations_dir, "*.json")

    for filepath in glob.glob(pattern):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                # Each file contains recommendations from one discovery run
                recs = data.get("recommendations", [])
                run_date = data.get("date", os.path.basename(filepath).replace(".json", ""))

                for rec in recs:
                    rec["discovery_date"] = run_date
                    all_recs.append(rec)
        except Exception as e:
            logger.error(f"Error loading {filepath}: {e}")

    return all_recs


def update_performance(recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Update performance metrics for all recommendations."""
    today = datetime.now().strftime("%Y-%m-%d")

    for rec in recommendations:
        ticker = rec.get("ticker")
        discovery_date = rec.get("discovery_date")
        entry_price = rec.get("entry_price")

        if not all([ticker, discovery_date, entry_price]):
            continue

        # Skip if already marked as closed
        if rec.get("status") == "closed":
            continue

        try:
            # Get current price
            current_price_data = get_stock_price(ticker, curr_date=today)

            # Parse the price from the response (it returns a markdown report)
            # Format is typically: "**Current Price**: $XXX.XX"
            import re

            price_match = re.search(r"\$([0-9,.]+)", current_price_data)
            if price_match:
                current_price = float(price_match.group(1).replace(",", ""))
            else:
                logger.warning(f"Could not parse price for {ticker}")
                continue

            # Calculate days since recommendation
            rec_date = datetime.strptime(discovery_date, "%Y-%m-%d")
            days_held = (datetime.now() - rec_date).days

            # Calculate return
            return_pct = ((current_price - entry_price) / entry_price) * 100

            # Update metrics
            rec["current_price"] = current_price
            rec["return_pct"] = round(return_pct, 2)
            rec["days_held"] = days_held
            rec["last_updated"] = today

            # Check specific time periods
            if days_held >= 7 and "return_7d" not in rec:
                rec["return_7d"] = round(return_pct, 2)

            if days_held >= 30 and "return_30d" not in rec:
                rec["return_30d"] = round(return_pct, 2)
                rec["status"] = "closed"  # Mark as complete after 30 days

            # Determine win/loss for completed periods
            if "return_7d" in rec:
                rec["win_7d"] = rec["return_7d"] > 0

            if "return_30d" in rec:
                rec["win_30d"] = rec["return_30d"] > 0

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
    """Calculate aggregate statistics from historical performance."""
    stats = {
        "total_recommendations": len(recommendations),
        "by_strategy": {},
        "overall_7d": {"count": 0, "wins": 0, "avg_return": 0},
        "overall_30d": {"count": 0, "wins": 0, "avg_return": 0},
    }

    # Calculate by strategy
    for rec in recommendations:
        strategy = rec.get("strategy_match", "unknown")

        if strategy not in stats["by_strategy"]:
            stats["by_strategy"][strategy] = {
                "count": 0,
                "wins_7d": 0,
                "losses_7d": 0,
                "wins_30d": 0,
                "losses_30d": 0,
                "avg_return_7d": 0,
                "avg_return_30d": 0,
            }

        stats["by_strategy"][strategy]["count"] += 1

        # 7-day stats
        if "return_7d" in rec:
            stats["overall_7d"]["count"] += 1
            if rec.get("win_7d"):
                stats["overall_7d"]["wins"] += 1
                stats["by_strategy"][strategy]["wins_7d"] += 1
            else:
                stats["by_strategy"][strategy]["losses_7d"] += 1
            stats["overall_7d"]["avg_return"] += rec["return_7d"]

        # 30-day stats
        if "return_30d" in rec:
            stats["overall_30d"]["count"] += 1
            if rec.get("win_30d"):
                stats["overall_30d"]["wins"] += 1
                stats["by_strategy"][strategy]["wins_30d"] += 1
            else:
                stats["by_strategy"][strategy]["losses_30d"] += 1
            stats["overall_30d"]["avg_return"] += rec["return_30d"]

    # Calculate averages and win rates
    if stats["overall_7d"]["count"] > 0:
        stats["overall_7d"]["win_rate"] = round(
            (stats["overall_7d"]["wins"] / stats["overall_7d"]["count"]) * 100, 1
        )
        stats["overall_7d"]["avg_return"] = round(
            stats["overall_7d"]["avg_return"] / stats["overall_7d"]["count"], 2
        )

    if stats["overall_30d"]["count"] > 0:
        stats["overall_30d"]["win_rate"] = round(
            (stats["overall_30d"]["wins"] / stats["overall_30d"]["count"]) * 100, 1
        )
        stats["overall_30d"]["avg_return"] = round(
            stats["overall_30d"]["avg_return"] / stats["overall_30d"]["count"], 2
        )

    # Calculate per-strategy stats
    for strategy, data in stats["by_strategy"].items():
        total_7d = data["wins_7d"] + data["losses_7d"]
        total_30d = data["wins_30d"] + data["losses_30d"]

        if total_7d > 0:
            data["win_rate_7d"] = round((data["wins_7d"] / total_7d) * 100, 1)

        if total_30d > 0:
            data["win_rate_30d"] = round((data["wins_30d"] / total_30d) * 100, 1)

    return stats


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

    logger.info("\n✅ Performance tracking complete!")


if __name__ == "__main__":
    main()
