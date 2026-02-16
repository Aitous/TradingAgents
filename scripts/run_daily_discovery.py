#!/usr/bin/env python3
"""
Daily Discovery Runner — non-interactive script for cron/launchd scheduling.

Runs the full discovery pipeline (scan → filter → rank), saves recommendations,
and updates position tracking.  Designed to run before market open (~8:30 AM ET).

Usage:
    python scripts/run_daily_discovery.py                   # Uses defaults
    python scripts/run_daily_discovery.py --date 2026-02-12 # Specific date
    python scripts/run_daily_discovery.py --provider google  # Override LLM provider

Scheduling (macOS launchd):
    See the companion plist at scripts/com.tradingagents.discovery.plist

Scheduling (cron):
    30 13 * * 1-5 cd /path/to/TradingAgents && .venv/bin/python scripts/run_daily_discovery.py >> logs/discovery_cron.log 2>&1
    (13:30 UTC = 8:30 AM ET, weekdays only)
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.discovery_graph import DiscoveryGraph
from tradingagents.utils.logger import get_logger

logger = get_logger("daily_discovery")


def parse_args():
    parser = argparse.ArgumentParser(description="Run daily discovery pipeline")
    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider override (openai, google, anthropic)",
    )
    parser.add_argument(
        "--shallow-model",
        default=None,
        help="Override quick_think_llm model name",
    )
    parser.add_argument(
        "--deep-model",
        default=None,
        help="Override deep_think_llm model name",
    )
    parser.add_argument(
        "--update-positions",
        action="store_true",
        default=True,
        help="Update position tracking after discovery (default: True)",
    )
    parser.add_argument(
        "--no-update-positions",
        action="store_false",
        dest="update_positions",
    )
    return parser.parse_args()


def run_discovery(args):
    """Run the discovery pipeline with the given arguments."""
    config = DEFAULT_CONFIG.copy()

    # Apply overrides
    if args.provider:
        config["llm_provider"] = args.provider.lower()
    if args.shallow_model:
        config["quick_think_llm"] = args.shallow_model
    if args.deep_model:
        config["deep_think_llm"] = args.deep_model

    set_config(config)

    # Create results directory
    run_timestamp = datetime.now().strftime("%H_%M_%S")
    results_dir = Path(config["results_dir"]) / "discovery" / args.date / f"run_{run_timestamp}"
    results_dir.mkdir(parents=True, exist_ok=True)
    config["discovery_run_dir"] = str(results_dir)

    logger.info(f"Starting daily discovery for {args.date}")
    logger.info(
        f"Provider: {config['llm_provider']} | "
        f"Shallow: {config['quick_think_llm']} | "
        f"Deep: {config['deep_think_llm']}"
    )

    # Run discovery
    graph = DiscoveryGraph(config=config)
    result = graph.run(trade_date=args.date)

    final_ranking = result.get("final_ranking", "No ranking available")
    logger.info(f"Discovery complete. Results saved to {results_dir}")

    return result


def update_positions():
    """Run position updates after discovery."""
    try:
        from scripts.update_positions import main as update_main

        logger.info("Updating position tracking...")
        update_main()
    except Exception as e:
        logger.error(f"Position update failed: {e}")


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info(f"DAILY DISCOVERY RUN — {datetime.now().isoformat()}")
    logger.info("=" * 60)

    try:
        result = run_discovery(args)

        if args.update_positions:
            update_positions()

        logger.info("Daily discovery completed successfully")

    except Exception as e:
        logger.error(f"Discovery failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
