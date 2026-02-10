#!/usr/bin/env python3
"""
Ticker Database Updater
Maintains and augments the ticker list in data/tickers.txt

Usage:
    python scripts/update_ticker_database.py [OPTIONS]

Examples:
    # Validate and clean existing list
    python scripts/update_ticker_database.py --validate

    # Add specific tickers
    python scripts/update_ticker_database.py --add NVDA,PLTR,HOOD

    # Fetch latest from Alpha Vantage
    python scripts/update_ticker_database.py --fetch-alphavantage
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Set

import requests
from dotenv import load_dotenv

from tradingagents.utils.logger import get_logger

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = get_logger(__name__)


class TickerDatabaseUpdater:
    def __init__(self, ticker_file: str = "data/tickers.txt"):
        self.ticker_file = ticker_file
        self.tickers: Set[str] = set()
        self.added_count = 0
        self.removed_count = 0

    def load_tickers(self) -> Set[str]:
        """Load existing tickers from file."""
        logger.info(f"📖 Loading tickers from {self.ticker_file}...")

        try:
            with open(self.ticker_file, "r") as f:
                for line in f:
                    symbol = line.strip()
                    if symbol and symbol.isalpha():
                        self.tickers.add(symbol.upper())

            logger.info(f"   ✓ Loaded {len(self.tickers)} tickers")
            return self.tickers

        except FileNotFoundError:
            logger.info("   ℹ️  File not found, starting fresh")
            return set()
        except Exception as e:
            logger.warning(f"   ⚠️  Error loading: {str(e)}")
            return set()

    def add_tickers(self, new_tickers: list):
        """Add new tickers to the database."""
        logger.info(f"\n➕ Adding tickers: {', '.join(new_tickers)}")

        for ticker in new_tickers:
            ticker = ticker.strip().upper()
            if ticker and ticker.isalpha():
                if ticker not in self.tickers:
                    self.tickers.add(ticker)
                    self.added_count += 1
                    logger.info(f"   ✓ Added {ticker}")
                else:
                    logger.info(f"   ℹ️  {ticker} already exists")

    def validate_and_clean(self, remove_warrants=False, remove_preferred=False):
        """Validate tickers and remove invalid ones."""
        logger.info(f"\n🔍 Validating {len(self.tickers)} tickers...")

        invalid = set()
        for ticker in self.tickers:
            # Remove if not alphabetic or too long
            if not ticker.isalpha() or len(ticker) > 5 or len(ticker) < 1:
                invalid.add(ticker)
                continue

            # Optionally remove warrants (ending in W)
            if remove_warrants and ticker.endswith("W") and len(ticker) > 1:
                invalid.add(ticker)
                continue

            # Optionally remove preferred shares (ending in P after checking it's not a regular stock)
            if remove_preferred and ticker.endswith("P") and len(ticker) > 1:
                invalid.add(ticker)

        if invalid:
            logger.warning(f"   ⚠️  Found {len(invalid)} problematic tickers")

            # Categorize for reporting
            warrants = [t for t in invalid if t.endswith("W")]
            preferred = [t for t in invalid if t.endswith("P")]
            other_invalid = [t for t in invalid if not (t.endswith("W") or t.endswith("P"))]

            if warrants and remove_warrants:
                logger.info(f"      Warrants (ending in W): {len(warrants)}")
            if preferred and remove_preferred:
                logger.info(f"      Preferred shares (ending in P): {len(preferred)}")
            if other_invalid:
                logger.info(f"      Other invalid: {len(other_invalid)}")
                for ticker in list(other_invalid)[:10]:
                    logger.debug(f"         - {ticker}")
                if len(other_invalid) > 10:
                    logger.debug(f"         ... and {len(other_invalid) - 10} more")

            for ticker in invalid:
                self.tickers.remove(ticker)
                self.removed_count += 1
        else:
            logger.info("   ✓ All tickers valid")

    def fetch_from_alphavantage(self):
        """Fetch tickers from Alpha Vantage LISTING_STATUS endpoint."""
        logger.info("\n📥 Fetching from Alpha Vantage...")

        api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        if not api_key or "placeholder" in api_key:
            logger.warning("   ⚠️  ALPHA_VANTAGE_API_KEY not configured")
            logger.info("   💡 Set in .env file to use this feature")
            return

        try:
            url = f"https://www.alphavantage.co/query?function=LISTING_STATUS&apikey={api_key}"
            logger.info("   Downloading listing data...")

            response = requests.get(url, timeout=60)
            if response.status_code != 200:
                logger.error(f"   ❌ Failed: HTTP {response.status_code}")
                return

            # Parse CSV response
            lines = response.text.strip().split("\n")
            if len(lines) < 2:
                logger.error("   ❌ Invalid response format")
                return

            header = lines[0].split(",")
            logger.debug(f"   Columns: {', '.join(header)}")

            # Find symbol and status columns
            try:
                symbol_idx = header.index("symbol")
                status_idx = header.index("status")
            except ValueError:
                # Try without quotes
                symbol_idx = 0  # Usually first column
                status_idx = None

            initial_count = len(self.tickers)

            for line in lines[1:]:
                parts = line.split(",")
                if len(parts) > symbol_idx:
                    symbol = parts[symbol_idx].strip().strip('"')

                    # Check if active (if status column exists)
                    if status_idx and len(parts) > status_idx:
                        status = parts[status_idx].strip().strip('"')
                        if status != "Active":
                            continue

                    # Only add alphabetic symbols
                    if symbol and symbol.isalpha() and len(symbol) <= 5:
                        self.tickers.add(symbol.upper())

            new_count = len(self.tickers) - initial_count
            self.added_count += new_count
            logger.info(f"   ✓ Added {new_count} new tickers from Alpha Vantage")

        except Exception as e:
            logger.error(f"   ❌ Error: {str(e)}")

    def save_tickers(self):
        """Save tickers back to file (sorted)."""
        output_path = Path(self.ticker_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        sorted_tickers = sorted(self.tickers)

        with open(output_path, "w") as f:
            for symbol in sorted_tickers:
                f.write(f"{symbol}\n")

        logger.info(f"\n✅ Saved {len(sorted_tickers)} tickers to: {self.ticker_file}")

    def print_summary(self):
        """Print summary."""
        logger.info("\n" + "=" * 70)
        logger.info("📊 SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Tickers: {len(self.tickers):,}")
        if self.added_count > 0:
            logger.info(f"Added: {self.added_count}")
        if self.removed_count > 0:
            logger.info(f"Removed: {self.removed_count}")
        logger.info("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Update and maintain ticker database")
    parser.add_argument(
        "--file",
        type=str,
        default="data/tickers.txt",
        help="Ticker file path (default: data/tickers.txt)",
    )
    parser.add_argument(
        "--add", type=str, help="Comma-separated list of tickers to add (e.g., NVDA,PLTR,HOOD)"
    )
    parser.add_argument(
        "--validate", action="store_true", help="Validate and clean existing tickers"
    )
    parser.add_argument(
        "--remove-warrants",
        action="store_true",
        help="Remove warrants (tickers ending in W) during validation",
    )
    parser.add_argument(
        "--remove-preferred",
        action="store_true",
        help="Remove preferred shares (tickers ending in P) during validation",
    )
    parser.add_argument(
        "--fetch-alphavantage", action="store_true", help="Fetch latest tickers from Alpha Vantage"
    )

    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("🔄 TICKER DATABASE UPDATER")
    logger.info("=" * 70)
    logger.info(f"File: {args.file}")
    logger.info("=" * 70 + "\n")

    updater = TickerDatabaseUpdater(args.file)

    # Load existing tickers
    updater.load_tickers()

    # Perform requested operations
    if args.add:
        new_tickers = [t.strip() for t in args.add.split(",")]
        updater.add_tickers(new_tickers)

    if args.validate or args.remove_warrants or args.remove_preferred:
        updater.validate_and_clean(
            remove_warrants=args.remove_warrants, remove_preferred=args.remove_preferred
        )

    if args.fetch_alphavantage:
        updater.fetch_from_alphavantage()

    # If no operations specified, just validate
    if not (
        args.add
        or args.validate
        or args.remove_warrants
        or args.remove_preferred
        or args.fetch_alphavantage
    ):
        logger.info("No operations specified. Use --help for options.")
        logger.info("\nRunning basic validation...")
        updater.validate_and_clean(remove_warrants=False, remove_preferred=False)

    # Save if any changes were made
    if updater.added_count > 0 or updater.removed_count > 0:
        updater.save_tickers()
    else:
        logger.info("\nℹ️  No changes made")

    # Print summary
    updater.print_summary()

    logger.info("💡 Usage examples:")
    logger.info("   python scripts/update_ticker_database.py --add NVDA,PLTR")
    logger.info("   python scripts/update_ticker_database.py --validate")
    logger.info("   python scripts/update_ticker_database.py --remove-warrants")
    logger.info("   python scripts/update_ticker_database.py --fetch-alphavantage\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("\n\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
