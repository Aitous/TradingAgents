"""
Delisted Cache System
---------------------
Track tickers that consistently fail data fetches (likely delisted).

SAFETY: Only cache tickers that:
- Passed initial format validation (not units/warrants/common words)
- Failed multiple times over multiple days
- Have consistent failure patterns (not temporary API issues)
"""

import json
from datetime import datetime
from pathlib import Path

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class DelistedCache:
    """
    Track tickers that consistently fail data fetches (likely delisted).

    SAFETY: Only cache tickers that:
    - Passed initial format validation (not units/warrants/common words)
    - Failed multiple times over multiple days
    - Have consistent failure patterns (not temporary API issues)
    """

    def __init__(self, cache_file="data/delisted_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()

    def _load_cache(self):
        if self.cache_file.exists():
            with open(self.cache_file, "r") as f:
                return json.load(f)
        return {}

    def mark_failed(self, ticker, reason="no_data", error_code=None):
        """
        Record a failed data fetch for a ticker.

        Args:
            ticker: Stock symbol
            reason: Human-readable failure reason
            error_code: Specific error (e.g., "404", "no_price_data", "empty_history")
        """
        ticker = ticker.upper()

        if ticker not in self.cache:
            self.cache[ticker] = {
                "first_failed": datetime.now().isoformat(),
                "last_failed": datetime.now().isoformat(),
                "fail_count": 1,
                "reason": reason,
                "error_code": error_code,
                "fail_dates": [datetime.now().date().isoformat()],
            }
        else:
            self.cache[ticker]["fail_count"] += 1
            self.cache[ticker]["last_failed"] = datetime.now().isoformat()
            self.cache[ticker]["reason"] = reason  # Update to latest reason

            # Track unique failure dates
            today = datetime.now().date().isoformat()
            if today not in self.cache[ticker].get("fail_dates", []):
                self.cache[ticker].setdefault("fail_dates", []).append(today)

        self._save_cache()

    def is_likely_delisted(self, ticker, fail_threshold=5, days_threshold=14, min_unique_days=3):
        """
        Conservative check: ticker must fail multiple times across multiple days.

        Args:
            fail_threshold: Minimum number of total failures (default: 5)
            days_threshold: Must have failed within this many days (default: 14)
            min_unique_days: Must have failed on at least this many different days (default: 3)

        Returns:
            bool: True if ticker is likely delisted
        """
        ticker = ticker.upper()
        if ticker not in self.cache:
            return False

        data = self.cache[ticker]
        last_failed = datetime.fromisoformat(data["last_failed"])
        days_since = (datetime.now() - last_failed).days

        # Count unique failure days
        unique_fail_days = len(set(data.get("fail_dates", [])))

        # Conservative criteria:
        # - Must have failed at least 5 times
        # - Must have failed on at least 3 different days (not just repeated same-day attempts)
        # - Last failure within 14 days (don't cache stale data)
        return (
            data["fail_count"] >= fail_threshold
            and unique_fail_days >= min_unique_days
            and days_since <= days_threshold
        )

    def get_failure_summary(self, ticker):
        """Get detailed failure info for manual review."""
        ticker = ticker.upper()
        if ticker not in self.cache:
            return None

        data = self.cache[ticker]
        return {
            "ticker": ticker,
            "fail_count": data["fail_count"],
            "unique_days": len(set(data.get("fail_dates", []))),
            "first_failed": data["first_failed"],
            "last_failed": data["last_failed"],
            "reason": data["reason"],
            "is_likely_delisted": self.is_likely_delisted(ticker),
        }

    def _save_cache(self):
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_file, "w") as f:
            json.dump(self.cache, f, indent=2)

    def export_review_list(self, output_file="data/delisted_review.txt"):
        """Export tickers that need manual review to add to DELISTED_TICKERS."""
        likely_delisted = [
            ticker for ticker in self.cache.keys() if self.is_likely_delisted(ticker)
        ]

        if not likely_delisted:
            return

        with open(output_file, "w") as f:
            f.write(
                "# Tickers that have failed consistently (review before adding to DELISTED_TICKERS)\n\n"
            )
            for ticker in sorted(likely_delisted):
                summary = self.get_failure_summary(ticker)
                f.write(
                    f"{ticker:8s} - Failed {summary['fail_count']:2d} times across {summary['unique_days']} days - {summary['reason']}\n"
                )

        logger.info(f"📝 Review list exported to: {output_file}")
