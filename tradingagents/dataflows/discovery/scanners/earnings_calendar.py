"""Earnings calendar scanner for upcoming earnings events."""
from typing import Any, Dict, List
from datetime import datetime, timedelta

from tradingagents.dataflows.discovery.scanner_registry import BaseScanner, SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.tools.executor import execute_tool


class EarningsCalendarScanner(BaseScanner):
    """Scan for stocks with upcoming earnings (volatility plays)."""

    name = "earnings_calendar"
    pipeline = "events"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_candidates = self.scanner_config.get("max_candidates", 25)
        self.max_days_until_earnings = self.scanner_config.get("max_days_until_earnings", 7)
        self.min_market_cap = self.scanner_config.get("min_market_cap", 0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        print(f"   📅 Scanning earnings calendar (next {self.max_days_until_earnings} days)...")

        try:
            # Get earnings calendar from Finnhub or Alpha Vantage
            from_date = datetime.now().strftime("%Y-%m-%d")
            to_date = (datetime.now() + timedelta(days=self.max_days_until_earnings)).strftime("%Y-%m-%d")

            result = execute_tool("get_earnings_calendar", from_date=from_date, to_date=to_date)

            if not result:
                print(f"      Found 0 earnings events")
                return []

            candidates = []
            seen_tickers = set()

            # Parse earnings data
            if isinstance(result, list):
                # Structured list of earnings
                candidates = self._parse_structured_earnings(result, seen_tickers)
            elif isinstance(result, dict):
                # Dict format
                earnings_list = result.get("earnings", result.get("data", []))
                candidates = self._parse_structured_earnings(earnings_list, seen_tickers)
            elif isinstance(result, str):
                # Text/markdown format
                candidates = self._parse_text_earnings(result, seen_tickers)

            # Sort by days until earnings (sooner = higher priority)
            candidates.sort(key=lambda x: x.get("days_until", 999))

            # Apply limit
            candidates = candidates[:self.limit]

            print(f"      Found {len(candidates)} upcoming earnings")
            return candidates

        except Exception as e:
            print(f"      ⚠️  Earnings calendar failed: {e}")
            return []

    def _parse_structured_earnings(self, earnings_list: List[Dict], seen_tickers: set) -> List[Dict[str, Any]]:
        """Parse structured earnings data."""
        candidates = []
        today = datetime.now().date()

        for event in earnings_list[:self.max_candidates * 2]:
            ticker = event.get("ticker", event.get("symbol", "")).upper()
            if not ticker or ticker in seen_tickers:
                continue

            # Get earnings date
            earnings_date_str = event.get("date", event.get("earnings_date", ""))
            if not earnings_date_str:
                continue

            try:
                # Parse date (handle different formats)
                if isinstance(earnings_date_str, str):
                    earnings_date = datetime.strptime(earnings_date_str.split()[0], "%Y-%m-%d").date()
                else:
                    earnings_date = earnings_date_str

                days_until = (earnings_date - today).days

                # Filter by max days
                if days_until < 0 or days_until > self.max_days_until_earnings:
                    continue

                # Filter by market cap if specified
                market_cap = event.get("market_cap", 0)
                if self.min_market_cap > 0 and market_cap < self.min_market_cap * 1e9:
                    continue

                seen_tickers.add(ticker)

                # Priority based on proximity to earnings
                if days_until <= 2:
                    priority = Priority.HIGH.value
                elif days_until <= 5:
                    priority = Priority.MEDIUM.value
                else:
                    priority = Priority.LOW.value

                candidates.append({
                    "ticker": ticker,
                    "source": self.name,
                    "context": f"Earnings in {days_until} day(s) on {earnings_date_str}",
                    "priority": priority,
                    "strategy": "pre_earnings_accumulation" if days_until > 1 else "earnings_play",
                    "days_until": days_until,
                    "earnings_date": earnings_date_str,
                })

                if len(candidates) >= self.max_candidates:
                    break

            except (ValueError, AttributeError):
                continue

        return candidates

    def _parse_text_earnings(self, text: str, seen_tickers: set) -> List[Dict[str, Any]]:
        """Parse earnings from text/markdown format."""
        import re

        candidates = []
        today = datetime.now().date()

        # Split by date sections (### 2026-02-05)
        date_sections = re.split(r'###\s+(\d{4}-\d{2}-\d{2})', text)

        current_date = None
        for i, section in enumerate(date_sections):
            # Check if this is a date line
            if re.match(r'\d{4}-\d{2}-\d{2}', section):
                current_date = section
                continue

            if not current_date:
                continue

            # Find tickers in this section (format: **TICKER** (timing))
            ticker_pattern = r'\*\*([A-Z]{2,5})\*\*\s*\(([^\)]+)\)'
            ticker_matches = re.findall(ticker_pattern, section)

            for ticker, timing in ticker_matches:
                if ticker in seen_tickers:
                    continue

                try:
                    earnings_date = datetime.strptime(current_date, "%Y-%m-%d").date()
                    days_until = (earnings_date - today).days

                    if days_until < 0 or days_until > self.max_days_until_earnings:
                        continue

                    seen_tickers.add(ticker)

                    # Priority based on proximity and timing
                    if days_until <= 1:
                        priority = Priority.HIGH.value
                    elif days_until <= 3:
                        priority = Priority.MEDIUM.value
                    else:
                        priority = Priority.LOW.value

                    # Strategy based on timing
                    if timing == "bmo":  # Before market open
                        strategy = "earnings_play"
                    elif timing == "amc":  # After market close
                        strategy = "pre_earnings_accumulation" if days_until > 0 else "earnings_play"
                    else:
                        strategy = "pre_earnings_accumulation"

                    candidates.append({
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Earnings {timing} in {days_until} day(s) on {current_date}",
                        "priority": priority,
                        "strategy": strategy,
                        "days_until": days_until,
                        "earnings_date": current_date,
                        "timing": timing,
                    })

                    if len(candidates) >= self.max_candidates:
                        return candidates

                except ValueError:
                    continue

        return candidates


SCANNER_REGISTRY.register(EarningsCalendarScanner)
