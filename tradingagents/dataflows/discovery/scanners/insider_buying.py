"""SEC Form 4 insider buying scanner."""
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import BaseScanner, SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import Priority


class InsiderBuyingScanner(BaseScanner):
    """Scan SEC Form 4 for insider purchases."""

    name = "insider_buying"
    pipeline = "edge"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_days = self.scanner_config.get("lookback_days", 7)
        self.min_transaction_value = self.scanner_config.get("min_transaction_value", 25000)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        print(f"   💼 Scanning insider buying (last {self.lookback_days} days)...")

        try:
            # Use Finviz insider buying screener
            from tradingagents.dataflows.finviz_scraper import get_finviz_insider_buying

            result = get_finviz_insider_buying(
                transaction_type="buy",
                lookback_days=self.lookback_days,
                min_value=self.min_transaction_value,
                top_n=self.limit
            )

            if not result or not isinstance(result, str):
                print(f"      Found 0 insider purchases")
                return []

            # Parse the markdown result
            candidates = []
            seen_tickers = set()

            # Extract tickers from markdown table
            import re
            lines = result.split('\n')
            for line in lines:
                if '|' not in line or 'Ticker' in line or '---' in line:
                    continue

                parts = [p.strip() for p in line.split('|')]
                if len(parts) < 3:
                    continue

                ticker = parts[1] if len(parts) > 1 else ""
                ticker = ticker.strip().upper()

                if not ticker or ticker in seen_tickers:
                    continue

                # Validate ticker format
                if not re.match(r'^[A-Z]{1,5}$', ticker):
                    continue

                seen_tickers.add(ticker)

                candidates.append({
                    "ticker": ticker,
                    "source": self.name,
                    "context": f"Insider purchase detected (Finviz)",
                    "priority": Priority.HIGH.value,
                    "strategy": "insider_buying",
                })

                if len(candidates) >= self.limit:
                    break

            print(f"      Found {len(candidates)} insider purchases")
            return candidates

        except Exception as e:
            print(f"      ⚠️  Insider buying failed: {e}")
            return []



SCANNER_REGISTRY.register(InsiderBuyingScanner)
