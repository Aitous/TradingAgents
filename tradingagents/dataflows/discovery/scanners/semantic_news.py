"""Semantic news scanner for early catalyst detection."""
from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import BaseScanner, SCANNER_REGISTRY
from tradingagents.dataflows.discovery.utils import Priority


class SemanticNewsScanner(BaseScanner):
    """Scan news for early catalysts using semantic analysis."""

    name = "semantic_news"
    pipeline = "news"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sources = self.scanner_config.get("sources", ["google_news"])
        self.lookback_hours = self.scanner_config.get("lookback_hours", 6)
        self.min_importance = self.scanner_config.get("min_news_importance", 5)
        self.min_similarity = self.scanner_config.get("min_similarity", 0.5)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        print(f"   📰 Scanning news catalysts...")

        try:
            from tradingagents.tools.executor import execute_tool
            from datetime import datetime

            # Get recent global news
            date_str = datetime.now().strftime("%Y-%m-%d")
            result = execute_tool("get_global_news", date=date_str)

            if not result or not isinstance(result, str):
                return []

            # Extract tickers mentioned in news
            import re
            ticker_pattern = r'\b([A-Z]{2,5})\b|\$([A-Z]{2,5})'
            matches = re.findall(ticker_pattern, result)

            tickers = list(set([t[0] or t[1] for t in matches if t[0] or t[1]]))
            stop_words = {'NYSE', 'NASDAQ', 'CEO', 'CFO', 'IPO', 'ETF', 'USA', 'SEC', 'NEWS', 'STOCK', 'MARKET'}
            tickers = [t for t in tickers if t not in stop_words]

            candidates = []
            for ticker in tickers[:self.limit]:
                candidates.append({
                    "ticker": ticker,
                    "source": self.name,
                    "context": "Mentioned in recent market news",
                    "priority": Priority.MEDIUM.value,
                    "strategy": "news_catalyst",
                })

            print(f"      Found {len(candidates)} news mentions")
            return candidates

        except Exception as e:
            print(f"      ⚠️  News scan failed: {e}")
            return []



SCANNER_REGISTRY.register(SemanticNewsScanner)
