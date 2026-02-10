"""Reddit trending scanner - migrated from legacy TraditionalScanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class RedditTrendingScanner(BaseScanner):
    """Scan for trending tickers on Reddit."""

    name = "reddit_trending"
    pipeline = "social"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📱 Scanning Reddit trending...")

        from tradingagents.tools.executor import execute_tool

        try:
            result = execute_tool("get_trending_tickers", limit=self.limit)

            if not result or not isinstance(result, str):
                return []

            if "Error" in result or "No trending" in result:
                logger.warning(f"⚠️  {result}")
                return []

            # Extract tickers using common utility
            from tradingagents.dataflows.discovery.common_utils import extract_tickers_from_text

            tickers_found = extract_tickers_from_text(result)

            candidates = []
            for ticker in tickers_found[: self.limit]:
                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": "Reddit trending discussion",
                        "priority": Priority.MEDIUM.value,
                        "strategy": "social_hype",
                    }
                )

            logger.info(f"Found {len(candidates)} Reddit trending tickers")
            return candidates

        except Exception as e:
            logger.warning(f"⚠️  Reddit trending failed: {e}")
            return []


SCANNER_REGISTRY.register(RedditTrendingScanner)
