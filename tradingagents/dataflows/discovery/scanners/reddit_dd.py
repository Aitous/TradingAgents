"""Reddit DD (Due Diligence) scanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.tools.executor import execute_tool
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class RedditDDScanner(BaseScanner):
    """Scan Reddit for high-quality DD posts."""

    name = "reddit_dd"
    pipeline = "social"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📝 Scanning Reddit DD posts...")

        try:
            # Use Reddit DD scanner tool
            result = execute_tool("scan_reddit_dd", limit=self.limit)

            if not result:
                logger.info("Found 0 DD posts")
                return []

            candidates = []

            # Handle different result formats
            if isinstance(result, list):
                # Structured result with DD posts
                for post in result[: self.limit]:
                    ticker = post.get("ticker", "").upper()
                    if not ticker:
                        continue

                    title = post.get("title", "")
                    score = post.get("score", 0)

                    # Higher score = higher priority
                    priority = Priority.HIGH.value if score > 1000 else Priority.MEDIUM.value

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": self.name,
                            "context": f"Reddit DD: {title[:80]}... (score: {score})",
                            "priority": priority,
                            "strategy": "undiscovered_dd",
                            "dd_score": score,
                        }
                    )

            elif isinstance(result, dict):
                # Dict format
                for ticker_data in result.get("posts", [])[: self.limit]:
                    ticker = ticker_data.get("ticker", "").upper()
                    if not ticker:
                        continue

                    candidates.append(
                        {
                            "ticker": ticker,
                            "source": self.name,
                            "context": "Reddit DD post",
                            "priority": Priority.MEDIUM.value,
                            "strategy": "undiscovered_dd",
                        }
                    )

            elif isinstance(result, str):
                # Text result - extract tickers
                candidates = self._parse_text_result(result)

            logger.info(f"Found {len(candidates)} DD posts")
            return candidates

        except Exception as e:
            logger.warning(f"⚠️  Reddit DD scan failed, using fallback: {e}")
            return self._fallback_dd_scan()

    def _fallback_dd_scan(self) -> List[Dict[str, Any]]:
        """Fallback using general Reddit API."""
        try:
            # Try to get Reddit posts with DD flair
            from tradingagents.dataflows.reddit_api import get_reddit_client

            reddit = get_reddit_client()
            subreddit = reddit.subreddit("wallstreetbets+stocks")

            candidates = []
            seen_tickers = set()

            # Look for DD posts
            for submission in subreddit.search("flair:DD", limit=self.limit * 2):
                # Extract ticker from title
                import re

                ticker_pattern = r"\$([A-Z]{2,5})\b|^([A-Z]{2,5})\s"
                matches = re.findall(ticker_pattern, submission.title)

                if not matches:
                    continue

                ticker = (matches[0][0] or matches[0][1]).upper()
                if ticker in seen_tickers:
                    continue

                seen_tickers.add(ticker)

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Reddit DD: {submission.title[:80]}...",
                        "priority": Priority.MEDIUM.value,
                        "strategy": "undiscovered_dd",
                    }
                )

                if len(candidates) >= self.limit:
                    break

            return candidates
        except Exception:
            return []

    def _parse_text_result(self, text: str) -> List[Dict[str, Any]]:
        """Parse tickers from text result."""
        import re

        candidates = []
        ticker_pattern = r"\$([A-Z]{2,5})\b|^([A-Z]{2,5})\s"
        matches = re.findall(ticker_pattern, text)

        tickers = list(set([t[0] or t[1] for t in matches if t[0] or t[1]]))

        for ticker in tickers[: self.limit]:
            candidates.append(
                {
                    "ticker": ticker,
                    "source": self.name,
                    "context": "Reddit DD post",
                    "priority": Priority.MEDIUM.value,
                    "strategy": "undiscovered_dd",
                }
            )

        return candidates


SCANNER_REGISTRY.register(RedditDDScanner)
