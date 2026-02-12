"""Market movers scanner - migrated from legacy TraditionalScanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class MarketMoversScanner(BaseScanner):
    """Scan for top gainers and losers."""

    name = "market_movers"
    pipeline = "momentum"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📈 Scanning market movers...")

        from tradingagents.tools.executor import execute_tool

        try:
            result = execute_tool("get_market_movers", return_structured=True)

            if not result or not isinstance(result, dict):
                return []

            if "error" in result:
                logger.warning(f"⚠️  API error: {result['error']}")
                return []

            candidates = []

            # Process gainers
            for gainer in result.get("gainers", [])[: self.limit // 2]:
                ticker = gainer.get("ticker", "").upper()
                if not ticker:
                    continue

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Top gainer: {gainer.get('change_percentage', 0)} change",
                        "priority": Priority.MEDIUM.value,
                        "strategy": "momentum",
                    }
                )

            # Process losers (potential reversal plays)
            for loser in result.get("losers", [])[: self.limit // 2]:
                ticker = loser.get("ticker", "").upper()
                if not ticker:
                    continue

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Top loser: {loser.get('change_percentage', 0)} change (reversal play)",
                        "priority": Priority.LOW.value,
                        "strategy": "oversold_reversal",
                    }
                )

            logger.info(f"Found {len(candidates)} market movers")
            return candidates

        except Exception as e:
            logger.warning(f"⚠️  Market movers failed: {e}")
            return []


SCANNER_REGISTRY.register(MarketMoversScanner)
