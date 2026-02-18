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
    strategy = "momentum"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_price = self.scanner_config.get("min_price", 5.0)
        self.min_volume = self.scanner_config.get("min_volume", 500_000)

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
            for gainer in result.get("gainers", [])[: self.limit]:
                ticker = gainer.get("ticker", "").upper()
                if not ticker:
                    continue

                if not self._validate_mover(ticker):
                    continue

                change_pct = gainer.get("change_percentage", 0)
                price = gainer.get("price", "")
                volume = gainer.get("volume", "")

                context = f"Top gainer: {change_pct} change"
                if price:
                    context += f" (${price})"
                if volume:
                    context += f" vol: {volume}"

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": context,
                        "priority": Priority.MEDIUM.value,
                        "strategy": self.strategy,
                    }
                )

                if len(candidates) >= self.limit // 2:
                    break

            # Process losers (potential reversal plays)
            loser_count = 0
            for loser in result.get("losers", [])[: self.limit]:
                ticker = loser.get("ticker", "").upper()
                if not ticker:
                    continue

                if not self._validate_mover(ticker):
                    continue

                change_pct = loser.get("change_percentage", 0)

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": f"Top loser: {change_pct} change (reversal play)",
                        "priority": Priority.LOW.value,
                        "strategy": self.strategy,
                    }
                )
                loser_count += 1
                if loser_count >= self.limit // 2:
                    break

            logger.info(f"Found {len(candidates)} market movers")
            return candidates

        except Exception as e:
            logger.warning(f"⚠️  Market movers failed: {e}")
            return []

    def _validate_mover(self, ticker: str) -> bool:
        """Quick validation: price and volume check to filter penny/illiquid stocks."""
        try:
            from tradingagents.dataflows.y_finance import get_stock_price, get_ticker_info

            price = get_stock_price(ticker)
            if price is not None and price < self.min_price:
                return False

            info = get_ticker_info(ticker)
            avg_vol = info.get("averageVolume", 0) if info else 0
            if avg_vol and avg_vol < self.min_volume:
                return False

            return True
        except Exception:
            return True  # Don't filter on errors


SCANNER_REGISTRY.register(MarketMoversScanner)
