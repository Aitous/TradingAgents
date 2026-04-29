"""Analyst recommendation breadth revision scanner.

Research: docs/iterations/research/2026-04-28-analyst-revision-breadth.md
"""

import time
from typing import Any, Dict, List, Optional

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.dataflows.finnhub_api import get_finnhub_client
from tradingagents.dataflows.universe import load_universe
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class AnalystRevisionBreadthScanner(BaseScanner):
    """Scan for stocks where net analyst buy count improved vs prior month."""

    name = "analyst_revision_breadth"
    pipeline = "edge"
    strategy = "analyst_revision_momentum"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_delta = self.scanner_config.get("min_delta", 2)
        self.min_analysts = self.scanner_config.get("min_analysts", 5)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(f"Scanning analyst revision breadth (min_delta={self.min_delta})...")

        tickers = load_universe(self.config)
        if not tickers:
            logger.warning("No tickers loaded for analyst revision breadth scan")
            return []

        client = get_finnhub_client()
        candidates = []

        for ticker in tickers:
            try:
                data = client.recommendation_trends(ticker.upper())
                result = self._check_revision(ticker, data)
                if result:
                    candidates.append(result)
                time.sleep(0.05)
            except Exception as e:
                logger.warning(f"analyst_revision_breadth: skipping {ticker}: {e}")

        candidates.sort(key=lambda c: c.get("_delta", 0), reverse=True)
        for c in candidates:
            c.pop("_delta", None)

        candidates = candidates[: self.limit]
        logger.info(f"Analyst revision breadth: {len(candidates)} candidates")
        return candidates

    def _check_revision(self, ticker: str, data: list) -> Optional[Dict[str, Any]]:
        try:
            if not data or len(data) < 2:
                return None

            cur = data[0]
            prev = data[1]

            total = (
                cur.get("strongBuy", 0)
                + cur.get("buy", 0)
                + cur.get("hold", 0)
                + cur.get("sell", 0)
                + cur.get("strongSell", 0)
            )
            if total < self.min_analysts:
                return None

            net_cur = (
                cur.get("strongBuy", 0) + cur.get("buy", 0)
                - cur.get("sell", 0) - cur.get("strongSell", 0)
            )
            net_prev = (
                prev.get("strongBuy", 0) + prev.get("buy", 0)
                - prev.get("sell", 0) - prev.get("strongSell", 0)
            )
            delta = net_cur - net_prev

            if delta < self.min_delta:
                return None

            buy0 = cur.get("strongBuy", 0) + cur.get("buy", 0)
            hold0 = cur.get("hold", 0)
            sell0 = cur.get("sell", 0) + cur.get("strongSell", 0)

            if delta >= 5 and total >= 8:
                priority = Priority.CRITICAL.value
            elif delta >= 3 and total >= 5:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

            context = (
                f"Analyst revision breadth: +{delta} net upgrades vs prior month "
                f"({buy0}B/{hold0}H/{sell0}S, {total} analysts) — revision momentum signal"
            )

            return {
                "ticker": ticker,
                "source": self.name,
                "context": context,
                "priority": priority,
                "strategy": self.strategy,
                "_delta": delta,
            }

        except Exception as e:
            logger.debug(f"analyst_revision_breadth check failed for {ticker}: {e}")
            return None


SCANNER_REGISTRY.register(AnalystRevisionBreadthScanner)
