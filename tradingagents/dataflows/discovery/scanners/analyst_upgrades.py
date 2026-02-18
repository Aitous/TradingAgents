"""Analyst upgrade and initiation scanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class AnalystUpgradeScanner(BaseScanner):
    """Scan for recent analyst upgrades and coverage initiations."""

    name = "analyst_upgrades"
    pipeline = "edge"
    strategy = "analyst_upgrade"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.lookback_days = self.scanner_config.get("lookback_days", 3)
        self.max_hours_old = self.scanner_config.get("max_hours_old", 72)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📊 Scanning analyst upgrades and initiations...")

        try:
            from tradingagents.dataflows.alpha_vantage_analysts import (
                get_analyst_rating_changes,
            )

            changes = get_analyst_rating_changes(
                lookback_days=self.lookback_days,
                change_types=["upgrade", "initiated"],
                top_n=self.limit * 2,
                return_structured=True,
            )

            if not changes:
                logger.info("No analyst upgrades found")
                return []

            candidates = []
            for change in changes:
                ticker = change.get("ticker", "").upper().strip()
                if not ticker:
                    continue

                action = change.get("action", "unknown")
                hours_old = change.get("hours_old", 999)
                headline = change.get("headline", "")
                source = change.get("source", "")

                if hours_old > self.max_hours_old:
                    continue

                # Priority by freshness and action type
                if action == "upgrade" and hours_old <= 24:
                    priority = Priority.HIGH.value
                elif action == "initiated" and hours_old <= 24:
                    priority = Priority.HIGH.value
                elif hours_old <= 48:
                    priority = Priority.MEDIUM.value
                else:
                    priority = Priority.LOW.value

                context = (
                    f"Analyst {action}: {headline}"
                    if headline
                    else f"Analyst {action} ({source})"
                )

                candidates.append(
                    {
                        "ticker": ticker,
                        "source": self.name,
                        "context": context,
                        "priority": priority,
                        "strategy": self.strategy,
                        "analyst_action": action,
                        "hours_old": hours_old,
                    }
                )

                if len(candidates) >= self.limit:
                    break

            logger.info(f"Analyst upgrades: {len(candidates)} candidates")
            return candidates

        except Exception as e:
            logger.error(f"Analyst upgrades scan failed: {e}", exc_info=True)
            return []


SCANNER_REGISTRY.register(AnalystUpgradeScanner)
