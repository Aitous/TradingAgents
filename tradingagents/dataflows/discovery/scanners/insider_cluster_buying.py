"""SEC Form 4 insider cluster buying scanner.

Detects coordinated insider purchases: when 3+ distinct insiders buy the same
stock within a rolling window, it signals unusually high insider conviction.
"""

from datetime import date, timedelta
from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Default title lists (lowercased for matching)
_DEFAULT_EXECUTIVE_TITLES = ["ceo", "cfo", "chairman", "president"]
_DEFAULT_DIRECTOR_TITLES = ["director", "officer"]


class InsiderClusterBuyingScanner(BaseScanner):
    """Scan SEC Form 4 filings for coordinated insider buying clusters."""

    name = "insider_cluster_buying"
    pipeline = "fundamental"
    strategy = "insider_cluster_buying"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cluster_window_days = self.scanner_config.get("cluster_window_days", 14)
        self.min_insiders = self.scanner_config.get("min_insiders", 3)
        self.executive_titles = self.scanner_config.get(
            "executive_titles", _DEFAULT_EXECUTIVE_TITLES
        )
        self.director_titles = self.scanner_config.get(
            "director_titles", _DEFAULT_DIRECTOR_TITLES
        )

    # ------------------------------------------------------------------
    # scan
    # ------------------------------------------------------------------

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(
            f"Scanning insider cluster buying (window={self.cluster_window_days}d, "
            f"min_insiders={self.min_insiders})..."
        )

        try:
            from tradingagents.dataflows.finviz_scraper import get_finviz_insider_buying

            # Fetch all raw transactions (no dedup) so we can count distinct insiders
            transactions = get_finviz_insider_buying(
                lookback_days=self.cluster_window_days,
                min_value=0,  # accept any value; cluster count is the filter
                return_structured=True,
                deduplicate=False,
            )

            if not transactions:
                logger.info("No insider transactions found")
                return []

            logger.info(f"Fetched {len(transactions)} raw insider transactions")

            # Group by ticker
            by_ticker: Dict[str, list] = {}
            for txn in transactions:
                ticker = txn.get("ticker", "").upper().strip()
                if not ticker:
                    continue
                by_ticker.setdefault(ticker, []).append(txn)

            candidates = []
            for ticker, txns in by_ticker.items():
                cluster = self._build_cluster(ticker, txns)
                if cluster is None:
                    continue
                candidates.append(cluster)

            # Sort by cluster score (stronger clusters first), then apply limit
            candidates.sort(key=lambda c: c.get("_cluster_score", 0), reverse=True)
            candidates = candidates[: self.limit]

            # Strip internal scoring key before returning
            for c in candidates:
                c.pop("_cluster_score", None)

            logger.info(f"Insider cluster buying: {len(candidates)} candidates")
            return candidates

        except Exception as e:
            logger.error(f"Insider cluster buying scan failed: {e}", exc_info=True)
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _title_flags(self, title: str):
        """Return (is_ceo, is_cfo, is_chairman, is_executive, is_director) booleans."""
        tl = title.lower()
        is_ceo = "ceo" in tl
        is_cfo = "cfo" in tl
        is_chairman = "chairman" in tl
        is_executive = any(kw in tl for kw in self.executive_titles)
        is_director = any(kw in tl for kw in self.director_titles)
        return is_ceo, is_cfo, is_chairman, is_executive, is_director

    def _parse_price(self, price_str: str) -> float:
        """Parse a price string like '$12.34' into a float; returns 0.0 on failure."""
        try:
            clean = price_str.replace("$", "").replace(",", "").strip()
            return float(clean) if clean else 0.0
        except (ValueError, AttributeError):
            return 0.0

    def _parse_qty(self, qty_str: str) -> int:
        """Parse a quantity string like '1,234' into an int; returns 0 on failure."""
        try:
            clean = qty_str.replace(",", "").replace("+", "").strip()
            return int(float(clean)) if clean else 0
        except (ValueError, AttributeError):
            return 0

    def _build_cluster(self, ticker: str, txns: list) -> Dict[str, Any]:
        """Aggregate transactions for one ticker; return candidate dict or None."""
        # Deduplicate by (insider_name, company) — same person can file multiple rows
        seen_insiders = set()
        unique_txns = []
        for txn in txns:
            insider_key = (
                txn.get("insider", "").strip().lower(),
                txn.get("company", "").strip().lower(),
            )
            if insider_key in seen_insiders:
                continue
            seen_insiders.add(insider_key)
            unique_txns.append(txn)

        insider_count = len(unique_txns)
        if insider_count < self.min_insiders:
            return None

        # Aggregate flags and metrics
        has_ceo = False
        has_cfo = False
        has_chairman = False
        has_executive = False
        total_shares = 0
        prices = []

        for txn in unique_txns:
            title = txn.get("title", "")
            is_ceo, is_cfo, is_chairman, is_executive, _ = self._title_flags(title)
            has_ceo = has_ceo or is_ceo
            has_cfo = has_cfo or is_cfo
            has_chairman = has_chairman or is_chairman
            has_executive = has_executive or is_executive

            qty = self._parse_qty(txn.get("qty", ""))
            total_shares += qty

            price = self._parse_price(txn.get("price", ""))
            if price > 0:
                prices.append(price)

        avg_price = sum(prices) / len(prices) if prices else 0.0

        # days_span: we don't have per-transaction dates from the scraper, so we
        # represent the window as the configured cluster_window_days.
        days_span = self.cluster_window_days

        # Priority rules from spec
        priority = self._assign_priority(insider_count, has_ceo, has_cfo, has_executive)

        # Cluster score for ranking: weight by count, executive presence, CEO+CFO combo
        cluster_score = insider_count * 10
        if has_executive:
            cluster_score += 20
        if has_ceo and has_cfo:
            cluster_score += 30
        elif has_ceo or has_cfo:
            cluster_score += 15
        if has_chairman:
            cluster_score += 10

        context = (
            f"Insider cluster: {insider_count} executives bought in {days_span}d "
            f"(CEO: {has_ceo}, CFO: {has_cfo}); "
            f"{total_shares:,} shares | avg price ${avg_price:.2f}"
        )

        return {
            "ticker": ticker,
            "source": self.name,
            "context": context,
            "priority": priority,
            "strategy": self.strategy,
            "insider_count": insider_count,
            "has_ceo": has_ceo,
            "has_cfo": has_cfo,
            "has_chairman": has_chairman,
            "total_shares": total_shares,
            "avg_price": avg_price,
            "days_span": days_span,
            "_cluster_score": cluster_score,
        }

    def _assign_priority(
        self, insider_count: int, has_ceo: bool, has_cfo: bool, has_executive: bool
    ) -> str:
        """Return a Priority string per spec rules."""
        if insider_count >= 4 and (has_ceo or has_cfo):
            return Priority.CRITICAL.value
        if insider_count >= 3 and (has_ceo or has_cfo or has_executive):
            return Priority.HIGH.value
        # >=3 insiders (directors ok)
        return Priority.MEDIUM.value


SCANNER_REGISTRY.register(InsiderClusterBuyingScanner)
