"""SEC Form 4 insider buying scanner with cluster detection.

Fetches insider transactions over a rolling window and groups by ticker.
Single-insider buys are scored by title/value; multi-insider clusters
(3+ distinct buyers) use a richer scoring model with CEO/CFO weighting.
Staleness suppression prevents the same Form 4 filing from re-appearing
across consecutive discovery runs.
"""

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Set

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class InsiderBuyingScanner(BaseScanner):
    """Scan SEC Form 4 for insider purchases, with cluster detection."""

    name = "insider_buying"
    pipeline = "edge"
    strategy = "insider_buying"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cluster_window_days = self.scanner_config.get("cluster_window_days", 14)
        self.min_transaction_value = self.scanner_config.get("min_transaction_value", 100_000)
        self.min_cluster_insiders = self.scanner_config.get("min_cluster_insiders", 3)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(
            f"Scanning insider buying (window={self.cluster_window_days}d, "
            f"min_txn=${self.min_transaction_value:,})..."
        )

        try:
            from tradingagents.dataflows.finviz_scraper import get_finviz_insider_buying

            transactions = get_finviz_insider_buying(
                lookback_days=self.cluster_window_days,
                min_value=self.min_transaction_value,
                return_structured=True,
                deduplicate=False,
            )

            if not transactions:
                logger.info("No insider buying transactions found")
                return []

            logger.info(f"Found {len(transactions)} insider transactions")

            # Group by ticker, deduplicate by (insider, company)
            by_ticker: Dict[str, list] = {}
            for txn in transactions:
                ticker = txn.get("ticker", "").upper().strip()
                if not ticker:
                    continue
                by_ticker.setdefault(ticker, []).append(txn)

            candidates = []
            for ticker, txns in by_ticker.items():
                candidate = self._build_candidate(ticker, txns)
                if candidate:
                    candidates.append(candidate)

            # Sort by signal quality, then limit
            candidates.sort(key=lambda c: c.pop("_score", 0), reverse=True)
            candidates = candidates[: self.limit]

            # Staleness suppression: filter tickers seen as insider_buying in past N days.
            # suppress_days=3 closes the gap found Apr 12: FUL appeared Apr 9 and Apr 12
            # (3-day spacing) from the same Form 4 filing.
            recently_seen = self._load_recent_insider_tickers(suppress_days=3)
            if recently_seen:
                before = {c["ticker"] for c in candidates}
                candidates = [c for c in candidates if c["ticker"] not in recently_seen]
                suppressed = before - {c["ticker"] for c in candidates}
                if suppressed:
                    logger.info(
                        f"Staleness filter: suppressed {len(suppressed)} ticker(s) "
                        f"already recommended in the past 3 days: {suppressed}"
                    )

            logger.info(f"Insider buying: {len(candidates)} candidates")
            return candidates

        except Exception as e:
            logger.error(f"Insider buying scan failed: {e}", exc_info=True)
            return []

    def _build_candidate(self, ticker: str, txns: list) -> Dict[str, Any]:
        """Build a candidate dict for one ticker from its transactions."""
        # Deduplicate by (insider_name, company)
        seen_keys: Set[tuple] = set()
        unique_txns = []
        for txn in txns:
            key = (txn.get("insider", "").strip().lower(), txn.get("company", "").strip().lower())
            if key not in seen_keys:
                seen_keys.add(key)
                unique_txns.append(txn)

        num_insiders = len(unique_txns)

        # Aggregate title flags and metrics
        has_ceo = has_cfo = has_chairman = has_executive = False
        total_value = 0
        prices = []

        for txn in unique_txns:
            tl = txn.get("title", "").lower()
            has_ceo = has_ceo or "ceo" in tl
            has_cfo = has_cfo or "cfo" in tl
            has_chairman = has_chairman or "chairman" in tl
            has_executive = has_executive or any(
                kw in tl for kw in ["ceo", "cfo", "chairman", "president", "coo", "cto"]
            )
            total_value += txn.get("value_num", 0)
            price = self._parse_price(txn.get("price", ""))
            if price > 0:
                prices.append(price)

        # Use largest transaction for single-insider context
        txns_sorted = sorted(unique_txns, key=lambda t: t.get("value_num", 0), reverse=True)
        primary = txns_sorted[0]
        primary_name = primary.get("insider", "Unknown")
        primary_title = primary.get("title", "")
        primary_value = primary.get("value_num", 0)
        primary_value_str = primary.get("value_str", f"${primary_value:,.0f}")
        avg_price = sum(prices) / len(prices) if prices else 0.0
        total_shares = sum(self._parse_qty(t.get("qty", "")) for t in unique_txns)

        is_cluster = num_insiders >= self.min_cluster_insiders

        # --- Priority ---
        if is_cluster:
            if num_insiders >= 4 and (has_ceo or has_cfo):
                priority = Priority.CRITICAL.value
            elif has_ceo or has_cfo or has_executive:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value
        else:
            # Single or 2-insider: use title + value tiering
            tl = primary_title.lower()
            is_c_suite = any(kw in tl for kw in ["ceo", "cfo", "coo", "cto", "president", "chairman"])
            is_director = "director" in tl
            if num_insiders >= 2 or (is_c_suite and primary_value >= 100_000):
                priority = Priority.CRITICAL.value
            elif is_c_suite or (is_director and primary_value >= 50_000):
                priority = Priority.HIGH.value
            elif primary_value >= 50_000:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

        # --- Context ---
        if is_cluster:
            context = (
                f"Insider cluster: {num_insiders} executives bought in {self.cluster_window_days}d "
                f"(CEO: {has_ceo}, CFO: {has_cfo}); "
                f"{total_shares:,} shares | avg price ${avg_price:.2f} | total ${total_value:,.0f}"
            )
        elif num_insiders == 2:
            context = (
                f"2 insiders buying {ticker}. "
                f"Largest: {primary_title} {primary_name} purchased {primary_value_str}"
            )
        else:
            context = f"{primary_title} {primary_name} purchased {primary_value_str} of {ticker}"

        # --- Score for ranking ---
        score = total_value
        score += num_insiders * 500_000
        if has_ceo or has_cfo:
            score += 1_500_000
        elif has_executive:
            score += 1_000_000
        if has_ceo and has_cfo:
            score += 1_000_000
        if has_chairman:
            score += 300_000

        return {
            "ticker": ticker,
            "source": self.name,
            "context": context,
            "priority": priority,
            "strategy": self.strategy,
            "insider_name": primary_name,
            "insider_title": primary_title,
            "transaction_value": primary_value,
            "total_transaction_value": total_value,
            "num_insiders_buying": num_insiders,
            "has_ceo": has_ceo,
            "has_cfo": has_cfo,
            "_score": score,
        }

    def _parse_price(self, price_str: str) -> float:
        try:
            return float(str(price_str).replace("$", "").replace(",", "").strip() or 0)
        except (ValueError, AttributeError):
            return 0.0

    def _parse_qty(self, qty_str: str) -> int:
        try:
            return int(float(str(qty_str).replace(",", "").replace("+", "").strip() or 0))
        except (ValueError, AttributeError):
            return 0

    def _load_recent_insider_tickers(self, suppress_days: int = 3) -> Set[str]:
        """Return tickers recommended as insider_buying in the past N days (and today).

        Covers both cross-day staleness (same Form 4 filing re-appearing each day)
        and same-day multi-run staleness (NKE appeared in 3/4 runs on Apr 20).
        """
        seen: Set[str] = set()
        data_dir = Path(self.config.get("data_dir", "data"))
        recs_dir = data_dir / "recommendations"

        if not recs_dir.exists():
            return seen

        today = date.today()
        # Range: today (same-day dedup) + past suppress_days
        for i in range(0, suppress_days + 1):
            check_date = today - timedelta(days=i)
            rec_file = recs_dir / f"{check_date.isoformat()}.json"
            if not rec_file.exists():
                continue
            try:
                with open(rec_file) as f:
                    data = json.load(f)
                for rec in data.get("recommendations", []):
                    if rec.get("strategy_match") in ("insider_buying", "insider_cluster_buying"):
                        ticker = rec.get("ticker", "").upper()
                        if ticker:
                            seen.add(ticker)
            except Exception:
                pass

        return seen


SCANNER_REGISTRY.register(InsiderBuyingScanner)
