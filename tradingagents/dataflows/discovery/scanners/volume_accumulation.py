"""Volume accumulation and compression scanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.tools.executor import execute_tool
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class VolumeAccumulationScanner(BaseScanner):
    """Scan for unusual volume accumulation patterns."""

    name = "volume_accumulation"
    pipeline = "momentum"
    strategy = "early_accumulation"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.unusual_volume_multiple = self.scanner_config.get("unusual_volume_multiple", 2.0)
        self.volume_cache_key = self.scanner_config.get("volume_cache_key", "default")

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("📊 Scanning volume accumulation...")

        try:
            # Use volume scanner tool
            result = execute_tool(
                "get_unusual_volume",
                min_volume_multiple=self.unusual_volume_multiple,
                top_n=self.limit,
            )

            if not result:
                logger.info("Found 0 volume accumulation candidates")
                return []

            raw_candidates = []

            # Handle different result formats
            if isinstance(result, str):
                # Parse markdown/text result
                raw_candidates = self._parse_text_result(result)
            elif isinstance(result, list):
                # Structured result
                for item in result[: self.limit * 2]:
                    ticker = item.get("ticker", "").upper()
                    if not ticker:
                        continue

                    volume_ratio = item.get("volume_ratio", 0)
                    avg_volume = item.get("avg_volume", 0)

                    raw_candidates.append(
                        {
                            "ticker": ticker,
                            "source": self.name,
                            "context": f"Unusual volume: {volume_ratio:.1f}x average ({avg_volume:,})",
                            "priority": (
                                Priority.MEDIUM.value if volume_ratio < 3.0 else Priority.HIGH.value
                            ),
                            "strategy": self.strategy,
                        }
                    )
            elif isinstance(result, dict):
                # Dict with tickers list
                for ticker in result.get("tickers", [])[: self.limit * 2]:
                    raw_candidates.append(
                        {
                            "ticker": ticker.upper(),
                            "source": self.name,
                            "context": "Unusual volume accumulation",
                            "priority": Priority.MEDIUM.value,
                            "strategy": self.strategy,
                        }
                    )

            # Enrich with price-change context and filter distribution
            candidates = []
            for cand in raw_candidates:
                cand = self._enrich_volume_candidate(cand["ticker"], cand)
                if cand.get("volume_signal") == "distribution":
                    continue
                candidates.append(cand)
                if len(candidates) >= self.limit:
                    break

            logger.info(f"Found {len(candidates)} volume accumulation candidates")
            return candidates

        except Exception as e:
            logger.warning(f"⚠️  Volume accumulation failed: {e}")
            return []

    def _enrich_volume_candidate(self, ticker: str, cand: Dict[str, Any]) -> Dict[str, Any]:
        """Add price-change context to distinguish accumulation from distribution."""
        try:
            from tradingagents.dataflows.y_finance import download_history

            hist = download_history(
                ticker, period="10d", interval="1d", auto_adjust=True, progress=False
            )
            if hist is None or hist.empty or len(hist) < 2:
                return cand

            # Handle MultiIndex from yfinance
            if isinstance(hist.columns, __import__("pandas").MultiIndex):
                tickers = hist.columns.get_level_values(1).unique()
                target = ticker if ticker in tickers else tickers[0]
                hist = hist.xs(target, level=1, axis=1)

            # Today's price change
            latest_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])
            if prev_close == 0:
                return cand
            day_change_pct = ((latest_close - prev_close) / prev_close) * 100

            cand["day_change_pct"] = round(day_change_pct, 2)

            # Multi-day volume pattern: count days with >1.5x avg volume in last 5 days
            if len(hist) >= 6:
                avg_vol = float(hist["Volume"].iloc[:-5].mean()) if len(hist) > 5 else float(
                    hist["Volume"].mean()
                )
                if avg_vol > 0:
                    recent_high_vol_days = sum(
                        1 for v in hist["Volume"].iloc[-5:] if float(v) > avg_vol * 1.5
                    )
                    cand["high_vol_days_5d"] = recent_high_vol_days
                    if recent_high_vol_days >= 3:
                        cand["context"] += (
                            f" | Sustained: {recent_high_vol_days}/5 days above 1.5x avg"
                        )

            # Classify signal
            if abs(day_change_pct) < 3:
                cand["volume_signal"] = "accumulation"
                cand["context"] += f" | Price flat ({day_change_pct:+.1f}%) — quiet accumulation"
            elif day_change_pct < -5:
                cand["volume_signal"] = "distribution"
                cand["priority"] = Priority.LOW.value
                cand["context"] += (
                    f" | Price dropped {day_change_pct:+.1f}% — possible distribution"
                )
            else:
                cand["volume_signal"] = "momentum"

        except Exception as e:
            logger.debug(f"Volume enrichment failed for {ticker}: {e}")

        return cand

    def _parse_text_result(self, text: str) -> List[Dict[str, Any]]:
        """Parse tickers from text result."""
        from tradingagents.dataflows.discovery.common_utils import extract_tickers_from_text

        candidates = []
        tickers = extract_tickers_from_text(text)

        for ticker in tickers[: self.limit]:
            candidates.append(
                {
                    "ticker": ticker,
                    "source": self.name,
                    "context": "Unusual volume detected",
                    "priority": Priority.MEDIUM.value,
                    "strategy": self.strategy,
                }
            )

        return candidates


SCANNER_REGISTRY.register(VolumeAccumulationScanner)
