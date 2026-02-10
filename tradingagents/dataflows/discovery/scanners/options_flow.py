"""Unusual options activity scanner."""

from typing import Any, Dict, List

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.y_finance import get_option_chain, get_ticker_options
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class OptionsFlowScanner(BaseScanner):
    """Scan for unusual options activity."""

    name = "options_flow"
    pipeline = "edge"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_volume_oi_ratio = self.scanner_config.get("unusual_volume_multiple", 2.0)
        self.min_volume = self.scanner_config.get("min_volume", 1000)
        self.min_premium = self.scanner_config.get("min_premium", 25000)
        self.ticker_universe = self.scanner_config.get(
            "ticker_universe", ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "TSLA"]
        )

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("Scanning unusual options activity...")

        candidates = []

        for ticker in self.ticker_universe[:20]:  # Limit for speed
            try:
                unusual = self._analyze_ticker_options(ticker)
                if unusual:
                    candidates.append(unusual)
                if len(candidates) >= self.limit:
                    break
            except Exception:
                continue

        logger.info(f"Found {len(candidates)} unusual options flows")
        return candidates

    def _analyze_ticker_options(self, ticker: str) -> Dict[str, Any]:
        try:
            expirations = get_ticker_options(ticker)
            if not expirations:
                return None

            options = get_option_chain(ticker, expirations[0])
            calls = options.calls
            puts = options.puts

            # Find unusual strikes
            unusual_strikes = []
            for _, opt in calls.iterrows():
                vol = opt.get("volume", 0)
                oi = opt.get("openInterest", 0)
                if oi > 0 and vol > self.min_volume and (vol / oi) >= self.min_volume_oi_ratio:
                    unusual_strikes.append(
                        {"type": "call", "strike": opt["strike"], "volume": vol, "oi": oi}
                    )

            if not unusual_strikes:
                return None

            # Calculate P/C ratio
            total_call_vol = calls["volume"].sum() if not calls.empty else 0
            total_put_vol = puts["volume"].sum() if not puts.empty else 0
            pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 0

            sentiment = "bullish" if pc_ratio < 0.7 else "bearish" if pc_ratio > 1.3 else "neutral"

            return {
                "ticker": ticker,
                "source": self.name,
                "context": f"Unusual options: {len(unusual_strikes)} strikes, P/C={pc_ratio:.2f} ({sentiment})",
                "priority": "high" if sentiment == "bullish" else "medium",
                "strategy": "options_flow",
                "put_call_ratio": round(pc_ratio, 2),
            }

        except Exception:
            return None


SCANNER_REGISTRY.register(OptionsFlowScanner)
