"""Unusual options activity scanner.

Scans a ticker universe (loaded from data/tickers.txt by default) for
unusual options volume relative to open interest.  Uses ThreadPoolExecutor
for parallel chain fetching so large universes remain practical.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.y_finance import get_option_chain, get_ticker_options
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TICKER_FILE = "data/tickers.txt"


def _load_tickers_from_file(path: str) -> List[str]:
    """Load ticker symbols from a text file (one per line, # comments allowed)."""
    try:
        with open(path) as f:
            tickers = [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
        if tickers:
            logger.info(f"Options scanner: loaded {len(tickers)} tickers from {path}")
            return tickers
    except FileNotFoundError:
        logger.warning(f"Ticker file not found: {path}")
    except Exception as e:
        logger.warning(f"Failed to load ticker file {path}: {e}")
    return []


class OptionsFlowScanner(BaseScanner):
    """Scan for unusual options activity across a ticker universe."""

    name = "options_flow"
    pipeline = "edge"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_volume_oi_ratio = self.scanner_config.get("unusual_volume_multiple", 2.0)
        self.min_volume = self.scanner_config.get("min_volume", 1000)
        self.min_premium = self.scanner_config.get("min_premium", 25000)
        self.max_tickers = self.scanner_config.get("max_tickers", 150)
        self.max_workers = self.scanner_config.get("max_workers", 8)

        # Load universe: explicit list > ticker_file > default file
        if "ticker_universe" in self.scanner_config:
            self.ticker_universe = self.scanner_config["ticker_universe"]
        else:
            ticker_file = self.scanner_config.get(
                "ticker_file",
                config.get("tickers_file", DEFAULT_TICKER_FILE),
            )
            self.ticker_universe = _load_tickers_from_file(ticker_file)
            if not self.ticker_universe:
                logger.warning("No tickers loaded — options scanner will be empty")

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        universe = self.ticker_universe[: self.max_tickers]
        logger.info(
            f"Scanning {len(universe)} tickers for unusual options activity "
            f"({self.max_workers} workers)..."
        )

        candidates: List[Dict[str, Any]] = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {
                pool.submit(self._analyze_ticker_options, ticker): ticker
                for ticker in universe
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)
                        if len(candidates) >= self.limit:
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            break
                except Exception:
                    continue

        logger.info(f"Found {len(candidates)} unusual options flows")
        return candidates

    def _analyze_ticker_options(self, ticker: str) -> Optional[Dict[str, Any]]:
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
                vol = opt.get("volume", 0) or 0
                oi = opt.get("openInterest", 0) or 0
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
                "context": (
                    f"Unusual options: {len(unusual_strikes)} strikes, "
                    f"P/C={pc_ratio:.2f} ({sentiment})"
                ),
                "priority": "high" if sentiment == "bullish" else "medium",
                "strategy": "options_flow",
                "put_call_ratio": round(pc_ratio, 2),
            }

        except Exception:
            return None


SCANNER_REGISTRY.register(OptionsFlowScanner)
