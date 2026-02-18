"""Sector rotation scanner — finds laggards in accelerating sectors."""

from typing import Any, Dict, List, Optional

import pandas as pd

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# SPDR Select Sector ETFs
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLB": "Materials",
    "XLRE": "Real Estate",
    "XLC": "Communication Services",
}

DEFAULT_TICKER_FILE = "data/tickers.txt"


def _load_tickers_from_file(path: str) -> List[str]:
    """Load ticker symbols from a text file."""
    try:
        with open(path) as f:
            return [
                line.strip().upper()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]
    except Exception:
        return []


class SectorRotationScanner(BaseScanner):
    """Detect sector momentum shifts and find laggards in accelerating sectors."""

    name = "sector_rotation"
    pipeline = "momentum"
    strategy = "sector_rotation"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.ticker_file = self.scanner_config.get(
            "ticker_file",
            config.get("tickers_file", DEFAULT_TICKER_FILE),
        )
        self.max_tickers = self.scanner_config.get("max_tickers", 100)
        self.min_sector_accel = self.scanner_config.get("min_sector_acceleration", 2.0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info("🔄 Scanning sector rotation...")

        from tradingagents.dataflows.y_finance import download_history, get_ticker_info

        # Step 1: Identify accelerating sectors
        try:
            etf_symbols = list(SECTOR_ETFS.keys())
            etf_data = download_history(
                etf_symbols, period="2mo", interval="1d", auto_adjust=True, progress=False
            )
        except Exception as e:
            logger.error(f"Failed to download sector ETF data: {e}")
            return []

        if etf_data is None or etf_data.empty:
            return []

        accelerating_sectors = self._find_accelerating_sectors(etf_data)
        if not accelerating_sectors:
            logger.info("No accelerating sectors detected")
            return []

        sector_names = [SECTOR_ETFS.get(etf, etf) for etf in accelerating_sectors]
        logger.info(f"Accelerating sectors: {', '.join(sector_names)}")

        # Step 2: Find laggard stocks in those sectors
        tickers = _load_tickers_from_file(self.ticker_file)
        if not tickers:
            return []

        tickers = tickers[: self.max_tickers]

        candidates = []
        for ticker in tickers:
            result = self._check_sector_laggard(ticker, accelerating_sectors, get_ticker_info)
            if result:
                candidates.append(result)
            if len(candidates) >= self.limit:
                break

        logger.info(f"Sector rotation: {len(candidates)} candidates")
        return candidates

    def _find_accelerating_sectors(self, data: pd.DataFrame) -> List[str]:
        """Find sectors where 5-day return is accelerating vs 20-day trend."""
        accelerating = []

        for etf in SECTOR_ETFS:
            try:
                if isinstance(data.columns, pd.MultiIndex):
                    if etf not in data.columns.get_level_values(1):
                        continue
                    close = data.xs(etf, axis=1, level=1)["Close"].dropna()
                else:
                    close = data["Close"].dropna()

                if len(close) < 21:
                    continue

                ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100
                ret_20d = (float(close.iloc[-1]) / float(close.iloc[-21]) - 1) * 100

                # Acceleration: 5-day annualized return significantly beats 20-day
                daily_rate_5d = ret_5d / 5
                daily_rate_20d = ret_20d / 20

                if daily_rate_20d != 0:
                    acceleration = daily_rate_5d / daily_rate_20d
                elif daily_rate_5d > 0:
                    acceleration = 10.0  # Strong acceleration from flat
                else:
                    acceleration = 0

                if acceleration >= self.min_sector_accel and ret_5d > 0:
                    accelerating.append(etf)
                    logger.debug(
                        f"{etf} ({SECTOR_ETFS[etf]}): 5d={ret_5d:+.1f}%, "
                        f"20d={ret_20d:+.1f}%, accel={acceleration:.1f}x"
                    )
            except Exception as e:
                logger.debug(f"Error analyzing {etf}: {e}")

        return accelerating

    def _check_sector_laggard(
        self, ticker: str, accelerating_sectors: List[str], get_info_fn
    ) -> Optional[Dict[str, Any]]:
        """Check if stock is in an accelerating sector but hasn't moved yet."""
        try:
            info = get_info_fn(ticker)
            if not info:
                return None

            stock_sector = info.get("sector", "")

            # Map stock sector to ETF
            sector_to_etf = {v: k for k, v in SECTOR_ETFS.items()}
            sector_etf = sector_to_etf.get(stock_sector)

            if not sector_etf or sector_etf not in accelerating_sectors:
                return None

            # Check if stock is lagging its sector
            from tradingagents.dataflows.y_finance import download_history

            hist = download_history(
                ticker, period="1mo", interval="1d", auto_adjust=True, progress=False
            )
            if hist is None or hist.empty or len(hist) < 6:
                return None

            # Handle MultiIndex
            if isinstance(hist.columns, pd.MultiIndex):
                tickers_in_data = hist.columns.get_level_values(1).unique()
                target = ticker if ticker in tickers_in_data else tickers_in_data[0]
                hist = hist.xs(target, level=1, axis=1)

            close = hist["Close"] if "Close" in hist.columns else hist.iloc[:, 0]
            ret_5d = (float(close.iloc[-1]) / float(close.iloc[-6]) - 1) * 100

            # Stock is a laggard if it moved less than 2% while sector is accelerating
            if ret_5d > 2.0:
                return None  # Already moved, not a laggard

            context = (
                f"Sector rotation: {stock_sector} sector accelerating, "
                f"{ticker} lagging at {ret_5d:+.1f}% (5d)"
            )

            return {
                "ticker": ticker,
                "source": self.name,
                "context": context,
                "priority": Priority.MEDIUM.value,
                "strategy": self.strategy,
                "sector": stock_sector,
                "sector_etf": sector_etf,
                "stock_5d_return": round(ret_5d, 2),
            }

        except Exception as e:
            logger.debug(f"Sector check failed for {ticker}: {e}")
            return None


SCANNER_REGISTRY.register(SectorRotationScanner)
