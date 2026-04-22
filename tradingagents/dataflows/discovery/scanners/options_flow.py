"""Unusual options activity scanner.

Scans a ticker universe (loaded from data/tickers.txt by default) for
unusual options volume relative to open interest.  Uses ThreadPoolExecutor
for parallel chain fetching so large universes remain practical.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.dataflows.universe import load_universe
import numpy as np

from tradingagents.dataflows.y_finance import get_option_chain, get_ticker_info, get_ticker_options
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class OptionsFlowScanner(BaseScanner):
    """Scan for unusual options activity across a ticker universe."""

    name = "options_flow"
    pipeline = "edge"
    strategy = "options_flow"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_volume_oi_ratio = self.scanner_config.get("unusual_volume_multiple", 2.0)
        self.min_volume = self.scanner_config.get("min_volume", 1000)
        self.min_premium = self.scanner_config.get("min_premium", 25000)
        self.max_tickers = self.scanner_config.get("max_tickers", 150)
        self.max_workers = self.scanner_config.get("max_workers", 8)

        # Load universe: explicit config list overrides the shared universe file
        if "ticker_universe" in self.scanner_config:
            self.ticker_universe = self.scanner_config["ticker_universe"]
        else:
            self.ticker_universe = load_universe(config)
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
                pool.submit(self._analyze_ticker_options, ticker): ticker for ticker in universe
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)
                except Exception:
                    continue

        # Sort by signal quality: unusual strike count, then bullish bias
        candidates.sort(key=lambda c: c.get("options_score", 0), reverse=True)
        candidates = candidates[: self.limit]

        logger.info(f"Found {len(candidates)} unusual options flows")
        return candidates

    def _analyze_ticker_options(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Scan a single ticker for unusual options activity across multiple expirations."""
        try:
            expirations = get_ticker_options(ticker)
            if not expirations:
                return None

            # Pre-fetch current price once for IV skew moneyness calculation
            current_price = None
            try:
                current_price = get_ticker_info(ticker).get("regularMarketPrice")
            except Exception:
                pass

            # Scan up to 3 nearest expirations — collect volume stats and IV data in one pass
            max_expirations = min(3, len(expirations))
            total_unusual_calls = 0
            total_unusual_puts = 0
            total_call_vol = 0
            total_put_vol = 0
            best_expiration = None
            best_unusual_count = 0
            atm_call_ivs: List[float] = []
            otm_put_ivs: List[float] = []

            for exp in expirations[:max_expirations]:
                try:
                    options = get_option_chain(ticker, exp)
                except Exception:
                    continue

                if options is None:
                    continue

                calls_df, puts_df = (None, None)
                if isinstance(options, tuple) and len(options) == 2:
                    calls_df, puts_df = options
                elif hasattr(options, "calls") and hasattr(options, "puts"):
                    calls_df, puts_df = options.calls, options.puts
                else:
                    continue

                exp_unusual_calls = 0
                exp_unusual_puts = 0

                # Analyze calls — volume stats + collect ATM IVs for skew
                if calls_df is not None and not calls_df.empty:
                    for _, opt in calls_df.iterrows():
                        vol = opt.get("volume", 0) or 0
                        oi = opt.get("openInterest", 0) or 0
                        price = opt.get("lastPrice", 0) or 0

                        if vol < self.min_volume:
                            continue
                        # Premium filter (volume * price * 100 shares per contract)
                        if (vol * price * 100) < self.min_premium:
                            continue
                        if oi > 0 and (vol / oi) >= self.min_volume_oi_ratio:
                            exp_unusual_calls += 1
                        total_call_vol += vol

                        # Collect ATM call IV for skew (moneyness within 5% of spot)
                        if current_price:
                            strike = opt.get("strike")
                            iv = opt.get("impliedVolatility")
                            if strike and iv and iv > 0:
                                if abs(strike / current_price - 1.0) <= 0.05:
                                    atm_call_ivs.append(float(iv))

                # Analyze puts — volume stats + collect OTM IVs for skew
                if puts_df is not None and not puts_df.empty:
                    for _, opt in puts_df.iterrows():
                        vol = opt.get("volume", 0) or 0
                        oi = opt.get("openInterest", 0) or 0
                        price = opt.get("lastPrice", 0) or 0

                        if vol < self.min_volume:
                            continue
                        if (vol * price * 100) < self.min_premium:
                            continue
                        if oi > 0 and (vol / oi) >= self.min_volume_oi_ratio:
                            exp_unusual_puts += 1
                        total_put_vol += vol

                        # Collect OTM put IV for skew (5-20% below spot = informed hedging zone)
                        if current_price:
                            strike = opt.get("strike")
                            iv = opt.get("impliedVolatility")
                            if strike and iv and iv > 0:
                                moneyness = (current_price - strike) / current_price
                                if 0.05 <= moneyness <= 0.20:
                                    otm_put_ivs.append(float(iv))

                total_unusual_calls += exp_unusual_calls
                total_unusual_puts += exp_unusual_puts

                exp_total = exp_unusual_calls + exp_unusual_puts
                if exp_total > best_unusual_count:
                    best_unusual_count = exp_total
                    best_expiration = exp

            total_unusual = total_unusual_calls + total_unusual_puts
            if total_unusual == 0:
                return None

            # Calculate put/call ratio
            pc_ratio = total_put_vol / total_call_vol if total_call_vol > 0 else 999

            if pc_ratio < 0.7:
                sentiment = "bullish"
            elif pc_ratio > 1.3:
                sentiment = "bearish"
            else:
                sentiment = "neutral"

            # IV skew = OTM put IV - ATM call IV (higher = more bearish informed positioning)
            iv_skew = (
                float(np.mean(otm_put_ivs) - np.mean(atm_call_ivs))
                if atm_call_ivs and otm_put_ivs
                else None
            )

            # Refine priority using IV skew (OTM put IV - ATM call IV)
            # High skew = bearish informed positioning; low skew = bullish
            if sentiment == "bullish":
                if iv_skew is not None and iv_skew < 0.02:
                    priority = Priority.CRITICAL.value  # unusual calls + low skew = strong bullish
                else:
                    priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

            context = (
                f"Unusual options: {total_unusual} strikes across {max_expirations} exp, "
                f"P/C={pc_ratio:.2f} ({sentiment}), "
                f"{total_unusual_calls} unusual calls / {total_unusual_puts} unusual puts"
            )
            if iv_skew is not None:
                context += f", IV_skew={iv_skew:.3f}"

            # Scoring: unusual strike count + bullish call bias bonus
            # Calls weighted 1.5x to favour bullish directional flow
            options_score = total_unusual_puts + (total_unusual_calls * 1.5)

            return {
                "ticker": ticker,
                "source": self.name,
                "context": context,
                "priority": priority,
                "strategy": self.strategy,
                "put_call_ratio": round(pc_ratio, 2),
                "unusual_calls": total_unusual_calls,
                "unusual_puts": total_unusual_puts,
                "best_expiration": best_expiration,
                "options_score": options_score,
                "iv_skew": round(iv_skew, 4) if iv_skew is not None else None,
            }

        except Exception as e:
            logger.debug(f"Error scanning {ticker}: {e}")
            return None


SCANNER_REGISTRY.register(OptionsFlowScanner)
