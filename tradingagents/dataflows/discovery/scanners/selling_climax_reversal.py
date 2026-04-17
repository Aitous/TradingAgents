"""Selling Climax Reversal scanner — extreme volume on a new multi-day low that reverses intraday.

Signal requires three independent extreme conditions to coincide:
  1. Volume >= 3x the 50-day average (panic selling climax, above O'Neil's 150% threshold)
  2. Close is a new N-day closing low (price exhaustion)
  3. Bullish intraday reversal bar: close >= open AND close in upper 40% of the day's range

Optional 4th condition upgrades priority to CRITICAL: RSI(14) < 35.

Evidence:
  - O'Neil Global Advisors ("Breakouts: Pump up the Volume"): Volume >150% above 50-day avg
    yields +5.04% avg over 63 days, 57.68% win rate (vs +2.61% at <50% above avg).
  - StockCharts (Arthur Hill, 2016): RSI-mean-reversion with volume filter on SPY:
    83–94% win rate, ~8% CAGR, only 30% time in market.
  - Lee & Swaminathan (2000): High-volume past losers outperform by 2–7% annually vs
    low-volume past losers — volume amplifies reversal signal.

Selectivity: all 3 conditions rarely coincide; expected <3 signals/day in 1,000-ticker universe.
Expected holding period: 5–20 days.

Research: docs/iterations/research/2026-04-16-volume-extreme-strategies.md
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from tradingagents.dataflows.data_cache.ohlcv_cache import download_ohlcv_cached
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.dataflows.universe import load_universe
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class SellingClimaxReversalScanner(BaseScanner):
    """Scan for selling climax exhaustion reversal setups.

    Identifies rare moments when panic selling (extreme volume) coincides with
    a multi-day closing low AND an intraday bullish reversal bar — signaling
    capitulation exhaustion where sellers have been absorbed and price closes
    near the high of the day despite setting a new low.

    Distinct from rsi_oversold (pure RSI, no volume or bar structure),
    obv_divergence (multi-week trend, no single-bar reversal), and
    volume_accumulation (unusual volume only, no price-low context).

    Data requirement: 65 trading days of OHLCV (50d vol avg + 20d low lookback + RSI buffer).
    Cost: single batch yfinance download via shared OHLCV cache.
    """

    name = "selling_climax_reversal"
    pipeline = "mean_reversion"
    strategy = "climax_reversal"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.vol_avg_days = self.scanner_config.get("vol_avg_days", 50)
        self.min_volume_multiple = self.scanner_config.get("min_volume_multiple", 3.0)
        self.price_low_days = self.scanner_config.get("price_low_days", 20)
        self.min_range_pct = self.scanner_config.get("min_range_pct", 0.40)
        self.rsi_period = self.scanner_config.get("rsi_period", 14)
        self.rsi_critical_threshold = self.scanner_config.get("rsi_critical_threshold", 35.0)
        self.min_price = self.scanner_config.get("min_price", 5.0)
        self.min_avg_volume = self.scanner_config.get("min_avg_volume", 100_000)
        self.max_tickers = self.scanner_config.get("max_tickers", 0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(
            f"🔻 Scanning selling climax reversals "
            f"(vol>={self.min_volume_multiple}x, {self.price_low_days}d low, bullish bar)..."
        )

        try:
            tickers = load_universe(max_tickers=self.max_tickers or None)
            if not tickers:
                logger.warning("No tickers in universe")
                return []

            # Need 65 days: 50d vol avg + RSI(14) buffer + 20d price low lookback
            history_days = max(self.vol_avg_days, self.price_low_days) + self.rsi_period + 5
            ohlcv = download_ohlcv_cached(tickers=tickers, period_days=history_days)

            if ohlcv is None or ohlcv.empty:
                logger.warning("OHLCV cache returned empty data")
                return []

            candidates = []
            for ticker in tickers:
                try:
                    result = self._check_ticker(ticker, ohlcv)
                    if result:
                        candidates.append(result)
                except Exception as e:
                    logger.debug(f"{ticker}: check failed — {e}")
                    continue

            # Sort: CRITICAL first, then HIGH, then MEDIUM; within each, by volume ratio
            priority_order = {
                Priority.CRITICAL.value: 0,
                Priority.HIGH.value: 1,
                Priority.MEDIUM.value: 2,
                Priority.LOW.value: 3,
            }
            candidates.sort(
                key=lambda x: (priority_order.get(x["priority"], 9), -x.get("_vol_ratio", 0))
            )

            # Strip internal sort key
            for c in candidates:
                c.pop("_vol_ratio", None)

            final = candidates[: self.limit]
            logger.info(f"Found {len(final)} selling climax reversal candidates")
            return final

        except Exception as e:
            logger.warning(f"⚠️  Selling climax reversal scan failed: {e}", exc_info=True)
            return []

    def _check_ticker(
        self, ticker: str, ohlcv: pd.DataFrame
    ) -> Optional[Dict[str, Any]]:
        """Return candidate dict if ticker meets selling climax reversal criteria, else None."""
        # Filter to this ticker
        if hasattr(ohlcv.index, "names") and "ticker" in (ohlcv.index.names or []):
            df = ohlcv.xs(ticker, level="ticker") if ticker in ohlcv.index.get_level_values("ticker") else None
        elif "ticker" in ohlcv.columns:
            df = ohlcv[ohlcv["ticker"] == ticker].copy()
        else:
            df = ohlcv

        if df is None or df.empty:
            return None

        # Normalize column names to lowercase
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]

        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            return None

        df = df.sort_index()
        if len(df) < self.vol_avg_days + 5:
            return None

        # Liquidity gate on 20-day average volume
        avg_vol_20 = df["volume"].iloc[-20:].mean() if len(df) >= 20 else df["volume"].mean()
        if avg_vol_20 < self.min_avg_volume:
            return None

        last = df.iloc[-1]

        # Price gate
        if last["close"] < self.min_price:
            return None

        # Condition 1: Volume climax — today's volume vs 50-day average
        vol_avg_50 = df["volume"].iloc[-(self.vol_avg_days + 1):-1].mean()
        if vol_avg_50 <= 0:
            return None
        vol_ratio = last["volume"] / vol_avg_50
        if vol_ratio < self.min_volume_multiple:
            return None

        # Condition 2: New N-day closing low — today's close is the lowest close in N days
        lookback_closes = df["close"].iloc[-(self.price_low_days + 1):-1]
        if last["close"] >= lookback_closes.min():
            return None

        # Condition 3: Bullish intraday reversal bar
        # close >= open (green candle despite hitting low)
        if last["close"] < last["open"]:
            return None
        # close in upper portion of the day's range (absorbed the selling)
        day_range = last["high"] - last["low"]
        if day_range <= 0:
            return None
        close_position = (last["close"] - last["low"]) / day_range
        if close_position < self.min_range_pct:
            return None

        # Condition 4 (priority): RSI(14)
        rsi = self._compute_rsi(df["close"], self.rsi_period)
        rsi_value = rsi.iloc[-1] if rsi is not None and not rsi.empty else None

        # Determine priority
        if rsi_value is not None and rsi_value < self.rsi_critical_threshold:
            priority = Priority.CRITICAL.value
        elif close_position >= 0.60:
            # Closes in upper 40% of range AND above open — strong reversal bar
            priority = Priority.HIGH.value
        else:
            priority = Priority.MEDIUM.value

        # Build context
        rsi_str = f"RSI({self.rsi_period})={rsi_value:.1f}" if rsi_value is not None else "RSI=N/A"
        context = (
            f"Selling climax: {vol_ratio:.1f}x volume on {self.price_low_days}-day closing low. "
            f"Closed in upper {close_position * 100:.0f}% of today's range (bullish reversal bar). "
            f"{rsi_str}. Potential exhaustion reversal — panic sellers absorbed."
        )

        return {
            "ticker": ticker,
            "source": self.name,
            "context": context,
            "priority": priority,
            "strategy": self.strategy,
            "_vol_ratio": vol_ratio,
        }

    def _compute_rsi(self, closes: pd.Series, period: int) -> Optional[pd.Series]:
        """Compute RSI using Wilder's smoothing."""
        try:
            delta = closes.diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
            rs = avg_gain / avg_loss.replace(0, float("nan"))
            return 100 - (100 / (1 + rs))
        except Exception:
            return None


SCANNER_REGISTRY.register(SellingClimaxReversalScanner)
