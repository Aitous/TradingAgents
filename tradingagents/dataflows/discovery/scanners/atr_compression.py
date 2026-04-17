"""ATR Compression Breakout scanner.

Practitioner consensus (Traderslog, VT Markets, VolatilityBox, TradingView community):
ATR(5) / ATR(20) falling to ≤0.75 signals volatility compression vs the 4-week baseline.
VT Markets (Dec 2025 cross-index backtest): combining ATR with directional signals improved
profitability 34% vs directional alone.

Signal: ATR(5) / ATR(20) ≤ atr_ratio_threshold (default 0.75) — short-term volatility
compressed vs recent baseline. PLUS today's close is above the 10-day high (breakout
confirmation — stock is resolving the squeeze upward). Trend filter: price above SMA(50).

Expected holding period: 5–15 days.
Research: docs/iterations/research/2026-04-16-volatility-batch.md
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from tradingagents.dataflows.data_cache.ohlcv_cache import download_ohlcv_cached
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.dataflows.universe import load_universe
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class ATRCompressionScanner(BaseScanner):
    """Scan for ATR compression → directional breakout.

    Identifies stocks where short-term realized volatility (ATR5) has compressed
    well below the 4-week baseline (ATR20), then price is breaking above the
    10-day high — confirming the squeeze is releasing upward.

    Data requirement: ~70 trading days of OHLCV.
    Cost: single batch yfinance download via shared OHLCV cache, zero per-ticker API calls.
    """

    name = "atr_compression"
    pipeline = "momentum"
    strategy = "volatility_contraction_breakout"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.atr_short = self.scanner_config.get("atr_short", 5)           # Short ATR period
        self.atr_long = self.scanner_config.get("atr_long", 20)            # Long ATR period (baseline)
        self.atr_ratio_max = self.scanner_config.get("atr_ratio_max", 0.75)  # Max ATR ratio for compression
        self.breakout_lookback = self.scanner_config.get("breakout_lookback", 10)  # Days for high
        self.sma_trend = self.scanner_config.get("sma_trend", 50)          # Trend filter SMA
        self.min_price = self.scanner_config.get("min_price", 5.0)
        self.min_avg_volume = self.scanner_config.get("min_avg_volume", 100_000)
        self.vol_avg_days = self.scanner_config.get("vol_avg_days", 20)
        self.max_tickers = self.scanner_config.get("max_tickers", 0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(
            f"📉 Scanning ATR compression breakouts "
            f"(ATR{self.atr_short}/ATR{self.atr_long} ≤ {self.atr_ratio_max})..."
        )

        tickers = load_universe(self.config)
        if not tickers:
            logger.warning("No tickers loaded for ATR compression scan")
            return []

        if self.max_tickers:
            tickers = tickers[: self.max_tickers]

        cache_dir = self.config.get("discovery", {}).get("ohlcv_cache_dir", "data/ohlcv_cache")
        data = download_ohlcv_cached(tickers, period="1y", cache_dir=cache_dir)

        if not data:
            return []

        candidates = []
        for ticker, df in data.items():
            result = self._check_compression(df)
            if result:
                result["ticker"] = ticker
                candidates.append(result)

        # Sort by tightest ATR ratio (most compressed first)
        candidates.sort(key=lambda c: c.pop("_atr_ratio", 1.0))
        candidates = candidates[: self.limit]
        logger.info(f"ATR compression: {len(candidates)} candidates")
        return candidates

    def _compute_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> float:
        """Compute ATR over the last `period` bars."""
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        return float(tr.iloc[-period:].mean())

    def _check_compression(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Return candidate dict if ATR compression + upward breakout fires, else None."""
        try:
            df = df.dropna(subset=["Close", "High", "Low", "Volume"])

            min_rows = self.atr_long + self.sma_trend + self.breakout_lookback + 5
            if len(df) < min_rows:
                return None

            close = df["Close"]
            high = df["High"]
            low = df["Low"]
            volume = df["Volume"]

            price = float(close.iloc[-1])

            # --- Liquidity gates ---
            avg_vol = float(volume.iloc[-(self.vol_avg_days + 1) : -1].mean())
            if avg_vol < self.min_avg_volume:
                return None
            if price < self.min_price:
                return None

            # --- Trend filter: price above SMA50 ---
            sma_trend = float(close.iloc[-self.sma_trend :].mean())
            if price <= sma_trend:
                return None

            # --- ATR compression: ATR(short) / ATR(long) ≤ threshold ---
            atr_short = self._compute_atr(high, low, close, self.atr_short)
            atr_long = self._compute_atr(high, low, close, self.atr_long)

            if atr_long <= 0:
                return None

            atr_ratio = atr_short / atr_long
            if atr_ratio > self.atr_ratio_max:
                return None  # Not compressed

            # --- Directional confirmation: close above N-day high (excluding today) ---
            lookback_high = float(high.iloc[-(self.breakout_lookback + 1) : -1].max())
            if price <= lookback_high:
                return None  # Not breaking out upward

            # --- Priority based on compression severity ---
            if atr_ratio < 0.50:
                priority = Priority.CRITICAL.value
            elif atr_ratio < 0.65:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

            pct_above_sma = ((price - sma_trend) / sma_trend) * 100
            breakout_pct = ((price - lookback_high) / lookback_high) * 100

            context = (
                f"ATR({self.atr_short})/ATR({self.atr_long}) = {atr_ratio:.2f} (compressed) | "
                f"Price ${price:.2f} breaks {self.breakout_lookback}d high +{breakout_pct:.1f}% | "
                f"SMA{self.sma_trend} ${sma_trend:.2f} (+{pct_above_sma:.1f}%) | "
                f"Volatility expansion → breakout entry"
            )

            return {
                "source": self.name,
                "context": context,
                "priority": priority,
                "strategy": self.strategy,
                "atr_ratio": round(atr_ratio, 3),
                "atr_short": round(atr_short, 4),
                "atr_long": round(atr_long, 4),
                "breakout_pct": round(breakout_pct, 2),
                "_atr_ratio": atr_ratio,  # for sorting, popped after sort
            }

        except Exception as e:
            logger.debug(f"ATR compression check failed: {e}")
            return None


SCANNER_REGISTRY.register(ATRCompressionScanner)
