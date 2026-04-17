"""Volume Dry-Up (VDU) Pocket Pivot scanner.

Based on Mark Minervini's SEPA methodology (US Investing Championship 1997,
audited 33,554% 5-year return) and Chris Kacher / Gil Morales "Pocket Pivot"
(IBD-lineage; systematized in "Trade Like an O'Neil Disciple", 2010).

The VDU is the key non-distribution confirmation in the VCP (Volatility Contraction
Pattern): if a stock in a strong uptrend pulls back on progressively shrinking volume,
institutions are NOT selling — they are holding (or quietly accumulating). The Pocket
Pivot entry fires when an up-close bar's volume exceeds any down-close bar's volume
in the prior 10 sessions, confirming demand absorption of the supply contraction.

Signal:
  1. Full trend template: close > SMA50 > SMA150 > SMA200 (aligned bull trend)
  2. Price within 25% of 52-week high (not extended/broken)
  3. Price at least 25% above 52-week low (confirmed recovery, not bottom-fishing)
  4. VDU: volume today < 60% of 50-day average (volume contraction on pullback)
  5. Pocket Pivot: today is an up-close bar AND today's volume > max volume of any
     down-close day in the prior 10 trading sessions

Selectivity: all 5 conditions + the 52-week structure requirements create a rare
conjunction; expected <5 picks/day in a 1000-ticker universe.

Expected holding period: 10–30 days (trend continuation entry).
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


class VolumeDryUpScanner(BaseScanner):
    """Scan for Pocket Pivot entries on Volume Dry-Up consolidations.

    Identifies stocks in confirmed multi-SMA uptrends that are consolidating on
    shrinking volume (VDU phase), then catching the precise entry bar where demand
    re-emerges stronger than any recent supply day.

    Distinct from atr_compression (ATR/volatility ratio, no SMA structure or pocket
    pivot trigger), high_52w_breakout (new 52w high breakout, not mid-consolidation),
    and minervini (broader 7-criteria screen, no VDU-specific volume filter).

    Data requirement: ~260 trading days of OHLCV (252d for 52w high/low + SMA200 warmup).
    Cost: single batch yfinance download via shared OHLCV cache, zero per-ticker API calls.
    """

    name = "volume_dry_up"
    pipeline = "momentum"
    strategy = "vdu_pocket_pivot"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.sma_fast = self.scanner_config.get("sma_fast", 50)
        self.sma_mid = self.scanner_config.get("sma_mid", 150)
        self.sma_slow = self.scanner_config.get("sma_slow", 200)
        self.vol_avg_days = self.scanner_config.get("vol_avg_days", 50)  # 50d volume baseline
        self.vdu_vol_pct = self.scanner_config.get(
            "vdu_vol_pct", 0.60
        )  # VDU: today < 60% of 50d avg
        self.pocket_pivot_lookback = self.scanner_config.get("pocket_pivot_lookback", 10)
        self.max_pct_from_52w_high = self.scanner_config.get("max_pct_from_52w_high", 25.0)
        self.min_pct_above_52w_low = self.scanner_config.get("min_pct_above_52w_low", 25.0)
        self.min_price = self.scanner_config.get("min_price", 5.0)
        self.min_avg_volume = self.scanner_config.get("min_avg_volume", 100_000)
        self.max_tickers = self.scanner_config.get("max_tickers", 0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(
            f"📦 Scanning Volume Dry-Up Pocket Pivot "
            f"(VDU < {self.vdu_vol_pct*100:.0f}% of {self.vol_avg_days}d avg, "
            f"pocket pivot trigger)..."
        )

        tickers = load_universe(self.config)
        if not tickers:
            logger.warning("No tickers loaded for volume dry-up scan")
            return []

        if self.max_tickers:
            tickers = tickers[: self.max_tickers]

        cache_dir = self.config.get("discovery", {}).get("ohlcv_cache_dir", "data/ohlcv_cache")
        data = download_ohlcv_cached(tickers, period="1y", cache_dir=cache_dir)

        if not data:
            return []

        candidates = []
        for ticker, df in data.items():
            result = self._check_vdu(df)
            if result:
                result["ticker"] = ticker
                candidates.append(result)

        # Sort by proximity to 52-week high (tightest base = highest quality)
        candidates.sort(key=lambda c: c.pop("_pct_from_high", 100))
        candidates = candidates[: self.limit]
        logger.info(f"volume_dry_up: {len(candidates)} candidates")
        return candidates

    def _check_vdu(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Return candidate dict if VDU Pocket Pivot detected, else None."""
        try:
            df = df.dropna(subset=["Close", "Volume"])

            # Need 252 bars for 52-week range + SMA200 warmup
            min_rows = 252 + self.vol_avg_days + 5
            if len(df) < min_rows:
                return None

            close = df["Close"]
            volume = df["Volume"]
            price = float(close.iloc[-1])
            today_vol = float(volume.iloc[-1])
            prev_close = float(close.iloc[-2])

            # --- Liquidity gates ---
            avg_vol_20 = float(volume.iloc[-(20 + 1) : -1].mean())
            if avg_vol_20 < self.min_avg_volume:
                return None
            if price < self.min_price:
                return None

            # --- Trend template: SMA alignment ---
            sma_fast = float(close.iloc[-self.sma_fast :].mean())
            sma_mid = float(close.iloc[-self.sma_mid :].mean())
            sma_slow = float(close.iloc[-self.sma_slow :].mean())

            if not (price > sma_fast > sma_mid > sma_slow):
                return None

            # --- 52-week structure ---
            hi_52w = float(close.iloc[-252:].max())
            lo_52w = float(close.iloc[-252:].min())

            pct_from_high = ((hi_52w - price) / hi_52w) * 100
            pct_above_low = ((price - lo_52w) / lo_52w) * 100

            if pct_from_high > self.max_pct_from_52w_high:
                return None  # Too far below 52w high — base broken or extended down
            if pct_above_low < self.min_pct_above_52w_low:
                return None  # Too close to 52w low — not a recovered uptrend

            # --- VDU: today's volume < threshold% of 50-day average ---
            vol_avg_50 = float(volume.iloc[-(self.vol_avg_days + 1) : -1].mean())
            if vol_avg_50 <= 0:
                return None
            vdu_ratio = today_vol / vol_avg_50
            if vdu_ratio >= self.vdu_vol_pct:
                return None  # Volume not contracted enough

            # --- Pocket Pivot: up-close day + volume beats any down-close day in last 10 ---
            if price <= prev_close:
                return None  # Must be an up-close bar

            # Find max volume on down-close days in prior pocket_pivot_lookback sessions
            # (exclude today — look at days [-lookback-1:-1])
            lookback_closes = close.iloc[-(self.pocket_pivot_lookback + 2) : -1].values
            lookback_vols = volume.iloc[-(self.pocket_pivot_lookback + 2) : -1].values

            down_day_vols = [
                lookback_vols[i]
                for i in range(1, len(lookback_closes))
                if lookback_closes[i] < lookback_closes[i - 1]
            ]

            if not down_day_vols:
                # No down days in the lookback — can't confirm pocket pivot
                return None

            max_down_vol = max(down_day_vols)
            if today_vol <= max_down_vol:
                return None  # Demand didn't absorb prior supply

            # Priority based on tightness of base (closer to 52w high = better)
            if pct_from_high <= 5:
                priority = Priority.CRITICAL.value
            elif pct_from_high <= 12:
                priority = Priority.HIGH.value
            else:
                priority = Priority.MEDIUM.value

            pocket_pivot_ratio = today_vol / max_down_vol

            context = (
                f"VDU Pocket Pivot: volume {vdu_ratio*100:.0f}% of 50d avg (dry-up phase) | "
                f"Up-close day {pocket_pivot_ratio:.1f}× max down-day volume (demand confirmed) | "
                f"{pct_from_high:.1f}% below 52w high | "
                f"SMA{self.sma_fast}>{self.sma_mid}>{self.sma_slow} trend template | "
                f"Non-distribution consolidation — trend continuation setup"
            )

            return {
                "source": self.name,
                "context": context,
                "priority": priority,
                "strategy": self.strategy,
                "pct_from_52w_high": round(pct_from_high, 1),
                "vdu_vol_ratio": round(vdu_ratio, 2),
                "pocket_pivot_ratio": round(pocket_pivot_ratio, 2),
                "_pct_from_high": pct_from_high,
            }

        except Exception as e:
            logger.debug(f"volume_dry_up check failed: {e}")
            return None


SCANNER_REGISTRY.register(VolumeDryUpScanner)
