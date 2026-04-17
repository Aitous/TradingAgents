"""Gap-Up Continuation (Breakaway Gap) scanner.

Detects stocks that opened significantly above the prior day's close with
above-average volume and held the gap intraday — a 'breakaway gap' pattern
indicating price discovery after new information. Targets the multi-day
continuation drift documented in academic research.

Research basis: docs/iterations/research/2026-04-17-gap-up-continuation.md
Key insight: PMC study (Fakhfakh 2023) measured 54-60% win rate with
+0.30-0.58% avg daily gain for positive gaps. Large gaps (>0.4%) fill
less than 50% of the time, confirming selectivity sharpens the edge.
"""

from typing import Any, Dict, List, Optional

import pandas as pd

from tradingagents.dataflows.data_cache.ohlcv_cache import download_ohlcv_cached
from tradingagents.dataflows.discovery.scanner_registry import SCANNER_REGISTRY, BaseScanner
from tradingagents.dataflows.discovery.utils import Priority
from tradingagents.dataflows.universe import load_universe
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class GapUpContinuationScanner(BaseScanner):
    """Scan for stocks with a recent breakaway gap-up still open for continuation.

    Signal: today's open ≥ prior_close × (1 + min_gap_pct/100)
            AND today's close ≥ today's open × (1 − max_reversal_pct/100)
            AND today's volume ≥ 20d-avg-volume × min_vol_multiple
            AND (optionally) today's close > 200d SMA

    Data requirement: ~200+ trading days of OHLCV (uses 1y lookback).
    Cost: single batch OHLCV cache download, zero per-ticker API calls.
    """

    name = "gap_up_continuation"
    pipeline = "momentum"
    strategy = "gap_continuation"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.min_gap_pct = self.scanner_config.get("min_gap_pct", 2.0)
        self.min_vol_multiple = self.scanner_config.get("min_vol_multiple", 1.5)
        self.max_intraday_reversal_pct = self.scanner_config.get("max_intraday_reversal_pct", 3.0)
        self.require_above_sma200 = self.scanner_config.get("require_above_sma200", False)
        self.sma200_days = self.scanner_config.get("sma200_days", 200)
        self.vol_avg_days = self.scanner_config.get("vol_avg_days", 20)
        self.lookback_period = self.scanner_config.get("lookback_period", "1y")
        self.max_tickers = self.scanner_config.get("max_tickers", 0)

    def scan(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self.is_enabled():
            return []

        logger.info(
            f"📈 Scanning gap-up continuation "
            f"(gap ≥{self.min_gap_pct}%, vol ≥{self.min_vol_multiple}x, "
            f"reversal ≤{self.max_intraday_reversal_pct}%)..."
        )

        tickers = load_universe(self.config)
        if not tickers:
            logger.warning("Gap-up scanner: no tickers loaded")
            return []

        if self.max_tickers and len(tickers) > self.max_tickers:
            tickers = tickers[: self.max_tickers]

        cache_dir = self.config.get("discovery", {}).get("ohlcv_cache_dir", "data/ohlcv_cache")
        data = download_ohlcv_cached(tickers, period=self.lookback_period, cache_dir=cache_dir)

        if not data:
            logger.warning("Gap-up scanner: no OHLCV data available")
            return []

        candidates = []

        for ticker, df in data.items():
            try:
                result = self._check_gap_up(ticker, df)
                if result is not None:
                    candidates.append(result)
            except Exception as e:
                logger.debug(f"Gap-up check failed for {ticker}: {e}")

        candidates.sort(key=lambda c: c.get("gap_pct", 0), reverse=True)
        candidates = candidates[: self.limit]

        logger.info(f"Gap-up continuation: {len(candidates)} candidates")
        return candidates

    def _check_gap_up(self, ticker: str, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        df = df.sort_values("Date").reset_index(drop=True)

        min_rows = self.sma200_days + 2
        if len(df) < min_rows:
            return None

        latest = df.iloc[-1]
        prior = df.iloc[-2]

        prior_close = float(prior["Close"])
        today_open = float(latest["Open"])
        today_close = float(latest["Close"])
        today_vol = float(latest["Volume"])

        if prior_close <= 0 or today_open <= 0:
            return None

        gap_pct = (today_open - prior_close) / prior_close * 100
        if gap_pct < self.min_gap_pct:
            return None

        close_vs_open_pct = (today_close / today_open - 1) * 100
        if close_vs_open_pct < -self.max_intraday_reversal_pct:
            return None

        vol_window = df["Volume"].iloc[-(self.vol_avg_days + 1) : -1]
        avg_vol = float(vol_window.mean()) if len(vol_window) > 0 else 0.0
        if avg_vol <= 0:
            return None
        vol_mult = today_vol / avg_vol
        if vol_mult < self.min_vol_multiple:
            return None

        sma200 = float(df["Close"].iloc[-(self.sma200_days + 1) : -1].mean())
        above_sma200 = today_close > sma200

        if self.require_above_sma200 and not above_sma200:
            return None

        if gap_pct >= 5.0 and vol_mult >= 2.0 and above_sma200:
            priority = Priority.CRITICAL.value
        elif gap_pct >= 3.0 and vol_mult >= 1.5:
            priority = Priority.HIGH.value
        else:
            priority = Priority.MEDIUM.value

        trend_str = "above 200d SMA" if above_sma200 else "below 200d SMA"
        context = (
            f"Gap-up: +{gap_pct:.1f}% above prior close | "
            f"vol {vol_mult:.1f}x avg | "
            f"held intraday ({close_vs_open_pct:+.1f}% vs open) | "
            f"{trend_str} — breakaway continuation setup"
        )

        return {
            "ticker": ticker,
            "source": self.name,
            "context": context,
            "priority": priority,
            "strategy": self.strategy,
            "gap_pct": round(gap_pct, 2),
            "vol_multiple": round(vol_mult, 2),
            "close_vs_open_pct": round(close_vs_open_pct, 2),
            "above_sma200": above_sma200,
        }


SCANNER_REGISTRY.register(GapUpContinuationScanner)
