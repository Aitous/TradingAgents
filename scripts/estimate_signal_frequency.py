#!/usr/bin/env python3
"""Pre-implementation signal frequency estimator.

Runs a scanner against the last N days of the local OHLCV cache and reports
picks/day. Use this BEFORE running a full walk-forward backtest to detect
unselective signals early.

Rule of thumb from empirical results:
  > 8 picks/day  → almost certainly won't meet promotion thresholds (discard)
  3–8 picks/day  → borderline, may pass with strong signal
  < 3 picks/day  → selective, worth full backtest

Usage:
    python scripts/estimate_signal_frequency.py --scanner volume_dry_up
    python scripts/estimate_signal_frequency.py --scanner atr_compression --days 30
    python scripts/estimate_signal_frequency.py --scanner my_new_scanner --days 14
"""

import argparse
import importlib
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from statistics import mean, median
from typing import List
from unittest.mock import patch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

logging.disable(logging.WARNING)

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

# Mirror of backtest_scanners.py — all registered scanners
SCANNER_MODULES = {
    "rsi_oversold": "tradingagents.dataflows.discovery.scanners.rsi_oversold",
    "high_52w_breakout": "tradingagents.dataflows.discovery.scanners.high_52w_breakout",
    "minervini": "tradingagents.dataflows.discovery.scanners.minervini",
    "technical_breakout": "tradingagents.dataflows.discovery.scanners.technical_breakout",
    "atr_compression": "tradingagents.dataflows.discovery.scanners.atr_compression",
    "volume_dry_up": "tradingagents.dataflows.discovery.scanners.volume_dry_up",
}


def load_parquet(cache_dir: str = "data/ohlcv_cache"):
    import pandas as pd

    files = list(Path(cache_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {cache_dir}")
    f = max(files, key=lambda p: p.stat().st_size)
    print(f"Cache: {f.name}")
    df = pd.read_parquet(f)
    df["Date"] = __import__("pandas").to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df


def slice_snapshot(df, as_of: date):
    import pandas as pd

    cutoff = pd.Timestamp(as_of)
    sliced = df[df["Date"] <= cutoff]
    result = {}
    for ticker, grp in sliced.groupby("Ticker"):
        result[str(ticker)] = grp.drop(columns=["Ticker"]).reset_index(drop=True)
    return result


def get_trading_days(df) -> List[date]:
    return sorted(df["Date"].dt.date.unique())


def run_scanner_on_day(scanner_name: str, module_path: str, snapshot: dict) -> int:
    mod = importlib.import_module(module_path)
    from tradingagents.dataflows.discovery.scanner_registry import BaseScanner

    scanner_cls = next(
        v
        for k, v in vars(mod).items()
        if isinstance(v, type) and issubclass(v, BaseScanner) and v is not BaseScanner
    )
    scanner = scanner_cls(DEFAULT_CONFIG)
    patch_target = f"{module_path}.download_ohlcv_cached"
    with patch(patch_target, return_value=snapshot):
        try:
            picks = scanner.scan({})
            return len(picks)
        except Exception as e:
            print(f"  ⚠️  {scanner_name} raised: {e}")
            return 0


def estimate(scanner_name: str, days: int = 20) -> None:
    import pandas as pd

    if scanner_name not in SCANNER_MODULES:
        # Try dynamic discovery
        module_path = f"tradingagents.dataflows.discovery.scanners.{scanner_name}"
        try:
            importlib.import_module(module_path)
            SCANNER_MODULES[scanner_name] = module_path
        except ImportError:
            print(f"❌ Scanner '{scanner_name}' not found in SCANNER_MODULES and not importable.")
            print(f"   Known scanners: {', '.join(SCANNER_MODULES)}")
            sys.exit(1)

    module_path = SCANNER_MODULES[scanner_name]
    df = load_parquet()
    trading_days = get_trading_days(df)

    # Use the last `days` trading days
    sim_days = trading_days[-days:] if len(trading_days) >= days else trading_days

    print(f"\nEstimating signal frequency for: {scanner_name}")
    print(f"Period: {sim_days[0]} → {sim_days[-1]} ({len(sim_days)} days)\n")

    daily_counts = []
    for sim_date in sim_days:
        snapshot = slice_snapshot(df, sim_date)
        count = run_scanner_on_day(scanner_name, module_path, snapshot)
        daily_counts.append(count)
        bar = "█" * count + "░" * max(0, 10 - count)
        print(f"  {sim_date}  [{bar}]  {count:3d} picks")

    avg = mean(daily_counts)
    med = median(daily_counts)
    hit_rate = sum(1 for c in daily_counts if c > 0) / len(daily_counts) * 100

    # Selectivity rating
    if avg > 8:
        rating = "🔴 UNSELECTIVE — very likely to fail promotion thresholds"
        verdict = "STOP: relax threshold or add a second independent condition before backtesting"
    elif avg > 5:
        rating = "🟡 BORDERLINE — may pass with a strong underlying signal"
        verdict = "CONSIDER: add one tightening condition or proceed cautiously"
    elif avg > 2:
        rating = "🟢 SELECTIVE — good candidate for full backtest"
        verdict = "PROCEED: signal fires rarely enough to be meaningful"
    else:
        rating = "🟢🟢 VERY SELECTIVE — rare signal, similar to volume_dry_up"
        verdict = "PROCEED: high precision expected; small-sample caveat applies"

    print(f"""
{'='*60}
  Signal Frequency Estimate: {scanner_name}
{'='*60}
  Avg picks/day  : {avg:.1f}
  Median picks/day: {med:.1f}
  Days with picks: {hit_rate:.0f}% of {len(sim_days)} trading days
  Total picks    : {sum(daily_counts)}

  Rating  : {rating}
  Verdict : {verdict}
{'='*60}

Benchmark (from empirical backtest results):
  atr_compression  3.3/day → WR-20d=59.3%  ✅ PROMOTED
  volume_dry_up    0.1/day → WR-20d=80.0%  ✅ PROMOTED
  macd_histogram   8.7/day → WR-20d=53.6%  ❌ DISCARDED
  consecutive_down 9.5/day → WR-20d=53.6%  ❌ DISCARDED
""")


def main():
    parser = argparse.ArgumentParser(description="Estimate scanner signal frequency")
    parser.add_argument("--scanner", required=True, help="Scanner name (e.g. volume_dry_up)")
    parser.add_argument("--days", type=int, default=20, help="Number of recent trading days to test (default: 20)")
    args = parser.parse_args()
    estimate(args.scanner, args.days)


if __name__ == "__main__":
    main()
