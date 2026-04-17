#!/usr/bin/env python3
"""Walk-forward backtest for OHLCV-based scanners.

Reads the local OHLCV parquet cache (no API calls), simulates running each
scanner on every historical trading day, records picks, and measures forward
returns at 1d / 5d / 10d / 20d horizons.

Usage:
    python scripts/backtest_scanners.py
    python scripts/backtest_scanners.py --scanners rsi_oversold high_52w_breakout
    python scripts/backtest_scanners.py --start 2025-10-01 --end 2026-03-01
    python scripts/backtest_scanners.py --out results/backtest/

Output:
    results/backtest/YYYY-MM-DD/picks.csv     — every pick with forward returns
    results/backtest/YYYY-MM-DD/summary.csv   — per-scanner win-rate summary
    results/backtest/YYYY-MM-DD/summary.json  — machine-readable version
"""

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import patch

import pandas as pd

# ── repo root on path ────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from tradingagents.default_config import DEFAULT_CONFIG  # noqa: E402

logging.disable(logging.WARNING)  # suppress scanner noise during backtest

# ── scanners to backtest (module path → scanner name) ────────────────────────
SCANNER_MODULES = {
    "rsi_oversold": "tradingagents.dataflows.discovery.scanners.rsi_oversold",
    "high_52w_breakout": "tradingagents.dataflows.discovery.scanners.high_52w_breakout",
    "minervini": "tradingagents.dataflows.discovery.scanners.minervini",
    "technical_breakout": "tradingagents.dataflows.discovery.scanners.technical_breakout",
    "nr7_breakout": "tradingagents.dataflows.discovery.scanners.nr7_breakout",
    "bb_squeeze": "tradingagents.dataflows.discovery.scanners.bb_squeeze",
    "atr_compression": "tradingagents.dataflows.discovery.scanners.atr_compression",
}

# Forward-return windows (trading days)
WINDOWS = [1, 5, 10, 20]

# Leave at least this many days of forward data — skip sim dates too close to end
MAX_FORWARD_DAYS = max(WINDOWS) * 2  # calendar-day buffer


# ── data loading ─────────────────────────────────────────────────────────────


def load_parquet(cache_dir: str = "data/ohlcv_cache") -> pd.DataFrame:
    """Load the largest parquet file (the full-universe nightly cache)."""
    files = list(Path(cache_dir).glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {cache_dir}")
    # Pick the largest file — that's the full 1003-ticker universe
    f = max(files, key=lambda p: p.stat().st_size)
    print(f"Loading cache: {f.name}")
    df = pd.read_parquet(f)
    df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None).dt.normalize()
    return df


def get_trading_days(df: pd.DataFrame) -> List[date]:
    """Return sorted list of unique trading dates in the cache."""
    return sorted(df["Date"].dt.date.unique())


def split_by_ticker(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Long-format DataFrame → {ticker: per-ticker DataFrame}."""
    result = {}
    for ticker, grp in df.groupby("Ticker"):
        result[str(ticker)] = grp.drop(columns=["Ticker"]).reset_index(drop=True)
    return result


def slice_snapshot(df: pd.DataFrame, as_of: date) -> Dict[str, pd.DataFrame]:
    """Return per-ticker DataFrames containing only rows up to as_of (inclusive)."""
    cutoff = pd.Timestamp(as_of)
    sliced = df[df["Date"] <= cutoff]
    return split_by_ticker(sliced)


# ── forward return computation ────────────────────────────────────────────────


def build_close_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot to Date × Ticker close-price matrix for fast forward-return lookup."""
    return df.pivot_table(index="Date", columns="Ticker", values="Close")


def forward_return(
    close: pd.DataFrame,
    ticker: str,
    entry_date: date,
    n_trading_days: int,
) -> Optional[float]:
    """Return n-day forward return from entry_date for ticker, or None if unavailable."""
    if ticker not in close.columns:
        return None
    ts = pd.Timestamp(entry_date)
    col = close[ticker].dropna()
    future = col[col.index > ts]
    if len(future) < n_trading_days:
        return None
    exit_price = future.iloc[n_trading_days - 1]
    entry_price = col.get(ts)
    if entry_price is None or pd.isna(entry_price) or entry_price == 0:
        # Try last available price on or before entry
        prev = col[col.index <= ts]
        if prev.empty:
            return None
        entry_price = prev.iloc[-1]
    if pd.isna(entry_price) or entry_price == 0:
        return None
    return (exit_price - entry_price) / entry_price


# ── scanner runner (with OHLCV injection) ────────────────────────────────────


def run_scanner(scanner_name: str, snapshot: Dict[str, pd.DataFrame]) -> List[Dict]:
    """Run a scanner against a pre-sliced OHLCV snapshot.

    Patches download_ohlcv_cached in the scanner's module so it returns the
    snapshot directly — no API call, no disk read during the backtest loop.
    """
    module_path = SCANNER_MODULES[scanner_name]

    # Import scanner class
    import importlib

    mod = importlib.import_module(module_path)
    # Find the scanner class by looking for BaseScanner subclasses in the module
    from tradingagents.dataflows.discovery.scanner_registry import BaseScanner

    scanner_cls = next(
        v
        for k, v in vars(mod).items()
        if isinstance(v, type) and issubclass(v, BaseScanner) and v is not BaseScanner
    )
    scanner = scanner_cls(DEFAULT_CONFIG)

    # Patch download_ohlcv_cached to return our snapshot
    patch_target = f"{module_path}.download_ohlcv_cached"
    with patch(patch_target, return_value=snapshot):
        try:
            return scanner.scan({})
        except Exception as e:
            print(f"    ⚠️  {scanner_name} raised: {e}")
            return []


# ── main backtest loop ────────────────────────────────────────────────────────


def run_backtest(
    scanner_names: List[str],
    start: Optional[date],
    end: Optional[date],
    out_dir: str,
) -> None:
    df = load_parquet()
    trading_days = get_trading_days(df)
    close = build_close_matrix(df)

    last_day = trading_days[-1]

    # Apply date range filters
    if start:
        trading_days = [d for d in trading_days if d >= start]
    if end:
        trading_days = [d for d in trading_days if d <= end]

    # Drop days too close to the end to have full forward-return windows
    cutoff_end = last_day - timedelta(days=MAX_FORWARD_DAYS)
    sim_days = [d for d in trading_days if d <= cutoff_end]

    if not sim_days:
        print("No simulation dates in range — try a wider date window.")
        return

    print(f"\nBacktest: {sim_days[0]} → {sim_days[-1]} ({len(sim_days)} days)")
    print(f"Scanners: {', '.join(scanner_names)}\n")

    all_picks = []

    for i, sim_date in enumerate(sim_days):
        snapshot = slice_snapshot(df, sim_date)
        print(f"[{i+1:3d}/{len(sim_days)}] {sim_date}", end="")

        for scanner_name in scanner_names:
            picks = run_scanner(scanner_name, snapshot)
            print(f"  {scanner_name}={len(picks)}", end="")

            for pick in picks:
                ticker = pick.get("ticker", "")
                row = {
                    "date": sim_date,
                    "scanner": scanner_name,
                    "ticker": ticker,
                    "priority": pick.get("priority", ""),
                    "context": pick.get("context", ""),
                }
                # Capture any extra numeric fields the scanner returns
                extra_keys = [k for k in pick if k not in row and not k.startswith("_")]
                for k in extra_keys:
                    row[k] = pick[k]
                for w in WINDOWS:
                    row[f"fwd_{w}d"] = forward_return(close, ticker, sim_date, w)
                all_picks.append(row)

        print()  # newline after each date

    if not all_picks:
        print("\nNo picks generated — nothing to report.")
        return

    picks_df = pd.DataFrame(all_picks)

    # ── summary stats per scanner ─────────────────────────────────────────────
    summary_rows = []
    for scanner_name in scanner_names:
        sub = picks_df[picks_df["scanner"] == scanner_name]
        if sub.empty:
            summary_rows.append({"scanner": scanner_name, "picks": 0})
            continue
        row = {
            "scanner": scanner_name,
            "picks": len(sub),
            "unique_tickers": sub["ticker"].nunique(),
        }
        for w in WINDOWS:
            col = f"fwd_{w}d"
            valid = sub[col].dropna()
            if valid.empty:
                row[f"win_rate_{w}d"] = None
                row[f"avg_return_{w}d"] = None
                row[f"median_return_{w}d"] = None
            else:
                row[f"win_rate_{w}d"] = round((valid > 0).mean() * 100, 1)
                row[f"avg_return_{w}d"] = round(valid.mean() * 100, 2)
                row[f"median_return_{w}d"] = round(valid.median() * 100, 2)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # ── save outputs ──────────────────────────────────────────────────────────
    out_path = Path(out_dir) / date.today().isoformat()
    out_path.mkdir(parents=True, exist_ok=True)

    picks_df.to_csv(out_path / "picks.csv", index=False)
    summary_df.to_csv(out_path / "summary.csv", index=False)
    summary_df.to_json(out_path / "summary.json", orient="records", indent=2)

    print(f"\nResults saved to {out_path}/")
    print("\n" + "=" * 72)
    print(summary_df.to_string(index=False))
    print("=" * 72)


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Walk-forward OHLCV scanner backtest")
    parser.add_argument(
        "--scanners",
        nargs="+",
        default=list(SCANNER_MODULES.keys()),
        choices=list(SCANNER_MODULES.keys()),
        help="Scanners to backtest (default: all)",
    )
    parser.add_argument(
        "--start",
        type=date.fromisoformat,
        default=None,
        help="Start date YYYY-MM-DD (default: full cache)",
    )
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="End date YYYY-MM-DD (default: full cache minus forward window)",
    )
    parser.add_argument(
        "--out", default="results/backtest", help="Output directory (default: results/backtest)"
    )
    args = parser.parse_args()

    run_backtest(
        scanner_names=args.scanners,
        start=args.start,
        end=args.end,
        out_dir=args.out,
    )


if __name__ == "__main__":
    main()
