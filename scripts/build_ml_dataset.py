#!/usr/bin/env python3
"""Build ML training dataset from historical OHLCV data.

Fetches price data for a universe of liquid stocks, computes features
locally via stockstats, and applies triple-barrier labels.

Usage:
    python scripts/build_ml_dataset.py
    python scripts/build_ml_dataset.py --stocks 100 --years 2
    python scripts/build_ml_dataset.py --ticker-file data/tickers_top50.txt
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from tradingagents.ml.feature_engineering import (
    FEATURE_COLUMNS,
    MIN_HISTORY_ROWS,
    apply_triple_barrier_labels,
    compute_features_bulk,
)
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Default universe: S&P 500 most liquid by volume (top ~200)
# Can be overridden via --ticker-file
DEFAULT_TICKERS = [
    # Mega-cap tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "AVGO",
    "ORCL",
    "CRM",
    "AMD",
    "INTC",
    "CSCO",
    "ADBE",
    "NFLX",
    "QCOM",
    "TXN",
    "AMAT",
    "MU",
    "LRCX",
    "KLAC",
    "MRVL",
    "SNPS",
    "CDNS",
    "PANW",
    "CRWD",
    "FTNT",
    "NOW",
    "UBER",
    "ABNB",
    # Financials
    "JPM",
    "BAC",
    "WFC",
    "GS",
    "MS",
    "C",
    "SCHW",
    "BLK",
    "AXP",
    "USB",
    "PNC",
    "TFC",
    "COF",
    "BK",
    "STT",
    "FITB",
    "HBAN",
    "RF",
    "CFG",
    "KEY",
    # Healthcare
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "DHR",
    "BMY",
    "AMGN",
    "GILD",
    "ISRG",
    "VRTX",
    "REGN",
    "MDT",
    "SYK",
    "BSX",
    "EW",
    "ZTS",
    # Consumer
    "WMT",
    "PG",
    "KO",
    "PEP",
    "COST",
    "MCD",
    "NKE",
    "SBUX",
    "TGT",
    "LOW",
    "HD",
    "TJX",
    "ROST",
    "DG",
    "DLTR",
    "EL",
    "CL",
    "KMB",
    "GIS",
    "K",
    # Energy
    "XOM",
    "CVX",
    "COP",
    "EOG",
    "SLB",
    "MPC",
    "PSX",
    "VLO",
    "OXY",
    "DVN",
    "HAL",
    "FANG",
    "HES",
    "BKR",
    "KMI",
    "WMB",
    "OKE",
    "ET",
    "TRGP",
    "LNG",
    # Industrials
    "CAT",
    "DE",
    "UNP",
    "UPS",
    "HON",
    "RTX",
    "BA",
    "LMT",
    "GD",
    "NOC",
    "GE",
    "MMM",
    "EMR",
    "ITW",
    "PH",
    "ROK",
    "ETN",
    "SWK",
    "CMI",
    "PCAR",
    # Materials & Utilities
    "LIN",
    "APD",
    "ECL",
    "SHW",
    "DD",
    "NEM",
    "FCX",
    "VMC",
    "MLM",
    "NUE",
    "NEE",
    "DUK",
    "SO",
    "D",
    "AEP",
    "EXC",
    "SRE",
    "XEL",
    "WEC",
    "ES",
    # REITs & Telecom
    "AMT",
    "PLD",
    "CCI",
    "EQIX",
    "SPG",
    "O",
    "PSA",
    "DLR",
    "WELL",
    "AVB",
    "T",
    "VZ",
    "TMUS",
    "CHTR",
    "CMCSA",
    # High-volatility / popular retail
    "COIN",
    "MARA",
    "RIOT",
    "PLTR",
    "SOFI",
    "HOOD",
    "RBLX",
    "SNAP",
    "PINS",
    "SQ",
    "SHOP",
    "SE",
    "ROKU",
    "DKNG",
    "PENN",
    "WYNN",
    "MGM",
    "LVS",
    "DASH",
    "TTD",
    # Biotech
    "MRNA",
    "BNTX",
    "BIIB",
    "SGEN",
    "ALNY",
    "BMRN",
    "EXAS",
    "DXCM",
    "HZNP",
    "INCY",
]

OUTPUT_DIR = Path("data/ml")


def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Fetch OHLCV data for a single ticker via yfinance."""
    from tradingagents.dataflows.y_finance import download_history

    df = download_history(
        ticker,
        start=start,
        end=end,
        multi_level_index=False,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return df

    df = df.reset_index()
    return df


def get_ticker_meta(ticker: str) -> tuple[float | None, str]:
    """Return (market_cap, sector_etf) in a single yfinance info call.

    Combining both lookups halves the number of HTTP requests during dataset build.
    sector_etf falls back to _DEFAULT_SECTOR_ETF when the sector is unknown.
    """
    from tradingagents.ml.feature_engineering import (
        YFINANCE_SECTOR_TO_ETF,
        _DEFAULT_SECTOR_ETF,
        SECTOR_ETF_MAP,
    )
    from tradingagents.dataflows.y_finance import get_ticker_info

    info = get_ticker_info(ticker)
    market_cap = info.get("marketCap")

    # Priority: static map (curated) > yfinance sector string > default
    if ticker in SECTOR_ETF_MAP:
        sector_etf = SECTOR_ETF_MAP[ticker]
    else:
        yf_sector = info.get("sector", "")
        sector_etf = YFINANCE_SECTOR_TO_ETF.get(yf_sector, _DEFAULT_SECTOR_ETF)

    return market_cap, sector_etf


def process_ticker(
    ticker: str,
    start: str,
    end: str,
    profit_target: float,
    stop_loss: float,
    max_holding_days: int,
    market_cap: float | None = None,
    binary: bool = True,
    market_ctx: "pd.DataFrame | None" = None,
    sector_series: "pd.Series | None" = None,
) -> "pd.DataFrame | None":
    """Process a single ticker: fetch data, compute features, apply labels."""
    try:
        ohlcv = fetch_ohlcv(ticker, start, end)
        if ohlcv.empty or len(ohlcv) < MIN_HISTORY_ROWS + max_holding_days:
            logger.debug(f"{ticker}: insufficient data ({len(ohlcv)} rows), skipping")
            return None

        # Compute features
        features = compute_features_bulk(ohlcv, market_cap=market_cap)
        if features.empty:
            logger.debug(f"{ticker}: feature computation failed, skipping")
            return None

        # Compute triple-barrier labels
        close = ohlcv.set_index("Date")["Close"] if "Date" in ohlcv.columns else ohlcv["Close"]
        if isinstance(close.index, pd.DatetimeIndex):
            pass
        else:
            close.index = pd.to_datetime(close.index)

        labels = apply_triple_barrier_labels(
            close,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_holding_days=max_holding_days,
            binary=binary,
        )

        # Align features and labels by date
        combined = features.join(labels, how="inner")

        # Drop rows with NaN features or labels
        combined = combined.dropna(subset=["label"])

        # Inject market regime features
        if market_ctx is not None and not market_ctx.empty:
            # Reindex market_ctx to the stock's trading calendar before joining
            ctx_aligned = market_ctx.reindex(combined.index, method="ffill")
            combined["spy_return_20d"] = ctx_aligned["spy_return_20d"].values
            combined["vix_level"] = ctx_aligned["vix_level"].values
            combined["vix_ma20_ratio"] = ctx_aligned["vix_ma20_ratio"].values
            combined["stock_vs_spy_20d"] = combined["return_20d"] - combined["spy_return_20d"]
        else:
            for col in ("spy_return_20d", "vix_level", "vix_ma20_ratio", "stock_vs_spy_20d"):
                combined[col] = np.nan

        if sector_series is not None and not sector_series.empty:
            sector_aligned = sector_series.reindex(combined.index, method="ffill")
            combined["sector_return_20d"] = sector_aligned.values
        else:
            combined["sector_return_20d"] = np.nan

        regime_cols = [
            "spy_return_20d",
            "vix_level",
            "vix_ma20_ratio",
            "stock_vs_spy_20d",
            "sector_return_20d",
        ]
        coverage = combined[regime_cols].notna().all(axis=1).mean()
        if coverage < 0.95 and market_ctx is not None:
            logger.warning(
                f"{ticker}: regime feature coverage only {coverage:.1%} — possible index mismatch"
            )

        combined = combined.dropna(subset=FEATURE_COLUMNS)

        if combined.empty:
            logger.debug(f"{ticker}: no valid rows after alignment, skipping")
            return None

        # Add metadata columns
        combined["ticker"] = ticker
        combined["date"] = combined.index

        if binary:
            logger.info(
                f"{ticker}: {len(combined)} samples "
                f"(WIN={int((combined['label'] == 1).sum())}, "
                f"NOT-WIN={int((combined['label'] == 0).sum())})"
            )
        else:
            logger.info(
                f"{ticker}: {len(combined)} samples "
                f"(WIN={int((combined['label'] == 1).sum())}, "
                f"LOSS={int((combined['label'] == -1).sum())}, "
                f"TIMEOUT={int((combined['label'] == 0).sum())})"
            )

        return combined

    except Exception as e:
        logger.warning(f"{ticker}: error processing — {e}")
        return None


def build_dataset(
    tickers: list[str],
    start: str = "2022-01-01",
    end: str = "2025-12-31",
    profit_target: float = 0.05,
    stop_loss: float = 0.03,
    max_holding_days: int = 20,
    binary: bool = True,
) -> pd.DataFrame:
    """Build the full training dataset across all tickers."""
    all_data = []
    total = len(tickers)

    logger.info(f"Building ML dataset: {total} tickers, {start} to {end}")
    logger.info(
        f"Triple-barrier: +{profit_target*100:.0f}% profit, "
        f"-{stop_loss*100:.0f}% stop, {max_holding_days}d timeout"
    )

    from tradingagents.ml.feature_engineering import (
        fetch_market_context,
        fetch_sector_context,
    )

    # Pre-fetch market-wide context once for all tickers
    logger.info("Fetching SPY/VIX market context...")
    market_ctx = fetch_market_context(start, end)
    logger.info(f"  Market context: {len(market_ctx)} trading days")

    # Resolve sector ETF + market cap for every ticker in one yfinance call each.
    # get_ticker_meta() uses static SECTOR_ETF_MAP first, then falls back to
    # yfinance info["sector"] — giving full universe coverage instead of ~100 tickers.
    logger.info("Fetching ticker metadata (market cap + sector)...")
    ticker_meta: dict[str, tuple] = {}
    for ticker in tickers:
        ticker_meta[ticker] = get_ticker_meta(ticker)
        time.sleep(0.05)  # rate limit courtesy

    # Pre-fetch sector ETF time-series for every unique ETF that was resolved
    unique_etfs = set(meta[1] for meta in ticker_meta.values())
    sector_data: dict[str, pd.Series] = {}
    for etf in unique_etfs:
        logger.info(f"  Fetching sector ETF {etf}...")
        sector_data[etf] = fetch_sector_context(etf, start, end)
        time.sleep(0.2)

    for i, ticker in enumerate(tickers):
        logger.info(f"[{i+1}/{total}] Processing {ticker}...")
        market_cap, sector_etf = ticker_meta.get(ticker, (None, "SPY"))
        result = process_ticker(
            ticker=ticker,
            start=start,
            end=end,
            profit_target=profit_target,
            stop_loss=stop_loss,
            max_holding_days=max_holding_days,
            market_cap=market_cap,
            binary=binary,
            market_ctx=market_ctx,
            sector_series=sector_data.get(sector_etf, pd.Series(dtype=float)),
        )
        if result is not None:
            all_data.append(result)

        # Brief pause between tickers to be polite to yfinance
        if (i + 1) % 50 == 0:
            logger.info(f"Progress: {i+1}/{total} tickers processed, pausing 2s...")
            time.sleep(2)

    if not all_data:
        logger.error("No data collected — check tickers and date range")
        return pd.DataFrame()

    dataset = pd.concat(all_data, ignore_index=True)

    logger.info(f"\n{'='*60}")
    logger.info(f"Dataset built: {len(dataset)} total samples from {len(all_data)} tickers")
    logger.info("Label distribution:")
    win_count = int((dataset["label"] == 1).sum())
    if binary:
        notwin_count = int((dataset["label"] == 0).sum())
        logger.info(f"  WIN  (1): {win_count:>7} ({win_count/len(dataset)*100:.1f}%)")
        logger.info(f"  NOT-WIN (0): {notwin_count:>7} ({notwin_count/len(dataset)*100:.1f}%)")
    else:
        loss_count = int((dataset["label"] == -1).sum())
        to_count = int((dataset["label"] == 0).sum())
        logger.info(f"  WIN (+1): {win_count:>7} ({win_count/len(dataset)*100:.1f}%)")
        logger.info(f"  LOSS (-1): {loss_count:>7} ({loss_count/len(dataset)*100:.1f}%)")
        logger.info(f"  TIMEOUT (0): {to_count:>7} ({to_count/len(dataset)*100:.1f}%)")
    logger.info(f"Features: {len(FEATURE_COLUMNS)}")
    logger.info(f"{'='*60}")

    return dataset


def main():
    parser = argparse.ArgumentParser(description="Build ML training dataset")
    parser.add_argument(
        "--stocks", type=int, default=None, help="Limit to N stocks from default universe"
    )
    parser.add_argument(
        "--ticker-file", type=str, default=None, help="File with tickers (one per line)"
    )
    parser.add_argument("--start", type=str, default="2022-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-12-31", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--profit-target", type=float, default=0.05, help="Profit target fraction (default: 0.05)"
    )
    parser.add_argument(
        "--stop-loss", type=float, default=0.03, help="Stop loss fraction (default: 0.03)"
    )
    parser.add_argument(
        "--holding-days", type=int, default=20, help="Max holding days (default: 20)"
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        default=True,
        help="Use binary labels WIN=1/NOT-WIN=0 (default: True)",
    )
    parser.add_argument(
        "--no-binary",
        dest="binary",
        action="store_false",
        help="Use 3-class labels WIN/TIMEOUT/LOSS",
    )
    parser.add_argument("--output", type=str, default=None, help="Output parquet path")
    args = parser.parse_args()

    # Determine ticker list
    if args.ticker_file:
        with open(args.ticker_file) as f:
            tickers = [
                line.strip().upper() for line in f if line.strip() and not line.startswith("#")
            ]
        logger.info(f"Loaded {len(tickers)} tickers from {args.ticker_file}")
    else:
        tickers = DEFAULT_TICKERS
        if args.stocks:
            tickers = tickers[: args.stocks]

    # Build dataset
    dataset = build_dataset(
        tickers=tickers,
        start=args.start,
        end=args.end,
        profit_target=args.profit_target,
        stop_loss=args.stop_loss,
        max_holding_days=args.holding_days,
        binary=args.binary,
    )

    if dataset.empty:
        logger.error("Empty dataset — aborting")
        sys.exit(1)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = args.output or str(OUTPUT_DIR / "training_dataset.parquet")
    dataset.to_parquet(output_path, index=False)
    logger.info(f"Saved dataset to {output_path} ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
