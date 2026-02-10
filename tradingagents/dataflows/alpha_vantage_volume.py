"""
Unusual Volume Detection using yfinance
Identifies stocks with unusual volume but minimal price movement (accumulation signal)
"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Annotated, Dict, List, Optional, Union

import pandas as pd
from tradingagents.dataflows.y_finance import _get_ticker_universe, get_ticker_history
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def _get_cache_path(ticker_universe: Union[str, List[str]]) -> Path:
    """
    Get the cache file path for unusual volume raw data.

    Args:
        ticker_universe: Universe identifier

    Returns:
        Path to cache file
    """
    # Get cache directory
    current_file = Path(__file__)
    cache_dir = current_file.parent / "data_cache"
    cache_dir.mkdir(exist_ok=True)

    # Create cache key from universe only (thresholds are applied later)
    if isinstance(ticker_universe, str):
        universe_key = ticker_universe
    else:
        # Stable hash for custom lists so different lists don't collide
        clean_tickers = [t.upper().strip() for t in ticker_universe if isinstance(t, str)]
        hash_suffix = hashlib.md5(",".join(sorted(clean_tickers)).encode()).hexdigest()[:8]
        universe_key = f"custom_{hash_suffix}"
    cache_key = f"unusual_volume_raw_{universe_key}".replace(".", "_")

    return cache_dir / f"{cache_key}.json"


def _load_cache(cache_path: Path) -> Optional[Dict]:
    """
    Load cached unusual volume raw data if it exists and is from today.

    Args:
        cache_path: Path to cache file

    Returns:
        Cached results dict if valid, None otherwise
    """
    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r") as f:
            cache_data = json.load(f)

        # Check if cache is from today
        cache_date = cache_data.get("date")
        today = datetime.now().strftime("%Y-%m-%d")
        has_raw_data = bool(cache_data.get("raw_data"))

        if cache_date == today and has_raw_data:
            return cache_data
        else:
            # Cache is stale, return None to trigger recompute
            return None

    except Exception:
        # If cache is corrupted, return None to trigger recompute
        return None


def _save_cache(cache_path: Path, raw_data: Dict[str, List[Dict]], date: str):
    """
    Save unusual volume raw data to cache.

    Args:
        cache_path: Path to cache file
        raw_data: Raw ticker data to cache
        date: Date string (YYYY-MM-DD)
    """
    try:
        cache_data = {"date": date, "raw_data": raw_data, "timestamp": datetime.now().isoformat()}

        with open(cache_path, "w") as f:
            json.dump(cache_data, f, indent=2)

    except Exception as e:
        # If caching fails, just continue without cache
        logger.warning(f"Could not save cache: {e}")


def _history_to_records(hist: pd.DataFrame) -> List[Dict[str, Union[str, float, int]]]:
    """Convert a yfinance history DataFrame to a cache-friendly list of dicts."""
    # Include Open price for intraday direction analysis (accumulation vs distribution)
    cols_to_use = ["Close", "Volume"]
    if "Open" in hist.columns:
        cols_to_use = ["Open", "Close", "Volume"]

    hist_for_cache = hist[cols_to_use].copy()
    hist_for_cache = hist_for_cache.reset_index()
    date_col = "Date" if "Date" in hist_for_cache.columns else hist_for_cache.columns[0]
    hist_for_cache.rename(columns={date_col: "Date"}, inplace=True)
    hist_for_cache["Date"] = pd.to_datetime(hist_for_cache["Date"]).dt.strftime("%Y-%m-%d")

    final_cols = ["Date"] + cols_to_use
    hist_for_cache = hist_for_cache[final_cols]
    return hist_for_cache.to_dict(orient="records")


def _records_to_dataframe(history_records: List[Dict[str, Union[str, float, int]]]) -> pd.DataFrame:
    """Convert cached history records back to a DataFrame for calculation."""
    hist_df = pd.DataFrame(history_records)
    if hist_df.empty:
        return hist_df
    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
    hist_df = hist_df.sort_values("Date")
    return hist_df


def get_cached_average_volume(
    symbol: str,
    lookback_days: int = 20,
    curr_date: Optional[str] = None,
    cache_key: str = "default",
    fallback_download: bool = True,
) -> Dict[str, Union[str, float, int, None]]:
    """Get average volume using cached unusual-volume data, with optional fallback download."""
    symbol = symbol.upper()
    cache_path = _get_cache_path(cache_key)
    cache_date = None
    history_records = None

    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            cache_date = cache_data.get("date")
            raw_data = cache_data.get("raw_data") or {}
            history_records = raw_data.get(symbol)
        except Exception:
            history_records = None

    source = "cache"
    if not history_records and fallback_download:
        history_records = _download_ticker_history(
            symbol, history_period_days=max(90, lookback_days * 2)
        )
        source = "download"

    if not history_records:
        return {
            "symbol": symbol,
            "average_volume": None,
            "latest_volume": None,
            "lookback_days": lookback_days,
            "source": source,
            "cache_date": cache_date,
            "error": "No volume data found",
        }

    hist_df = _records_to_dataframe(history_records)
    if hist_df.empty or "Volume" not in hist_df.columns:
        return {
            "symbol": symbol,
            "average_volume": None,
            "latest_volume": None,
            "lookback_days": lookback_days,
            "source": source,
            "cache_date": cache_date,
            "error": "No volume data found",
        }

    if curr_date:
        curr_dt = pd.to_datetime(curr_date)
        hist_df = hist_df[hist_df["Date"] <= curr_dt]

    recent = hist_df.tail(lookback_days)
    if recent.empty:
        return {
            "symbol": symbol,
            "average_volume": None,
            "latest_volume": None,
            "lookback_days": lookback_days,
            "source": source,
            "cache_date": cache_date,
            "error": "No recent volume data found",
        }

    average_volume = float(recent["Volume"].mean())
    latest_volume = float(recent["Volume"].iloc[-1])

    return {
        "symbol": symbol,
        "average_volume": average_volume,
        "latest_volume": latest_volume,
        "lookback_days": lookback_days,
        "source": source,
        "cache_date": cache_date,
    }


def get_cached_average_volume_batch(
    symbols: List[str],
    lookback_days: int = 20,
    curr_date: Optional[str] = None,
    cache_key: str = "default",
    fallback_download: bool = True,
) -> Dict[str, Dict[str, Union[str, float, int, None]]]:
    """Get average volumes for multiple tickers using the cache once."""
    cache_path = _get_cache_path(cache_key)
    cache_date = None
    raw_data = {}

    if cache_path.exists():
        try:
            with open(cache_path, "r") as f:
                cache_data = json.load(f)
            cache_date = cache_data.get("date")
            raw_data = cache_data.get("raw_data") or {}
        except Exception:
            raw_data = {}

    results: Dict[str, Dict[str, Union[str, float, int, None]]] = {}
    symbols_upper = [s.upper() for s in symbols if isinstance(s, str)]

    def compute_from_records(symbol: str, history_records: List[Dict[str, Union[str, float, int]]]):
        hist_df = _records_to_dataframe(history_records)
        if hist_df.empty or "Volume" not in hist_df.columns:
            return None, None, "No volume data found"
        if curr_date:
            curr_dt = pd.to_datetime(curr_date)
            hist_df = hist_df[hist_df["Date"] <= curr_dt]
        recent = hist_df.tail(lookback_days)
        if recent.empty:
            return None, None, "No recent volume data found"
        avg_volume = float(recent["Volume"].mean())
        latest_volume = float(recent["Volume"].iloc[-1])
        return avg_volume, latest_volume, None

    missing = []
    for symbol in symbols_upper:
        history_records = raw_data.get(symbol)
        if history_records:
            avg_volume, latest_volume, error = compute_from_records(symbol, history_records)
            results[symbol] = {
                "symbol": symbol,
                "average_volume": avg_volume,
                "latest_volume": latest_volume,
                "lookback_days": lookback_days,
                "source": "cache",
                "cache_date": cache_date,
                "error": error,
            }
        else:
            missing.append(symbol)

    if fallback_download and missing:
        for symbol in missing:
            history_records = _download_ticker_history(
                symbol, history_period_days=max(90, lookback_days * 2)
            )
            if history_records:
                avg_volume, latest_volume, error = compute_from_records(symbol, history_records)
                results[symbol] = {
                    "symbol": symbol,
                    "average_volume": avg_volume,
                    "latest_volume": latest_volume,
                    "lookback_days": lookback_days,
                    "source": "download",
                    "cache_date": cache_date,
                    "error": error,
                }
            else:
                results[symbol] = {
                    "symbol": symbol,
                    "average_volume": None,
                    "latest_volume": None,
                    "lookback_days": lookback_days,
                    "source": "download",
                    "cache_date": cache_date,
                    "error": "No volume data found",
                }

    return results


def _evaluate_unusual_volume_from_history(
    ticker: str,
    history_records: List[Dict[str, Union[str, float, int]]],
    min_volume_multiple: float,
    max_price_change: float,
    lookback_days: int = 30,
) -> Optional[Dict]:
    """
    Evaluate a ticker's cached history for unusual volume patterns.

    Now includes DIRECTION ANALYSIS to distinguish:
    - Accumulation (high volume + price holds/rises) = BULLISH - keep
    - Distribution (high volume + price drops) = BEARISH - skip

    Args:
        ticker: Stock ticker symbol
        history_records: Cached price/volume history records
        min_volume_multiple: Minimum volume multiple vs average
        max_price_change: Maximum absolute price change percentage
        lookback_days: Days to look back for average volume calculation

    Returns:
        Dict with ticker data if unusual volume detected, None otherwise
    """
    try:
        hist = _records_to_dataframe(history_records)
        if hist.empty or len(hist) < lookback_days + 1:
            return None

        current_data = hist.iloc[-1]
        current_volume = current_data["Volume"]
        current_price = current_data["Close"]

        avg_volume = hist["Volume"].iloc[-(lookback_days + 1) : -1].mean()
        if pd.isna(avg_volume) or avg_volume <= 0:
            return None

        volume_ratio = current_volume / avg_volume

        price_start = hist["Close"].iloc[-(lookback_days + 1)]
        price_end = current_price
        price_change_pct = ((price_end - price_start) / price_start) * 100

        # === DIRECTION ANALYSIS (NEW) ===
        # Check intraday direction to distinguish accumulation from distribution
        intraday_change_pct = 0.0
        direction = "neutral"

        if "Open" in current_data and pd.notna(current_data["Open"]):
            open_price = current_data["Open"]
            if open_price > 0:
                intraday_change_pct = ((current_price - open_price) / open_price) * 100

                # Classify direction based on intraday movement
                if intraday_change_pct > 0.5:
                    direction = "bullish"  # Closed higher than open
                elif intraday_change_pct < -1.5:
                    direction = "bearish"  # Closed significantly lower than open
                else:
                    direction = "neutral"  # Flat intraday

        # === DISTRIBUTION FILTER (NEW) ===
        # Skip if high volume + bearish direction = likely distribution (selling)
        if volume_ratio >= min_volume_multiple and direction == "bearish":
            # This is likely DISTRIBUTION - smart money selling, not accumulation
            # Return None to filter it out
            return None

        # Filter: High volume multiple AND low price change (accumulation signal)
        if volume_ratio >= min_volume_multiple and abs(price_change_pct) < max_price_change:
            # Determine signal type with direction context
            if direction == "bullish" and abs(price_change_pct) < 3.0:
                signal = "strong_accumulation"  # Best signal: high volume, rising intraday
            elif abs(price_change_pct) < 2.0:
                signal = "accumulation"
            elif abs(price_change_pct) < 5.0:
                signal = "moderate_activity"
            else:
                signal = "building_momentum"

            return {
                "ticker": ticker.upper(),
                "volume": int(current_volume),
                "price": round(float(current_price), 2),
                "price_change_pct": round(price_change_pct, 2),
                "intraday_change_pct": round(intraday_change_pct, 2),
                "direction": direction,
                "volume_ratio": round(volume_ratio, 2),
                "avg_volume": int(avg_volume),
                "signal": signal,
            }

        return None

    except Exception:
        return None


def _download_ticker_history(
    ticker: str, history_period_days: int = 90
) -> Optional[List[Dict[str, Union[str, float, int]]]]:
    """
    Download raw history for a ticker and return cache-friendly records.

    Args:
        ticker: Stock ticker symbol
        history_period_days: Total days of history to download (default: 90)

    Returns:
        List of history records or None if insufficient data
    """
    try:
        hist = get_ticker_history(ticker, period=f"{history_period_days}d")

        if hist.empty:
            return None

        if hist.index.tz is not None:
            hist.index = hist.index.tz_localize(None)

        return _history_to_records(hist)
    except Exception:
        return None


def download_volume_data(
    tickers: List[str],
    history_period_days: int = 90,
    use_cache: bool = True,
    cache_key: str = "default",
) -> Dict[str, List[Dict[str, Union[str, float, int]]]]:
    """
    Download or load cached volume data for a list of tickers.

    This is the main data fetching function that:
    1. If use_cache=True: Check if cache exists and is fresh (from today)
    2. If cache is stale or use_cache=False: Download fresh data
    3. Always save downloaded data to cache (for next time)

    Args:
        tickers: List of ticker symbols to download
        history_period_days: Total days of history to download (default: 90)
        use_cache: Whether to USE existing cache (fresh data always gets saved)
        cache_key: Identifier for cache file (default: "default")

    Returns:
        Dict mapping ticker symbols to their history records
    """
    today = datetime.now().strftime("%Y-%m-%d")

    # Get cache path (we always need it for saving)
    cache_path = _get_cache_path(cache_key)

    # Try to load cache only if use_cache=True
    if use_cache:
        cached_data = _load_cache(cache_path)

        # Check if cache is fresh (from today)
        if cached_data and cached_data.get("date") == today:
            logger.info(f"Using cached volume data from {cached_data['date']}")
            return cached_data["raw_data"]
        elif cached_data:
            logger.info(f"Cache is stale (from {cached_data.get('date')}), re-downloading...")
    else:
        logger.info("Skipping cache (use_cache=False), forcing fresh download...")

    # Download fresh data
    logger.info(f"Downloading {history_period_days} days of volume data for {len(tickers)} tickers...")
    raw_data = {}

    with ThreadPoolExecutor(max_workers=15) as executor:
        futures = {
            executor.submit(_download_ticker_history, ticker, history_period_days): ticker
            for ticker in tickers
        }

        completed = 0
        for future in as_completed(futures):
            completed += 1
            if completed % 50 == 0:
                logger.info(f"Progress: {completed}/{len(tickers)} tickers downloaded...")

            ticker_symbol = futures[future].upper()
            history_records = future.result()
            if history_records:
                raw_data[ticker_symbol] = history_records

    # Always save fresh data to cache (so it's available next time)
    if cache_path and raw_data:
        logger.info(f"Saving {len(raw_data)} tickers to cache...")
        _save_cache(cache_path, raw_data, today)

    return raw_data


def get_unusual_volume(
    date: Annotated[str, "Analysis date in yyyy-mm-dd format"] = None,
    min_volume_multiple: Annotated[float, "Minimum volume multiple vs average"] = 3.0,
    max_price_change: Annotated[float, "Maximum price change percentage"] = 5.0,
    top_n: Annotated[int, "Number of top results to return"] = 20,
    tickers: Annotated[Optional[List[str]], "Custom ticker list or None to use config file"] = None,
    max_tickers_to_scan: Annotated[int, "Maximum number of tickers to scan"] = 3000,
    use_cache: Annotated[bool, "Use cached raw data when available"] = True,
    return_structured: Annotated[bool, "Return list of dicts instead of markdown"] = False,
):
    """
    Find stocks with unusual volume but minimal price movement.

    This is a strong accumulation signal - smart money buying before a breakout.
    Scans all major US stocks (3000+ including S&P 500, NASDAQ, small caps, meme stocks) using yfinance.

    Args:
        date: Analysis date in yyyy-mm-dd format (for reporting only)
        min_volume_multiple: Minimum volume multiple vs 30-day average (e.g., 3.0 = 3x average volume)
        max_price_change: Maximum absolute price change percentage
        top_n: Number of top results to return
        tickers: Custom list of ticker symbols, or None to load from config file
        max_tickers_to_scan: Maximum number of tickers to scan (default: 3000, scans all)
        use_cache: Whether to reuse/save cached raw data
        return_structured: If True, returns list of candidate dicts instead of markdown

    Returns:
        If return_structured=True: list of candidate dicts with ticker, volume_ratio, signal, etc.
        If return_structured=False: Formatted markdown report
    """
    try:
        lookback_days = 30
        today = datetime.now().strftime("%Y-%m-%d")
        analysis_date = date or today

        ticker_list = _get_ticker_universe(tickers=tickers, max_tickers=max_tickers_to_scan)
        ticker_count = len(ticker_list) if ticker_list else 0
        if not ticker_list:
            return "Error: No tickers found"

        # Use the new helper function to download/load data
        # Create cache key from ticker list or "default"
        if isinstance(tickers, list):
            import hashlib

            cache_key = "custom_" + hashlib.md5(",".join(sorted(tickers)).encode()).hexdigest()[:8]
        else:
            cache_key = "default"

        raw_data = download_volume_data(
            tickers=ticker_list, history_period_days=90, use_cache=use_cache, cache_key=cache_key
        )

        if not raw_data:
            return "Error: Unable to retrieve volume data for requested tickers"

        unusual_candidates = []
        for ticker in ticker_list:
            history_records = raw_data.get(ticker.upper())
            if not history_records:
                continue

            candidate = _evaluate_unusual_volume_from_history(
                ticker,
                history_records,
                min_volume_multiple,
                max_price_change,
                lookback_days=lookback_days,
            )
            if candidate:
                unusual_candidates.append(candidate)

        if not unusual_candidates:
            if return_structured:
                return []
            return f"No stocks found with unusual volume patterns matching criteria\n\nScanned {len(ticker_list)} tickers."

        # Sort by volume ratio (highest first)
        sorted_candidates = sorted(
            unusual_candidates, key=lambda x: (x.get("volume_ratio", 0), x["volume"]), reverse=True
        )

        # Take top N for display
        sorted_candidates = sorted_candidates[:top_n]

        # Return structured data if requested
        if return_structured:
            return sorted_candidates

        # Format output
        report = f"# Unusual Volume Detected - {analysis_date}\n\n"
        report += "**Criteria**: \n"
        report += f"- Price Change: <{max_price_change}% (accumulation pattern)\n"
        report += f"- Volume Multiple: Current volume ≥ {min_volume_multiple}x 30-day average\n"
        report += f"- Tickers Scanned: {ticker_count}\n\n"
        report += f"**Found**: {len(sorted_candidates)} stocks with unusual activity\n\n"
        report += "## Top Unusual Volume Candidates\n\n"
        report += (
            "| Ticker | Price | Volume | Avg Volume | Volume Ratio | Price Change % | Signal |\n"
        )
        report += (
            "|--------|-------|--------|------------|--------------|----------------|--------|\n"
        )

        for candidate in sorted_candidates:
            volume_ratio_str = (
                f"{candidate.get('volume_ratio', 'N/A')}x"
                if candidate.get("volume_ratio")
                else "N/A"
            )
            avg_vol_str = (
                f"{candidate.get('avg_volume', 0):,}" if candidate.get("avg_volume") else "N/A"
            )
            report += f"| {candidate['ticker']} | "
            report += f"${candidate['price']:.2f} | "
            report += f"{candidate['volume']:,} | "
            report += f"{avg_vol_str} | "
            report += f"{volume_ratio_str} | "
            report += f"{candidate['price_change_pct']:.2f}% | "
            report += f"{candidate['signal']} |\n"

        report += "\n\n## Signal Definitions\n\n"
        report += "- **strong_accumulation**: High volume + bullish intraday direction - Strongest buy signal\n"
        report += "- **accumulation**: High volume, minimal price change (<2%) - Smart money building position\n"
        report += (
            "- **moderate_activity**: Elevated volume with 2-5% price change - Early momentum\n"
        )
        report += "- **building_momentum**: High volume with moderate price change - Conviction building\n"
        report += "\n**Note**: Distribution patterns (high volume + bearish direction) are automatically filtered out.\n"

        return report

    except Exception as e:
        if return_structured:
            return []
        return f"Unexpected error in unusual volume detection: {str(e)}"


def get_alpha_vantage_unusual_volume(
    date: str = None,
    min_volume_multiple: float = 3.0,
    max_price_change: float = 5.0,
    top_n: int = 20,
    tickers: Optional[List[str]] = None,
    max_tickers_to_scan: int = 3000,
    use_cache: bool = True,
) -> str:
    """Alias for get_unusual_volume to match registry naming convention"""
    return get_unusual_volume(
        date, min_volume_multiple, max_price_change, top_n, tickers, max_tickers_to_scan, use_cache
    )
