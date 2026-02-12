"""
Ticker Matching Utility

Maps company names to ticker symbols using fuzzy string matching
with the ticker universe CSV.

Usage:
    from tradingagents.dataflows.discovery.ticker_matcher import match_company_to_ticker

    ticker = match_company_to_ticker("Apple Inc")
    # Returns: "AAPL"
"""

import csv
import re
from pathlib import Path
from typing import Dict, Optional

from rapidfuzz import fuzz, process

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)

# Global cache
_TICKER_UNIVERSE: Optional[Dict[str, str]] = None  # ticker -> name
_NAME_TO_TICKER: Optional[Dict[str, str]] = None  # normalized_name -> ticker
_MATCH_CACHE: Dict[str, Optional[str]] = {}  # company_name -> ticker


def _normalize_company_name(name: str) -> str:
    """
    Normalize company name for matching.

    Removes common suffixes, punctuation, and standardizes format.
    """
    if not name:
        return ""

    # Convert to uppercase
    name = name.upper()

    # Remove common suffixes
    suffixes = [
        r"\s+INC\.?",
        r"\s+INCORPORATED",
        r"\s+CORP\.?",
        r"\s+CORPORATION",
        r"\s+LTD\.?",
        r"\s+LIMITED",
        r"\s+LLC",
        r"\s+L\.?L\.?C\.?",
        r"\s+PLC",
        r"\s+CO\.?",
        r"\s+COMPANY",
        r"\s+CLASS [A-Z]",
        r"\s+COMMON STOCK",
        r"\s+ORDINARY SHARES?",
        r"\s+-\s+.*$",  # Remove everything after dash
        r"\s+\(.*?\)",  # Remove parenthetical
    ]

    for suffix in suffixes:
        name = re.sub(suffix, "", name, flags=re.IGNORECASE)

    # Remove punctuation except spaces
    name = re.sub(r"[^\w\s]", "", name)

    # Normalize whitespace
    name = " ".join(name.split())

    return name.strip()


def load_ticker_universe(force_reload: bool = False) -> Dict[str, str]:
    """
    Load ticker universe from CSV.

    Args:
        force_reload: Force reload even if already loaded

    Returns:
        Dict mapping ticker -> company name
    """
    global _TICKER_UNIVERSE, _NAME_TO_TICKER

    if _TICKER_UNIVERSE is not None and not force_reload:
        return _TICKER_UNIVERSE

    # Find CSV file
    project_root = Path(__file__).parent.parent.parent.parent
    csv_path = project_root / "data" / "ticker_universe.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Ticker universe not found: {csv_path}")

    ticker_universe = {}
    name_to_ticker = {}

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ticker = row["ticker"]
            name = row["name"]

            # Store ticker -> name mapping
            ticker_universe[ticker] = name

            # Build reverse index (normalized name -> ticker)
            normalized = _normalize_company_name(name)
            if normalized:
                # If multiple tickers have same normalized name, prefer common stocks
                if normalized not in name_to_ticker:
                    name_to_ticker[normalized] = ticker
                elif (
                    "COMMON" in name.upper()
                    and "COMMON" not in ticker_universe.get(name_to_ticker[normalized], "").upper()
                ):
                    # Prefer common stock over other securities
                    name_to_ticker[normalized] = ticker

    _TICKER_UNIVERSE = ticker_universe
    _NAME_TO_TICKER = name_to_ticker

    logger.info(f"Loaded {len(ticker_universe)} tickers from universe")

    return ticker_universe


def match_company_to_ticker(
    company_name: str,
    min_confidence: float = 80.0,
    use_cache: bool = True,
) -> Optional[str]:
    """
    Match a company name to a ticker symbol using fuzzy matching.

    Args:
        company_name: Company name from 13F filing
        min_confidence: Minimum fuzzy match score (0-100)
        use_cache: Use cached results

    Returns:
        Ticker symbol or None if no good match found

    Examples:
        >>> match_company_to_ticker("Apple Inc")
        'AAPL'
        >>> match_company_to_ticker("MICROSOFT CORP")
        'MSFT'
        >>> match_company_to_ticker("Berkshire Hathaway Inc")
        'BRK.B'
    """
    if not company_name:
        return None

    # Check cache
    if use_cache and company_name in _MATCH_CACHE:
        return _MATCH_CACHE[company_name]

    # Ensure universe is loaded
    if _TICKER_UNIVERSE is None or _NAME_TO_TICKER is None:
        load_ticker_universe()

    # Normalize input
    normalized_input = _normalize_company_name(company_name)

    if not normalized_input:
        return None

    # Try exact match first
    if normalized_input in _NAME_TO_TICKER:
        result = _NAME_TO_TICKER[normalized_input]
        _MATCH_CACHE[company_name] = result
        return result

    # Fuzzy match against all normalized names
    choices = list(_NAME_TO_TICKER.keys())

    # Use token_sort_ratio for best results with company names
    match_result = process.extractOne(
        normalized_input, choices, scorer=fuzz.token_sort_ratio, score_cutoff=min_confidence
    )

    if match_result:
        matched_name, score, _ = match_result
        ticker = _NAME_TO_TICKER[matched_name]

        # Log match for debugging
        if score < 95:
            logger.info(f"Fuzzy match: '{company_name}' -> {ticker} (score: {score:.1f})")

        _MATCH_CACHE[company_name] = ticker
        return ticker

    # No match found
    logger.info(f"No ticker match for: '{company_name}'")
    _MATCH_CACHE[company_name] = None
    return None


def get_match_confidence(company_name: str, ticker: str) -> float:
    """
    Get confidence score for a company name -> ticker match.

    Args:
        company_name: Company name
        ticker: Ticker symbol

    Returns:
        Confidence score (0-100)
    """
    if _TICKER_UNIVERSE is None:
        load_ticker_universe()

    if ticker not in _TICKER_UNIVERSE:
        return 0.0

    ticker_name = _TICKER_UNIVERSE[ticker]

    # Normalize both names
    norm_input = _normalize_company_name(company_name)
    norm_ticker = _normalize_company_name(ticker_name)

    # Calculate similarity
    return fuzz.token_sort_ratio(norm_input, norm_ticker)


def clear_cache():
    """Clear the match cache."""
    global _MATCH_CACHE
    _MATCH_CACHE = {}
