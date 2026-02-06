"""Common utilities for discovery scanners."""
import re
import logging
from typing import List, Set, Optional

logger = logging.getLogger(__name__)


def get_common_stopwords() -> Set[str]:
    """Get common words that look like tickers but aren't.

    Returns:
        Set of uppercase words to filter out from ticker extraction
    """
    return {
        # Common words
        'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL', 'CAN',
        'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'DAY', 'WHO', 'HAS', 'HAD',
        'NEW', 'NOW', 'GET', 'GOT', 'PUT', 'SET', 'RUN', 'TOP', 'BIG',
        # Financial terms
        'CEO', 'CFO', 'CTO', 'COO', 'USD', 'USA', 'SEC', 'IPO', 'ETF',
        'NYSE', 'NASDAQ', 'WSB', 'DD', 'YOLO', 'FD', 'ATH', 'ATL', 'GDP',
        'STOCK', 'STOCKS', 'MARKET', 'NEWS', 'PRICE', 'TRADE', 'SALES',
        # Time
        'JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP',
        'OCT', 'NOV', 'DEC', 'MON', 'TUE', 'WED', 'THU', 'FRI', 'SAT', 'SUN',
    }


def extract_tickers_from_text(
    text: str,
    stop_words: Optional[Set[str]] = None,
    max_text_length: int = 100_000
) -> List[str]:
    """Extract valid ticker symbols from text.

    Uses regex patterns to find potential tickers ($TICKER or standalone TICKER),
    filters out common stopwords, and returns deduplicated list.

    Args:
        text: Text to extract tickers from
        stop_words: Custom stopwords to filter (uses defaults if None)
        max_text_length: Maximum text length to process (prevents ReDoS)

    Returns:
        List of unique ticker symbols found in text

    Example:
        >>> extract_tickers_from_text("I like $AAPL and MSFT stocks")
        ['AAPL', 'MSFT']
    """
    # Truncate oversized text to prevent ReDoS
    if len(text) > max_text_length:
        logger.warning(
            f"Truncating oversized text from {len(text)} to {max_text_length} chars"
        )
        text = text[:max_text_length]

    # Match: $TICKER or standalone TICKER (2-5 uppercase letters)
    ticker_pattern = r'\b([A-Z]{2,5})\b|\$([A-Z]{2,5})'
    matches = re.findall(ticker_pattern, text)

    # Flatten tuples and deduplicate
    tickers = list(set([t[0] or t[1] for t in matches if t[0] or t[1]]))

    # Filter stopwords
    stop_words = stop_words or get_common_stopwords()
    filtered_tickers = [t for t in tickers if t not in stop_words]

    return filtered_tickers


def validate_ticker_format(ticker: str) -> bool:
    """Validate ticker symbol format.

    Args:
        ticker: Ticker symbol to validate

    Returns:
        True if ticker matches expected format (2-5 uppercase letters)
    """
    if not ticker or not isinstance(ticker, str):
        return False

    return bool(re.match(r'^[A-Z]{2,5}$', ticker.strip().upper()))


def validate_candidate_structure(candidate: dict) -> bool:
    """Validate candidate dictionary has required keys.

    Args:
        candidate: Candidate dictionary to validate

    Returns:
        True if candidate has all required keys with valid types
    """
    required_keys = {'ticker', 'source', 'context', 'priority'}

    if not isinstance(candidate, dict):
        return False

    if not required_keys.issubset(candidate.keys()):
        missing = required_keys - set(candidate.keys())
        logger.warning(f"Candidate missing required keys: {missing}")
        return False

    # Validate ticker format
    if not validate_ticker_format(candidate.get('ticker', '')):
        logger.warning(f"Invalid ticker format: {candidate.get('ticker')}")
        return False

    # Validate priority is string
    if not isinstance(candidate.get('priority'), str):
        logger.warning(f"Invalid priority type: {type(candidate.get('priority'))}")
        return False

    return True
