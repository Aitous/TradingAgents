import re
from typing import Any, List


def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    """
    Format a list of rows into a Markdown table.

    Args:
        headers: List of column headers
        rows: List of rows, where each row is a list of values

    Returns:
        Formatted Markdown table string
    """
    if not headers:
        return ""

    # Create header row
    header_str = "| " + " | ".join(headers) + " |\n"

    # Create separator row
    separator_str = "| " + " | ".join(["---"] * len(headers)) + " |\n"

    # Create data rows
    body_str = ""
    for row in rows:
        # Convert all values to string and handle None
        formatted_row = [str(val) if val is not None else "" for val in row]
        body_str += "| " + " | ".join(formatted_row) + " |\n"

    return header_str + separator_str + body_str


def parse_market_cap(market_cap_text: str) -> float:
    """Parse market cap from string format (e.g., '1.23B', '456M')."""
    if not market_cap_text or market_cap_text == "-":
        return 0.0

    market_cap_text = str(market_cap_text).upper().strip()

    # Extract number and multiplier
    match = re.match(r"([0-9.]+)([BMK])?", market_cap_text)
    if not match:
        try:
            return float(market_cap_text)
        except ValueError:
            return 0.0

    number = float(match.group(1))
    multiplier = match.group(2)

    if multiplier == "B":
        return number * 1_000_000_000
    elif multiplier == "M":
        return number * 1_000_000
    elif multiplier == "K":
        return number * 1_000
    else:
        return number


def format_market_cap(market_cap: float) -> str:
    """Format market cap for display (e.g. 1.5B, 200M)."""
    if not isinstance(market_cap, (int, float)):
        return str(market_cap)

    if market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.2f}M"
    else:
        return f"${market_cap:,.0f}"
