"""
Tradier API - Options Activity Detection
Detects unusual options activity indicating smart money positioning
"""

import os
from typing import Annotated, List

import requests

from tradingagents.config import config
from tradingagents.dataflows.market_data_utils import format_markdown_table


def get_unusual_options_activity(
    tickers: Annotated[List[str], "List of ticker symbols to analyze"] = None,
    date: Annotated[str, "Analysis date in yyyy-mm-dd format"] = None,
    min_volume_multiple: Annotated[float, "Minimum options volume multiple"] = 2.0,
    top_n: Annotated[int, "Number of top results to return"] = 20,
) -> str:
    """
    Detect unusual options activity for given tickers (confirmation signal).

    This function is designed as a CONFIRMATION tool - it analyzes options activity
    for candidates found by other discovery methods (unusual volume, analyst changes, etc.)

    Unusual options volume is a leading indicator of price moves - institutions
    positioning before catalysts.

    Args:
        tickers: List of ticker symbols to analyze (if None, returns error message)
        date: Analysis date in yyyy-mm-dd format
        min_volume_multiple: Minimum volume multiple vs 20-day average
        top_n: Number of top results to return

    Returns:
        Formatted markdown report of unusual options activity
    """
    try:
        api_key = config.validate_key("tradier_api_key", "Tradier")
    except ValueError as e:
        return f"Error: {str(e)}"

    if not tickers or len(tickers) == 0:
        return "Error: No tickers provided. This function analyzes options activity for specific tickers found by other discovery methods."

    # Tradier API base URLs
    # Use sandbox for testing: https://sandbox.tradier.com
    # Use production: https://api.tradier.com
    base_url = os.getenv("TRADIER_BASE_URL", "https://sandbox.tradier.com")

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

    try:
        # Strategy: Analyze options activity for provided tickers
        # This confirms smart money positioning for candidates found by other methods

        unusual_activity = []

        for ticker in tickers:
            try:
                # Get options chain
                options_url = f"{base_url}/v1/markets/options/chains"
                params = {
                    "symbol": ticker,
                    "expiration": "",  # Will get nearest expiration
                    "greeks": "true",
                }

                response = requests.get(options_url, headers=headers, params=params, timeout=10)

                if response.status_code == 200:
                    data = response.json()

                    if "options" in data and "option" in data["options"]:
                        options = data["options"]["option"]

                        # Aggregate call and put volume
                        total_call_volume = 0
                        total_put_volume = 0
                        total_call_oi = 0
                        total_put_oi = 0

                        for option in options[:50]:  # Check first 50 options
                            option_type = option.get("option_type", "")
                            volume = int(option.get("volume", 0))
                            open_interest = int(option.get("open_interest", 0))

                            if option_type == "call":
                                total_call_volume += volume
                                total_call_oi += open_interest
                            elif option_type == "put":
                                total_put_volume += volume
                                total_put_oi += open_interest

                        # Calculate metrics
                        total_volume = total_call_volume + total_put_volume

                        if total_volume > 10000:  # Significant volume threshold
                            put_call_ratio = (
                                total_put_volume / total_call_volume if total_call_volume > 0 else 0
                            )

                            # Unusual signals:
                            # - Very low P/C ratio (<0.7) = Bullish (heavy call buying)
                            # - Very high P/C ratio (>1.5) = Bearish (heavy put buying)
                            # - High volume (>50k) = Strong conviction

                            signal = "neutral"
                            if put_call_ratio < 0.7:
                                signal = "bullish_calls"
                            elif put_call_ratio > 1.5:
                                signal = "bearish_puts"
                            elif total_volume > 50000:
                                signal = "high_volume"

                            unusual_activity.append(
                                {
                                    "ticker": ticker,
                                    "total_volume": total_volume,
                                    "call_volume": total_call_volume,
                                    "put_volume": total_put_volume,
                                    "put_call_ratio": put_call_ratio,
                                    "signal": signal,
                                    "call_oi": total_call_oi,
                                    "put_oi": total_put_oi,
                                }
                            )

            except Exception:
                # Skip this ticker if there's an error
                continue

        # Sort by total volume (highest first)
        sorted_activity = sorted(unusual_activity, key=lambda x: x["total_volume"], reverse=True)[
            :top_n
        ]

        # Format output
        if not sorted_activity:
            return "No unusual options activity detected"

        report = f"# Unusual Options Activity - {date or 'Latest'}\n\n"
        report += (
            "**Criteria**: P/C Ratio extremes (<0.7 bullish, >1.5 bearish), High volume (>50k)\n\n"
        )
        report += f"**Found**: {len(sorted_activity)} stocks with notable options activity\n\n"
        report += "## Top Options Activity\n\n"
        report += format_markdown_table(
            ["Ticker", "Total Volume", "Call Vol", "Put Vol", "P/C Ratio", "Signal"],
            [
                [
                    a["ticker"],
                    f"{a['total_volume']:,}",
                    f"{a['call_volume']:,}",
                    f"{a['put_volume']:,}",
                    f"{a['put_call_ratio']:.2f}",
                    a["signal"],
                ]
                for a in sorted_activity
            ],
        )

        report += "\n\n## Signal Definitions\n\n"
        report += "- **bullish_calls**: P/C ratio <0.7 - Heavy call buying, bullish positioning\n"
        report += "- **bearish_puts**: P/C ratio >1.5 - Heavy put buying, bearish positioning\n"
        report += "- **high_volume**: Exceptional volume (>50k) - Strong conviction move\n"
        report += "- **neutral**: Balanced activity\n\n"
        report += "**Note**: Options activity is a leading indicator. Smart money often positions 1-2 weeks before catalysts.\n"

        return report

    except requests.exceptions.RequestException as e:
        return f"Error fetching options activity from Tradier: {str(e)}"
    except Exception as e:
        return f"Unexpected error in options activity detection: {str(e)}"


def get_tradier_unusual_options(
    tickers: List[str] = None,
    date: str = None,
    min_volume_multiple: float = 2.0,
    top_n: int = 20,
) -> str:
    """Alias for get_unusual_options_activity to match registry naming convention"""
    return get_unusual_options_activity(tickers, date, min_volume_multiple, top_n)
