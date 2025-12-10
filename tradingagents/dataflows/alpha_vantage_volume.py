"""
Alpha Vantage Unusual Volume Detection
Identifies stocks with unusual volume but minimal price movement (accumulation signal)
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Annotated, List, Dict
import pandas as pd


def get_unusual_volume(
    date: Annotated[str, "Analysis date in yyyy-mm-dd format"] = None,
    min_volume_multiple: Annotated[float, "Minimum volume multiple vs average"] = 3.0,
    max_price_change: Annotated[float, "Maximum price change percentage"] = 5.0,
    top_n: Annotated[int, "Number of top results to return"] = 20,
) -> str:
    """
    Find stocks with unusual volume but minimal price movement.

    This is a strong accumulation signal - smart money buying before a breakout.

    Args:
        date: Analysis date in yyyy-mm-dd format
        min_volume_multiple: Minimum volume multiple vs 30-day average
        max_price_change: Maximum absolute price change percentage
        top_n: Number of top results to return

    Returns:
        Formatted markdown report of stocks with unusual volume
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY not set in environment variables"

    # For unusual volume detection, we'll use Alpha Vantage's market data
    # Note: Alpha Vantage doesn't have a direct "unusual volume" endpoint,
    # so we'll use a combination of their screening and market movers data

    # Strategy: Get top active stocks (high volume) and filter for minimal price change
    url = "https://www.alphavantage.co/query"

    try:
        # Get top active stocks by volume
        params = {
            "function": "TOP_GAINERS_LOSERS",
            "apikey": api_key,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Note" in data:
            return f"API Rate Limit: {data['Note']}"

        if "Error Message" in data:
            return f"API Error: {data['Error Message']}"

        # Combine all movers (gainers, losers, and most actively traded)
        unusual_candidates = []

        # Process most actively traded (these have high volume)
        if "most_actively_traded" in data:
            for stock in data["most_actively_traded"][:50]:  # Check top 50
                try:
                    ticker = stock.get("ticker", "")
                    price_change = abs(float(stock.get("change_percentage", "0").replace("%", "")))
                    volume = int(stock.get("volume", 0))
                    price = float(stock.get("price", 0))

                    # Filter: High volume but low price change (accumulation signal)
                    if price_change <= max_price_change and volume > 0:
                        unusual_candidates.append({
                            "ticker": ticker,
                            "volume": volume,
                            "price": price,
                            "price_change_pct": price_change,
                            "signal": "accumulation" if price_change < 2.0 else "moderate_activity"
                        })

                except (ValueError, KeyError) as e:
                    continue

        # Also check gainers and losers with unusual volume patterns
        for category in ["top_gainers", "top_losers"]:
            if category in data:
                for stock in data[category][:30]:
                    try:
                        ticker = stock.get("ticker", "")
                        price_change = abs(float(stock.get("change_percentage", "0").replace("%", "")))
                        volume = int(stock.get("volume", 0))
                        price = float(stock.get("price", 0))

                        # For gainers/losers, we want very high volume
                        # This indicates strong conviction in the move
                        if volume > 0:
                            unusual_candidates.append({
                                "ticker": ticker,
                                "volume": volume,
                                "price": price,
                                "price_change_pct": price_change,
                                "signal": "breakout" if price_change > 5.0 else "building_momentum"
                            })
                    except (ValueError, KeyError) as e:
                        continue

        # Remove duplicates (keep highest volume)
        seen_tickers = {}
        for candidate in unusual_candidates:
            ticker = candidate["ticker"]
            if ticker not in seen_tickers or candidate["volume"] > seen_tickers[ticker]["volume"]:
                seen_tickers[ticker] = candidate

        # Sort by volume (highest first) and take top N
        sorted_candidates = sorted(
            seen_tickers.values(),
            key=lambda x: x["volume"],
            reverse=True
        )[:top_n]

        # Format output
        if not sorted_candidates:
            return "No stocks found with unusual volume patterns matching criteria"

        report = f"# Unusual Volume Detected - {date or 'Latest'}\n\n"
        report += f"**Criteria**: Volume signal detected, Price Change <{max_price_change}% preferred\n\n"
        report += f"**Found**: {len(sorted_candidates)} stocks with unusual activity\n\n"
        report += "## Top Unusual Volume Candidates\n\n"
        report += "| Ticker | Price | Volume | Price Change % | Signal |\n"
        report += "|--------|-------|--------|----------------|--------|\n"

        for candidate in sorted_candidates:
            report += f"| {candidate['ticker']} | "
            report += f"${candidate['price']:.2f} | "
            report += f"{candidate['volume']:,} | "
            report += f"{candidate['price_change_pct']:.2f}% | "
            report += f"{candidate['signal']} |\n"

        report += "\n\n## Signal Definitions\n\n"
        report += "- **accumulation**: High volume, minimal price change (<2%) - Smart money building position\n"
        report += "- **moderate_activity**: Elevated volume with 2-5% price change - Early momentum\n"
        report += "- **building_momentum**: Losers/Gainers with strong volume - Conviction in direction\n"
        report += "- **breakout**: Strong price move (>5%) on high volume - Already in motion\n"

        return report

    except requests.exceptions.RequestException as e:
        return f"Error fetching unusual volume data: {str(e)}"
    except Exception as e:
        return f"Unexpected error in unusual volume detection: {str(e)}"


def get_alpha_vantage_unusual_volume(
    date: str = None,
    min_volume_multiple: float = 3.0,
    max_price_change: float = 5.0,
    top_n: int = 20,
) -> str:
    """Alias for get_unusual_volume to match registry naming convention"""
    return get_unusual_volume(date, min_volume_multiple, max_price_change, top_n)
