from datetime import datetime

from .alpha_vantage_common import _filter_csv_by_date_range, _make_api_request


def get_stock(symbol: str, start_date: str, end_date: str) -> str:
    """
    Returns raw daily OHLCV values, adjusted close values, and historical split/dividend events
    filtered to the specified date range.

    Args:
        symbol: The name of the equity. For example: symbol=IBM
        start_date: Start date in yyyy-mm-dd format
        end_date: End date in yyyy-mm-dd format

    Returns:
        CSV string containing the daily adjusted time series data filtered to the date range.
    """
    # Parse dates to determine the range
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    today = datetime.now()

    # Choose outputsize based on whether the requested range is within the latest 100 days
    # Compact returns latest 100 data points, so check if start_date is recent enough
    days_from_today_to_start = (today - start_dt).days
    outputsize = "compact" if days_from_today_to_start < 100 else "full"

    params = {
        "symbol": symbol,
        "outputsize": outputsize,
        "datatype": "csv",
    }

    response = _make_api_request("TIME_SERIES_DAILY_ADJUSTED", params)

    return _filter_csv_by_date_range(response, start_date, end_date)


def get_top_gainers_losers(limit: int = 10, return_structured: bool = False):
    """
    Returns the top gainers, losers, and most active stocks from Alpha Vantage.

    Args:
        limit: Maximum number of items per category
        return_structured: If True, returns dict with raw data instead of markdown

    Returns:
        If return_structured=True: dict with 'gainers', 'losers', 'most_active' lists
        If return_structured=False: Formatted markdown string
    """
    params = {}

    # This returns a JSON string
    response_text = _make_api_request("TOP_GAINERS_LOSERS", params)

    try:
        import json

        data = json.loads(response_text)

        if "top_gainers" not in data:
            if return_structured:
                return {"error": f"Unexpected response format: {response_text[:200]}..."}
            return f"Error: Unexpected response format: {response_text[:200]}..."

        # Apply limit to data
        gainers = data.get("top_gainers", [])[:limit]
        losers = data.get("top_losers", [])[:limit]
        most_active = data.get("most_actively_traded", [])[:limit]

        # Return structured data if requested
        if return_structured:
            return {
                "gainers": gainers,
                "losers": losers,
                "most_active": most_active,
            }

        # Format as markdown report
        report = "## Top Market Movers (Alpha Vantage)\n\n"

        # Top Gainers
        report += "### Top Gainers\n"
        report += "| Ticker | Price | Change % | Volume |\n"
        report += "|--------|-------|----------|--------|\n"
        for item in gainers:
            report += f"| {item['ticker']} | {item['price']} | {item['change_percentage']} | {item['volume']} |\n"

        # Top Losers
        report += "\n### Top Losers\n"
        report += "| Ticker | Price | Change % | Volume |\n"
        report += "|--------|-------|----------|--------|\n"
        for item in losers:
            report += f"| {item['ticker']} | {item['price']} | {item['change_percentage']} | {item['volume']} |\n"

        # Most Active
        report += "\n### Most Active\n"
        report += "| Ticker | Price | Change % | Volume |\n"
        report += "|--------|-------|----------|--------|\n"
        for item in most_active:
            report += f"| {item['ticker']} | {item['price']} | {item['change_percentage']} | {item['volume']} |\n"

        return report

    except json.JSONDecodeError:
        if return_structured:
            return {"error": f"Failed to parse JSON response: {response_text[:200]}..."}
        return f"Error: Failed to parse JSON response: {response_text[:200]}..."
    except Exception as e:
        if return_structured:
            return {"error": str(e)}
        return f"Error processing market movers: {str(e)}"
