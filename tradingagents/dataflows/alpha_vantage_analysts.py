"""
Alpha Vantage Analyst Rating Changes Detection
Tracks recent analyst upgrades/downgrades and price target changes
"""

import os
import requests
from datetime import datetime, timedelta
from typing import Annotated, List


def get_analyst_rating_changes(
    lookback_days: Annotated[int, "Number of days to look back for rating changes"] = 7,
    change_types: Annotated[List[str], "Types of changes to track"] = None,
    top_n: Annotated[int, "Number of top results to return"] = 20,
) -> str:
    """
    Track recent analyst upgrades/downgrades and rating changes.

    Fresh analyst actions (<72 hours) are strong catalysts for short-term moves.

    Args:
        lookback_days: Number of days to look back (default 7)
        change_types: Types of changes ["upgrade", "downgrade", "initiated", "reiterated"]
        top_n: Maximum number of results to return

    Returns:
        Formatted markdown report of recent analyst rating changes
    """
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY")
    if not api_key:
        return "Error: ALPHA_VANTAGE_API_KEY not set in environment variables"

    if change_types is None:
        change_types = ["upgrade", "downgrade", "initiated"]

    # Note: Alpha Vantage doesn't have a direct analyst ratings endpoint in the free tier
    # We'll use news sentiment API which includes analyst actions
    # For production, consider using Financial Modeling Prep or Benzinga API

    url = "https://www.alphavantage.co/query"

    try:
        # Get market news which includes analyst actions
        params = {
            "function": "NEWS_SENTIMENT",
            "topics": "earnings,technology,finance",
            "sort": "LATEST",
            "limit": 200,  # Get more news to find analyst actions
            "apikey": api_key,
        }

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "Note" in data:
            return f"API Rate Limit: {data['Note']}"

        if "Error Message" in data:
            return f"API Error: {data['Error Message']}"

        # Parse news for analyst actions
        analyst_changes = []
        cutoff_date = datetime.now() - timedelta(days=lookback_days)

        if "feed" in data:
            for article in data["feed"]:
                try:
                    # Check article time
                    time_published = article.get("time_published", "")
                    if time_published:
                        article_date = datetime.strptime(time_published[:8], "%Y%m%d")
                        if article_date < cutoff_date:
                            continue

                    title = article.get("title", "").lower()
                    summary = article.get("summary", "").lower()
                    text = f"{title} {summary}"

                    # Look for analyst action keywords
                    is_upgrade = any(word in text for word in ["upgrade", "upgrades", "raised", "raises rating"])
                    is_downgrade = any(word in text for word in ["downgrade", "downgrades", "lowered", "lowers rating"])
                    is_initiated = any(word in text for word in ["initiates", "initiated", "coverage", "starts coverage"])
                    is_reiterated = any(word in text for word in ["reiterates", "reiterated", "maintains", "confirms"])

                    # Extract tickers from article
                    tickers = []
                    if "ticker_sentiment" in article:
                        for ticker_data in article["ticker_sentiment"]:
                            ticker = ticker_data.get("ticker", "")
                            if ticker and len(ticker) <= 5:  # Valid ticker format
                                tickers.append(ticker)

                    # Classify action type
                    action_type = None
                    if is_upgrade and "upgrade" in change_types:
                        action_type = "upgrade"
                    elif is_downgrade and "downgrade" in change_types:
                        action_type = "downgrade"
                    elif is_initiated and "initiated" in change_types:
                        action_type = "initiated"
                    elif is_reiterated and "reiterated" in change_types:
                        action_type = "reiterated"

                    if action_type and tickers:
                        # Calculate freshness (hours since published)
                        hours_old = (datetime.now() - article_date).total_seconds() / 3600

                        for ticker in tickers[:3]:  # Max 3 tickers per article
                            analyst_changes.append({
                                "ticker": ticker,
                                "action": action_type,
                                "date": time_published[:8],
                                "hours_old": int(hours_old),
                                "headline": article.get("title", "")[:100],
                                "source": article.get("source", "Unknown"),
                                "url": article.get("url", ""),
                            })

                except (ValueError, KeyError) as e:
                    continue

        # Remove duplicates (keep most recent per ticker)
        seen_tickers = {}
        for change in analyst_changes:
            ticker = change["ticker"]
            if ticker not in seen_tickers or change["hours_old"] < seen_tickers[ticker]["hours_old"]:
                seen_tickers[ticker] = change

        # Sort by freshness (most recent first)
        sorted_changes = sorted(
            seen_tickers.values(),
            key=lambda x: x["hours_old"]
        )[:top_n]

        # Format output
        if not sorted_changes:
            return f"No analyst rating changes found in the last {lookback_days} days"

        report = f"# Analyst Rating Changes - Last {lookback_days} Days\n\n"
        report += f"**Tracking**: {', '.join(change_types)}\n\n"
        report += f"**Found**: {len(sorted_changes)} recent analyst actions\n\n"
        report += "## Recent Analyst Actions\n\n"
        report += "| Ticker | Action | Source | Hours Ago | Headline |\n"
        report += "|--------|--------|--------|-----------|----------|\n"

        for change in sorted_changes:
            freshness = "🔥 FRESH" if change["hours_old"] < 24 else "🟢 Recent" if change["hours_old"] < 72 else "Older"

            report += f"| {change['ticker']} | "
            report += f"{change['action'].upper()} | "
            report += f"{change['source']} | "
            report += f"{change['hours_old']}h ({freshness}) | "
            report += f"{change['headline']} |\n"

        report += "\n\n## Freshness Legend\n\n"
        report += "- 🔥 **FRESH** (<24h): Highest impact, market may not have fully reacted\n"
        report += "- 🟢 **Recent** (24-72h): Still relevant for short-term trading\n"
        report += "- **Older** (>72h): Lower priority, likely partially priced in\n"

        return report

    except requests.exceptions.RequestException as e:
        return f"Error fetching analyst rating changes: {str(e)}"
    except Exception as e:
        return f"Unexpected error in analyst rating detection: {str(e)}"


def get_alpha_vantage_analyst_changes(
    lookback_days: int = 7,
    change_types: List[str] = None,
    top_n: int = 20,
) -> str:
    """Alias for get_analyst_rating_changes to match registry naming convention"""
    return get_analyst_rating_changes(lookback_days, change_types, top_n)
