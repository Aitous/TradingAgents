"""
Yahoo Finance API - Short Interest Data using yfinance
Identifies potential short squeeze candidates with high short interest
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated

from tradingagents.dataflows.market_data_utils import format_markdown_table, format_market_cap
from tradingagents.dataflows.y_finance import get_ticker_info
from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


def get_short_interest(
    min_short_interest_pct: Annotated[float, "Minimum short interest % of float"] = 10.0,
    min_days_to_cover: Annotated[float, "Minimum days to cover ratio"] = 2.0,
    top_n: Annotated[int, "Number of top results to return"] = 20,
) -> str:
    """
    Get stocks with high short interest using yfinance (FREE data source).

    Checks a watchlist of stocks for high short interest data from Yahoo Finance.
    High short interest + positive catalyst = short squeeze potential.

    Note: This scans a predefined universe of stocks. For comprehensive scanning,
    consider using a stock screener API with short interest filters.

    Args:
        min_short_interest_pct: Minimum short interest as % of float
        min_days_to_cover: Minimum days to cover ratio
        top_n: Number of top results to return

    Returns:
        Formatted markdown report of high short interest stocks
    """
    try:
        # Curated watchlist of stocks known for volatility/short interest
        # In a production system, this would come from a screener API
        watchlist = [
            # Meme stocks & high short interest candidates
            "GME",
            "AMC",
            "BBBY",
            "BYND",
            "CLOV",
            "WISH",
            "PLTR",
            "SPCE",
            # EV & Tech
            "RIVN",
            "LCID",
            "NIO",
            "TSLA",
            "NKLA",
            "PLUG",
            "FCEL",
            # Biotech (often heavily shorted)
            "SAVA",
            "NVAX",
            "MRNA",
            "BNTX",
            "VXRT",
            "SESN",
            "OCGN",
            # Retail & Consumer
            "PTON",
            "W",
            "CVNA",
            "DASH",
            "UBER",
            "LYFT",
            # Finance & REITs
            "SOFI",
            "HOOD",
            "COIN",
            "SQ",
            "AFRM",
            # Small caps with squeeze potential
            "APRN",
            "ATER",
            "BBIG",
            "CEI",
            "PROG",
            "SNDL",
            # Others
            "TDOC",
            "ZM",
            "PTON",
            "NFLX",
            "SNAP",
            "PINS",
        ]

        logger.info(f"Checking short interest for {len(watchlist)} tickers...")

        high_si_candidates = []

        # Use threading to speed up API calls
        def fetch_short_data(ticker):
            try:
                info = get_ticker_info(ticker)

                # Get short interest data
                short_pct = info.get("shortPercentOfFloat", info.get("sharesPercentSharesOut", 0))
                if short_pct and isinstance(short_pct, (int, float)):
                    short_pct = short_pct * 100  # Convert to percentage
                else:
                    return None

                # Only include if meets criteria
                if short_pct >= min_short_interest_pct:
                    # Get other data
                    price = info.get("currentPrice", info.get("regularMarketPrice", 0))
                    market_cap = info.get("marketCap", 0)
                    volume = info.get("volume", info.get("regularMarketVolume", 0))

                    # Categorize squeeze potential
                    if short_pct >= 30:
                        signal = "extreme_squeeze_risk"
                    elif short_pct >= 20:
                        signal = "high_squeeze_potential"
                    elif short_pct >= 15:
                        signal = "moderate_squeeze_potential"
                    else:
                        signal = "low_squeeze_potential"

                    return {
                        "ticker": ticker,
                        "price": price,
                        "market_cap": market_cap,
                        "volume": volume,
                        "short_interest_pct": short_pct,
                        "signal": signal,
                    }
            except Exception:
                return None

        # Fetch data in parallel (faster)
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_short_data, ticker): ticker for ticker in watchlist}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    high_si_candidates.append(result)

        if not high_si_candidates:
            return f"# High Short Interest Stocks\n\n**No stocks found** matching criteria: SI% >{min_short_interest_pct}%\n\n**Note**: Checked {len(watchlist)} tickers from watchlist."

        # Sort by short interest percentage (highest first)
        sorted_candidates = sorted(
            high_si_candidates, key=lambda x: x["short_interest_pct"], reverse=True
        )[:top_n]

        # Format output
        report = "# High Short Interest Stocks (Yahoo Finance Data)\n\n"
        report += f"**Criteria**: Short Interest >{min_short_interest_pct}%\n"
        report += "**Data Source**: Yahoo Finance via yfinance\n"
        report += f"**Checked**: {len(watchlist)} tickers from watchlist\n\n"
        report += f"**Found**: {len(sorted_candidates)} stocks with high short interest\n\n"
        report += f"**Found**: {len(sorted_candidates)} stocks with high short interest\n\n"
        report += "## Potential Short Squeeze Candidates\n\n"

        headers = ["Ticker", "Price", "Market Cap", "Volume", "Short %", "Signal"]
        rows = []
        for candidate in sorted_candidates:
            rows.append(
                [
                    candidate["ticker"],
                    f"${candidate['price']:.2f}",
                    format_market_cap(candidate["market_cap"]),
                    f"{candidate['volume']:,}",
                    f"{candidate['short_interest_pct']:.1f}%",
                    candidate["signal"],
                ]
            )

        report += format_markdown_table(headers, rows)

        report += "\n\n## Signal Definitions\n\n"
        report += "- **extreme_squeeze_risk**: Short interest >30% - Very high squeeze potential\n"
        report += "- **high_squeeze_potential**: Short interest 20-30% - High squeeze risk\n"
        report += (
            "- **moderate_squeeze_potential**: Short interest 15-20% - Moderate squeeze risk\n"
        )
        report += "- **low_squeeze_potential**: Short interest 10-15% - Lower squeeze risk\n\n"
        report += "**Note**: High short interest alone doesn't guarantee a squeeze. Look for positive catalysts.\n"
        report += "**Limitation**: This checks a curated watchlist. For comprehensive scanning, use a stock screener with short interest filters.\n"

        return report

    except Exception as e:
        return f"Unexpected error in short interest detection: {str(e)}"


def get_fmp_short_interest(
    min_short_interest_pct: float = 10.0,
    min_days_to_cover: float = 2.0,
    top_n: int = 20,
) -> str:
    """Alias for get_short_interest to match registry naming convention"""
    return get_short_interest(min_short_interest_pct, min_days_to_cover, top_n)


def get_finra_short_interest(
    min_short_interest_pct: float = 10.0,
    min_days_to_cover: float = 2.0,
    top_n: int = 20,
) -> str:
    """
    Alternative: Get short interest from Finra public data.
    Note: Finra data is updated bi-monthly and requires parsing from their website.
    """
    # This would require web scraping or using Finra's data API
    # For now, return a message directing to manual sources
    return """# Finra Short Interest Data

**Note**: Finra short interest data is publicly available but requires specialized parsing.

## Access Finra Data:
1. Visit: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data
2. Download latest settlement date files
3. Parse for high short interest stocks

## Alternative Free Sources:
- **Market Beat**: https://www.marketbeat.com/short-interest/
- **Finviz Screener**: Filter by "Short Float >20%"
- **Yahoo Finance**: Individual stock pages show short % of float

For automated access, consider FMP Premium API or implementing Finra data parser.
"""
