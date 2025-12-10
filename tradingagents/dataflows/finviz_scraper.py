"""
Finviz + Yahoo Finance Hybrid - Short Interest Discovery
Uses Finviz to discover tickers with high short interest, then Yahoo Finance for exact data
"""

import requests
from bs4 import BeautifulSoup
from typing import Annotated
import re
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_short_interest(
    min_short_interest_pct: Annotated[float, "Minimum short interest % of float"] = 10.0,
    min_days_to_cover: Annotated[float, "Minimum days to cover ratio"] = 2.0,
    top_n: Annotated[int, "Number of top results to return"] = 20,
) -> str:
    """
    Discover stocks with high short interest using Finviz + Yahoo Finance.

    Strategy: Finviz filters stocks by short interest (discovery),
    then Yahoo Finance provides exact short % data.

    This is a TRUE DISCOVERY tool - finds stocks we may not know about,
    not checking a predefined watchlist.

    Args:
        min_short_interest_pct: Minimum short interest as % of float
        min_days_to_cover: Minimum days to cover ratio
        top_n: Number of top results to return

    Returns:
        Formatted markdown report of discovered high short interest stocks
    """
    try:
        # Step 1: Use Finviz screener to DISCOVER tickers with high short interest
        print(f"   Discovering tickers with short interest >{min_short_interest_pct}% from Finviz...")

        # Determine Finviz filter
        if min_short_interest_pct >= 20:
            short_filter = "sh_short_o20"
        elif min_short_interest_pct >= 15:
            short_filter = "sh_short_o15"
        elif min_short_interest_pct >= 10:
            short_filter = "sh_short_o10"
        else:
            short_filter = "sh_short_o5"

        # Build Finviz URL (v=152 is simple view)
        base_url = f"https://finviz.com/screener.ashx?v=152&f={short_filter}"

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html',
        }

        discovered_tickers = []

        # Scrape first 3 pages (60 stocks)
        for page_num in range(1, 4):
            if page_num == 1:
                url = base_url
            else:
                offset = (page_num - 1) * 20 + 1
                url = f"{base_url}&r={offset}"

            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'html.parser')

            # Find ticker links in the page
            ticker_links = soup.find_all('a', href=re.compile(r'quote\.ashx\?t='))

            for link in ticker_links:
                ticker = link.get_text(strip=True)
                # Validate it's a ticker (1-5 uppercase letters)
                if re.match(r'^[A-Z]{1,5}$', ticker) and ticker not in discovered_tickers:
                    discovered_tickers.append(ticker)

        if not discovered_tickers:
            return f"No stocks discovered with short interest >{min_short_interest_pct}% on Finviz."

        print(f"   Discovered {len(discovered_tickers)} tickers from Finviz")
        print(f"   Fetching detailed short interest data from Yahoo Finance...")

        # Step 2: Use Yahoo Finance to get EXACT short interest data for discovered tickers
        def fetch_short_data(ticker):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info

                # Get short interest data
                short_pct = info.get('shortPercentOfFloat', info.get('sharesPercentSharesOut', 0))
                if short_pct and isinstance(short_pct, (int, float)):
                    short_pct = short_pct * 100  # Convert to percentage
                else:
                    return None

                # Verify it meets criteria (Finviz filter might be outdated)
                if short_pct >= min_short_interest_pct:
                    price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                    market_cap = info.get('marketCap', 0)
                    volume = info.get('volume', info.get('regularMarketVolume', 0))

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
        all_candidates = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = {executor.submit(fetch_short_data, ticker): ticker for ticker in discovered_tickers}

            for future in as_completed(futures):
                result = future.result()
                if result:
                    all_candidates.append(result)

        if not all_candidates:
            return f"No stocks with verified short interest >{min_short_interest_pct}% (Finviz found {len(discovered_tickers)} tickers but Yahoo Finance data didn't confirm)."

        # Sort by short interest percentage (highest first)
        sorted_candidates = sorted(
            all_candidates,
            key=lambda x: x["short_interest_pct"],
            reverse=True
        )[:top_n]

        # Format output
        report = f"# Discovered High Short Interest Stocks\n\n"
        report += f"**Criteria**: Short Interest >{min_short_interest_pct}%\n"
        report += f"**Data Source**: Finviz Screener (Web Scraping)\n"
        report += f"**Total Discovered**: {len(all_candidates)} stocks\n\n"
        report += f"**Top {len(sorted_candidates)} Candidates**:\n\n"
        report += "| Ticker | Price | Market Cap | Volume | Short % | Signal |\n"
        report += "|--------|-------|------------|--------|---------|--------|\n"

        for candidate in sorted_candidates:
            market_cap_str = format_market_cap(candidate['market_cap'])
            report += f"| {candidate['ticker']} | "
            report += f"${candidate['price']:.2f} | "
            report += f"{market_cap_str} | "
            report += f"{candidate['volume']:,} | "
            report += f"{candidate['short_interest_pct']:.1f}% | "
            report += f"{candidate['signal']} |\n"

        report += "\n\n## Signal Definitions\n\n"
        report += "- **extreme_squeeze_risk**: Short interest >30% - Very high squeeze potential\n"
        report += "- **high_squeeze_potential**: Short interest 20-30% - High squeeze risk\n"
        report += "- **moderate_squeeze_potential**: Short interest 15-20% - Moderate squeeze risk\n"
        report += "- **low_squeeze_potential**: Short interest 10-15% - Lower squeeze risk\n\n"
        report += "**Note**: High short interest alone doesn't guarantee a squeeze. Look for positive catalysts.\n"

        return report

    except requests.exceptions.RequestException as e:
        return f"Error scraping Finviz: {str(e)}"
    except Exception as e:
        return f"Unexpected error discovering short interest stocks: {str(e)}"


def parse_market_cap(market_cap_text: str) -> float:
    """Parse market cap from Finviz format (e.g., '1.23B', '456M')."""
    if not market_cap_text or market_cap_text == '-':
        return 0.0

    market_cap_text = market_cap_text.upper().strip()

    # Extract number and multiplier
    match = re.match(r'([0-9.]+)([BMK])?', market_cap_text)
    if not match:
        return 0.0

    number = float(match.group(1))
    multiplier = match.group(2)

    if multiplier == 'B':
        return number * 1_000_000_000
    elif multiplier == 'M':
        return number * 1_000_000
    elif multiplier == 'K':
        return number * 1_000
    else:
        return number


def format_market_cap(market_cap: float) -> str:
    """Format market cap for display."""
    if market_cap >= 1_000_000_000:
        return f"${market_cap / 1_000_000_000:.2f}B"
    elif market_cap >= 1_000_000:
        return f"${market_cap / 1_000_000:.2f}M"
    else:
        return f"${market_cap:,.0f}"


def get_finviz_short_interest(
    min_short_interest_pct: float = 10.0,
    min_days_to_cover: float = 2.0,
    top_n: int = 20,
) -> str:
    """Alias for get_short_interest to match registry naming convention"""
    return get_short_interest(min_short_interest_pct, min_days_to_cover, top_n)
