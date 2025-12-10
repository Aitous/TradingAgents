from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pandas as pd
import os
from .stockstats_utils import StockstatsUtils

def get_YFin_data_online(
    symbol: Annotated[str, "ticker symbol of the company"],
    start_date: Annotated[str, "Start date in yyyy-mm-dd format"],
    end_date: Annotated[str, "End date in yyyy-mm-dd format"],
):

    datetime.strptime(start_date, "%Y-%m-%d")
    datetime.strptime(end_date, "%Y-%m-%d")

    # Create ticker object
    ticker = yf.Ticker(symbol.upper())

    # Fetch historical data for the specified date range
    data = ticker.history(start=start_date, end=end_date)

    # Check if data is empty
    if data.empty:
        return (
            f"No data found for symbol '{symbol}' between {start_date} and {end_date}"
        )

    # Remove timezone info from index for cleaner output
    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    # Round numerical values to 2 decimal places for cleaner display
    numeric_columns = ["Open", "High", "Low", "Close", "Adj Close"]
    for col in numeric_columns:
        if col in data.columns:
            data[col] = data[col].round(2)

    # Convert DataFrame to CSV string
    csv_string = data.to_csv()

    # Add header information
    header = f"# Stock data for {symbol.upper()} from {start_date} to {end_date}\n"
    header += f"# Total records: {len(data)}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return header + csv_string

def get_stock_stats_indicators_window(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:

    best_ind_params = {
        # Moving Averages
        "close_50_sma": (
            "50 SMA: A medium-term trend indicator. "
            "Usage: Identify trend direction and serve as dynamic support/resistance. "
            "Tips: It lags price; combine with faster indicators for timely signals."
        ),
        "close_200_sma": (
            "200 SMA: A long-term trend benchmark. "
            "Usage: Confirm overall market trend and identify golden/death cross setups. "
            "Tips: It reacts slowly; best for strategic trend confirmation rather than frequent trading entries."
        ),
        "close_10_ema": (
            "10 EMA: A responsive short-term average. "
            "Usage: Capture quick shifts in momentum and potential entry points. "
            "Tips: Prone to noise in choppy markets; use alongside longer averages for filtering false signals."
        ),
        # MACD Related
        "macd": (
            "MACD: Computes momentum via differences of EMAs. "
            "Usage: Look for crossovers and divergence as signals of trend changes. "
            "Tips: Confirm with other indicators in low-volatility or sideways markets."
        ),
        "macds": (
            "MACD Signal: An EMA smoothing of the MACD line. "
            "Usage: Use crossovers with the MACD line to trigger trades. "
            "Tips: Should be part of a broader strategy to avoid false positives."
        ),
        "macdh": (
            "MACD Histogram: Shows the gap between the MACD line and its signal. "
            "Usage: Visualize momentum strength and spot divergence early. "
            "Tips: Can be volatile; complement with additional filters in fast-moving markets."
        ),
        # Momentum Indicators
        "rsi": (
            "RSI: Measures momentum to flag overbought/oversold conditions. "
            "Usage: Apply 70/30 thresholds and watch for divergence to signal reversals. "
            "Tips: In strong trends, RSI may remain extreme; always cross-check with trend analysis."
        ),
        # Volatility Indicators
        "boll": (
            "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands. "
            "Usage: Acts as a dynamic benchmark for price movement. "
            "Tips: Combine with the upper and lower bands to effectively spot breakouts or reversals."
        ),
        "boll_ub": (
            "Bollinger Upper Band: Typically 2 standard deviations above the middle line. "
            "Usage: Signals potential overbought conditions and breakout zones. "
            "Tips: Confirm signals with other tools; prices may ride the band in strong trends."
        ),
        "boll_lb": (
            "Bollinger Lower Band: Typically 2 standard deviations below the middle line. "
            "Usage: Indicates potential oversold conditions. "
            "Tips: Use additional analysis to avoid false reversal signals."
        ),
        "atr": (
            "ATR: Averages true range to measure volatility. "
            "Usage: Set stop-loss levels and adjust position sizes based on current market volatility. "
            "Tips: It's a reactive measure, so use it as part of a broader risk management strategy."
        ),
        # Volume-Based Indicators
        "vwma": (
            "VWMA: A moving average weighted by volume. "
            "Usage: Confirm trends by integrating price action with volume data. "
            "Tips: Watch for skewed results from volume spikes; use in combination with other volume analyses."
        ),
        "mfi": (
            "MFI: The Money Flow Index is a momentum indicator that uses both price and volume to measure buying and selling pressure. "
            "Usage: Identify overbought (>80) or oversold (<20) conditions and confirm the strength of trends or reversals. "
            "Tips: Use alongside RSI or MACD to confirm signals; divergence between price and MFI can indicate potential reversals."
        ),
    }

    if indicator not in best_ind_params:
        raise ValueError(
            f"Indicator {indicator} is not supported. Please choose from: {list(best_ind_params.keys())}"
        )

    end_date = curr_date
    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before = curr_date_dt - relativedelta(days=look_back_days)

    # Optimized: Get stock data once and calculate indicators for all dates
    try:
        indicator_data = _get_stock_stats_bulk(symbol, indicator, curr_date)
        
        # Generate the date range we need
        current_dt = curr_date_dt
        date_values = []
        
        while current_dt >= before:
            date_str = current_dt.strftime('%Y-%m-%d')
            
            # Look up the indicator value for this date
            if date_str in indicator_data:
                indicator_value = indicator_data[date_str]
            else:
                indicator_value = "N/A: Not a trading day (weekend or holiday)"
            
            date_values.append((date_str, indicator_value))
            current_dt = current_dt - relativedelta(days=1)
        
        # Build the result string
        ind_string = ""
        for date_str, value in date_values:
            ind_string += f"{date_str}: {value}\n"
        
    except Exception as e:
        print(f"Error getting bulk stockstats data: {e}")
        # Fallback to original implementation if bulk method fails
        ind_string = ""
        curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
        while curr_date_dt >= before:
            indicator_value = get_stockstats_indicator(
                symbol, indicator, curr_date_dt.strftime("%Y-%m-%d")
            )
            ind_string += f"{curr_date_dt.strftime('%Y-%m-%d')}: {indicator_value}\n"
            curr_date_dt = curr_date_dt - relativedelta(days=1)

    result_str = (
        f"## {indicator} values from {before.strftime('%Y-%m-%d')} to {end_date}:\n\n"
        + ind_string
        + "\n\n"
        + best_ind_params.get(indicator, "No description available.")
    )

    return result_str


def _get_stock_stats_bulk(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to calculate"],
    curr_date: Annotated[str, "current date for reference"]
) -> dict:
    """
    Optimized bulk calculation of stock stats indicators.
    Fetches data once and calculates indicator for all available dates.
    Returns dict mapping date strings to indicator values.
    """
    from .config import get_config
    import pandas as pd
    from stockstats import wrap
    import os
    
    config = get_config()
    online = config["data_vendors"]["technical_indicators"] != "local"
    
    if not online:
        # Local data path
        try:
            data = pd.read_csv(
                os.path.join(
                    config.get("data_cache_dir", "data"),
                    f"{symbol}-YFin-data-2015-01-01-2025-03-25.csv",
                )
            )
            df = wrap(data)
        except FileNotFoundError:
            raise Exception("Stockstats fail: Yahoo Finance data not fetched yet!")
    else:
        # Online data fetching with caching
        today_date = pd.Timestamp.today()
        curr_date_dt = pd.to_datetime(curr_date)
        
        end_date = today_date
        start_date = today_date - pd.DateOffset(years=2)
        start_date_str = start_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")
        
        os.makedirs(config["data_cache_dir"], exist_ok=True)
        
        data_file = os.path.join(
            config["data_cache_dir"],
            f"{symbol}-YFin-data-{start_date_str}-{end_date_str}.csv",
        )
        
        if os.path.exists(data_file):
            data = pd.read_csv(data_file)
            data["Date"] = pd.to_datetime(data["Date"])
        else:
            data = yf.download(
                symbol,
                start=start_date_str,
                end=end_date_str,
                multi_level_index=False,
                progress=False,
                auto_adjust=True,
            )
            data = data.reset_index()
            data.to_csv(data_file, index=False)
        
        df = wrap(data)
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
    
    # Calculate the indicator for all rows at once
    df[indicator]  # This triggers stockstats to calculate the indicator
    
    # Create a dictionary mapping date strings to indicator values
    result_dict = {}
    for _, row in df.iterrows():
        date_str = row["Date"]
        indicator_value = row[indicator]
        
        # Handle NaN/None values
        if pd.isna(indicator_value):
            result_dict[date_str] = "N/A"
        else:
            result_dict[date_str] = str(indicator_value)
    
    return result_dict


def get_stockstats_indicator(
    symbol: Annotated[str, "ticker symbol of the company"],
    indicator: Annotated[str, "technical indicator to get the analysis and report of"],
    curr_date: Annotated[
        str, "The current trading date you are trading on, YYYY-mm-dd"
    ],
) -> str:

    curr_date_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    curr_date = curr_date_dt.strftime("%Y-%m-%d")

    try:
        indicator_value = StockstatsUtils.get_stock_stats(
            symbol,
            indicator,
            curr_date,
        )
    except Exception as e:
        print(
            f"Error getting stockstats indicator data for indicator {indicator} on {curr_date}: {e}"
        )
        return ""

    return str(indicator_value)


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get balance sheet data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_balance_sheet
        else:
            data = ticker_obj.balance_sheet
            
        if data.empty:
            return f"No balance sheet data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Balance Sheet data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving balance sheet for {ticker}: {str(e)}"


def get_cashflow(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get cash flow data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_cashflow
        else:
            data = ticker_obj.cashflow
            
        if data.empty:
            return f"No cash flow data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Cash Flow data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving cash flow for {ticker}: {str(e)}"


def get_income_statement(
    ticker: Annotated[str, "ticker symbol of the company"],
    freq: Annotated[str, "frequency of data: 'annual' or 'quarterly'"] = "quarterly",
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get income statement data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        if freq.lower() == "quarterly":
            data = ticker_obj.quarterly_income_stmt
        else:
            data = ticker_obj.income_stmt
            
        if data.empty:
            return f"No income statement data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Income Statement data for {ticker.upper()} ({freq})\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving income statement for {ticker}: {str(e)}"


def get_insider_transactions(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (not used for yfinance)"] = None
):
    """Get insider transactions data from yfinance."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
            
        # Convert to CSV string for consistency with other functions
        csv_string = data.to_csv()
        
        # Add header information
        header = f"# Insider Transactions data for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"

def validate_ticker(symbol: str) -> bool:
    """
    Validate if a ticker symbol exists and has trading data.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        # Try to fetch 1 day of history
        # Suppress yfinance error output
        import sys
        from io import StringIO
        
        # Redirect stderr to suppress yfinance error messages
        original_stderr = sys.stderr
        sys.stderr = StringIO()
        
        try:
            history = ticker.history(period="1d")
            return not history.empty
        finally:
            # Restore stderr
            sys.stderr = original_stderr
            
    except Exception:
        return False


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date (for reference)"] = None
) -> str:
    """
    Get comprehensive fundamental data for a ticker using yfinance.
    Returns data in a format similar to Alpha Vantage's OVERVIEW endpoint.
    
    This is a FREE alternative to Alpha Vantage with no rate limits.
    """
    import json
    
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        info = ticker_obj.info
        
        if not info or info.get('regularMarketPrice') is None:
            return f"No fundamental data found for symbol '{ticker}'"
        
        # Build a structured response similar to Alpha Vantage
        fundamentals = {
            # Company Info
            "Symbol": ticker.upper(),
            "AssetType": info.get("quoteType", "N/A"),
            "Name": info.get("longName", info.get("shortName", "N/A")),
            "Description": info.get("longBusinessSummary", "N/A"),
            "Exchange": info.get("exchange", "N/A"),
            "Currency": info.get("currency", "USD"),
            "Country": info.get("country", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Industry": info.get("industry", "N/A"),
            "Address": f"{info.get('address1', '')} {info.get('city', '')}, {info.get('state', '')} {info.get('zip', '')}".strip(),
            "OfficialSite": info.get("website", "N/A"),
            "FiscalYearEnd": info.get("fiscalYearEnd", "N/A"),
            
            # Valuation
            "MarketCapitalization": str(info.get("marketCap", "N/A")),
            "EBITDA": str(info.get("ebitda", "N/A")),
            "PERatio": str(info.get("trailingPE", "N/A")),
            "ForwardPE": str(info.get("forwardPE", "N/A")),
            "PEGRatio": str(info.get("pegRatio", "N/A")),
            "BookValue": str(info.get("bookValue", "N/A")),
            "PriceToBookRatio": str(info.get("priceToBook", "N/A")),
            "PriceToSalesRatioTTM": str(info.get("priceToSalesTrailing12Months", "N/A")),
            "EVToRevenue": str(info.get("enterpriseToRevenue", "N/A")),
            "EVToEBITDA": str(info.get("enterpriseToEbitda", "N/A")),
            
            # Earnings & Revenue
            "EPS": str(info.get("trailingEps", "N/A")),
            "ForwardEPS": str(info.get("forwardEps", "N/A")),
            "RevenueTTM": str(info.get("totalRevenue", "N/A")),
            "RevenuePerShareTTM": str(info.get("revenuePerShare", "N/A")),
            "GrossProfitTTM": str(info.get("grossProfits", "N/A")),
            "QuarterlyRevenueGrowthYOY": str(info.get("revenueGrowth", "N/A")),
            "QuarterlyEarningsGrowthYOY": str(info.get("earningsGrowth", "N/A")),
            
            # Margins & Returns
            "ProfitMargin": str(info.get("profitMargins", "N/A")),
            "OperatingMarginTTM": str(info.get("operatingMargins", "N/A")),
            "GrossMargins": str(info.get("grossMargins", "N/A")),
            "ReturnOnAssetsTTM": str(info.get("returnOnAssets", "N/A")),
            "ReturnOnEquityTTM": str(info.get("returnOnEquity", "N/A")),
            
            # Dividend
            "DividendPerShare": str(info.get("dividendRate", "N/A")),
            "DividendYield": str(info.get("dividendYield", "N/A")),
            "ExDividendDate": str(info.get("exDividendDate", "N/A")),
            "PayoutRatio": str(info.get("payoutRatio", "N/A")),
            
            # Balance Sheet
            "TotalCash": str(info.get("totalCash", "N/A")),
            "TotalDebt": str(info.get("totalDebt", "N/A")),
            "CurrentRatio": str(info.get("currentRatio", "N/A")),
            "QuickRatio": str(info.get("quickRatio", "N/A")),
            "DebtToEquity": str(info.get("debtToEquity", "N/A")),
            "FreeCashFlow": str(info.get("freeCashflow", "N/A")),
            "OperatingCashFlow": str(info.get("operatingCashflow", "N/A")),
            
            # Trading Info
            "Beta": str(info.get("beta", "N/A")),
            "52WeekHigh": str(info.get("fiftyTwoWeekHigh", "N/A")),
            "52WeekLow": str(info.get("fiftyTwoWeekLow", "N/A")),
            "50DayMovingAverage": str(info.get("fiftyDayAverage", "N/A")),
            "200DayMovingAverage": str(info.get("twoHundredDayAverage", "N/A")),
            "SharesOutstanding": str(info.get("sharesOutstanding", "N/A")),
            "SharesFloat": str(info.get("floatShares", "N/A")),
            "SharesShort": str(info.get("sharesShort", "N/A")),
            "ShortRatio": str(info.get("shortRatio", "N/A")),
            "ShortPercentOfFloat": str(info.get("shortPercentOfFloat", "N/A")),
            
            # Ownership
            "PercentInsiders": str(info.get("heldPercentInsiders", "N/A")),
            "PercentInstitutions": str(info.get("heldPercentInstitutions", "N/A")),
            
            # Analyst
            "AnalystTargetPrice": str(info.get("targetMeanPrice", "N/A")),
            "AnalystTargetHigh": str(info.get("targetHighPrice", "N/A")),
            "AnalystTargetLow": str(info.get("targetLowPrice", "N/A")),
            "NumberOfAnalysts": str(info.get("numberOfAnalystOpinions", "N/A")),
            "RecommendationKey": info.get("recommendationKey", "N/A"),
            "RecommendationMean": str(info.get("recommendationMean", "N/A")),
        }
        
        # Return as formatted JSON string
        return json.dumps(fundamentals, indent=4)
        
    except Exception as e:
        return f"Error retrieving fundamentals for {ticker}: {str(e)}"


def get_options_activity(
    ticker: Annotated[str, "ticker symbol of the company"],
    num_expirations: Annotated[int, "number of nearest expiration dates to analyze"] = 3,
    curr_date: Annotated[str, "current date (for reference)"] = None
) -> str:
    """
    Get options activity for a specific ticker using yfinance.
    Analyzes volume, open interest, and put/call ratios.
    
    This is a FREE alternative to Tradier with no API key required.
    """
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        
        # Get available expiration dates
        expirations = ticker_obj.options
        if not expirations:
            return f"No options data available for {ticker}"
        
        # Analyze the nearest N expiration dates
        expirations_to_analyze = expirations[:min(num_expirations, len(expirations))]
        
        report = f"## Options Activity for {ticker.upper()}\n\n"
        report += f"**Available Expirations:** {len(expirations)} dates\n"
        report += f"**Analyzing:** {', '.join(expirations_to_analyze)}\n\n"
        
        total_call_volume = 0
        total_put_volume = 0
        total_call_oi = 0
        total_put_oi = 0
        
        unusual_activity = []
        
        for exp_date in expirations_to_analyze:
            try:
                opt = ticker_obj.option_chain(exp_date)
                calls = opt.calls
                puts = opt.puts
                
                if calls.empty and puts.empty:
                    continue
                
                # Calculate totals for this expiration
                call_vol = calls['volume'].sum() if 'volume' in calls.columns else 0
                put_vol = puts['volume'].sum() if 'volume' in puts.columns else 0
                call_oi = calls['openInterest'].sum() if 'openInterest' in calls.columns else 0
                put_oi = puts['openInterest'].sum() if 'openInterest' in puts.columns else 0
                
                # Handle NaN values
                call_vol = 0 if pd.isna(call_vol) else int(call_vol)
                put_vol = 0 if pd.isna(put_vol) else int(put_vol)
                call_oi = 0 if pd.isna(call_oi) else int(call_oi)
                put_oi = 0 if pd.isna(put_oi) else int(put_oi)
                
                total_call_volume += call_vol
                total_put_volume += put_vol
                total_call_oi += call_oi
                total_put_oi += put_oi
                
                # Find unusual activity (high volume relative to OI)
                for _, row in calls.iterrows():
                    vol = row.get('volume', 0)
                    oi = row.get('openInterest', 0)
                    if pd.notna(vol) and pd.notna(oi) and oi > 0 and vol > oi * 0.5 and vol > 100:
                        unusual_activity.append({
                            'type': 'CALL',
                            'expiration': exp_date,
                            'strike': row['strike'],
                            'volume': int(vol),
                            'openInterest': int(oi),
                            'vol_oi_ratio': round(vol / oi, 2) if oi > 0 else 0,
                            'impliedVolatility': round(row.get('impliedVolatility', 0) * 100, 1)
                        })
                
                for _, row in puts.iterrows():
                    vol = row.get('volume', 0)
                    oi = row.get('openInterest', 0)
                    if pd.notna(vol) and pd.notna(oi) and oi > 0 and vol > oi * 0.5 and vol > 100:
                        unusual_activity.append({
                            'type': 'PUT',
                            'expiration': exp_date,
                            'strike': row['strike'],
                            'volume': int(vol),
                            'openInterest': int(oi),
                            'vol_oi_ratio': round(vol / oi, 2) if oi > 0 else 0,
                            'impliedVolatility': round(row.get('impliedVolatility', 0) * 100, 1)
                        })
                        
            except Exception as e:
                report += f"*Error fetching {exp_date}: {str(e)}*\n"
                continue
        
        # Calculate put/call ratios
        pc_volume_ratio = round(total_put_volume / total_call_volume, 3) if total_call_volume > 0 else 0
        pc_oi_ratio = round(total_put_oi / total_call_oi, 3) if total_call_oi > 0 else 0
        
        # Summary
        report += "### Summary\n"
        report += "| Metric | Calls | Puts | Put/Call Ratio |\n"
        report += "|--------|-------|------|----------------|\n"
        report += f"| Volume | {total_call_volume:,} | {total_put_volume:,} | {pc_volume_ratio} |\n"
        report += f"| Open Interest | {total_call_oi:,} | {total_put_oi:,} | {pc_oi_ratio} |\n\n"
        
        # Sentiment interpretation
        report += "### Sentiment Analysis\n"
        if pc_volume_ratio < 0.7:
            report += "- **Volume P/C Ratio:** Bullish (more call volume)\n"
        elif pc_volume_ratio > 1.3:
            report += "- **Volume P/C Ratio:** Bearish (more put volume)\n"
        else:
            report += "- **Volume P/C Ratio:** Neutral\n"
            
        if pc_oi_ratio < 0.7:
            report += "- **OI P/C Ratio:** Bullish positioning\n"
        elif pc_oi_ratio > 1.3:
            report += "- **OI P/C Ratio:** Bearish positioning\n"
        else:
            report += "- **OI P/C Ratio:** Neutral positioning\n"
        
        # Unusual activity
        if unusual_activity:
            # Sort by volume/OI ratio
            unusual_activity.sort(key=lambda x: x['vol_oi_ratio'], reverse=True)
            top_unusual = unusual_activity[:10]
            
            report += "\n### Unusual Activity (High Volume vs Open Interest)\n"
            report += "| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |\n"
            report += "|------|--------|--------|--------|----|---------|----|---|\n"
            for item in top_unusual:
                report += f"| {item['type']} | {item['expiration']} | ${item['strike']} | {item['volume']:,} | {item['openInterest']:,} | {item['vol_oi_ratio']}x | {item['impliedVolatility']}% |\n"
        else:
            report += "\n*No unusual options activity detected.*\n"
        
        return report
        
    except Exception as e:
        return f"Error retrieving options activity for {ticker}: {str(e)}"