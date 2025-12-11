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


def get_technical_analysis(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date, YYYY-mm-dd"],
) -> str:
    """
    Get a concise technical analysis summary with key indicators, signals, and trend interpretation.
    
    Returns analysis-ready output instead of verbose day-by-day data.
    """
    from .config import get_config
    from stockstats import wrap
    
    # Default indicators to analyze
    indicators = ["rsi", "stoch", "macd", "adx", "close_20_ema", "close_50_sma", "close_200_sma", "boll", "atr", "obv", "vwap", "fib"]
    
    # Fetch price data (last 60 days for indicator calculation)
    curr_date_dt = pd.to_datetime(curr_date)
    start_date = curr_date_dt - pd.DateOffset(days=200)  # Need enough history for 200 SMA
    
    try:
        data = yf.download(
            symbol,
            start=start_date.strftime("%Y-%m-%d"),
            end=curr_date_dt.strftime("%Y-%m-%d"),
            multi_level_index=False,
            progress=False,
            auto_adjust=True,
        )
        
        if data.empty:
            return f"No data found for {symbol}"
        
        data = data.reset_index()
        df = wrap(data)
        
        # Get latest values
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        prev_5 = df.iloc[-5] if len(df) > 5 else latest
        
        current_price = float(latest['close'])
        
        # Build analysis
        analysis = []
        analysis.append(f"# Technical Analysis for {symbol.upper()}")
        analysis.append(f"**Date:** {curr_date}")
        analysis.append(f"**Current Price:** ${current_price:.2f}")
        analysis.append("")
        
        # Price action summary
        daily_change = ((current_price - float(prev['close'])) / float(prev['close'])) * 100
        weekly_change = ((current_price - float(prev_5['close'])) / float(prev_5['close'])) * 100
        analysis.append(f"## Price Action")
        analysis.append(f"- **Daily Change:** {daily_change:+.2f}%")
        analysis.append(f"- **5-Day Change:** {weekly_change:+.2f}%")
        analysis.append("")
        
        # RSI Analysis
        if 'rsi' in indicators:
            try:
                df['rsi']  # Trigger calculation
                rsi = float(df.iloc[-1]['rsi'])
                rsi_prev = float(df.iloc[-5]['rsi']) if len(df) > 5 else rsi
                
                if rsi > 70:
                    rsi_signal = "OVERBOUGHT ⚠️"
                elif rsi < 30:
                    rsi_signal = "OVERSOLD ⚡"
                elif rsi > 50:
                    rsi_signal = "Bullish"
                else:
                    rsi_signal = "Bearish"
                    
                rsi_trend = "↑" if rsi > rsi_prev else "↓"
                analysis.append(f"## RSI (14)")
                analysis.append(f"- **Value:** {rsi:.1f} {rsi_trend}")
                analysis.append(f"- **Signal:** {rsi_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # MACD Analysis
        if 'macd' in indicators:
            try:
                df['macd']
                df['macds']
                df['macdh']
                macd = float(df.iloc[-1]['macd'])
                signal = float(df.iloc[-1]['macds'])
                histogram = float(df.iloc[-1]['macdh'])
                hist_prev = float(df.iloc[-2]['macdh']) if len(df) > 1 else histogram
                
                if macd > signal and histogram > 0:
                    macd_signal = "BULLISH CROSSOVER ⚡" if histogram > hist_prev else "Bullish"
                elif macd < signal and histogram < 0:
                    macd_signal = "BEARISH CROSSOVER ⚠️" if histogram < hist_prev else "Bearish"
                else:
                    macd_signal = "Neutral"
                
                momentum = "Strengthening ↑" if abs(histogram) > abs(hist_prev) else "Weakening ↓"
                analysis.append(f"## MACD")
                analysis.append(f"- **MACD Line:** {macd:.3f}")
                analysis.append(f"- **Signal Line:** {signal:.3f}")
                analysis.append(f"- **Histogram:** {histogram:.3f} ({momentum})")
                analysis.append(f"- **Signal:** {macd_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # Moving Averages
        if 'close_50_sma' in indicators or 'close_200_sma' in indicators:
            try:
                df['close_50_sma']
                df['close_200_sma']
                sma_50 = float(df.iloc[-1]['close_50_sma'])
                sma_200 = float(df.iloc[-1]['close_200_sma'])
                
                # Trend determination
                if current_price > sma_50 > sma_200:
                    trend = "STRONG UPTREND ⚡"
                elif current_price > sma_50:
                    trend = "Uptrend"
                elif current_price < sma_50 < sma_200:
                    trend = "STRONG DOWNTREND ⚠️"
                elif current_price < sma_50:
                    trend = "Downtrend"
                else:
                    trend = "Sideways"
                
                # Golden/Death cross detection
                sma_50_prev = float(df.iloc[-5]['close_50_sma']) if len(df) > 5 else sma_50
                sma_200_prev = float(df.iloc[-5]['close_200_sma']) if len(df) > 5 else sma_200
                
                cross = ""
                if sma_50 > sma_200 and sma_50_prev < sma_200_prev:
                    cross = " (GOLDEN CROSS ⚡)"
                elif sma_50 < sma_200 and sma_50_prev > sma_200_prev:
                    cross = " (DEATH CROSS ⚠️)"
                
                analysis.append(f"## Moving Averages")
                analysis.append(f"- **50 SMA:** ${sma_50:.2f} ({'+' if current_price > sma_50 else ''}{((current_price - sma_50) / sma_50 * 100):.1f}% from price)")
                analysis.append(f"- **200 SMA:** ${sma_200:.2f} ({'+' if current_price > sma_200 else ''}{((current_price - sma_200) / sma_200 * 100):.1f}% from price)")
                analysis.append(f"- **Trend:** {trend}{cross}")
                analysis.append("")
            except Exception as e:
                pass
        
        # Bollinger Bands
        if 'boll' in indicators:
            try:
                df['boll']
                df['boll_ub']
                df['boll_lb']
                middle = float(df.iloc[-1]['boll'])
                upper = float(df.iloc[-1]['boll_ub'])
                lower = float(df.iloc[-1]['boll_lb'])
                
                # Position within bands (0 = lower, 1 = upper)
                band_position = (current_price - lower) / (upper - lower) if upper != lower else 0.5
                
                if band_position > 0.95:
                    bb_signal = "AT UPPER BAND - Potential reversal ⚠️"
                elif band_position < 0.05:
                    bb_signal = "AT LOWER BAND - Potential bounce ⚡"
                elif band_position > 0.8:
                    bb_signal = "Near upper band"
                elif band_position < 0.2:
                    bb_signal = "Near lower band"
                else:
                    bb_signal = "Within bands"
                
                bandwidth = ((upper - lower) / middle) * 100
                analysis.append(f"## Bollinger Bands (20,2)")
                analysis.append(f"- **Upper:** ${upper:.2f}")
                analysis.append(f"- **Middle:** ${middle:.2f}")
                analysis.append(f"- **Lower:** ${lower:.2f}")
                analysis.append(f"- **Band Position:** {band_position:.0%}")
                analysis.append(f"- **Bandwidth:** {bandwidth:.1f}% (volatility indicator)")
                analysis.append(f"- **Signal:** {bb_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # ATR (Volatility)
        if 'atr' in indicators:
            try:
                df['atr']
                atr = float(df.iloc[-1]['atr'])
                atr_pct = (atr / current_price) * 100
                
                if atr_pct > 5:
                    vol_level = "HIGH VOLATILITY ⚠️"
                elif atr_pct > 2:
                    vol_level = "Moderate volatility"
                else:
                    vol_level = "Low volatility"
                
                analysis.append(f"## ATR (Volatility)")
                analysis.append(f"- **ATR:** ${atr:.2f} ({atr_pct:.1f}% of price)")
                analysis.append(f"- **Level:** {vol_level}")
                analysis.append(f"- **Suggested Stop-Loss:** ${current_price - (1.5 * atr):.2f} (1.5x ATR)")
                analysis.append("")
            except Exception as e:
                pass
        
        # Stochastic Oscillator
        if 'stoch' in indicators:
            try:
                df['kdjk']  # Stochastic %K
                df['kdjd']  # Stochastic %D
                stoch_k = float(df.iloc[-1]['kdjk'])
                stoch_d = float(df.iloc[-1]['kdjd'])
                stoch_k_prev = float(df.iloc[-2]['kdjk']) if len(df) > 1 else stoch_k
                
                if stoch_k > 80 and stoch_d > 80:
                    stoch_signal = "OVERBOUGHT ⚠️"
                elif stoch_k < 20 and stoch_d < 20:
                    stoch_signal = "OVERSOLD ⚡"
                elif stoch_k > stoch_d and stoch_k_prev < stoch_d:
                    stoch_signal = "Bullish crossover ⚡"
                elif stoch_k < stoch_d and stoch_k_prev > stoch_d:
                    stoch_signal = "Bearish crossover ⚠️"
                elif stoch_k > 50:
                    stoch_signal = "Bullish"
                else:
                    stoch_signal = "Bearish"
                
                analysis.append(f"## Stochastic (14,3,3)")
                analysis.append(f"- **%K:** {stoch_k:.1f}")
                analysis.append(f"- **%D:** {stoch_d:.1f}")
                analysis.append(f"- **Signal:** {stoch_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # ADX (Trend Strength)
        if 'adx' in indicators:
            try:
                df['adx']
                df['dx']  
                adx = float(df.iloc[-1]['adx'])
                adx_prev = float(df.iloc[-5]['adx']) if len(df) > 5 else adx
                
                if adx > 50:
                    trend_strength = "VERY STRONG TREND ⚡"
                elif adx > 25:
                    trend_strength = "Strong trend"
                elif adx > 20:
                    trend_strength = "Trending"
                else:
                    trend_strength = "WEAK/NO TREND (range-bound) ⚠️"
                
                adx_direction = "Strengthening ↑" if adx > adx_prev else "Weakening ↓"
                analysis.append(f"## ADX (Trend Strength)")
                analysis.append(f"- **ADX:** {adx:.1f} ({adx_direction})")
                analysis.append(f"- **Interpretation:** {trend_strength}")
                analysis.append("")
            except Exception as e:
                pass
        
        # 20 EMA (Short-term trend)
        if 'close_20_ema' in indicators:
            try:
                df['close_20_ema']
                ema_20 = float(df.iloc[-1]['close_20_ema'])
                
                pct_from_ema = ((current_price - ema_20) / ema_20) * 100
                if current_price > ema_20:
                    ema_signal = "Price ABOVE 20 EMA (short-term bullish)"
                else:
                    ema_signal = "Price BELOW 20 EMA (short-term bearish)"
                
                analysis.append(f"## 20 EMA")
                analysis.append(f"- **Value:** ${ema_20:.2f} ({pct_from_ema:+.1f}% from price)")
                analysis.append(f"- **Signal:** {ema_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # OBV (On-Balance Volume)
        if 'obv' in indicators:
            try:
                # Calculate OBV manually since stockstats may not have it
                obv = 0
                obv_values = [0]
                for i in range(1, len(df)):
                    if float(df.iloc[i]['close']) > float(df.iloc[i-1]['close']):
                        obv += float(df.iloc[i]['volume'])
                    elif float(df.iloc[i]['close']) < float(df.iloc[i-1]['close']):
                        obv -= float(df.iloc[i]['volume'])
                    obv_values.append(obv)
                
                current_obv = obv_values[-1]
                obv_5_ago = obv_values[-5] if len(obv_values) > 5 else obv_values[0]
                
                if current_obv > obv_5_ago and current_price > float(df.iloc[-5]['close']):
                    obv_signal = "Confirmed uptrend (price & volume rising)"
                elif current_obv < obv_5_ago and current_price < float(df.iloc[-5]['close']):
                    obv_signal = "Confirmed downtrend (price & volume falling)"
                elif current_obv > obv_5_ago and current_price < float(df.iloc[-5]['close']):
                    obv_signal = "BULLISH DIVERGENCE ⚡ (accumulation)"
                elif current_obv < obv_5_ago and current_price > float(df.iloc[-5]['close']):
                    obv_signal = "BEARISH DIVERGENCE ⚠️ (distribution)"
                else:
                    obv_signal = "Neutral"
                
                obv_formatted = f"{current_obv/1e6:.1f}M" if abs(current_obv) > 1e6 else f"{current_obv/1e3:.1f}K"
                analysis.append(f"## OBV (On-Balance Volume)")
                analysis.append(f"- **Value:** {obv_formatted}")
                analysis.append(f"- **5-Day Trend:** {'Rising ↑' if current_obv > obv_5_ago else 'Falling ↓'}")
                analysis.append(f"- **Signal:** {obv_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # VWAP (Volume Weighted Average Price)
        if 'vwap' in indicators:
            try:
                # Calculate VWAP for today (simplified - using recent data)
                typical_price = (float(df.iloc[-1]['high']) + float(df.iloc[-1]['low']) + float(df.iloc[-1]['close'])) / 3
                
                # Calculate cumulative VWAP (last 20 periods approximation)
                recent_df = df.tail(20)
                tp_vol = ((recent_df['high'] + recent_df['low'] + recent_df['close']) / 3) * recent_df['volume']
                vwap = float(tp_vol.sum() / recent_df['volume'].sum())
                
                pct_from_vwap = ((current_price - vwap) / vwap) * 100
                if current_price > vwap:
                    vwap_signal = "Price ABOVE VWAP (institutional buying)"
                else:
                    vwap_signal = "Price BELOW VWAP (institutional selling)"
                
                analysis.append(f"## VWAP (20-period)")
                analysis.append(f"- **VWAP:** ${vwap:.2f}")
                analysis.append(f"- **Current vs VWAP:** {pct_from_vwap:+.1f}%")
                analysis.append(f"- **Signal:** {vwap_signal}")
                analysis.append("")
            except Exception as e:
                pass
        
        # Fibonacci Retracement Levels
        if 'fib' in indicators:
            try:
                # Get high and low from last 50 periods
                recent_high = float(df.tail(50)['high'].max())
                recent_low = float(df.tail(50)['low'].min())
                diff = recent_high - recent_low
                
                fib_levels = {
                    "0.0% (High)": recent_high,
                    "23.6%": recent_high - (diff * 0.236),
                    "38.2%": recent_high - (diff * 0.382),
                    "50.0%": recent_high - (diff * 0.5),
                    "61.8%": recent_high - (diff * 0.618),
                    "78.6%": recent_high - (diff * 0.786),
                    "100% (Low)": recent_low,
                }
                
                # Find nearest support and resistance
                support = None
                resistance = None
                for level_name, level_price in fib_levels.items():
                    if level_price < current_price and (support is None or level_price > support[1]):
                        support = (level_name, level_price)
                    if level_price > current_price and (resistance is None or level_price < resistance[1]):
                        resistance = (level_name, level_price)
                
                analysis.append(f"## Fibonacci Levels (50-period)")
                analysis.append(f"- **Recent High:** ${recent_high:.2f}")
                analysis.append(f"- **Recent Low:** ${recent_low:.2f}")
                if resistance:
                    analysis.append(f"- **Next Resistance:** ${resistance[1]:.2f} ({resistance[0]})")
                if support:
                    analysis.append(f"- **Next Support:** ${support[1]:.2f} ({support[0]})")
                analysis.append("")
            except Exception as e:
                pass
        
        # Overall Summary
        analysis.append("## Summary")
        signals = []
        
        # Collect all signals for summary
        try:
            rsi = float(df.iloc[-1]['rsi'])
            if rsi > 70:
                signals.append("RSI overbought")
            elif rsi < 30:
                signals.append("RSI oversold")
        except:
            pass
            
        try:
            if current_price > float(df.iloc[-1]['close_50_sma']):
                signals.append("Above 50 SMA")
            else:
                signals.append("Below 50 SMA")
        except:
            pass
        
        if signals:
            analysis.append(f"- **Key Signals:** {', '.join(signals)}")
        
        return "\n".join(analysis)
        
    except Exception as e:
        return f"Error analyzing {symbol}: {str(e)}"


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
    """Get insider transactions data from yfinance with parsed transaction types."""
    try:
        ticker_obj = yf.Ticker(ticker.upper())
        data = ticker_obj.insider_transactions
        
        if data is None or data.empty:
            return f"No insider transactions data found for symbol '{ticker}'"
        
        # Parse the Text column to populate Transaction type
        def classify_transaction(text):
            if pd.isna(text) or text == '':
                return 'Unknown'
            text_lower = str(text).lower()
            if 'sale' in text_lower:
                return 'Sale'
            elif 'purchase' in text_lower or 'buy' in text_lower:
                return 'Purchase'
            elif 'gift' in text_lower:
                return 'Gift'
            elif 'exercise' in text_lower or 'option' in text_lower:
                return 'Option Exercise'
            elif 'award' in text_lower or 'grant' in text_lower:
                return 'Award/Grant'
            elif 'conversion' in text_lower:
                return 'Conversion'
            else:
                return 'Other'
        
        # Apply classification
        data['Transaction'] = data['Text'].apply(classify_transaction)
        
        # Calculate summary statistics
        transaction_counts = data['Transaction'].value_counts().to_dict()
        total_sales_value = data[data['Transaction'] == 'Sale']['Value'].sum()
        total_purchases_value = data[data['Transaction'] == 'Purchase']['Value'].sum()
        
        # Determine insider sentiment
        sales_count = transaction_counts.get('Sale', 0)
        purchases_count = transaction_counts.get('Purchase', 0)
        
        if purchases_count > sales_count:
            sentiment = "BULLISH ⚡ (more buying than selling)"
        elif sales_count > purchases_count * 2:
            sentiment = "BEARISH ⚠️ (significant insider selling)"
        elif sales_count > purchases_count:
            sentiment = "Slightly bearish (more selling than buying)"
        else:
            sentiment = "Neutral"
        
        # Build summary header
        header = f"# Insider Transactions for {ticker.upper()}\n"
        header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        header += "## Summary\n"
        header += f"- **Insider Sentiment:** {sentiment}\n"
        for tx_type, count in sorted(transaction_counts.items(), key=lambda x: -x[1]):
            header += f"- **{tx_type}:** {count} transactions\n"
        if total_sales_value > 0:
            header += f"- **Total Sales Value:** ${total_sales_value:,.0f}\n"
        if total_purchases_value > 0:
            header += f"- **Total Purchases Value:** ${total_purchases_value:,.0f}\n"
        header += "\n## Transaction Details\n\n"
        
        # Select key columns for output
        output_cols = ['Start Date', 'Insider', 'Position', 'Transaction', 'Shares', 'Value', 'Ownership']
        available_cols = [c for c in output_cols if c in data.columns]
        
        csv_string = data[available_cols].to_csv(index=False)
        
        return header + csv_string
        
    except Exception as e:
        return f"Error retrieving insider transactions for {ticker}: {str(e)}"

def validate_ticker(symbol: str) -> bool:
    """
    Validate if a ticker symbol exists and has trading data.
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        # Use fast_info for lighter validation (no historical download needed)
        # fast_info attributes are lazy-loaded
        _ = ticker.fast_info.get("lastPrice")
        return True
            
    except Exception:
        # Fallback to older method if fast_info fails or is missing
        try:
            return not ticker.history(period="1d", progress=False).empty
        except:
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