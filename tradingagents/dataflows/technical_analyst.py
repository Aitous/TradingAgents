from typing import List

import pandas as pd

from tradingagents.utils.logger import get_logger

logger = get_logger(__name__)


class TechnicalAnalyst:
    """
    Performs comprehensive technical analysis on stock data.
    """

    def __init__(self, df: pd.DataFrame, current_price: float):
        """
        Initialize with stock dataframe and current price.

        Args:
            df: DataFrame with stock data (must contain 'close', 'high', 'low', 'volume')
            current_price: The latest price of the stock
        """
        self.df = df
        self.current_price = current_price
        self.analysis_report = []

    def add_section(self, title: str, content: List[str]):
        """Add a formatted section to the report."""
        self.analysis_report.append(f"## {title}")
        self.analysis_report.extend(content)
        self.analysis_report.append("")

    def analyze_price_action(self):
        """Analyze recent price movements."""
        latest = self.df.iloc[-1]
        prev = self.df.iloc[-2] if len(self.df) > 1 else latest
        prev_5 = self.df.iloc[-5] if len(self.df) > 5 else latest

        daily_change = ((self.current_price - float(prev["close"])) / float(prev["close"])) * 100
        weekly_change = (
            (self.current_price - float(prev_5["close"])) / float(prev_5["close"])
        ) * 100

        self.add_section(
            "Price Action",
            [
                f"- **Daily Change:** {daily_change:+.2f}%",
                f"- **5-Day Change:** {weekly_change:+.2f}%",
            ],
        )

    def analyze_rsi(self):
        """Analyze Relative Strength Index."""
        try:
            self.df["rsi"]  # Trigger calculation
            rsi = float(self.df.iloc[-1]["rsi"])
            rsi_prev = float(self.df.iloc[-5]["rsi"]) if len(self.df) > 5 else rsi

            if rsi > 70:
                rsi_signal = "OVERBOUGHT ⚠️"
            elif rsi < 30:
                rsi_signal = "OVERSOLD ⚡"
            elif rsi > 50:
                rsi_signal = "Bullish"
            else:
                rsi_signal = "Bearish"

            rsi_trend = "↑" if rsi > rsi_prev else "↓"

            self.add_section(
                "RSI (14)", [f"- **Value:** {rsi:.1f} {rsi_trend}", f"- **Signal:** {rsi_signal}"]
            )
        except Exception as e:
            logger.warning(f"RSI analysis failed: {e}")

    def analyze_macd(self):
        """Analyze MACD."""
        try:
            self.df["macd"]
            self.df["macds"]
            self.df["macdh"]
            macd = float(self.df.iloc[-1]["macd"])
            signal = float(self.df.iloc[-1]["macds"])
            histogram = float(self.df.iloc[-1]["macdh"])
            hist_prev = float(self.df.iloc[-2]["macdh"]) if len(self.df) > 1 else histogram

            if macd > signal and histogram > 0:
                macd_signal = "BULLISH CROSSOVER ⚡" if histogram > hist_prev else "Bullish"
            elif macd < signal and histogram < 0:
                macd_signal = "BEARISH CROSSOVER ⚠️" if histogram < hist_prev else "Bearish"
            else:
                macd_signal = "Neutral"

            momentum = "Strengthening ↑" if abs(histogram) > abs(hist_prev) else "Weakening ↓"

            self.add_section(
                "MACD",
                [
                    f"- **MACD Line:** {macd:.3f}",
                    f"- **Signal Line:** {signal:.3f}",
                    f"- **Histogram:** {histogram:.3f} ({momentum})",
                    f"- **Signal:** {macd_signal}",
                ],
            )
        except Exception as e:
            logger.warning(f"MACD analysis failed: {e}")

    def analyze_moving_averages(self):
        """Analyze Moving Averages."""
        try:
            self.df["close_50_sma"]
            self.df["close_200_sma"]
            sma_50 = float(self.df.iloc[-1]["close_50_sma"])
            sma_200 = float(self.df.iloc[-1]["close_200_sma"])

            # Trend determination
            if self.current_price > sma_50 > sma_200:
                trend = "STRONG UPTREND ⚡"
            elif self.current_price > sma_50:
                trend = "Uptrend"
            elif self.current_price < sma_50 < sma_200:
                trend = "STRONG DOWNTREND ⚠️"
            elif self.current_price < sma_50:
                trend = "Downtrend"
            else:
                trend = "Sideways"

            # Golden/Death cross detection
            sma_50_prev = float(self.df.iloc[-5]["close_50_sma"]) if len(self.df) > 5 else sma_50
            sma_200_prev = float(self.df.iloc[-5]["close_200_sma"]) if len(self.df) > 5 else sma_200

            cross = ""
            if sma_50 > sma_200 and sma_50_prev < sma_200_prev:
                cross = " (GOLDEN CROSS ⚡)"
            elif sma_50 < sma_200 and sma_50_prev > sma_200_prev:
                cross = " (DEATH CROSS ⚠️)"

            self.add_section(
                "Moving Averages",
                [
                    f"- **50 SMA:** ${sma_50:.2f} ({'+' if self.current_price > sma_50 else ''}{((self.current_price - sma_50) / sma_50 * 100):.1f}% from price)",
                    f"- **200 SMA:** ${sma_200:.2f} ({'+' if self.current_price > sma_200 else ''}{((self.current_price - sma_200) / sma_200 * 100):.1f}% from price)",
                    f"- **Trend:** {trend}{cross}",
                ],
            )
        except Exception as e:
            logger.warning(f"Moving averages analysis failed: {e}")

    def analyze_bollinger_bands(self):
        """Analyze Bollinger Bands."""
        try:
            self.df["boll"]
            self.df["boll_ub"]
            self.df["boll_lb"]
            middle = float(self.df.iloc[-1]["boll"])
            upper = float(self.df.iloc[-1]["boll_ub"])
            lower = float(self.df.iloc[-1]["boll_lb"])

            band_position = (
                (self.current_price - lower) / (upper - lower) if upper != lower else 0.5
            )

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

            self.add_section(
                "Bollinger Bands (20,2)",
                [
                    f"- **Upper:** ${upper:.2f}",
                    f"- **Middle:** ${middle:.2f}",
                    f"- **Lower:** ${lower:.2f}",
                    f"- **Band Position:** {band_position:.0%}",
                    f"- **Bandwidth:** {bandwidth:.1f}% (volatility indicator)",
                    f"- **Signal:** {bb_signal}",
                ],
            )
        except Exception as e:
            logger.warning(f"Bollinger bands analysis failed: {e}")

    def analyze_atr(self):
        """Analyze ATR (Volatility)."""
        try:
            self.df["atr"]
            atr = float(self.df.iloc[-1]["atr"])
            atr_pct = (atr / self.current_price) * 100

            if atr_pct > 5:
                vol_level = "HIGH VOLATILITY ⚠️"
            elif atr_pct > 2:
                vol_level = "Moderate volatility"
            else:
                vol_level = "Low volatility"

            self.add_section(
                "ATR (Volatility)",
                [
                    f"- **ATR:** ${atr:.2f} ({atr_pct:.1f}% of price)",
                    f"- **Level:** {vol_level}",
                    f"- **Suggested Stop-Loss:** ${self.current_price - (1.5 * atr):.2f} (1.5x ATR)",
                ],
            )
        except Exception as e:
            logger.warning(f"ATR analysis failed: {e}")

    def analyze_stochastic(self):
        """Analyze Stochastic Oscillator."""
        try:
            self.df["kdjk"]
            self.df["kdjd"]
            stoch_k = float(self.df.iloc[-1]["kdjk"])
            stoch_d = float(self.df.iloc[-1]["kdjd"])
            stoch_k_prev = float(self.df.iloc[-2]["kdjk"]) if len(self.df) > 1 else stoch_k

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

            self.add_section(
                "Stochastic (14,3,3)",
                [
                    f"- **%K:** {stoch_k:.1f}",
                    f"- **%D:** {stoch_d:.1f}",
                    f"- **Signal:** {stoch_signal}",
                ],
            )
        except Exception as e:
            logger.warning(f"Stochastic analysis failed: {e}")

    def analyze_adx(self):
        """Analyze ADX (Trend Strength)."""
        try:
            self.df["adx"]
            adx = float(self.df.iloc[-1]["adx"])
            adx_prev = float(self.df.iloc[-5]["adx"]) if len(self.df) > 5 else adx

            if adx > 50:
                trend_strength = "VERY STRONG TREND ⚡"
            elif adx > 25:
                trend_strength = "Strong trend"
            elif adx > 20:
                trend_strength = "Trending"
            else:
                trend_strength = "WEAK/NO TREND (range-bound) ⚠️"

            adx_direction = "Strengthening ↑" if adx > adx_prev else "Weakening ↓"

            self.add_section(
                "ADX (Trend Strength)",
                [
                    f"- **ADX:** {adx:.1f} ({adx_direction})",
                    f"- **Interpretation:** {trend_strength}",
                ],
            )
        except Exception as e:
            logger.warning(f"ADX analysis failed: {e}")

    def analyze_ema(self):
        """Analyze 20 EMA."""
        try:
            self.df["close_20_ema"]
            ema_20 = float(self.df.iloc[-1]["close_20_ema"])

            pct_from_ema = ((self.current_price - ema_20) / ema_20) * 100
            if self.current_price > ema_20:
                ema_signal = "Price ABOVE 20 EMA (short-term bullish)"
            else:
                ema_signal = "Price BELOW 20 EMA (short-term bearish)"

            self.add_section(
                "20 EMA",
                [
                    f"- **Value:** ${ema_20:.2f} ({pct_from_ema:+.1f}% from price)",
                    f"- **Signal:** {ema_signal}",
                ],
            )
        except Exception as e:
            logger.warning(f"EMA analysis failed: {e}")

    def analyze_obv(self):
        """Analyze On-Balance Volume."""
        try:
            # Check if we have enough data
            if len(self.df) < 2:
                logger.warning("Insufficient data for OBV analysis (need at least 2 days)")
                return

            obv = 0
            obv_values = [0]
            for i in range(1, len(self.df)):
                if float(self.df.iloc[i]["close"]) > float(self.df.iloc[i - 1]["close"]):
                    obv += float(self.df.iloc[i]["volume"])
                elif float(self.df.iloc[i]["close"]) < float(self.df.iloc[i - 1]["close"]):
                    obv -= float(self.df.iloc[i]["volume"])
                obv_values.append(obv)

            current_obv = obv_values[-1]
            obv_5_ago = obv_values[-5] if len(obv_values) > 5 else obv_values[0]

            # Check if we have enough data for price comparison
            if len(self.df) >= 5:
                price_5_ago = float(self.df.iloc[-5]["close"])
            else:
                price_5_ago = float(self.df.iloc[0]["close"])

            if current_obv > obv_5_ago and self.current_price > price_5_ago:
                obv_signal = "Confirmed uptrend (price & volume rising)"
            elif current_obv < obv_5_ago and self.current_price < price_5_ago:
                obv_signal = "Confirmed downtrend (price & volume falling)"
            elif current_obv > obv_5_ago and self.current_price < price_5_ago:
                obv_signal = "BULLISH DIVERGENCE ⚡ (accumulation)"
            elif current_obv < obv_5_ago and self.current_price > price_5_ago:
                obv_signal = "BEARISH DIVERGENCE ⚠️ (distribution)"
            else:
                obv_signal = "Neutral"

            obv_formatted = (
                f"{current_obv/1e6:.1f}M" if abs(current_obv) > 1e6 else f"{current_obv/1e3:.1f}K"
            )

            self.add_section(
                "OBV (On-Balance Volume)",
                [
                    f"- **Value:** {obv_formatted}",
                    f"- **5-Day Trend:** {'Rising ↑' if current_obv > obv_5_ago else 'Falling ↓'}",
                    f"- **Signal:** {obv_signal}",
                ],
            )
        except Exception as e:
            logger.warning(f"OBV analysis failed: {e}")

    def analyze_vwap(self):
        """Analyze VWAP."""
        try:
            # Calculate VWAP for today (simplified - using recent data)
            # Calculate cumulative VWAP (last 20 periods approximation)
            recent_df = self.df.tail(20)
            tp_vol = ((recent_df["high"] + recent_df["low"] + recent_df["close"]) / 3) * recent_df[
                "volume"
            ]
            vwap = float(tp_vol.sum() / recent_df["volume"].sum())

            pct_from_vwap = ((self.current_price - vwap) / vwap) * 100
            if self.current_price > vwap:
                vwap_signal = "Price ABOVE VWAP (institutional buying)"
            else:
                vwap_signal = "Price BELOW VWAP (institutional selling)"

            self.add_section(
                "VWAP (20-period)",
                [
                    f"- **VWAP:** ${vwap:.2f}",
                    f"- **Current vs VWAP:** {pct_from_vwap:+.1f}%",
                    f"- **Signal:** {vwap_signal}",
                ],
            )
        except Exception as e:
            logger.warning(f"VWAP analysis failed: {e}")

    def analyze_fibonacci(self):
        """Analyze Fibonacci Retracement."""
        try:
            # Get high and low from last 50 periods
            recent_high = float(self.df.tail(50)["high"].max())
            recent_low = float(self.df.tail(50)["low"].min())
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
                if level_price < self.current_price and (
                    support is None or level_price > support[1]
                ):
                    support = (level_name, level_price)
                if level_price > self.current_price and (
                    resistance is None or level_price < resistance[1]
                ):
                    resistance = (level_name, level_price)

            content = [
                f"- **Recent High:** ${recent_high:.2f}",
                f"- **Recent Low:** ${recent_low:.2f}",
            ]
            if resistance:
                content.append(f"- **Next Resistance:** ${resistance[1]:.2f} ({resistance[0]})")
            if support:
                content.append(f"- **Next Support:** ${support[1]:.2f} ({support[0]})")

            self.add_section("Fibonacci Levels (50-period)", content)

        except Exception as e:
            logger.warning(f"Fibonacci analysis failed: {e}")

    def generate_summary(self):
        """Generate final summary section."""
        signals = []
        try:
            rsi = float(self.df.iloc[-1]["rsi"])
            if rsi > 70:
                signals.append("RSI overbought")
            elif rsi < 30:
                signals.append("RSI oversold")
        except Exception:
            pass

        try:
            if self.current_price > float(self.df.iloc[-1]["close_50_sma"]):
                signals.append("Above 50 SMA")
            else:
                signals.append("Below 50 SMA")
        except Exception:
            pass

        content = []
        if signals:
            content.append(f"- **Key Signals:** {', '.join(signals)}")

        self.add_section("Summary", content)

    def generate_report(self, symbol: str, date: str) -> str:
        """Run all analyses and generate the markdown report."""
        self.df = self.df.copy()  # Avoid modifying original

        # Header
        self.analysis_report = [
            f"# Technical Analysis for {symbol.upper()}",
            f"**Date:** {date}",
            f"**Current Price:** ${self.current_price:.2f}",
            "",
        ]

        # Run analyses
        self.analyze_price_action()
        self.analyze_rsi()
        self.analyze_macd()
        self.analyze_moving_averages()
        self.analyze_bollinger_bands()
        self.analyze_atr()
        self.analyze_stochastic()
        self.analyze_adx()
        self.analyze_ema()
        self.analyze_obv()
        self.analyze_vwap()
        self.analyze_fibonacci()
        self.generate_summary()

        return "\n".join(self.analysis_report)
