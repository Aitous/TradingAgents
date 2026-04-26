# Tool Execution Logs - 2026-04-25

## 1. Check average volume (batch)

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:46:57.012061
- **Tool:** `get_average_volume_batch`
- **Parameters:** `{'symbols': ['COE', 'PRLD', 'TORO', 'ALMR', 'HTO', 'NKE', 'ABX', 'ACHC', 'BEAM', 'ARCT', 'AI', 'INTC', 'VNCE', 'PINE', 'DTST', 'NTST', 'DLR', 'PECO', 'KOPN', 'TAL', 'CCI', 'TXN', 'MASI', 'VOYA', 'PENN', 'WST', 'URI', 'JKHY', 'BBY', 'ADI', 'SVRE', 'SMC', 'LNBIX', 'THM', 'MESO', 'AVEX', 'BMI', 'BORR', 'NE', 'ON', 'NEE', 'MCHP', 'R', 'UNP', 'ARMK', 'FIX', 'SNA', 'MOH', 'SEIC', 'PTC', 'OKTA', 'KLRA', 'ARXS', 'TSLA', 'AMD', 'NVDA', 'AMZN', 'META', 'AAPL', 'GOOGL', 'MSFT', 'QCOM', 'AVGO', 'PLTR', 'GOOG', 'VZ', 'CBK', 'LTBR', 'CDP', 'BLX', 'DEA', 'KRC', 'FULC', 'ARLP', 'ARE', 'KEYS', 'HXL', 'WSC', 'HOLX', 'EXAS', 'EA', 'CPRT', 'DRI', 'CPB', 'DTE', 'POET', 'RDDT', 'IV'], 'lookback_days': 10, 'curr_date': '2026-04-25', 'cache_key': 'default'}`
- **Output Length:** 14228 chars

### Output
```
{"COE": {"symbol": "COE", "average_volume": 9020.0, "latest_volume": 7700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "PRLD": {"symbol": "PRLD", "average_volume": 583400.0, "latest_volume": 864700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "TORO": {"symbol": "TORO", "average_volume": 2410200.0, "latest_volume": 410000.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ALMR": {"symbol": "ALMR", "average_volume": 1276500.0, "latest_volume": 707500.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "HTO": {"symbol": "HTO", "average_volume": 769090.0, "latest_volume": 1489900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "NKE": {"symbol": "NKE", "average_volume": 22087000.0, "latest_volume": 15859200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ABX": {"symbol": "ABX", "average_volume": 465680.0, "latest_volume": 420700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ACHC": {"symbol": "ACHC", "average_volume": 2568710.0, "latest_volume": 2961600.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "BEAM": {"symbol": "BEAM", "average_volume": 2312020.0, "latest_volume": 4712300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ARCT": {"symbol": "ARCT", "average_volume": 333260.0, "latest_volume": 266300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "AI": {"symbol": "AI", "average_volume": 5198240.0, "latest_volume": 4160800.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "INTC": {"symbol": "INTC", "average_volume": 125829080.0, "latest_volume": 280509900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "VNCE": {"symbol": "VNCE", "average_volume": 437590.0, "latest_volume": 535200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "PINE": {"symbol": "PINE", "average_volume": 117180.0, "latest_volume": 216700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "DTST": {"symbol": "DTST", "average_volume": 79460.0, "latest_volume": 23400.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "NTST": {"symbol": "NTST", "average_volume": 1556710.0, "latest_volume": 945300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "DLR": {"symbol": "DLR", "average_volume": 2155630.0, "latest_volume": 4627500.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "PECO": {"symbol": "PECO", "average_volume": 898940.0, "latest_volume": 1530200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "KOPN": {"symbol": "KOPN", "average_volume": 4906660.0, "latest_volume": 8289400.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "TAL": {"symbol": "TAL", "average_volume": 4972290.0, "latest_volume": 3651300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "CCI": {"symbol": "CCI", "average_volume": 2940870.0, "latest_volume": 2893800.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "TXN": {"symbol": "TXN", "average_volume": 9240450.0, "latest_volume": 13939700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "MASI": {"symbol": "MASI", "average_volume": 548350.0, "latest_volume": 576600.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "VOYA": {"symbol": "VOYA", "average_volume": 1508740.0, "latest_volume": 1724100.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "PENN": {"symbol": "PENN", "average_volume": 5137190.0, "latest_volume": 7882200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "WST": {"symbol": "WST", "average_volume": 1085830.0, "latest_volume": 1286000.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "URI": {"symbol": "URI", "average_volume": 658580.0, "latest_volume": 1043600.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "JKHY": {"symbol": "JKHY", "average_volume": 867190.0, "latest_volume": 685300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "BBY": {"symbol": "BBY", "average_volume": 3520600.0, "latest_volume": 3095300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ADI": {"symbol": "ADI", "average_volume": 3347360.0, "latest_volume": 4512300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "SVRE": {"symbol": "SVRE", "average_volume": 47450.0, "latest_volume": 57100.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "SMC": {"symbol": "SMC", "average_volume": 40150.0, "latest_volume": 32800.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "LNBIX": {"symbol": "LNBIX", "average_volume": 0.0, "latest_volume": 0.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "THM": {"symbol": "THM", "average_volume": 1195420.0, "latest_volume": 1512600.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "MESO": {"symbol": "MESO", "average_volume": 224750.0, "latest_volume": 87800.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "AVEX": {"symbol": "AVEX", "average_volume": 8440600.0, "latest_volume": 2467000.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "BMI": {"symbol": "BMI", "average_volume": 1309720.0, "latest_volume": 1139900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "BORR": {"symbol": "BORR", "average_volume": 8515520.0, "latest_volume": 4321500.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "NE": {"symbol": "NE", "average_volume": 1324750.0, "latest_volume": 1076000.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ON": {"symbol": "ON", "average_volume": 12152580.0, "latest_volume": 13366900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "NEE": {"symbol": "NEE", "average_volume": 9363380.0, "latest_volume": 9761200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "MCHP": {"symbol": "MCHP", "average_volume": 9572340.0, "latest_volume": 11341200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "R": {"symbol": "R", "average_volume": 518060.0, "latest_volume": 535300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "UNP": {"symbol": "UNP", "average_volume": 3570650.0, "latest_volume": 3565700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ARMK": {"symbol": "ARMK", "average_volume": 2783130.0, "latest_volume": 2133700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "FIX": {"symbol": "FIX", "average_volume": 428390.0, "latest_volume": 832000.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "SNA": {"symbol": "SNA", "average_volume": 475480.0, "latest_volume": 590900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "MOH": {"symbol": "MOH", "average_volume": 1549480.0, "latest_volume": 1773800.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "SEIC": {"symbol": "SEIC", "average_volume": 1345410.0, "latest_volume": 1516900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "PTC": {"symbol": "PTC", "average_volume": 1179740.0, "latest_volume": 1358700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "OKTA": {"symbol": "OKTA", "average_volume": 4291880.0, "latest_volume": 2667300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "KLRA": {"symbol": "KLRA", "average_volume": 2732633.3333333335, "latest_volume": 1209700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "ARXS": {"symbol": "ARXS", "average_volume": 5021757.142857143, "latest_volume": 1507600.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "TSLA": {"symbol": "TSLA", "average_volume": 71578310.0, "latest_volume": 62753700.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "AMD": {"symbol": "AMD", "average_volume": 42346890.0, "latest_volume": 81429300.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "NVDA": {"symbol": "NVDA", "average_volume": 143680050.0, "latest_volume": 213780100.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "AMZN": {"symbol": "AMZN", "average_volume": 46305860.0, "latest_volume": 53695200.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "META": {"symbol": "META", "average_volume": 12354790.0, "latest_volume": 13333400.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "AAPL": {"symbol": "AAPL", "average_volume": 44085150.0, "latest_volume": 38124500.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "GOOGL": {"symbol": "GOOGL", "average_volume": 22488130.0, "latest_volume": 26400000.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "MSFT": {"symbol": "MSFT", "average_volume": 36325510.0, "latest_volume": 27413900.0, "lookback_days": 10, "source": "download", "cache_date": null, "error": null}, "QCOM": {"symbol": "QCOM", "average_volume": 11258440.0, "latest_volume": 29412300.0, "lookback_days": 10, "source": "download",... [truncated]
```

---

## 2. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:12.130261
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PRLD', 'curr_date': '2026-04-25'}`
- **Output Length:** 1485 chars

### Output
```
# Technical Analysis for PRLD
**Date:** 2026-04-25
**Current Price:** $4.70

## Price Action
- **Daily Change:** +6.82%
- **5-Day Change:** +4.21%

## RSI (14)
- **Value:** 60.0 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.359
- **Signal Line:** 0.375
- **Histogram:** -0.016 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $3.44 (+36.8% from price)
- **200 SMA:** $2.00 (+134.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $5.37
- **Middle:** $4.15
- **Lower:** $2.92
- **Band Position:** 73%
- **Bandwidth:** 59.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.54 (11.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $3.88 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 41.2
- **%D:** 45.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 44.8 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $4.20 (+12.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 18.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $4.44
- **Current vs VWAP:** +5.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $5.54
- **Recent Low:** $2.01
- **Next Resistance:** $4.71 (23.6%)
- **Next Support:** $4.19 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 3. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:12.180086
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PRLD'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for PRLD: Too Many Requests. Rate limited. Try after a while.
```

---

## 4. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:12.442270
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PRLD'}`
- **Output Length:** 707 chars

### Output
```
## Analyst Recommendation Trends for PRLD

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish


```

---

## 5. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:12.459548
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PRLD', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for PRLD: Too Many Requests. Rate limited. Try after a while.
```

---

## 6. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:13.220556
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TORO', 'curr_date': '2026-04-25'}`
- **Output Length:** 1567 chars

### Output
```
# Technical Analysis for TORO
**Date:** 2026-04-25
**Current Price:** $6.90

## Price Action
- **Daily Change:** +3.45%
- **5-Day Change:** +73.37%

## RSI (14)
- **Value:** 87.9 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.597
- **Signal Line:** 0.254
- **Histogram:** 0.342 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $3.92 (+75.9% from price)
- **200 SMA:** $3.03 (+127.6% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $6.44
- **Middle:** $4.27
- **Lower:** $2.09
- **Band Position:** 110%
- **Bandwidth:** 102.1% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.51 (7.4% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $6.14 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 70.8
- **%D:** 64.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 61.2 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $4.60 (+50.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 41.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $6.16
- **Current vs VWAP:** +12.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $7.70
- **Recent Low:** $3.37
- **Next Resistance:** $7.70 (0.0% (High))
- **Next Support:** $6.68 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 7. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:13.270696
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TORO'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for TORO: Too Many Requests. Rate limited. Try after a while.
```

---

## 8. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:13.566738
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TORO'}`
- **Output Length:** 44 chars

### Output
```
No recommendation trends data found for TORO
```

---

## 9. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:13.583144
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TORO', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for TORO: Too Many Requests. Rate limited. Try after a while.
```

---

## 10. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:14.270966
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ALMR', 'curr_date': '2026-04-25'}`
- **Output Length:** 1539 chars

### Output
```
# Technical Analysis for ALMR
**Date:** 2026-04-25
**Current Price:** $24.84

## Price Action
- **Daily Change:** +2.35%
- **5-Day Change:** +12.91%

## RSI (14)
- **Value:** 100.0 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.158
- **Signal Line:** 0.090
- **Histogram:** 0.068 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $23.44 (+6.0% from price)
- **200 SMA:** $23.44 (+6.0% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $25.89
- **Middle:** $23.44
- **Lower:** $20.98
- **Band Position:** 79%
- **Bandwidth:** 20.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.93 (7.8% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $21.95 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.5
- **%D:** 57.8
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 81.0 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $23.62 (+5.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $22.97
- **Current vs VWAP:** +8.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $25.03
- **Recent Low:** $21.50
- **Next Resistance:** $25.03 (0.0% (High))
- **Next Support:** $24.20 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 11. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:14.328541
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ALMR'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for ALMR: Too Many Requests. Rate limited. Try after a while.
```

---

## 12. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:14.583679
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ALMR'}`
- **Output Length:** 44 chars

### Output
```
No recommendation trends data found for ALMR
```

---

## 13. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:14.601542
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ALMR', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for ALMR: Too Many Requests. Rate limited. Try after a while.
```

---

## 14. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:15.411296
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'HTO', 'curr_date': '2026-04-25'}`
- **Output Length:** 1501 chars

### Output
```
# Technical Analysis for HTO
**Date:** 2026-04-25
**Current Price:** $59.72

## Price Action
- **Daily Change:** -2.59%
- **5-Day Change:** -0.73%

## RSI (14)
- **Value:** 55.0 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.812
- **Signal Line:** 0.764
- **Histogram:** 0.048 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $57.35 (+4.1% from price)
- **200 SMA:** $50.84 (+17.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $60.82
- **Middle:** $59.05
- **Lower:** $57.28
- **Band Position:** 69%
- **Bandwidth:** 6.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.69 (2.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $57.18 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 68.8
- **%D:** 67.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 22.4 (Strengthening ↑)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $58.97 (+1.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 6.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $58.96
- **Current vs VWAP:** +1.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $61.87
- **Recent Low:** $52.33
- **Next Resistance:** $61.87 (0.0% (High))
- **Next Support:** $59.62 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 15. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:15.461449
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'HTO'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for HTO: Too Many Requests. Rate limited. Try after a while.
```

---

## 16. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:15.743580
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'HTO'}`
- **Output Length:** 706 chars

### Output
```
## Analyst Recommendation Trends for HTO

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 4
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 4
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 8

**Sentiment**: 87.5% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 3
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 8

**Sentiment**: 87.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 3
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 8

**Sentiment**: 75.0% Bullish, 0.0% Bearish


```

---

## 17. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:15.759241
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'HTO', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for HTO: Too Many Requests. Rate limited. Try after a while.
```

---

## 18. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:16.539918
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NKE', 'curr_date': '2026-04-25'}`
- **Output Length:** 1524 chars

### Output
```
# Technical Analysis for NKE
**Date:** 2026-04-25
**Current Price:** $44.69

## Price Action
- **Daily Change:** -0.20%
- **5-Day Change:** -3.85%

## RSI (14)
- **Value:** 35.2 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.168
- **Signal Line:** -2.711
- **Histogram:** 0.544 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $53.23 (-16.0% from price)
- **200 SMA:** $64.08 (-30.3% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $51.47
- **Middle:** $45.65
- **Lower:** $39.83
- **Band Position:** 42%
- **Bandwidth:** 25.5% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.50 (3.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $42.44 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 61.0
- **%D:** 65.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 28.7 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $46.67 (-4.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -414.9M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $45.43
- **Current vs VWAP:** -1.6%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $68.04
- **Recent Low:** $42.09
- **Next Resistance:** $47.64 (78.6%)
- **Next Support:** $42.09 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 19. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:16.588448
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NKE'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for NKE: Too Many Requests. Rate limited. Try after a while.
```

---

## 20. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:16.833863
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NKE'}`
- **Output Length:** 722 chars

### Output
```
## Analyst Recommendation Trends for NKE

### 2026-04-01
- **Strong Buy**: 11
- **Buy**: 16
- **Hold**: 14
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 62.8% Bullish, 4.7% Bearish

### 2026-03-01
- **Strong Buy**: 11
- **Buy**: 15
- **Hold**: 16
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 59.1% Bullish, 4.5% Bearish

### 2026-02-01
- **Strong Buy**: 11
- **Buy**: 15
- **Hold**: 16
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 59.1% Bullish, 4.5% Bearish

### 2026-01-01
- **Strong Buy**: 11
- **Buy**: 15
- **Hold**: 15
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 60.5% Bullish, 4.7% Bearish


```

---

## 21. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:16.850405
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NKE', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for NKE: Too Many Requests. Rate limited. Try after a while.
```

---

## 22. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:17.500206
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ACHC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1526 chars

### Output
```
# Technical Analysis for ACHC
**Date:** 2026-04-25
**Current Price:** $27.38

## Price Action
- **Daily Change:** +0.15%
- **5-Day Change:** -1.65%

## RSI (14)
- **Value:** 64.1 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 1.315
- **Signal Line:** 1.350
- **Histogram:** -0.035 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $23.41 (+17.0% from price)
- **200 SMA:** $20.25 (+35.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $29.31
- **Middle:** $25.91
- **Lower:** $22.51
- **Band Position:** 72%
- **Bandwidth:** 26.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.16 (4.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $25.64 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.0
- **%D:** 75.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 34.5 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $26.13 (+4.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 87.7M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $25.83
- **Current vs VWAP:** +6.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $28.27
- **Recent Low:** $15.34
- **Next Resistance:** $28.27 (0.0% (High))
- **Next Support:** $25.22 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 23. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:17.552236
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ACHC'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for ACHC: Too Many Requests. Rate limited. Try after a while.
```

---

## 24. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:17.811405
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ACHC'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for ACHC

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 5.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 5.0% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 5.0% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 65.0% Bullish, 5.0% Bearish


```

---

## 25. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:17.827636
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ACHC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for ACHC: Too Many Requests. Rate limited. Try after a while.
```

---

## 26. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:18.562908
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BEAM', 'curr_date': '2026-04-25'}`
- **Output Length:** 1507 chars

### Output
```
# Technical Analysis for BEAM
**Date:** 2026-04-25
**Current Price:** $27.11

## Price Action
- **Daily Change:** -10.11%
- **5-Day Change:** -13.08%

## RSI (14)
- **Value:** 46.8 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 1.180
- **Signal Line:** 1.044
- **Histogram:** 0.136 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $26.99 (+0.4% from price)
- **200 SMA:** $24.80 (+9.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $34.13
- **Middle:** $27.42
- **Lower:** $20.71
- **Band Position:** 48%
- **Bandwidth:** 48.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.77 (6.5% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $24.46 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 49.5
- **%D:** 70.5
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 24.5 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $28.36 (-4.4% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 4.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $27.42
- **Current vs VWAP:** -1.1%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $34.20
- **Recent Low:** $21.63
- **Next Resistance:** $27.91 (50.0%)
- **Next Support:** $26.43 (61.8%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 27. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:18.610938
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BEAM'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for BEAM: Too Many Requests. Rate limited. Try after a while.
```

---

## 28. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:18.845777
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'BEAM'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for BEAM

### 2026-04-01
- **Strong Buy**: 11
- **Buy**: 11
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 88.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 11
- **Buy**: 11
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 88.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 11
- **Buy**: 10
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 87.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 10
- **Buy**: 10
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 87.0% Bullish, 0.0% Bearish


```

---

## 29. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:18.868614
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BEAM', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for BEAM: Too Many Requests. Rate limited. Try after a while.
```

---

## 30. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:19.543021
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AI', 'curr_date': '2026-04-25'}`
- **Output Length:** 1510 chars

### Output
```
# Technical Analysis for AI
**Date:** 2026-04-25
**Current Price:** $8.64

## Price Action
- **Daily Change:** +1.17%
- **5-Day Change:** -7.89%

## RSI (14)
- **Value:** 45.9 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.014
- **Signal Line:** -0.064
- **Histogram:** 0.050 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $9.04 (-4.4% from price)
- **200 SMA:** $15.03 (-42.5% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $9.72
- **Middle:** $8.72
- **Lower:** $7.72
- **Band Position:** 46%
- **Bandwidth:** 23.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.54 (6.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $7.83 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 43.4
- **%D:** 56.3
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 10.1 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $8.88 (-2.7% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -142.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $8.72
- **Current vs VWAP:** -0.9%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $11.08
- **Recent Low:** $7.68
- **Next Resistance:** $8.98 (61.8%)
- **Next Support:** $8.40 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 31. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:19.592775
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AI'}`
- **Output Length:** 97 chars

### Output
```
Error retrieving insider transactions for AI: Too Many Requests. Rate limited. Try after a while.
```

---

## 32. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:19.855774
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AI'}`
- **Output Length:** 713 chars

### Output
```
## Analyst Recommendation Trends for AI

### 2026-04-01
- **Strong Buy**: 1
- **Buy**: 1
- **Hold**: 10
- **Sell**: 8
- **Strong Sell**: 4
- **Total Analysts**: 24

**Sentiment**: 8.3% Bullish, 50.0% Bearish

### 2026-03-01
- **Strong Buy**: 1
- **Buy**: 1
- **Hold**: 10
- **Sell**: 8
- **Strong Sell**: 4
- **Total Analysts**: 24

**Sentiment**: 8.3% Bullish, 50.0% Bearish

### 2026-02-01
- **Strong Buy**: 1
- **Buy**: 2
- **Hold**: 9
- **Sell**: 8
- **Strong Sell**: 4
- **Total Analysts**: 24

**Sentiment**: 12.5% Bullish, 50.0% Bearish

### 2026-01-01
- **Strong Buy**: 1
- **Buy**: 2
- **Hold**: 9
- **Sell**: 8
- **Strong Sell**: 4
- **Total Analysts**: 24

**Sentiment**: 12.5% Bullish, 50.0% Bearish


```

---

## 33. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:19.877625
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AI', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 93 chars

### Output
```
Error retrieving options activity for AI: Too Many Requests. Rate limited. Try after a while.
```

---

## 34. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:20.619629
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'INTC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1582 chars

### Output
```
# Technical Analysis for INTC
**Date:** 2026-04-25
**Current Price:** $82.54

## Price Action
- **Daily Change:** +23.60%
- **5-Day Change:** +25.63%

## RSI (14)
- **Value:** 82.2 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 6.608
- **Signal Line:** 5.266
- **Histogram:** 1.342 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $51.03 (+61.7% from price)
- **200 SMA:** $38.45 (+114.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $80.73
- **Middle:** $59.55
- **Lower:** $38.37
- **Band Position:** 104%
- **Bandwidth:** 71.1% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $4.29 (5.2% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $76.11 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.6
- **%D:** 74.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 58.5 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $61.95 (+33.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 3538.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $61.36
- **Current vs VWAP:** +34.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $85.22
- **Recent Low:** $40.63
- **Next Resistance:** $85.22 (0.0% (High))
- **Next Support:** $74.70 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 35. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:20.668330
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'INTC'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for INTC: Too Many Requests. Rate limited. Try after a while.
```

---

## 36. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:20.892181
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'INTC'}`
- **Output Length:** 716 chars

### Output
```
## Analyst Recommendation Trends for INTC

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 35
- **Sell**: 3
- **Strong Sell**: 1
- **Total Analysts**: 52

**Sentiment**: 25.0% Bullish, 7.7% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 36
- **Sell**: 3
- **Strong Sell**: 1
- **Total Analysts**: 53

**Sentiment**: 24.5% Bullish, 7.5% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 34
- **Sell**: 4
- **Strong Sell**: 1
- **Total Analysts**: 52

**Sentiment**: 25.0% Bullish, 9.6% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 4
- **Hold**: 34
- **Sell**: 6
- **Strong Sell**: 4
- **Total Analysts**: 50

**Sentiment**: 12.0% Bullish, 20.0% Bearish


```

---

## 37. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:20.907600
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'INTC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for INTC: Too Many Requests. Rate limited. Try after a while.
```

---

## 38. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:21.691650
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NTST', 'curr_date': '2026-04-25'}`
- **Output Length:** 1522 chars

### Output
```
# Technical Analysis for NTST
**Date:** 2026-04-25
**Current Price:** $20.56

## Price Action
- **Daily Change:** -0.92%
- **5-Day Change:** -1.58%

## RSI (14)
- **Value:** 60.4 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.337
- **Signal Line:** 0.248
- **Histogram:** 0.088 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $19.99 (+2.9% from price)
- **200 SMA:** $18.37 (+11.9% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $21.47
- **Middle:** $19.96
- **Lower:** $18.45
- **Band Position:** 70%
- **Bandwidth:** 15.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.45 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $19.89 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.8
- **%D:** 81.8
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 16.2 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $20.22 (+1.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 38.4M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $20.03
- **Current vs VWAP:** +2.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $21.18
- **Recent Low:** $18.47
- **Next Resistance:** $21.18 (0.0% (High))
- **Next Support:** $20.54 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 39. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:21.740418
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NTST'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for NTST: Too Many Requests. Rate limited. Try after a while.
```

---

## 40. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:21.987431
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NTST'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for NTST

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 12
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 80.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 12
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 80.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 12
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 80.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 12
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 83.3% Bullish, 0.0% Bearish


```

---

## 41. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:22.005507
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NTST', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for NTST: Too Many Requests. Rate limited. Try after a while.
```

---

## 42. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:22.697210
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DLR', 'curr_date': '2026-04-25'}`
- **Output Length:** 1545 chars

### Output
```
# Technical Analysis for DLR
**Date:** 2026-04-25
**Current Price:** $200.00

## Price Action
- **Daily Change:** +0.00%
- **5-Day Change:** -1.92%

## RSI (14)
- **Value:** 72.2 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 6.356
- **Signal Line:** 5.680
- **Histogram:** 0.676 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $182.75 (+9.4% from price)
- **200 SMA:** $168.68 (+18.6% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $210.08
- **Middle:** $190.60
- **Lower:** $171.13
- **Band Position:** 74%
- **Bandwidth:** 20.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.26 (2.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $193.61 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.0
- **%D:** 82.3
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 51.7 (Weakening ↓)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $192.81 (+3.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 29.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $190.90
- **Current vs VWAP:** +4.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $208.14
- **Recent Low:** $171.74
- **Next Resistance:** $208.14 (0.0% (High))
- **Next Support:** $199.55 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 43. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:22.753585
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DLR'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for DLR: Too Many Requests. Rate limited. Try after a while.
```

---

## 44. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:23.035600
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DLR'}`
- **Output Length:** 716 chars

### Output
```
## Analyst Recommendation Trends for DLR

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 72.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 17
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 17
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 71.4% Bullish, 2.9% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 15
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 34

**Sentiment**: 70.6% Bullish, 2.9% Bearish


```

---

## 45. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:23.052405
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DLR', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for DLR: Too Many Requests. Rate limited. Try after a while.
```

---

## 46. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:23.871712
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PECO', 'curr_date': '2026-04-25'}`
- **Output Length:** 1531 chars

### Output
```
# Technical Analysis for PECO
**Date:** 2026-04-25
**Current Price:** $39.46

## Price Action
- **Daily Change:** +1.08%
- **5-Day Change:** +0.33%

## RSI (14)
- **Value:** 62.4 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.412
- **Signal Line:** 0.311
- **Histogram:** 0.101 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $38.17 (+3.4% from price)
- **200 SMA:** $35.25 (+12.0% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $39.93
- **Middle:** $38.22
- **Lower:** $36.51
- **Band Position:** 86%
- **Bandwidth:** 9.0% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $0.72 (1.8% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $38.38 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.3
- **%D:** 71.4
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 10.1 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $38.46 (+2.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -7.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $38.18
- **Current vs VWAP:** +3.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $39.83
- **Recent Low:** $36.40
- **Next Resistance:** $39.83 (0.0% (High))
- **Next Support:** $39.02 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 47. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:23.921040
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PECO'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for PECO: Too Many Requests. Rate limited. Try after a while.
```

---

## 48. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:24.175533
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PECO'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for PECO

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 6
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 19

**Sentiment**: 57.9% Bullish, 0.0% Bearish


```

---

## 49. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:24.192056
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PECO', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for PECO: Too Many Requests. Rate limited. Try after a while.
```

---

## 50. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:24.977300
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'KOPN', 'curr_date': '2026-04-25'}`
- **Output Length:** 1556 chars

### Output
```
# Technical Analysis for KOPN
**Date:** 2026-04-25
**Current Price:** $3.70

## Price Action
- **Daily Change:** +4.82%
- **5-Day Change:** +30.28%

## RSI (14)
- **Value:** 76.9 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.290
- **Signal Line:** 0.197
- **Histogram:** 0.093 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $2.41 (+53.8% from price)
- **200 SMA:** $2.51 (+47.3% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $3.64
- **Middle:** $2.73
- **Lower:** $1.82
- **Band Position:** 103%
- **Bandwidth:** 66.6% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.25 (6.9% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $3.32 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.7
- **%D:** 77.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 51.6 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $2.85 (+29.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 97.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $2.79
- **Current vs VWAP:** +32.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $3.90
- **Recent Low:** $1.81
- **Next Resistance:** $3.90 (0.0% (High))
- **Next Support:** $3.41 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 51. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:25.028541
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'KOPN'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for KOPN: Too Many Requests. Rate limited. Try after a while.
```

---

## 52. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:25.278156
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'KOPN'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for KOPN

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 6
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 10

**Sentiment**: 90.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 6
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 10

**Sentiment**: 90.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 6
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 10

**Sentiment**: 90.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 6
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 10

**Sentiment**: 90.0% Bullish, 0.0% Bearish


```

---

## 53. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:25.296011
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'KOPN', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for KOPN: Too Many Requests. Rate limited. Try after a while.
```

---

## 54. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:26.091023
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TAL', 'curr_date': '2026-04-25'}`
- **Output Length:** 1541 chars

### Output
```
# Technical Analysis for TAL
**Date:** 2026-04-25
**Current Price:** $11.24

## Price Action
- **Daily Change:** +3.02%
- **5-Day Change:** -8.84%

## RSI (14)
- **Value:** 46.8 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 0.134
- **Signal Line:** 0.182
- **Histogram:** -0.048 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $11.18 (+0.5% from price)
- **200 SMA:** $11.19 (+0.5% from price)
- **Trend:** Uptrend (DEATH CROSS ⚠️)

## Bollinger Bands (20,2)
- **Upper:** $12.57
- **Middle:** $11.50
- **Lower:** $10.43
- **Band Position:** 38%
- **Bandwidth:** 18.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.53 (4.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $10.45 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 43.2
- **%D:** 55.5
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 19.2 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $11.55 (-2.7% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 34.4M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $11.57
- **Current vs VWAP:** -2.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $12.84
- **Recent Low:** $10.15
- **Next Resistance:** $11.49 (50.0%)
- **Next Support:** $11.18 (61.8%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 55. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:26.143348
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TAL'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for TAL: Too Many Requests. Rate limited. Try after a while.
```

---

## 56. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:26.369086
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TAL'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for TAL

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 88.5% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 88.5% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 88.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 15
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 84.6% Bullish, 0.0% Bearish


```

---

## 57. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:26.399364
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TAL', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for TAL: Too Many Requests. Rate limited. Try after a while.
```

---

## 58. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:27.066853
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CCI', 'curr_date': '2026-04-25'}`
- **Output Length:** 1493 chars

### Output
```
# Technical Analysis for CCI
**Date:** 2026-04-25
**Current Price:** $86.34

## Price Action
- **Daily Change:** -1.35%
- **5-Day Change:** -1.39%

## RSI (14)
- **Value:** 52.4 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.665
- **Signal Line:** 0.489
- **Histogram:** 0.176 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $85.68 (+0.8% from price)
- **200 SMA:** $90.44 (-4.5% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $90.65
- **Middle:** $85.10
- **Lower:** $79.55
- **Band Position:** 61%
- **Bandwidth:** 13.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.51 (2.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $82.57 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 48.8
- **%D:** 57.9
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 7.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $85.72 (+0.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 2.1M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $84.99
- **Current vs VWAP:** +1.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $90.81
- **Recent Low:** $75.96
- **Next Resistance:** $87.31 (23.6%)
- **Next Support:** $85.14 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 59. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:27.120519
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CCI'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for CCI: Too Many Requests. Rate limited. Try after a while.
```

---

## 60. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:27.349263
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CCI'}`
- **Output Length:** 712 chars

### Output
```
## Analyst Recommendation Trends for CCI

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 8
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 58.3% Bullish, 4.2% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 4.3% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 65.2% Bullish, 4.3% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 65.2% Bullish, 4.3% Bearish


```

---

## 61. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:27.365071
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CCI', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for CCI: Too Many Requests. Rate limited. Try after a while.
```

---

## 62. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:28.021228
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MASI', 'curr_date': '2026-04-25'}`
- **Output Length:** 1515 chars

### Output
```
# Technical Analysis for MASI
**Date:** 2026-04-25
**Current Price:** $178.63

## Price Action
- **Daily Change:** -0.04%
- **5-Day Change:** +0.10%

## RSI (14)
- **Value:** 69.8 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 1.402
- **Signal Line:** 1.888
- **Histogram:** -0.486 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $175.14 (+2.0% from price)
- **200 SMA:** $152.31 (+17.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $179.13
- **Middle:** $178.29
- **Lower:** $177.45
- **Band Position:** 70%
- **Bandwidth:** 0.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.67 (0.4% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $177.63 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.8
- **%D:** 63.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 51.9 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $177.86 (+0.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 24.1M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $178.13
- **Current vs VWAP:** +0.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $179.00
- **Recent Low:** $129.21
- **Next Resistance:** $179.00 (0.0% (High))
- **Next Support:** $167.25 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 63. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:28.070216
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MASI'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for MASI: Too Many Requests. Rate limited. Try after a while.
```

---

## 64. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:28.316171
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MASI'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for MASI

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 4
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 15

**Sentiment**: 40.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 5
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 15

**Sentiment**: 46.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 16

**Sentiment**: 75.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 16

**Sentiment**: 75.0% Bullish, 0.0% Bearish


```

---

## 65. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:28.332970
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MASI', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for MASI: Too Many Requests. Rate limited. Try after a while.
```

---

## 66. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:28.998672
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'VOYA', 'curr_date': '2026-04-25'}`
- **Output Length:** 1568 chars

### Output
```
# Technical Analysis for VOYA
**Date:** 2026-04-25
**Current Price:** $80.35

## Price Action
- **Daily Change:** +1.88%
- **5-Day Change:** +6.42%

## RSI (14)
- **Value:** 77.0 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 2.406
- **Signal Line:** 1.381
- **Histogram:** 1.025 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $70.20 (+14.5% from price)
- **200 SMA:** $72.27 (+11.2% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $79.83
- **Middle:** $71.55
- **Lower:** $63.28
- **Band Position:** 103%
- **Bandwidth:** 23.1% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $2.35 (2.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $76.83 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 89.0
- **%D:** 86.7
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 42.7 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $73.05 (+10.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 17.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $72.80
- **Current vs VWAP:** +10.4%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $80.76
- **Recent Low:** $64.50
- **Next Resistance:** $80.76 (0.0% (High))
- **Next Support:** $76.92 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 67. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:29.045386
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'VOYA'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for VOYA: Too Many Requests. Rate limited. Try after a while.
```

---

## 68. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:29.287030
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'VOYA'}`
- **Output Length:** 712 chars

### Output
```
## Analyst Recommendation Trends for VOYA

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 10
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 66.7% Bullish, 5.6% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 9
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 64.7% Bullish, 5.9% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 9
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 64.7% Bullish, 5.9% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 8
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 58.8% Bullish, 5.9% Bearish


```

---

## 69. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:29.302451
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'VOYA', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for VOYA: Too Many Requests. Rate limited. Try after a while.
```

---

## 70. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:30.029988
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'JKHY', 'curr_date': '2026-04-25'}`
- **Output Length:** 1522 chars

### Output
```
# Technical Analysis for JKHY
**Date:** 2026-04-25
**Current Price:** $151.15

## Price Action
- **Daily Change:** +0.37%
- **5-Day Change:** -1.77%

## RSI (14)
- **Value:** 39.2 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.513
- **Signal Line:** -2.730
- **Histogram:** 0.216 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $159.48 (-5.2% from price)
- **200 SMA:** $165.47 (-8.7% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $160.38
- **Middle:** $154.36
- **Lower:** $148.35
- **Band Position:** 23%
- **Bandwidth:** 7.8% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.10 (2.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $145.00 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 50.3
- **%D:** 53.0
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 27.9 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $155.07 (-2.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 1.9M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $154.21
- **Current vs VWAP:** -2.0%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $172.84
- **Recent Low:** $146.92
- **Next Resistance:** $152.47 (78.6%)
- **Next Support:** $146.92 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 71. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:30.077945
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'JKHY'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for JKHY: Too Many Requests. Rate limited. Try after a while.
```

---

## 72. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:30.330534
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'JKHY'}`
- **Output Length:** 712 chars

### Output
```
## Analyst Recommendation Trends for JKHY

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 8
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 72.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 8
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 73.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 6
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 52.2% Bullish, 0.0% Bearish


```

---

## 73. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:30.356211
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'JKHY', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for JKHY: Too Many Requests. Rate limited. Try after a while.
```

---

## 74. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:31.062792
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BBY', 'curr_date': '2026-04-25'}`
- **Output Length:** 1565 chars

### Output
```
# Technical Analysis for BBY
**Date:** 2026-04-25
**Current Price:** $60.45

## Price Action
- **Daily Change:** -1.71%
- **5-Day Change:** -9.70%

## RSI (14)
- **Value:** 39.7 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.117
- **Signal Line:** 0.133
- **Histogram:** -0.250 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $63.35 (-4.6% from price)
- **200 SMA:** $68.96 (-12.3% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $67.18
- **Middle:** $63.76
- **Lower:** $60.34
- **Band Position:** 2%
- **Bandwidth:** 10.7% (volatility indicator)
- **Signal:** AT LOWER BAND - Potential bounce ⚡

## ATR (Volatility)
- **ATR:** $2.26 (3.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $57.06 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 34.9
- **%D:** 52.5
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 11.3 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $63.50 (-4.8% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -38.4M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $63.63
- **Current vs VWAP:** -5.0%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $68.21
- **Recent Low:** $59.60
- **Next Resistance:** $61.44 (78.6%)
- **Next Support:** $59.60 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 75. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:31.113284
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BBY'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for BBY: Too Many Requests. Rate limited. Try after a while.
```

---

## 76. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:31.373033
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'BBY'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for BBY

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 7
- **Hold**: 19
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 30

**Sentiment**: 33.3% Bullish, 3.3% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 7
- **Hold**: 20
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 32.3% Bullish, 3.2% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 8
- **Hold**: 19
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 35.5% Bullish, 3.2% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 8
- **Hold**: 19
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 35.5% Bullish, 3.2% Bearish


```

---

## 77. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:31.390936
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BBY', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for BBY: Too Many Requests. Rate limited. Try after a while.
```

---

## 78. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:32.071264
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ADI', 'curr_date': '2026-04-25'}`
- **Output Length:** 1549 chars

### Output
```
# Technical Analysis for ADI
**Date:** 2026-04-25
**Current Price:** $399.57

## Price Action
- **Daily Change:** -1.07%
- **5-Day Change:** +4.86%

## RSI (14)
- **Value:** 76.8 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 18.408
- **Signal Line:** 12.472
- **Histogram:** 5.936 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $337.54 (+18.4% from price)
- **200 SMA:** $276.06 (+44.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $407.82
- **Middle:** $349.15
- **Lower:** $290.48
- **Band Position:** 93%
- **Bandwidth:** 33.6% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $11.25 (2.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $382.69 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.9
- **%D:** 89.1
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 44.9 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $358.43 (+11.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 89.6M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $351.00
- **Current vs VWAP:** +13.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $408.37
- **Recent Low:** $300.56
- **Next Resistance:** $408.37 (0.0% (High))
- **Next Support:** $382.93 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 79. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:32.119060
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ADI'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for ADI: Too Many Requests. Rate limited. Try after a while.
```

---

## 80. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:32.376253
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ADI'}`
- **Output Length:** 721 chars

### Output
```
## Analyst Recommendation Trends for ADI

### 2026-04-01
- **Strong Buy**: 12
- **Buy**: 21
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 41

**Sentiment**: 80.5% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 12
- **Buy**: 20
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 74.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 12
- **Buy**: 18
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 69.8% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 12
- **Buy**: 18
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 41

**Sentiment**: 73.2% Bullish, 0.0% Bearish


```

---

## 81. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:32.394943
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ADI', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for ADI: Too Many Requests. Rate limited. Try after a while.
```

---

## 82. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:33.165170
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'THM', 'curr_date': '2026-04-25'}`
- **Output Length:** 1501 chars

### Output
```
# Technical Analysis for THM
**Date:** 2026-04-25
**Current Price:** $2.63

## Price Action
- **Daily Change:** +3.95%
- **5-Day Change:** +1.94%

## RSI (14)
- **Value:** 55.3 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.032
- **Signal Line:** 0.008
- **Histogram:** 0.024 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $2.60 (+1.0% from price)
- **200 SMA:** $1.97 (+33.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $2.80
- **Middle:** $2.47
- **Lower:** $2.13
- **Band Position:** 75%
- **Bandwidth:** 27.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.19 (7.2% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $2.35 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 47.4
- **%D:** 53.1
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 13.2 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $2.52 (+4.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 19.3M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $2.48
- **Current vs VWAP:** +6.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $3.62
- **Recent Low:** $1.89
- **Next Resistance:** $2.75 (50.0%)
- **Next Support:** $2.55 (61.8%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 83. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:33.215832
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'THM'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for THM: Too Many Requests. Rate limited. Try after a while.
```

---

## 84. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:33.467128
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'THM'}`
- **Output Length:** 43 chars

### Output
```
No recommendation trends data found for THM
```

---

## 85. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:33.485296
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'THM', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for THM: Too Many Requests. Rate limited. Try after a while.
```

---

## 86. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:34.238021
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AVEX', 'curr_date': '2026-04-25'}`
- **Output Length:** 1501 chars

### Output
```
# Technical Analysis for AVEX
**Date:** 2026-04-25
**Current Price:** $29.70

## Price Action
- **Daily Change:** -6.52%
- **5-Day Change:** -11.10%

## RSI (14)
- **Value:** 54.1 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.029
- **Signal Line:** 0.079
- **Histogram:** -0.109 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $31.33 (-5.2% from price)
- **200 SMA:** $31.33 (-5.2% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $37.63
- **Middle:** $31.33
- **Lower:** $25.03
- **Band Position:** 37%
- **Bandwidth:** 40.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $6.47 (21.8% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $20.00 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 45.2
- **%D:** 50.9
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 48.5 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $31.33 (-5.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 18.1M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $31.53
- **Current vs VWAP:** -5.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $42.34
- **Recent Low:** $23.00
- **Next Resistance:** $30.39 (61.8%)
- **Next Support:** $27.14 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 87. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:34.288864
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AVEX'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for AVEX: Too Many Requests. Rate limited. Try after a while.
```

---

## 88. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:34.550597
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AVEX'}`
- **Output Length:** 44 chars

### Output
```
No recommendation trends data found for AVEX
```

---

## 89. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:34.575949
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AVEX', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for AVEX: Too Many Requests. Rate limited. Try after a while.
```

---

## 90. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:35.425913
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BMI', 'curr_date': '2026-04-25'}`
- **Output Length:** 1527 chars

### Output
```
# Technical Analysis for BMI
**Date:** 2026-04-25
**Current Price:** $121.83

## Price Action
- **Daily Change:** -0.95%
- **5-Day Change:** +7.42%

## RSI (14)
- **Value:** 32.6 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -8.149
- **Signal Line:** -4.661
- **Histogram:** -3.488 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $148.81 (-18.1% from price)
- **200 SMA:** $172.75 (-29.5% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $177.59
- **Middle:** $143.97
- **Lower:** $110.35
- **Band Position:** 17%
- **Bandwidth:** 46.7% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $6.85 (5.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $111.55 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 21.3
- **%D:** 26.6
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 38.6 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $138.86 (-12.3% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -15.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $129.63
- **Current vs VWAP:** -6.0%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $162.74
- **Recent Low:** $112.09
- **Next Resistance:** $122.93 (78.6%)
- **Next Support:** $112.09 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 91. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:35.473575
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BMI'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for BMI: Too Many Requests. Rate limited. Try after a while.
```

---

## 92. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:35.728270
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'BMI'}`
- **Output Length:** 710 chars

### Output
```
## Analyst Recommendation Trends for BMI

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 6
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 58.8% Bullish, 5.9% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 6
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 52.9% Bullish, 5.9% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 6
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 58.8% Bullish, 5.9% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 6
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 16

**Sentiment**: 62.5% Bullish, 0.0% Bearish


```

---

## 93. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:35.744185
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BMI', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for BMI: Too Many Requests. Rate limited. Try after a while.
```

---

## 94. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:36.583513
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BORR', 'curr_date': '2026-04-25'}`
- **Output Length:** 1508 chars

### Output
```
# Technical Analysis for BORR
**Date:** 2026-04-25
**Current Price:** $5.69

## Price Action
- **Daily Change:** +2.15%
- **5-Day Change:** +2.71%

## RSI (14)
- **Value:** 53.0 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.012
- **Signal Line:** 0.045
- **Histogram:** -0.034 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $5.62 (+1.2% from price)
- **200 SMA:** $3.84 (+48.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $6.15
- **Middle:** $5.75
- **Lower:** $5.34
- **Band Position:** 43%
- **Bandwidth:** 14.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.30 (5.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $5.24 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 38.6
- **%D:** 32.6
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 5.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $5.62 (+1.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 171.9M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $5.73
- **Current vs VWAP:** -0.7%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $6.25
- **Recent Low:** $4.64
- **Next Resistance:** $5.87 (23.6%)
- **Next Support:** $5.64 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 95. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:36.630670
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BORR'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for BORR: Too Many Requests. Rate limited. Try after a while.
```

---

## 96. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:36.869474
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'BORR'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for BORR

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 7
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 7
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 6
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 6
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 71.4% Bullish, 0.0% Bearish


```

---

## 97. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:36.885544
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BORR', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for BORR: Too Many Requests. Rate limited. Try after a while.
```

---

## 98. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:37.630317
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NE', 'curr_date': '2026-04-25'}`
- **Output Length:** 1516 chars

### Output
```
# Technical Analysis for NE
**Date:** 2026-04-25
**Current Price:** $49.54

## Price Action
- **Daily Change:** +1.23%
- **5-Day Change:** +6.24%

## RSI (14)
- **Value:** 58.0 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.434
- **Signal Line:** 0.630
- **Histogram:** -0.196 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $47.00 (+5.4% from price)
- **200 SMA:** $33.66 (+47.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $51.10
- **Middle:** $48.78
- **Lower:** $46.45
- **Band Position:** 66%
- **Bandwidth:** 9.5% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.65 (3.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $47.07 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 64.1
- **%D:** 46.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 12.0 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $48.24 (+2.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 15.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $48.74
- **Current vs VWAP:** +1.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $51.61
- **Recent Low:** $40.24
- **Next Resistance:** $51.61 (0.0% (High))
- **Next Support:** $48.93 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 99. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:37.691330
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NE'}`
- **Output Length:** 97 chars

### Output
```
Error retrieving insider transactions for NE: Too Many Requests. Rate limited. Try after a while.
```

---

## 100. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:37.947354
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NE'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for NE

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 5
- **Hold**: 12
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 21

**Sentiment**: 42.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 5
- **Hold**: 12
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 21

**Sentiment**: 42.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 21

**Sentiment**: 57.1% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 0.0% Bearish


```

---

## 101. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:37.969983
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NE', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 93 chars

### Output
```
Error retrieving options activity for NE: Too Many Requests. Rate limited. Try after a while.
```

---

## 102. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:38.671059
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ON', 'curr_date': '2026-04-25'}`
- **Output Length:** 1585 chars

### Output
```
# Technical Analysis for ON
**Date:** 2026-04-25
**Current Price:** $98.40

## Price Action
- **Daily Change:** +0.63%
- **5-Day Change:** +15.01%

## RSI (14)
- **Value:** 89.1 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 8.274
- **Signal Line:** 5.287
- **Histogram:** 2.987 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $67.72 (+45.3% from price)
- **200 SMA:** $57.19 (+72.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $99.10
- **Middle:** $73.46
- **Lower:** $47.82
- **Band Position:** 99%
- **Bandwidth:** 69.8% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $3.63 (3.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $92.96 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 95.6
- **%D:** 96.1
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 61.4 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $77.83 (+26.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 123.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $76.57
- **Current vs VWAP:** +28.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $100.30
- **Recent Low:** $54.99
- **Next Resistance:** $100.30 (0.0% (High))
- **Next Support:** $89.61 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 103. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:38.718336
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ON'}`
- **Output Length:** 97 chars

### Output
```
Error retrieving insider transactions for ON: Too Many Requests. Rate limited. Try after a while.
```

---

## 104. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:38.973405
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ON'}`
- **Output Length:** 717 chars

### Output
```
## Analyst Recommendation Trends for ON

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 16
- **Hold**: 24
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 45.5% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 17
- **Hold**: 23
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 47.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 18
- **Hold**: 22
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 50.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 19
- **Hold**: 20
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 52.3% Bullish, 2.3% Bearish


```

---

## 105. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:38.990341
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ON', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 93 chars

### Output
```
Error retrieving options activity for ON: Too Many Requests. Rate limited. Try after a while.
```

---

## 106. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:39.706397
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NEE', 'curr_date': '2026-04-25'}`
- **Output Length:** 1530 chars

### Output
```
# Technical Analysis for NEE
**Date:** 2026-04-25
**Current Price:** $95.28

## Price Action
- **Daily Change:** -1.01%
- **5-Day Change:** +3.55%

## RSI (14)
- **Value:** 60.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.446
- **Signal Line:** 0.273
- **Histogram:** 0.173 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $92.28 (+3.2% from price)
- **200 SMA:** $81.91 (+16.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $95.87
- **Middle:** $92.71
- **Lower:** $89.55
- **Band Position:** 91%
- **Bandwidth:** 6.8% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $2.18 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $92.01 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 51.6
- **%D:** 38.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 19.0 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $92.60 (+2.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 248.9M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $92.78
- **Current vs VWAP:** +2.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $97.63
- **Recent Low:** $89.17
- **Next Resistance:** $95.63 (23.6%)
- **Next Support:** $94.40 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 107. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:39.754836
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NEE'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for NEE: Too Many Requests. Rate limited. Try after a while.
```

---

## 108. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:40.039282
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NEE'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for NEE

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 67.7% Bullish, 3.2% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 67.7% Bullish, 3.2% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 67.7% Bullish, 3.2% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 30

**Sentiment**: 70.0% Bullish, 3.3% Bearish


```

---

## 109. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:40.056979
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NEE', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for NEE: Too Many Requests. Rate limited. Try after a while.
```

---

## 110. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:40.785797
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MCHP', 'curr_date': '2026-04-25'}`
- **Output Length:** 1584 chars

### Output
```
# Technical Analysis for MCHP
**Date:** 2026-04-25
**Current Price:** $89.44

## Price Action
- **Daily Change:** -1.32%
- **5-Day Change:** +11.26%

## RSI (14)
- **Value:** 82.0 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 5.011
- **Signal Line:** 2.964
- **Histogram:** 2.047 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $71.22 (+25.6% from price)
- **200 SMA:** $66.92 (+33.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $90.47
- **Middle:** $73.40
- **Lower:** $56.33
- **Band Position:** 97%
- **Bandwidth:** 46.5% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $3.14 (3.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $84.74 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 91.2
- **%D:** 92.9
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 54.4 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $76.41 (+17.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 14.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $74.36
- **Current vs VWAP:** +20.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $92.24
- **Recent Low:** $59.31
- **Next Resistance:** $92.24 (0.0% (High))
- **Next Support:** $84.47 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 111. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:40.839273
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MCHP'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for MCHP: Too Many Requests. Rate limited. Try after a while.
```

---

## 112. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:41.118415
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MCHP'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for MCHP

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 81.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 18
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 81.8% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 78.8% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 16
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 72.7% Bullish, 0.0% Bearish


```

---

## 113. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:41.134594
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MCHP', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for MCHP: Too Many Requests. Rate limited. Try after a while.
```

---

## 114. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:41.862135
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'R', 'curr_date': '2026-04-25'}`
- **Output Length:** 1588 chars

### Output
```
# Technical Analysis for R
**Date:** 2026-04-25
**Current Price:** $253.34

## Price Action
- **Daily Change:** +4.43%
- **5-Day Change:** +10.11%

## RSI (14)
- **Value:** 76.5 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 9.548
- **Signal Line:** 6.258
- **Histogram:** 3.290 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $211.70 (+19.7% from price)
- **200 SMA:** $189.51 (+33.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $247.55
- **Middle:** $218.62
- **Lower:** $189.70
- **Band Position:** 110%
- **Bandwidth:** 26.5% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $8.37 (3.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $240.78 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 89.3
- **%D:** 85.5
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 37.7 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $222.54 (+13.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 15.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $221.45
- **Current vs VWAP:** +14.4%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $254.16
- **Recent Low:** $185.50
- **Next Resistance:** $254.16 (0.0% (High))
- **Next Support:** $237.96 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 115. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:41.911185
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'R'}`
- **Output Length:** 96 chars

### Output
```
Error retrieving insider transactions for R: Too Many Requests. Rate limited. Try after a while.
```

---

## 116. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:42.155916
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'R'}`
- **Output Length:** 709 chars

### Output
```
## Analyst Recommendation Trends for R

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 76.5% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 76.5% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 10
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 82.4% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 82.4% Bullish, 0.0% Bearish


```

---

## 117. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:42.199162
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'R', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 92 chars

### Output
```
Error retrieving options activity for R: Too Many Requests. Rate limited. Try after a while.
```

---

## 118. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:42.940015
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'UNP', 'curr_date': '2026-04-25'}`
- **Output Length:** 1588 chars

### Output
```
# Technical Analysis for UNP
**Date:** 2026-04-25
**Current Price:** $268.70

## Price Action
- **Daily Change:** -0.94%
- **5-Day Change:** +6.55%

## RSI (14)
- **Value:** 71.7 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 3.983
- **Signal Line:** 1.716
- **Histogram:** 2.267 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $251.84 (+6.7% from price)
- **200 SMA:** $232.26 (+15.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $265.95
- **Middle:** $249.86
- **Lower:** $233.78
- **Band Position:** 109%
- **Bandwidth:** 12.9% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $5.95 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $259.77 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 70.7
- **%D:** 67.1
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 31.4 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $252.41 (+6.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 84.5M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $251.43
- **Current vs VWAP:** +6.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $274.79
- **Recent Low:** $231.72
- **Next Resistance:** $274.79 (0.0% (High))
- **Next Support:** $264.63 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 119. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:42.990996
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'UNP'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for UNP: Too Many Requests. Rate limited. Try after a while.
```

---

## 120. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:43.243308
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'UNP'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for UNP

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 12
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 67.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 10
- **Hold**: 12
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 61.3% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 10
- **Hold**: 12
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 61.3% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 12
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 67.7% Bullish, 0.0% Bearish


```

---

## 121. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:43.259740
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'UNP', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for UNP: Too Many Requests. Rate limited. Try after a while.
```

---

## 122. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:43.934695
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ARMK', 'curr_date': '2026-04-25'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for ARMK
**Date:** 2026-04-25
**Current Price:** $46.28

## Price Action
- **Daily Change:** +0.24%
- **5-Day Change:** +1.49%

## RSI (14)
- **Value:** 72.3 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 1.382
- **Signal Line:** 1.149
- **Histogram:** 0.233 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $41.60 (+11.3% from price)
- **200 SMA:** $39.54 (+17.0% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $47.26
- **Middle:** $43.44
- **Lower:** $39.61
- **Band Position:** 87%
- **Bandwidth:** 17.6% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $1.12 (2.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $44.60 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 83.8
- **%D:** 84.9
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 40.1 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $43.84 (+5.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 31.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $43.68
- **Current vs VWAP:** +6.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $46.88
- **Recent Low:** $38.03
- **Next Resistance:** $46.88 (0.0% (High))
- **Next Support:** $44.79 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 123. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:43.982932
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ARMK'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for ARMK: Too Many Requests. Rate limited. Try after a while.
```

---

## 124. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:44.249444
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ARMK'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for ARMK: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 125. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:44.268255
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ARMK', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for ARMK: Too Many Requests. Rate limited. Try after a while.
```

---

## 126. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:44.972080
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'SEIC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1544 chars

### Output
```
# Technical Analysis for SEIC
**Date:** 2026-04-25
**Current Price:** $88.84

## Price Action
- **Daily Change:** -2.43%
- **5-Day Change:** +9.13%

## RSI (14)
- **Value:** 69.9 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 2.044
- **Signal Line:** 0.635
- **Histogram:** 1.409 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $79.83 (+11.3% from price)
- **200 SMA:** $83.59 (+6.3% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $88.26
- **Middle:** $79.80
- **Lower:** $71.34
- **Band Position:** 103%
- **Bandwidth:** 21.2% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $2.58 (2.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $84.97 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 83.3
- **%D:** 82.7
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 38.4 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $81.54 (+9.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -84.0K
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $80.96
- **Current vs VWAP:** +9.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $92.87
- **Recent Low:** $75.08
- **Next Resistance:** $92.87 (0.0% (High))
- **Next Support:** $88.67 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 127. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:45.022265
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'SEIC'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for SEIC: Too Many Requests. Rate limited. Try after a while.
```

---

## 128. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:45.278667
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'SEIC'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for SEIC: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 129. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:45.294065
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'SEIC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for SEIC: Too Many Requests. Rate limited. Try after a while.
```

---

## 130. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:45.974467
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PTC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1530 chars

### Output
```
# Technical Analysis for PTC
**Date:** 2026-04-25
**Current Price:** $137.17

## Price Action
- **Daily Change:** +0.99%
- **5-Day Change:** -2.54%

## RSI (14)
- **Value:** 39.6 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -3.134
- **Signal Line:** -3.670
- **Histogram:** 0.536 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $148.66 (-7.7% from price)
- **200 SMA:** $179.32 (-23.5% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $146.36
- **Middle:** $139.39
- **Lower:** $132.43
- **Band Position:** 34%
- **Bandwidth:** 10.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.43 (3.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $130.52 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 50.2
- **%D:** 50.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 23.5 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $140.62 (-2.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -9.3M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $138.72
- **Current vs VWAP:** -1.1%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $168.55
- **Recent Low:** $130.89
- **Next Resistance:** $138.95 (78.6%)
- **Next Support:** $130.89 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 131. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:46.021162
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PTC'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for PTC: Too Many Requests. Rate limited. Try after a while.
```

---

## 132. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:46.261262
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PTC'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for PTC: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 133. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:46.275418
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PTC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for PTC: Too Many Requests. Rate limited. Try after a while.
```

---

## 134. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:46.916314
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'OKTA', 'curr_date': '2026-04-25'}`
- **Output Length:** 1515 chars

### Output
```
# Technical Analysis for OKTA
**Date:** 2026-04-25
**Current Price:** $75.98

## Price Action
- **Daily Change:** -0.08%
- **5-Day Change:** +0.29%

## RSI (14)
- **Value:** 51.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.559
- **Signal Line:** -1.478
- **Histogram:** 0.919 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $76.38 (-0.5% from price)
- **200 SMA:** $86.42 (-12.1% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $85.13
- **Middle:** $73.93
- **Lower:** $62.73
- **Band Position:** 59%
- **Bandwidth:** 30.3% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $3.97 (5.2% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $70.02 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 74.2
- **%D:** 64.7
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 8.3 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $74.57 (+1.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 38.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $73.25
- **Current vs VWAP:** +3.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $89.50
- **Recent Low:** $62.66
- **Next Resistance:** $76.08 (50.0%)
- **Next Support:** $72.91 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 135. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:46.970174
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'OKTA'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for OKTA: Too Many Requests. Rate limited. Try after a while.
```

---

## 136. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:47.223316
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'OKTA'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for OKTA: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 137. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:47.238913
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'OKTA', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for OKTA: Too Many Requests. Rate limited. Try after a while.
```

---

## 138. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:48.154814
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'KLRA', 'curr_date': '2026-04-25'}`
- **Output Length:** 1537 chars

### Output
```
# Technical Analysis for KLRA
**Date:** 2026-04-25
**Current Price:** $22.47

## Price Action
- **Daily Change:** -10.41%
- **5-Day Change:** -8.70%

## RSI (14)
- **Value:** 21.5 ↑
- **Signal:** OVERSOLD ⚡

## MACD
- **MACD Line:** -0.133
- **Signal Line:** -0.058
- **Histogram:** -0.075 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $24.61 (-8.7% from price)
- **200 SMA:** $24.61 (-8.7% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $27.01
- **Middle:** $24.61
- **Lower:** $22.21
- **Band Position:** 5%
- **Bandwidth:** 19.5% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $2.77 (12.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $18.31 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 22.8
- **%D:** 32.5
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 74.3 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $24.47 (-8.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 1.4M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $25.22
- **Current vs VWAP:** -10.9%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $28.23
- **Recent Low:** $22.10
- **Next Resistance:** $23.41 (78.6%)
- **Next Support:** $22.10 (100% (Low))

## Summary
- **Key Signals:** RSI oversold, Below 50 SMA

```

---

## 139. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:48.207313
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'KLRA'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for KLRA: Too Many Requests. Rate limited. Try after a while.
```

---

## 140. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:48.453231
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'KLRA'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for KLRA: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 141. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:48.468989
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'KLRA', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for KLRA: Too Many Requests. Rate limited. Try after a while.
```

---

## 142. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:49.105029
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ARXS', 'curr_date': '2026-04-25'}`
- **Output Length:** 1530 chars

### Output
```
# Technical Analysis for ARXS
**Date:** 2026-04-25
**Current Price:** $35.00

## Price Action
- **Daily Change:** -6.04%
- **5-Day Change:** -7.04%

## RSI (14)
- **Value:** 18.8 ↑
- **Signal:** OVERSOLD ⚡

## MACD
- **MACD Line:** -0.182
- **Signal Line:** -0.088
- **Histogram:** -0.094 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $37.24 (-6.0% from price)
- **200 SMA:** $37.24 (-6.0% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $39.72
- **Middle:** $37.24
- **Lower:** $34.75
- **Band Position:** 5%
- **Bandwidth:** 13.3% (volatility indicator)
- **Signal:** AT LOWER BAND - Potential bounce ⚡

## ATR (Volatility)
- **ATR:** $2.23 (6.4% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $31.65 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 32.6
- **%D:** 47.2
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 43.3 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $37.03 (-5.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -8.7M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $37.54
- **Current vs VWAP:** -6.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $39.45
- **Recent Low:** $35.00
- **Next Resistance:** $35.95 (78.6%)

## Summary
- **Key Signals:** RSI oversold, Below 50 SMA

```

---

## 143. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:49.154432
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ARXS'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for ARXS: Too Many Requests. Rate limited. Try after a while.
```

---

## 144. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:49.393931
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ARXS'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for ARXS: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 145. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:49.410872
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ARXS', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for ARXS: Too Many Requests. Rate limited. Try after a while.
```

---

## 146. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:50.004000
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TSLA', 'curr_date': '2026-04-25'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for TSLA
**Date:** 2026-04-25
**Current Price:** $376.30

## Price Action
- **Daily Change:** +0.69%
- **5-Day Change:** -4.13%

## RSI (14)
- **Value:** 48.9 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.275
- **Signal Line:** -3.275
- **Histogram:** 2.999 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $387.08 (-2.8% from price)
- **200 SMA:** $400.86 (-6.1% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $405.65
- **Middle:** $369.13
- **Lower:** $332.61
- **Band Position:** 60%
- **Bandwidth:** 19.8% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $14.94 (4.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $353.90 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 53.9
- **%D:** 61.6
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 10.9 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $377.09 (-0.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 1195.5M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $369.99
- **Current vs VWAP:** +1.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $436.23
- **Recent Low:** $337.24
- **Next Resistance:** $386.74 (50.0%)
- **Next Support:** $375.05 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 147. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:50.055657
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TSLA'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for TSLA: Too Many Requests. Rate limited. Try after a while.
```

---

## 148. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:50.284265
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TSLA'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for TSLA: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 149. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:50.300093
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TSLA', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for TSLA: Too Many Requests. Rate limited. Try after a while.
```

---

## 150. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:51.120852
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMD', 'curr_date': '2026-04-25'}`
- **Output Length:** 1603 chars

### Output
```
# Technical Analysis for AMD
**Date:** 2026-04-25
**Current Price:** $347.81

## Price Action
- **Daily Change:** +13.91%
- **5-Day Change:** +26.50%

## RSI (14)
- **Value:** 88.9 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 27.631
- **Signal Line:** 18.653
- **Histogram:** 8.978 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $221.57 (+57.0% from price)
- **200 SMA:** $206.60 (+68.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $331.96
- **Middle:** $250.85
- **Lower:** $169.75
- **Band Position:** 110%
- **Bandwidth:** 64.7% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $13.82 (4.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $327.08 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 94.2
- **%D:** 93.5
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 71.0 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $263.31 (+32.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1175.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $257.50
- **Current vs VWAP:** +35.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $352.99
- **Recent Low:** $188.22
- **Next Resistance:** $352.99 (0.0% (High))
- **Next Support:** $314.10 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 151. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:51.170713
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMD'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for AMD: Too Many Requests. Rate limited. Try after a while.
```

---

## 152. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:51.406636
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMD'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for AMD: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 153. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:51.426621
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMD', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for AMD: Too Many Requests. Rate limited. Try after a while.
```

---

## 154. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:52.124047
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NVDA', 'curr_date': '2026-04-25'}`
- **Output Length:** 1568 chars

### Output
```
# Technical Analysis for NVDA
**Date:** 2026-04-25
**Current Price:** $208.27

## Price Action
- **Daily Change:** +4.32%
- **5-Day Change:** +3.07%

## RSI (14)
- **Value:** 71.5 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 6.508
- **Signal Line:** 4.578
- **Histogram:** 1.930 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $185.02 (+12.6% from price)
- **200 SMA:** $182.83 (+13.9% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $214.44
- **Middle:** $188.38
- **Lower:** $162.33
- **Band Position:** 88%
- **Bandwidth:** 27.7% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $5.45 (2.6% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $200.10 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.3
- **%D:** 90.9
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 29.1 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $192.81 (+8.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1744.3M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $186.87
- **Current vs VWAP:** +11.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $210.95
- **Recent Low:** $164.27
- **Next Resistance:** $210.95 (0.0% (High))
- **Next Support:** $199.93 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 155. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:52.180945
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NVDA'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for NVDA: Too Many Requests. Rate limited. Try after a while.
```

---

## 156. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:52.410238
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NVDA'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for NVDA: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 157. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:52.430877
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NVDA', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for NVDA: Too Many Requests. Rate limited. Try after a while.
```

---

## 158. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:53.179643
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMZN', 'curr_date': '2026-04-25'}`
- **Output Length:** 1560 chars

### Output
```
# Technical Analysis for AMZN
**Date:** 2026-04-25
**Current Price:** $263.99

## Price Action
- **Daily Change:** +3.49%
- **5-Day Change:** +6.33%

## RSI (14)
- **Value:** 80.4 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 12.028
- **Signal Line:** 9.374
- **Histogram:** 2.655 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $218.64 (+20.7% from price)
- **200 SMA:** $226.40 (+16.6% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $274.90
- **Middle:** $232.95
- **Lower:** $191.00
- **Band Position:** 87%
- **Bandwidth:** 36.0% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $6.64 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $254.03 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 90.7
- **%D:** 88.8
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 45.8 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $238.89 (+10.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 839.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $233.15
- **Current vs VWAP:** +13.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $264.50
- **Recent Low:** $196.00
- **Next Resistance:** $264.50 (0.0% (High))
- **Next Support:** $248.33 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 159. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:53.237462
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMZN'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for AMZN: Too Many Requests. Rate limited. Try after a while.
```

---

## 160. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:53.483987
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMZN'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for AMZN: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 161. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:53.501583
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMZN', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for AMZN: Too Many Requests. Rate limited. Try after a while.
```

---

## 162. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:54.162640
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'META', 'curr_date': '2026-04-25'}`
- **Output Length:** 1505 chars

### Output
```
# Technical Analysis for META
**Date:** 2026-04-25
**Current Price:** $675.03

## Price Action
- **Daily Change:** +2.41%
- **5-Day Change:** +0.61%

## RSI (14)
- **Value:** 61.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 17.122
- **Signal Line:** 11.095
- **Histogram:** 6.027 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $629.62 (+7.2% from price)
- **200 SMA:** $679.03 (-0.6% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $728.19
- **Middle:** $624.47
- **Lower:** $520.74
- **Band Position:** 74%
- **Bandwidth:** 33.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $19.43 (2.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $645.89 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 69.7
- **%D:** 76.4
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 23.9 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $643.81 (+4.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -152.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $610.80
- **Current vs VWAP:** +10.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $691.52
- **Recent Low:** $520.26
- **Next Resistance:** $691.52 (0.0% (High))
- **Next Support:** $651.10 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 163. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:54.211890
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'META'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for META: Too Many Requests. Rate limited. Try after a while.
```

---

## 164. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:54.460429
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'META'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for META: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 165. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:54.478435
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'META', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for META: Too Many Requests. Rate limited. Try after a while.
```

---

## 166. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:55.198207
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AAPL', 'curr_date': '2026-04-25'}`
- **Output Length:** 1544 chars

### Output
```
# Technical Analysis for AAPL
**Date:** 2026-04-25
**Current Price:** $271.06

## Price Action
- **Daily Change:** -0.87%
- **5-Day Change:** -0.73%

## RSI (14)
- **Value:** 59.7 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 4.019
- **Signal Line:** 2.491
- **Histogram:** 1.528 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $260.15 (+4.2% from price)
- **200 SMA:** $253.34 (+7.0% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $277.48
- **Middle:** $261.40
- **Lower:** $245.32
- **Band Position:** 80%
- **Bandwidth:** 12.3% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $5.88 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $262.24 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 81.4
- **%D:** 83.0
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 11.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $264.06 (+2.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 914.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $260.96
- **Current vs VWAP:** +3.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $276.11
- **Recent Low:** $245.51
- **Next Resistance:** $276.11 (0.0% (High))
- **Next Support:** $268.89 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 167. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:55.245655
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AAPL'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for AAPL: Too Many Requests. Rate limited. Try after a while.
```

---

## 168. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:55.489883
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AAPL'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for AAPL: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 169. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:55.505374
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AAPL', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for AAPL: Too Many Requests. Rate limited. Try after a while.
```

---

## 170. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:56.178332
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'GOOGL', 'curr_date': '2026-04-25'}`
- **Output Length:** 1524 chars

### Output
```
# Technical Analysis for GOOGL
**Date:** 2026-04-25
**Current Price:** $344.40

## Price Action
- **Daily Change:** +1.63%
- **5-Day Change:** +2.07%

## RSI (14)
- **Value:** 69.0 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 10.419
- **Signal Line:** 7.811
- **Histogram:** 2.608 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $309.49 (+11.3% from price)
- **200 SMA:** $276.28 (+24.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $362.99
- **Middle:** $317.42
- **Lower:** $271.86
- **Band Position:** 80%
- **Bandwidth:** 28.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $7.90 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $332.55 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 89.4
- **%D:** 88.0
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 33.6 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $324.50 (+6.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 851.7M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $312.87
- **Current vs VWAP:** +10.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $345.27
- **Recent Low:** $272.11
- **Next Resistance:** $345.27 (0.0% (High))
- **Next Support:** $328.00 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 171. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:56.225754
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'GOOGL'}`
- **Output Length:** 100 chars

### Output
```
Error retrieving insider transactions for GOOGL: Too Many Requests. Rate limited. Try after a while.
```

---

## 172. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:56.448948
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'GOOGL'}`
- **Output Length:** 148 chars

### Output
```
Error fetching recommendation trends for GOOGL: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 173. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:56.467080
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'GOOGL', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 96 chars

### Output
```
Error retrieving options activity for GOOGL: Too Many Requests. Rate limited. Try after a while.
```

---

## 174. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:57.158103
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MSFT', 'curr_date': '2026-04-25'}`
- **Output Length:** 1515 chars

### Output
```
# Technical Analysis for MSFT
**Date:** 2026-04-25
**Current Price:** $424.62

## Price Action
- **Daily Change:** +2.13%
- **5-Day Change:** +1.57%

## RSI (14)
- **Value:** 63.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 9.852
- **Signal Line:** 4.644
- **Histogram:** 5.208 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $393.98 (+7.8% from price)
- **200 SMA:** $468.66 (-9.4% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $443.74
- **Middle:** $391.97
- **Lower:** $340.21
- **Band Position:** 82%
- **Bandwidth:** 26.4% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $10.49 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $408.88 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 82.9
- **%D:** 84.3
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 32.9 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $403.01 (+5.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -302.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $392.27
- **Current vs VWAP:** +8.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $433.70
- **Recent Low:** $356.28
- **Next Resistance:** $433.70 (0.0% (High))
- **Next Support:** $415.43 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 175. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:57.207766
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MSFT'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for MSFT: Too Many Requests. Rate limited. Try after a while.
```

---

## 176. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:57.444847
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MSFT'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for MSFT: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 177. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:57.460659
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MSFT', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for MSFT: Too Many Requests. Rate limited. Try after a while.
```

---

## 178. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:58.167755
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'QCOM', 'curr_date': '2026-04-25'}`
- **Output Length:** 1572 chars

### Output
```
# Technical Analysis for QCOM
**Date:** 2026-04-25
**Current Price:** $148.85

## Price Action
- **Daily Change:** +11.12%
- **5-Day Change:** +8.24%

## RSI (14)
- **Value:** 73.4 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 1.788
- **Signal Line:** -0.074
- **Histogram:** 1.862 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $134.31 (+10.8% from price)
- **200 SMA:** $155.11 (-4.0% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $143.03
- **Middle:** $131.50
- **Lower:** $119.96
- **Band Position:** 125%
- **Bandwidth:** 17.5% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $4.40 (3.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $142.25 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 79.8
- **%D:** 80.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 20.0 (Strengthening ↑)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $134.16 (+11.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 26.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $132.93
- **Current vs VWAP:** +12.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $151.54
- **Recent Low:** $121.99
- **Next Resistance:** $151.54 (0.0% (High))
- **Next Support:** $144.57 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 179. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:58.214912
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'QCOM'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for QCOM: Too Many Requests. Rate limited. Try after a while.
```

---

## 180. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:58.449518
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'QCOM'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for QCOM: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 181. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:58.472461
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'QCOM', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for QCOM: Too Many Requests. Rate limited. Try after a while.
```

---

## 182. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:59.138637
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AVGO', 'curr_date': '2026-04-25'}`
- **Output Length:** 1555 chars

### Output
```
# Technical Analysis for AVGO
**Date:** 2026-04-25
**Current Price:** $422.76

## Price Action
- **Daily Change:** +0.67%
- **5-Day Change:** +5.79%

## RSI (14)
- **Value:** 77.8 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 25.325
- **Signal Line:** 19.254
- **Histogram:** 6.070 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $341.00 (+24.0% from price)
- **200 SMA:** $335.60 (+26.0% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $453.75
- **Middle:** $364.33
- **Lower:** $274.91
- **Band Position:** 83%
- **Bandwidth:** 49.1% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $12.78 (3.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $403.59 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 90.4
- **%D:** 92.5
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 47.2 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $378.14 (+11.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 441.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $361.39
- **Current vs VWAP:** +17.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $429.31
- **Recent Low:** $289.96
- **Next Resistance:** $429.31 (0.0% (High))
- **Next Support:** $396.42 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 183. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:59.239332
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AVGO'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for AVGO: Too Many Requests. Rate limited. Try after a while.
```

---

## 184. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:59.480093
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AVGO'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for AVGO: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 185. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:49:59.496154
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AVGO', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for AVGO: Too Many Requests. Rate limited. Try after a while.
```

---

## 186. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:00.159760
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PLTR', 'curr_date': '2026-04-25'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for PLTR
**Date:** 2026-04-25
**Current Price:** $143.09

## Price Action
- **Daily Change:** +1.07%
- **5-Day Change:** -1.92%

## RSI (14)
- **Value:** 48.6 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.748
- **Signal Line:** -1.491
- **Histogram:** 0.742 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $144.35 (-0.9% from price)
- **200 SMA:** $164.43 (-13.0% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $155.56
- **Middle:** $142.38
- **Lower:** $129.21
- **Band Position:** 53%
- **Bandwidth:** 18.5% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $7.20 (5.0% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $132.29 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 62.9
- **%D:** 64.4
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 6.7 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $144.04 (-0.7% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 435.1M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $140.18
- **Current vs VWAP:** +2.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $162.40
- **Recent Low:** $122.68
- **Next Resistance:** $147.23 (38.2%)
- **Next Support:** $142.54 (50.0%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 187. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:00.208928
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PLTR'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for PLTR: Too Many Requests. Rate limited. Try after a while.
```

---

## 188. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:00.464766
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PLTR'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for PLTR: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 189. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:00.484864
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PLTR', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for PLTR: Too Many Requests. Rate limited. Try after a while.
```

---

## 190. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:01.133840
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'GOOG', 'curr_date': '2026-04-25'}`
- **Output Length:** 1523 chars

### Output
```
# Technical Analysis for GOOG
**Date:** 2026-04-25
**Current Price:** $342.32

## Price Action
- **Daily Change:** +1.35%
- **5-Day Change:** +2.06%

## RSI (14)
- **Value:** 69.7 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 10.182
- **Signal Line:** 7.455
- **Histogram:** 2.727 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $308.50 (+11.0% from price)
- **200 SMA:** $276.52 (+23.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $360.25
- **Middle:** $315.58
- **Lower:** $270.92
- **Band Position:** 80%
- **Bandwidth:** 28.3% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $7.63 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $330.88 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 90.0
- **%D:** 88.9
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 34.0 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $322.71 (+6.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 634.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $311.05
- **Current vs VWAP:** +10.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $343.69
- **Recent Low:** $271.54
- **Next Resistance:** $343.69 (0.0% (High))
- **Next Support:** $326.66 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 191. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:01.187822
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'GOOG'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for GOOG: Too Many Requests. Rate limited. Try after a while.
```

---

## 192. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:01.444478
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'GOOG'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for GOOG: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 193. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:01.460760
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'GOOG', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for GOOG: Too Many Requests. Rate limited. Try after a while.
```

---

## 194. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:02.178246
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'VZ', 'curr_date': '2026-04-25'}`
- **Output Length:** 1512 chars

### Output
```
# Technical Analysis for VZ
**Date:** 2026-04-25
**Current Price:** $46.38

## Price Action
- **Daily Change:** -1.78%
- **5-Day Change:** -0.13%

## RSI (14)
- **Value:** 43.4 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.591
- **Signal Line:** -0.572
- **Histogram:** -0.018 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $48.49 (-4.3% from price)
- **200 SMA:** $42.38 (+9.4% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $50.07
- **Middle:** $47.21
- **Lower:** $44.35
- **Band Position:** 35%
- **Bandwidth:** 12.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.02 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $44.85 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 66.2
- **%D:** 56.6
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 18.7 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $47.07 (-1.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -161.1M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $47.20
- **Current vs VWAP:** -1.7%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $50.91
- **Recent Low:** $44.27
- **Next Resistance:** $46.81 (61.8%)
- **Next Support:** $45.69 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 195. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:02.273039
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'VZ'}`
- **Output Length:** 97 chars

### Output
```
Error retrieving insider transactions for VZ: Too Many Requests. Rate limited. Try after a while.
```

---

## 196. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:02.503222
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'VZ'}`
- **Output Length:** 145 chars

### Output
```
Error fetching recommendation trends for VZ: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 197. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:02.523129
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'VZ', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 93 chars

### Output
```
Error retrieving options activity for VZ: Too Many Requests. Rate limited. Try after a while.
```

---

## 198. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:03.308030
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'LTBR', 'curr_date': '2026-04-25'}`
- **Output Length:** 1503 chars

### Output
```
# Technical Analysis for LTBR
**Date:** 2026-04-25
**Current Price:** $12.61

## Price Action
- **Daily Change:** -3.00%
- **5-Day Change:** -2.47%

## RSI (14)
- **Value:** 55.1 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.349
- **Signal Line:** 0.095
- **Histogram:** 0.254 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $11.92 (+5.8% from price)
- **200 SMA:** $15.83 (-20.4% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $13.73
- **Middle:** $11.58
- **Lower:** $9.43
- **Band Position:** 74%
- **Bandwidth:** 37.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.96 (7.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $11.17 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 66.0
- **%D:** 74.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 25.4 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $12.05 (+4.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 24.5M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $11.76
- **Current vs VWAP:** +7.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $14.20
- **Recent Low:** $9.78
- **Next Resistance:** $13.16 (23.6%)
- **Next Support:** $12.51 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 199. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:03.396073
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'LTBR'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for LTBR: Too Many Requests. Rate limited. Try after a while.
```

---

## 200. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:03.621975
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'LTBR'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for LTBR: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 201. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:03.640153
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'LTBR', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for LTBR: Too Many Requests. Rate limited. Try after a while.
```

---

## 202. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:04.466633
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CDP', 'curr_date': '2026-04-25'}`
- **Output Length:** 1525 chars

### Output
```
# Technical Analysis for CDP
**Date:** 2026-04-25
**Current Price:** $32.11

## Price Action
- **Daily Change:** +0.60%
- **5-Day Change:** -0.99%

## RSI (14)
- **Value:** 55.6 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.228
- **Signal Line:** 0.232
- **Histogram:** -0.004 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $31.62 (+1.5% from price)
- **200 SMA:** $29.23 (+9.9% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $33.15
- **Middle:** $31.70
- **Lower:** $30.25
- **Band Position:** 64%
- **Bandwidth:** 9.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.56 (1.7% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $31.28 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 39.6
- **%D:** 52.1
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 14.2 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $31.88 (+0.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 6.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $31.64
- **Current vs VWAP:** +1.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $32.96
- **Recent Low:** $30.24
- **Next Resistance:** $32.32 (23.6%)
- **Next Support:** $31.92 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 203. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:04.521172
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CDP'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for CDP: Too Many Requests. Rate limited. Try after a while.
```

---

## 204. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:04.767607
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CDP'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for CDP: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 205. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:04.784021
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CDP', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for CDP: Too Many Requests. Rate limited. Try after a while.
```

---

## 206. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:05.507412
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'KRC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1508 chars

### Output
```
# Technical Analysis for KRC
**Date:** 2026-04-25
**Current Price:** $31.88

## Price Action
- **Daily Change:** +1.34%
- **5-Day Change:** -0.81%

## RSI (14)
- **Value:** 65.9 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.721
- **Signal Line:** 0.329
- **Histogram:** 0.392 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $29.51 (+8.0% from price)
- **200 SMA:** $36.06 (-11.6% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $32.93
- **Middle:** $29.52
- **Lower:** $26.12
- **Band Position:** 85%
- **Bandwidth:** 23.1% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $0.96 (3.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $30.45 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 82.6
- **%D:** 83.0
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 31.9 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $30.19 (+5.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -24.3M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $29.30
- **Current vs VWAP:** +8.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $32.58
- **Recent Low:** $27.27
- **Next Resistance:** $32.58 (0.0% (High))
- **Next Support:** $31.33 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 207. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:05.555108
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'KRC'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for KRC: Too Many Requests. Rate limited. Try after a while.
```

---

## 208. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:05.805750
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'KRC'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for KRC: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 209. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:05.822769
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'KRC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for KRC: Too Many Requests. Rate limited. Try after a while.
```

---

## 210. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:06.612094
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'FULC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1506 chars

### Output
```
# Technical Analysis for FULC
**Date:** 2026-04-25
**Current Price:** $7.84

## Price Action
- **Daily Change:** -2.97%
- **5-Day Change:** -1.26%

## RSI (14)
- **Value:** 49.6 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.006
- **Signal Line:** -0.062
- **Histogram:** 0.056 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $8.17 (-4.1% from price)
- **200 SMA:** $8.85 (-11.4% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $8.60
- **Middle:** $7.74
- **Lower:** $6.88
- **Band Position:** 56%
- **Bandwidth:** 22.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.51 (6.5% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $7.08 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 49.2
- **%D:** 54.6
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 5.0 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $7.82 (+0.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 9.1M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $7.72
- **Current vs VWAP:** +1.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $12.03
- **Recent Low:** $6.47
- **Next Resistance:** $8.59 (61.8%)
- **Next Support:** $7.66 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 211. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:06.679975
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'FULC'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for FULC: Too Many Requests. Rate limited. Try after a while.
```

---

## 212. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:06.928985
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'FULC'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for FULC: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 213. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:06.947154
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'FULC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for FULC: Too Many Requests. Rate limited. Try after a while.
```

---

## 214. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:07.576416
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ARE', 'curr_date': '2026-04-25'}`
- **Output Length:** 1525 chars

### Output
```
# Technical Analysis for ARE
**Date:** 2026-04-25
**Current Price:** $47.40

## Price Action
- **Daily Change:** +2.20%
- **5-Day Change:** -3.23%

## RSI (14)
- **Value:** 51.8 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.144
- **Signal Line:** -0.587
- **Histogram:** 0.444 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $48.48 (-2.2% from price)
- **200 SMA:** $60.30 (-21.4% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $49.65
- **Middle:** $45.65
- **Lower:** $41.65
- **Band Position:** 72%
- **Bandwidth:** 17.5% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.87 (3.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $44.59 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.3
- **%D:** 73.7
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 13.4 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $46.69 (+1.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -28.6M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $45.36
- **Current vs VWAP:** +4.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $54.41
- **Recent Low:** $41.44
- **Next Resistance:** $47.93 (50.0%)
- **Next Support:** $46.39 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 215. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:07.683628
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ARE'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for ARE: Too Many Requests. Rate limited. Try after a while.
```

---

## 216. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:07.932978
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ARE'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for ARE: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 217. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:07.958661
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ARE', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for ARE: Too Many Requests. Rate limited. Try after a while.
```

---

## 218. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:08.705087
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'KEYS', 'curr_date': '2026-04-25'}`
- **Output Length:** 1552 chars

### Output
```
# Technical Analysis for KEYS
**Date:** 2026-04-25
**Current Price:** $346.96

## Price Action
- **Daily Change:** +0.12%
- **5-Day Change:** +3.53%

## RSI (14)
- **Value:** 75.0 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 16.281
- **Signal Line:** 14.636
- **Histogram:** 1.645 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $293.72 (+18.1% from price)
- **200 SMA:** $211.67 (+63.9% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $365.07
- **Middle:** $316.45
- **Lower:** $267.82
- **Band Position:** 81%
- **Bandwidth:** 30.7% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $9.99 (2.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $331.97 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 85.9
- **%D:** 87.2
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 41.0 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $321.90 (+7.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 58.8M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $315.21
- **Current vs VWAP:** +10.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $352.78
- **Recent Low:** $228.35
- **Next Resistance:** $352.78 (0.0% (High))
- **Next Support:** $323.41 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 219. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:08.755012
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'KEYS'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for KEYS: Too Many Requests. Rate limited. Try after a while.
```

---

## 220. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:08.998437
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'KEYS'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for KEYS: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 221. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:09.014184
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'KEYS', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for KEYS: Too Many Requests. Rate limited. Try after a while.
```

---

## 222. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:09.704917
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'HXL', 'curr_date': '2026-04-25'}`
- **Output Length:** 1500 chars

### Output
```
# Technical Analysis for HXL
**Date:** 2026-04-25
**Current Price:** $89.39

## Price Action
- **Daily Change:** -2.11%
- **5-Day Change:** -0.15%

## RSI (14)
- **Value:** 59.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 1.641
- **Signal Line:** 0.748
- **Histogram:** 0.893 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $85.49 (+4.6% from price)
- **200 SMA:** $73.26 (+22.0% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $91.87
- **Middle:** $83.93
- **Lower:** $75.99
- **Band Position:** 84%
- **Bandwidth:** 18.9% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $3.64 (4.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $83.93 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 61.8
- **%D:** 70.6
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 20.8 (Strengthening ↑)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $85.50 (+4.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 26.9M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $84.89
- **Current vs VWAP:** +5.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $98.26
- **Recent Low:** $76.27
- **Next Resistance:** $89.86 (38.2%)
- **Next Support:** $87.26 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 223. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:09.805721
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'HXL'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for HXL: Too Many Requests. Rate limited. Try after a while.
```

---

## 224. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:10.100659
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'HXL'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for HXL: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 225. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:10.126402
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'HXL', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for HXL: Too Many Requests. Rate limited. Try after a while.
```

---

## 226. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:10.888702
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'WSC', 'curr_date': '2026-04-25'}`
- **Output Length:** 1566 chars

### Output
```
# Technical Analysis for WSC
**Date:** 2026-04-25
**Current Price:** $22.66

## Price Action
- **Daily Change:** +0.76%
- **5-Day Change:** +6.38%

## RSI (14)
- **Value:** 74.7 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.852
- **Signal Line:** 0.336
- **Histogram:** 0.516 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $19.73 (+14.9% from price)
- **200 SMA:** $21.62 (+4.8% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $22.91
- **Middle:** $19.24
- **Lower:** $15.57
- **Band Position:** 97%
- **Bandwidth:** 38.1% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.98 (4.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $21.19 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 86.3
- **%D:** 84.0
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 42.0 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $20.04 (+13.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -39.4M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $19.69
- **Current vs VWAP:** +15.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $24.47
- **Recent Low:** $16.52
- **Next Resistance:** $24.47 (0.0% (High))
- **Next Support:** $22.59 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 227. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:10.943350
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'WSC'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for WSC: Too Many Requests. Rate limited. Try after a while.
```

---

## 228. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:11.174874
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'WSC'}`
- **Output Length:** 146 chars

### Output
```
Error fetching recommendation trends for WSC: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 229. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:11.191030
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'WSC', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for WSC: Too Many Requests. Rate limited. Try after a while.
```

---

## 230. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:12.068043
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'HOLX', 'curr_date': '2026-04-25'}`
- **Output Length:** 1539 chars

### Output
```
# Technical Analysis for HOLX
**Date:** 2026-04-25
**Current Price:** $76.01

## Price Action
- **Daily Change:** +0.00%
- **5-Day Change:** +0.56%

## RSI (14)
- **Value:** 68.7 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.152
- **Signal Line:** 0.108
- **Histogram:** 0.044 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $75.31 (+0.9% from price)
- **200 SMA:** $71.54 (+6.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $76.03
- **Middle:** $75.49
- **Lower:** $74.95
- **Band Position:** 98%
- **Bandwidth:** 1.4% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.21 (0.3% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $75.69 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 81.5
- **%D:** 76.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 32.0 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $75.58 (+0.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 120.8M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $75.78
- **Current vs VWAP:** +0.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $76.07
- **Recent Low:** $74.50
- **Next Resistance:** $76.07 (0.0% (High))
- **Next Support:** $75.70 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 231. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:12.116571
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'HOLX'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for HOLX: Too Many Requests. Rate limited. Try after a while.
```

---

## 232. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:12.340267
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'HOLX'}`
- **Output Length:** 147 chars

### Output
```
Error fetching recommendation trends for HOLX: FinnhubAPIException(status_code: 429): API limit reached. Please try again later. Remaining Limit: 0
```

---

## 233. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:12.360746
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'HOLX', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for HOLX: Too Many Requests. Rate limited. Try after a while.
```

---

## 234. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:13.084558
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'EXAS', 'curr_date': '2026-04-25'}`
- **Output Length:** 1588 chars

### Output
```
# Technical Analysis for EXAS
**Date:** 2026-04-25
**Current Price:** $104.91

## Price Action
- **Daily Change:** +0.00%
- **5-Day Change:** +0.93%

## RSI (14)
- **Value:** 76.1 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.442
- **Signal Line:** 0.359
- **Histogram:** 0.083 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $103.09 (+1.8% from price)
- **200 SMA:** $76.38 (+37.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $104.62
- **Middle:** $103.65
- **Lower:** $102.69
- **Band Position:** 115%
- **Bandwidth:** 1.9% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.35 (0.3% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $104.39 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.5
- **%D:** 80.8
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 58.0 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $103.75 (+1.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 220.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $104.05
- **Current vs VWAP:** +0.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $104.98
- **Recent Low:** $101.83
- **Next Resistance:** $104.98 (0.0% (High))
- **Next Support:** $104.24 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 235. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:13.132143
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'EXAS'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for EXAS: Too Many Requests. Rate limited. Try after a while.
```

---

## 236. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:13.379439
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'EXAS'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for EXAS

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 6
- **Hold**: 21
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 27.6% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 6
- **Hold**: 21
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 27.6% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 6
- **Hold**: 21
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 27.6% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 8
- **Hold**: 19
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 34.5% Bullish, 0.0% Bearish


```

---

## 237. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:13.396157
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'EXAS', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for EXAS: Too Many Requests. Rate limited. Try after a while.
```

---

## 238. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:14.035031
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'EA', 'curr_date': '2026-04-25'}`
- **Output Length:** 1544 chars

### Output
```
# Technical Analysis for EA
**Date:** 2026-04-25
**Current Price:** $202.67

## Price Action
- **Daily Change:** +0.08%
- **5-Day Change:** -0.34%

## RSI (14)
- **Value:** 50.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.342
- **Signal Line:** 0.492
- **Histogram:** -0.149 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $201.55 (+0.6% from price)
- **200 SMA:** $191.49 (+5.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $204.46
- **Middle:** $203.17
- **Lower:** $201.88
- **Band Position:** 31%
- **Bandwidth:** 1.3% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.91 (0.4% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $201.31 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 38.3
- **%D:** 49.2
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 8.8 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $202.80 (-0.1% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 42.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $203.14
- **Current vs VWAP:** -0.2%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $204.22
- **Recent Low:** $197.01
- **Next Resistance:** $204.22 (0.0% (High))
- **Next Support:** $202.52 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 239. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:14.106238
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'EA'}`
- **Output Length:** 97 chars

### Output
```
Error retrieving insider transactions for EA: Too Many Requests. Rate limited. Try after a while.
```

---

## 240. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:14.371658
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'EA'}`
- **Output Length:** 713 chars

### Output
```
## Analyst Recommendation Trends for EA

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 6
- **Hold**: 21
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 34.4% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 6
- **Hold**: 21
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 33.3% Bullish, 3.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 6
- **Hold**: 21
- **Sell**: 1
- **Strong Sell**: 1
- **Total Analysts**: 34

**Sentiment**: 32.4% Bullish, 5.9% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 6
- **Hold**: 21
- **Sell**: 1
- **Strong Sell**: 1
- **Total Analysts**: 34

**Sentiment**: 32.4% Bullish, 5.9% Bearish


```

---

## 241. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:14.387826
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'EA', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 93 chars

### Output
```
Error retrieving options activity for EA: Too Many Requests. Rate limited. Try after a while.
```

---

## 242. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:15.106440
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CPRT', 'curr_date': '2026-04-25'}`
- **Output Length:** 1541 chars

### Output
```
# Technical Analysis for CPRT
**Date:** 2026-04-25
**Current Price:** $33.07

## Price Action
- **Daily Change:** -2.39%
- **5-Day Change:** -2.04%

## RSI (14)
- **Value:** 41.5 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.294
- **Signal Line:** -0.479
- **Histogram:** 0.185 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $34.72 (-4.8% from price)
- **200 SMA:** $41.17 (-19.7% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $33.98
- **Middle:** $33.21
- **Lower:** $32.45
- **Band Position:** 41%
- **Bandwidth:** 4.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.83 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $31.82 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 49.0
- **%D:** 56.7
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 11.0 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $33.53 (-1.4% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -152.2M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $33.26
- **Current vs VWAP:** -0.6%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $39.61
- **Recent Low:** $32.20
- **Next Resistance:** $33.79 (78.6%)
- **Next Support:** $32.20 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 243. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:15.159294
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CPRT'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for CPRT: Too Many Requests. Rate limited. Try after a while.
```

---

## 244. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:15.422808
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CPRT'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for CPRT

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 5.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 5.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 20

**Sentiment**: 60.0% Bullish, 5.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 7
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 19

**Sentiment**: 63.2% Bullish, 5.3% Bearish


```

---

## 245. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:15.440644
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CPRT', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for CPRT: Too Many Requests. Rate limited. Try after a while.
```

---

## 246. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:16.141463
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DRI', 'curr_date': '2026-04-25'}`
- **Output Length:** 1512 chars

### Output
```
# Technical Analysis for DRI
**Date:** 2026-04-25
**Current Price:** $201.06

## Price Action
- **Daily Change:** -0.63%
- **5-Day Change:** -0.39%

## RSI (14)
- **Value:** 53.4 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.309
- **Signal Line:** -0.758
- **Histogram:** 1.067 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $201.80 (-0.4% from price)
- **200 SMA:** $194.19 (+3.5% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $203.89
- **Middle:** $196.02
- **Lower:** $188.15
- **Band Position:** 82%
- **Bandwidth:** 8.0% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $5.36 (2.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $193.01 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 76.2
- **%D:** 73.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 6.9 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $198.32 (+1.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -2.7M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $195.95
- **Current vs VWAP:** +2.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $218.96
- **Recent Low:** $186.91
- **Next Resistance:** $202.94 (50.0%)
- **Next Support:** $199.15 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 247. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:16.192561
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DRI'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for DRI: Too Many Requests. Rate limited. Try after a while.
```

---

## 248. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:16.423517
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DRI'}`
- **Output Length:** 722 chars

### Output
```
## Analyst Recommendation Trends for DRI

### 2026-04-01
- **Strong Buy**: 10
- **Buy**: 13
- **Hold**: 12
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 63.9% Bullish, 2.8% Bearish

### 2026-03-01
- **Strong Buy**: 10
- **Buy**: 13
- **Hold**: 13
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 37

**Sentiment**: 62.2% Bullish, 2.7% Bearish

### 2026-02-01
- **Strong Buy**: 10
- **Buy**: 13
- **Hold**: 13
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 37

**Sentiment**: 62.2% Bullish, 2.7% Bearish

### 2026-01-01
- **Strong Buy**: 10
- **Buy**: 13
- **Hold**: 13
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 37

**Sentiment**: 62.2% Bullish, 2.7% Bearish


```

---

## 249. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:16.443185
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DRI', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for DRI: Too Many Requests. Rate limited. Try after a while.
```

---

## 250. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:17.190867
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CPB', 'curr_date': '2026-04-25'}`
- **Output Length:** 1507 chars

### Output
```
# Technical Analysis for CPB
**Date:** 2026-04-25
**Current Price:** $20.62

## Price Action
- **Daily Change:** -1.62%
- **5-Day Change:** -3.01%

## RSI (14)
- **Value:** 39.6 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.497
- **Signal Line:** -0.651
- **Histogram:** 0.153 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $22.90 (-10.0% from price)
- **200 SMA:** $27.77 (-25.8% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $22.31
- **Middle:** $21.06
- **Lower:** $19.81
- **Band Position:** 32%
- **Bandwidth:** 11.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.63 (3.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $19.67 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 59.8
- **%D:** 57.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 22.8 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $21.10 (-2.3% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -195.3M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $21.02
- **Current vs VWAP:** -1.9%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $29.47
- **Recent Low:** $19.76
- **Next Resistance:** $21.84 (78.6%)
- **Next Support:** $19.76 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 251. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:17.241958
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CPB'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for CPB: Too Many Requests. Rate limited. Try after a while.
```

---

## 252. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:17.503293
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CPB'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for CPB

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 0
- **Hold**: 13
- **Sell**: 11
- **Strong Sell**: 1
- **Total Analysts**: 28

**Sentiment**: 10.7% Bullish, 42.9% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 0
- **Hold**: 14
- **Sell**: 10
- **Strong Sell**: 1
- **Total Analysts**: 27

**Sentiment**: 7.4% Bullish, 40.7% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 0
- **Hold**: 14
- **Sell**: 10
- **Strong Sell**: 1
- **Total Analysts**: 27

**Sentiment**: 7.4% Bullish, 40.7% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 0
- **Hold**: 15
- **Sell**: 9
- **Strong Sell**: 1
- **Total Analysts**: 27

**Sentiment**: 7.4% Bullish, 37.0% Bearish


```

---

## 253. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:17.520785
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CPB', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for CPB: Too Many Requests. Rate limited. Try after a while.
```

---

## 254. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:18.342932
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DTE', 'curr_date': '2026-04-25'}`
- **Output Length:** 1534 chars

### Output
```
# Technical Analysis for DTE
**Date:** 2026-04-25
**Current Price:** $146.88

## Price Action
- **Daily Change:** -0.35%
- **5-Day Change:** +0.69%

## RSI (14)
- **Value:** 52.2 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.153
- **Signal Line:** 0.481
- **Histogram:** -0.328 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $146.11 (+0.5% from price)
- **200 SMA:** $136.92 (+7.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $150.69
- **Middle:** $147.02
- **Lower:** $143.36
- **Band Position:** 48%
- **Bandwidth:** 5.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.48 (1.7% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $143.16 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 45.2
- **%D:** 34.3
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 6.9 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $146.46 (+0.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 44.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $146.96
- **Current vs VWAP:** -0.1%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $153.43
- **Recent Low:** $138.13
- **Next Resistance:** $147.58 (38.2%)
- **Next Support:** $145.78 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 255. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:18.397221
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DTE'}`
- **Output Length:** 98 chars

### Output
```
Error retrieving insider transactions for DTE: Too Many Requests. Rate limited. Try after a while.
```

---

## 256. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:18.656052
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DTE'}`
- **Output Length:** 710 chars

### Output
```
## Analyst Recommendation Trends for DTE

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 9
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 65.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 8
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 8
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 8
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish


```

---

## 257. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:18.686152
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DTE', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 94 chars

### Output
```
Error retrieving options activity for DTE: Too Many Requests. Rate limited. Try after a while.
```

---

## 258. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:19.630144
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'POET', 'curr_date': '2026-04-25'}`
- **Output Length:** 1582 chars

### Output
```
# Technical Analysis for POET
**Date:** 2026-04-25
**Current Price:** $15.10

## Price Action
- **Daily Change:** +28.84%
- **5-Day Change:** +75.79%

## RSI (14)
- **Value:** 81.2 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 1.593
- **Signal Line:** 0.751
- **Histogram:** 0.841 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $6.83 (+121.1% from price)
- **200 SMA:** $6.33 (+138.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $13.09
- **Middle:** $7.67
- **Lower:** $2.24
- **Band Position:** 118%
- **Bandwidth:** 141.6% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $1.24 (8.2% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $13.24 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 87.3
- **%D:** 82.5
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 55.5 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $8.67 (+74.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 178.7M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $9.87
- **Current vs VWAP:** +53.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $15.50
- **Recent Low:** $4.87
- **Next Resistance:** $15.50 (0.0% (High))
- **Next Support:** $12.99 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 259. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:19.691437
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'POET'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for POET: Too Many Requests. Rate limited. Try after a while.
```

---

## 260. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:19.926779
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'POET'}`
- **Output Length:** 707 chars

### Output
```
## Analyst Recommendation Trends for POET

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 4
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 7

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 4
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 7

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 4
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 7

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 4
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 7

**Sentiment**: 85.7% Bullish, 0.0% Bearish


```

---

## 261. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:19.944877
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'POET', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for POET: Too Many Requests. Rate limited. Try after a while.
```

---

## 262. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:20.548146
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'RDDT', 'curr_date': '2026-04-25'}`
- **Output Length:** 1533 chars

### Output
```
# Technical Analysis for RDDT
**Date:** 2026-04-25
**Current Price:** $154.89

## Price Action
- **Daily Change:** +1.26%
- **5-Day Change:** -6.85%

## RSI (14)
- **Value:** 53.9 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 4.546
- **Signal Line:** 3.249
- **Histogram:** 1.296 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $143.44 (+8.0% from price)
- **200 SMA:** $192.74 (-19.6% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $173.59
- **Middle:** $146.96
- **Lower:** $120.34
- **Band Position:** 65%
- **Bandwidth:** 36.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $8.19 (5.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $142.61 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 55.2
- **%D:** 70.8
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 15.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $151.12 (+2.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -11.3M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $145.75
- **Current vs VWAP:** +6.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $168.70
- **Recent Low:** $119.27
- **Next Resistance:** $157.03 (23.6%)
- **Next Support:** $149.82 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 263. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:20.598088
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'RDDT'}`
- **Output Length:** 99 chars

### Output
```
Error retrieving insider transactions for RDDT: Too Many Requests. Rate limited. Try after a while.
```

---

## 264. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:20.828322
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'RDDT'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for RDDT

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 16
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 68.6% Bullish, 2.9% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 16
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 68.6% Bullish, 2.9% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 16
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 68.6% Bullish, 2.9% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 15
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 69.7% Bullish, 3.0% Bearish


```

---

## 265. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-25T13:50:20.861241
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'RDDT', 'num_expirations': 3, 'curr_date': '2026-04-25'}`
- **Output Length:** 95 chars

### Output
```
Error retrieving options activity for RDDT: Too Many Requests. Rate limited. Try after a while.
```

---

## 266. Rank candidates

- **Type:** `llm`
- **Node:** ranker
- **Timestamp:** 2026-04-25T13:51:30.645510
- **Model:** `gemini-3-pro-preview`
- **Prompt Length:** 210527 chars
- **Output Length:** 2000 chars

### Prompt
```
You are a professional stock analyst selecting the best short-term trading opportunities from a pre-filtered candidate list.

CURRENT DATE: 2026-04-25

GOAL: Select UP TO 15 stocks with the highest probability of generating >5.0% returns within 1-7 days. If fewer than 15 candidates meet the quality bar, return only the ones that do. Quality over quantity — never pad the list with weak picks.

MINIMUM QUALITY BAR:
- Only include candidates where you have genuine conviction (final_score >= 65).
- If a candidate lacks a clear catalyst or has contradictory signals, SKIP it.
- It is better to return 5 excellent picks than 15 mediocre ones.

STRATEGY-SPECIFIC EVALUATION CRITERIA:
Each candidate was discovered by a specific scanner. Evaluate them using the criteria most relevant to their strategy:
- **insider_buying**: Focus on insider transaction SIZE relative to market cap, insider ROLE (CEO/CFO > Director), number of distinct insiders buying, and whether the stock is near support. Large cluster buys are strongest.
- **options_flow**: Focus on put/call ratio, absolute call VOLUME vs open interest, premium size, and whether flow aligns with the technical trend. Unusually low P/C ratios (<0.1) with high volume are strongest.
- **momentum / technical_breakout**: Focus on volume confirmation (>2x average), trend alignment (above key SMAs), and whether momentum is accelerating or fading. Avoid chasing extended moves (RSI >80).
- **earnings_play**: Focus on short interest (squeeze potential), pre-earnings accumulation signals, analyst estimate trends, and historical earnings surprise rate. Binary risk must be acknowledged.
- **social_dd**: Has shown 55% 30d win rate — strongest long-hold scanner. These setups combine social sentiment WITH technical confirmation (OBV, short interest, MACD). Score based on quality of technical/fundamental corroboration. A strong OBV + high short interest + bullish MACD warrants 65-75. DO NOT conflate with social_hype.
- **social_hype**: Treat as SPECULATIVE (14.3% 7d win rate, -4.84% avg 7d return). Require strong corroborating evidence. Pure social sentiment without data backing should score below 50.
- **short_squeeze**: Focus on short interest %, days to cover, cost to borrow, and whether a catalyst exists to trigger covering. High SI alone is not enough.
- **contrarian_value**: Focus on oversold technicals (RSI <30), fundamental support (earnings stability), and a clear reason why the selloff is overdone.
- **news_catalyst**: **AVOID by default** — 0% historical 7d win rate (-8.37% avg 7d return, n=8). Only score ≥55 if the catalyst is (1) not yet reflected in the intraday move, (2) mechanistic and specific (FDA decision, contract win, regulatory approval), NOT macroeconomic framing ('geopolitical tension', 'oil price', 'rate expectations'). Macro news_catalyst setups should score <50.
- **sector_rotation**: Focus on relative strength vs sector ETF, whether the stock is a laggard in an accelerating sector.
- **minervini**: Focus on the RS Rating (top 30% = RS>=70, top 10% = RS>=90) as the primary signal. Verify all 6 trend template conditions are met (price structure above rising SMAs). Strongest setups combine RS>=85 with price consolidating near highs (within 10-15% of 52w high) — these have minimal overhead supply. Penalize if RS Rating is borderline (70-75) without other confirming signals.
- **ml_signal**: Use the ML Win Probability as a strong quantitative signal. Scores above 65% deserve significant weight.

HISTORICAL INSIGHTS:
"Historical 1-day win rate: 40.9% (662 tracked)\nHistorical 7-day win rate: 44.4% (610 tracked)\nHistorical 30-day win rate: 39.0% (541 tracked)\n\nBest performing strategies (7-day):\n  - short_squeeze: 63.6% win rate, avg +2.3% return (11 samples)\n  - analyst_upgrade: 55.9% win rate, avg +0.2% return (34 samples)\n  - earnings_play: 47.6% win rate, avg -0.2% return (63 samples)\n\nWORST performing strategies (7-day) \u2014 penalize these heavily in scoring:\n  - pre_earnings_accumulation: 28.6% win rate, avg -1.9% return (7 samples)\n  - social_hype: 18.2% win rate, avg -4.1% return (22 samples)\n  - news_catalyst: 0.0% win rate, avg -9.1% return (8 samples)"

CANDIDATES FOR REVIEW:
### PRLD (Priority: CRITICAL)
- **Strategy Match**: insider_buying
- **Sources**: insider_buying | **Confluence**: 1 source(s)
- **Quant Pre-Score**: 15/100 | **ML Win Probability**: 41.0% (Predicted: NOT-WIN) | **Altman Z-Score**: None | **Piotroski F-Score**: None
- **Price**: $4.70 | **Current Price (numeric)**: 4.699999809265137 | **Intraday**: N/A | **Avg Volume**: 583,400
- **Short Interest**: N/A
- **Discovery Context**: 2 insiders buying PRLD. Largest: Dir, 10% Bonita David P purchased +$12,499,999
- **Business**: PRLD - Business description not available.
- **News**: No significant business catalysts found for this ticker during the period of April 24 to April 25, 2026.

**Technical Analysis**:
# Technical Analysis for PRLD
**Date:** 2026-04-25
**Current Price:** $4.70

## Price Action
- **Daily Change:** +6.82%
- **5-Day Change:** +4.21%

## RSI (14)
- **Value:** 60.0 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.359
- **Signal Line:** 0.375
- **Histogram:** -0.016 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $3.44 (+36.8% from price)
- **200 SMA:** $2.00 (+134.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $5.37
- **Middle:** $4.15
- **Lower:** $2.92
- **Band Position:** 73%
- **Bandwidth:** 59.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.54 (11.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $3.88 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 41.2
- **%D:** 45.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 44.8 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $4.20 (+12.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 18.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $4.44
- **Current vs VWAP:** +5.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $5.54
- **Recent Low:** $2.01
- **Next Resistance:** $4.71 (23.6%)
- **Next Support:** $4.19 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA


**Fundamentals**: N/A

**Insider Transactions**:
Error retrieving insider transactions for PRLD: Too Many Requests. Rate limited. Try after a while.

**Analyst Recommendations**:
## Analyst Recommendation Trends for PRLD

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish



**Options Activity**:
Error retrieving options activity for PRLD: Too Many Requests. Rate limited. Try after a while.

**Upcoming Earnings**: Earnings in 17 days (2026-05-12): EPS Est $-0.24, Rev Est $0

### TORO (Priority: CRITICAL)
- **Strategy Match**: insider_buying
- **Sources**: insider_buying | **Confluence**: 1 source(s)
- **Quant Pre-Score**: 10/100 | **ML Win Probability**: 41.0% (Predicted: NOT-WIN) | **Altman Z-Score**: None | **Piotroski F-Score**: None
- **Price**: $6.90 | **Current Price (numeric)**: 6.900000095367432 | **Intraday**: N/A | **Avg Volume**: 2,410,200
- **Short Interest**: N/A
- **Discovery Context**: CEO Panagiotidis Petros Panag purchased +$14,254,646 of TORO
- **Business**: TORO - Business description not available.
- **News**: No significant business catalysts found for this ticker during the period of April 24 to April 25, 2026.

**Technical Analysis**:
# Technical Analysis for TORO
**Date:** 2026-04-25
**Current Price:** $6.90

## Price Action
- **Daily Change:** +3.45%
- **5-Day Change:** +73.37%

## RSI (14)
- **Value:** 87.9 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.597
- **Signal Line:** 0.254
- **Histogram:** 0.342 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $3.92 (+75.9% from price)
- **200 SMA:** $3.03 (+127.6% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $6.44
- **Middle:** $4.27
- **Lower:** $2.09
- **Band Position:** 110%
- **Bandwidth:** 102.1% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.51 (7.4% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $6.14 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 70.8
- **%D:** 64.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 61.2 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $4.60 (+50.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 41.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $6.16
- **Current vs VWAP:** +12.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $7.70
- **Recent Low:** $3.37
- **Next Resistance:** $7.70 (0.0% (High))
- **Next Support:** $6.68 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA


**Fundamentals**: N/A

**Insider Transactions**:
Error retrieving insider transactions for TORO: Too Many Requests. Rate limited. Try after a while.

**Analyst Recommendations**:
No recommendation trends data found for TORO... [truncated]
```

### Output
```
```json
{
  "rankings": [
    {
      "rank": 1,
      "ticker": "KLRA",
      "company_name": "KLRA",
      "current_price": 22.47,
      "strategy_match": "insider_buying",
      "final_score": 90,
      "confidence": 9,
      "risk_level": "high",
      "reason": "A massive cluster of 5 insiders purchased over $326 million worth of shares at an average price of $16.00, providing an incredibly strong fundamental conviction signal. The stock recently debuted and saw extreme initial volatility, leaving it technically oversold with an RSI of 21.5. This severe technical disconnect combined with immense insider support creates an asymmetric reward profile for a sharp mean-reversion bounce. The primary risk is the inherent volatility of a newly listed IPO finding its price discovery equilibrium.",
      "description": "A newly public biotechnology company focused on obesity drug development."
    },
    {
      "rank": 2,
      "ticker": "ARXS",
      "company_name": "ARXS",
      "current_price": 35.00,
      "strategy_match": "insider_buying",
      "final_score": 85,
      "confidence": 8,
      "risk_level": "moderate",
      "reason": "ARXS displays a rare combination of a severely oversold technical state (RSI of 18.8) directly coinciding with significant insider accumulation. A distinct cluster of 7 executives deployed $5.64 million to purchase shares at an average price of $28.00 within the last 14 days, establishing a firm fundamental floor. With the stock price now trading at the lower Bollinger Band, the stage is set for a robust technical bounce off extreme conditions. The main risk is whether broader sector weakness continues to suppress the stock in the near term despite insider conviction.",
      "description": "ARXS - Business description not available."
    },
    {
      "rank": 3,
      "ticker": "INTC",
      "company_name": "Intel Corporation",
      "current_price": 66.78,
      "strategy_match": "pead_drift",
      "final_score": 80,
      "confi
```

---

