# Tool Execution Logs - 2026-04-20

## 1. Check average volume (batch)

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:26:32.178718
- **Tool:** `get_average_volume_batch`
- **Parameters:** `{'symbols': ['BEAT', 'BZUN', 'NKE', 'CAG', 'GLOO', 'FTLF', 'ABX', 'VNCE', 'DTST', 'KOPN', 'CCEL', 'GOVX', 'PXED', 'BEEM', 'ARAI', 'KALA', 'SVRE', 'BORR', 'CGCT', 'VANI', 'SST', 'WKSP', 'ESEA', 'HERZ', 'SVBL', 'XZO', 'LOVE', 'TXN', 'ALV', 'CDNS', 'VZ', 'QCOM', 'AAPL', 'ABCL', 'FMBL', 'HBCP', 'BOKF', 'ZION', 'WASH', 'SMBK', 'ISTR', 'SFBS', 'WTFC', 'RBB', 'OKTA', 'TSLA', 'AMD', 'MRVL', 'WMT', 'JPM', 'NFLX', 'APP', 'BSX', 'ORCL', 'IBM', 'UBER', 'CMCSA', 'HOLX', 'EXAS', 'EA', 'CTAS', 'GTLS', 'SEE', 'DGX', 'SO', 'DDOG', 'SNX', 'PRGO', 'PWR', 'BURL', 'Q', 'MTSI', 'ASTS', 'SEC', 'DERM'], 'lookback_days': 10, 'curr_date': '2026-04-20', 'cache_key': 'default'}`
- **Output Length:** 12551 chars

### Output
```
{"NKE": {"symbol": "NKE", "average_volume": 24950914.7, "latest_volume": 3220647.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "CAG": {"symbol": "CAG", "average_volume": 13795431.8, "latest_volume": 1676218.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "TXN": {"symbol": "TXN", "average_volume": 5153005.9, "latest_volume": 1271059.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "CDNS": {"symbol": "CDNS", "average_volume": 2091857.7, "latest_volume": 311677.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "VZ": {"symbol": "VZ", "average_volume": 24954628.1, "latest_volume": 2489781.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "QCOM": {"symbol": "QCOM", "average_volume": 8401240.0, "latest_volume": 1190200.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "AAPL": {"symbol": "AAPL", "average_volume": 41080342.6, "latest_volume": 9052726.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "ZION": {"symbol": "ZION", "average_volume": 1362232.8, "latest_volume": 301828.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "WTFC": {"symbol": "WTFC", "average_volume": 353306.0, "latest_volume": 36860.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "OKTA": {"symbol": "OKTA", "average_volume": 4743526.1, "latest_volume": 1087261.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "TSLA": {"symbol": "TSLA", "average_volume": 66796286.8, "latest_volume": 19797868.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "AMD": {"symbol": "AMD", "average_volume": 31094270.7, "latest_volume": 11857907.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "MRVL": {"symbol": "MRVL", "average_volume": 27290947.0, "latest_volume": 15129870.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "WMT": {"symbol": "WMT", "average_volume": 15665113.7, "latest_volume": 2739937.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "JPM": {"symbol": "JPM", "average_volume": 8135906.6, "latest_volume": 1316266.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "NFLX": {"symbol": "NFLX", "average_volume": 42458854.7, "latest_volume": 12573647.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "BSX": {"symbol": "BSX", "average_volume": 12251483.8, "latest_volume": 2381038.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "ORCL": {"symbol": "ORCL", "average_volume": 34846071.8, "latest_volume": 6259918.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "IBM": {"symbol": "IBM", "average_volume": 4274198.2, "latest_volume": 1218082.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "CMCSA": {"symbol": "CMCSA", "average_volume": 22072376.6, "latest_volume": 2631066.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "HOLX": {"symbol": "HOLX", "average_volume": 12273089.2, "latest_volume": 0.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "EXAS": {"symbol": "EXAS", "average_volume": 3743537.9, "latest_volume": 0.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "EA": {"symbol": "EA", "average_volume": 1552043.7, "latest_volume": 103137.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "CTAS": {"symbol": "CTAS", "average_volume": 1883916.9, "latest_volume": 174969.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "GTLS": {"symbol": "GTLS", "average_volume": 1197603.3, "latest_volume": 299233.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "SEE": {"symbol": "SEE", "average_volume": 3048100.0, "latest_volume": 0.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "DGX": {"symbol": "DGX", "average_volume": 819066.7, "latest_volume": 70467.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "SO": {"symbol": "SO", "average_volume": 3925262.5, "latest_volume": 274725.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "DDOG": {"symbol": "DDOG", "average_volume": 4675101.3, "latest_volume": 472613.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "PRGO": {"symbol": "PRGO", "average_volume": 2534696.2, "latest_volume": 186062.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "PWR": {"symbol": "PWR", "average_volume": 788530.8, "latest_volume": 95808.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-20", "error": null}, "BEAT": {"symbol": "BEAT", "average_volume": 2170301.2, "latest_volume": 810312.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "BZUN": {"symbol": "BZUN", "average_volume": 576968.4, "latest_volume": 147884.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "GLOO": {"symbol": "GLOO", "average_volume": 208134.9, "latest_volume": 31149.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "FTLF": {"symbol": "FTLF", "average_volume": 45152.3, "latest_volume": 1223.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "ABX": {"symbol": "ABX", "average_volume": 593577.8, "latest_volume": 122978.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "VNCE": {"symbol": "VNCE", "average_volume": 199632.6, "latest_volume": 411926.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "DTST": {"symbol": "DTST", "average_volume": 76270.0, "latest_volume": 2500.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "KOPN": {"symbol": "KOPN", "average_volume": 3139141.3, "latest_volume": 537813.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "CCEL": {"symbol": "CCEL", "average_volume": 9975.8, "latest_volume": 1058.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "GOVX": {"symbol": "GOVX", "average_volume": 77645.7, "latest_volume": 84557.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "PXED": {"symbol": "PXED", "average_volume": 166219.0, "latest_volume": 13790.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "BEEM": {"symbol": "BEEM", "average_volume": 358075.4, "latest_volume": 55554.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "ARAI": {"symbol": "ARAI", "average_volume": 53178270.9, "latest_volume": 1528509.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "KALA": {"symbol": "KALA", "average_volume": 2406279.4, "latest_volume": 308294.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "SVRE": {"symbol": "SVRE", "average_volume": 43117.1, "latest_volume": 5571.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "BORR": {"symbol": "BORR", "average_volume": 7602642.9, "latest_volume": 1314329.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "CGCT": {"symbol": "CGCT", "average_volume": 142664.2, "latest_volume": 18442.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "VANI": {"symbol": "VANI", "average_volume": 292428.7, "latest_volume": 124987.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "SST": {"symbol": "SST", "average_volume": 1021292.5, "latest_volume": 5582725.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "WKSP": {"symbol": "WKSP", "average_volume": 289712.7, "latest_volume": 67327.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "ESEA": {"symbol": "ESEA", "average_volume": 78648.6, "latest_volume": 8986.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "HERZ": {"symbol": "HERZ", "average_volume": 6532.2, "latest_volume": 2222.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "SVBL": {"symbol": "SVBL", "average_volume": 26820.0, "latest_volume": 137100.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "XZO": {"symbol": "XZO", "average_volume": 149666.6, "latest_volume": 12666.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "LOVE": {"symbol": "LOVE", "average_volume": 323476.4, "latest_volume": 95864.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "ALV": {"symbol": "ALV", "average_volume": 977264.9, "latest_volume": 305549.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "ABCL": {"symbol": "ABCL", "average_volume": 2926059.0, "latest_volume": 1563890.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "FMBL": {"symbol": "FMBL", "average_volume": 90.2, "latest_volume": 2.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-20", "error": null}, "HBCP": {"symbol": "HBCP", "average_volume": 143119.8, "latest_volume": 69798.0, "lookback_days": 10, "source": "download", "cache_date": "2026-... [truncated]
```

---

## 2. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:05.966248
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NKE', 'curr_date': '2026-04-20'}`
- **Output Length:** 1523 chars

### Output
```
# Technical Analysis for NKE
**Date:** 2026-04-20
**Current Price:** $46.03

## Price Action
- **Daily Change:** +0.72%
- **5-Day Change:** +7.27%

## RSI (14)
- **Value:** 38.8 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -3.045
- **Signal Line:** -3.464
- **Histogram:** 0.419 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $54.92 (-16.2% from price)
- **200 SMA:** $64.79 (-29.0% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $55.97
- **Middle:** $47.43
- **Lower:** $38.90
- **Band Position:** 42%
- **Bandwidth:** 36.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.65 (3.6% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $43.55 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 65.2
- **%D:** 44.3
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 37.2 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $47.43 (-2.9% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -246.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $46.33
- **Current vs VWAP:** -0.6%
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

## 3. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:06.344352
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NKE'}`
- **Output Length:** 681 chars

### Output
```
# Insider Transactions for NKE
# Data retrieved on: 2026-04-20 14:28:06

## Summary
- **Insider Sentiment:** BULLISH ⚡ (more buying than selling)
- **Purchase:** 4 transactions
- **Total Purchases Value:** $2,734,112
- **Largest Purchase (3mo):** 2026-04-10 | COOK TIMOTHY D (Director) | 25000 shares | $1060750.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-13,HILL ELLIOTT,Chief Executive Officer,Purchase,23660,1000000.0,D
2026-04-10,COOK TIMOTHY D,Director,Purchase,25000,1060750.0,D
2026-04-09,ROGERS JOHN W JR,Director,Purchase,4000,173360.0,D
2026-04-07,SWAN ROBERT HOLMES,Director,Purchase,11781,500002.0,D

```

---

## 4. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:06.557506
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

## 5. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:07.048836
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NKE', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 927 chars

### Output
```
## Options Activity for NKE

**Available Expirations:** 18 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 9,039 | 22,366 | 2.474 |
| Open Interest | 99,081 | 30,832 | 0.311 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-04-24 | $46.0 | 16,242 | 1,001 | 16.23x | 29.9% |
| PUT | 2026-04-24 | $46.5 | 616 | 95 | 6.48x | 31.0% |
| PUT | 2026-04-24 | $45.5 | 2,201 | 2,194 | 1.0x | 31.2% |
| PUT | 2026-05-01 | $43.5 | 400 | 610 | 0.66x | 31.4% |
| CALL | 2026-05-01 | $40.0 | 104 | 181 | 0.57x | 51.0% |
| PUT | 2026-04-24 | $44.5 | 396 | 753 | 0.53x | 31.5% |

```

---

## 6. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:09.584146
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ABX', 'curr_date': '2026-04-20'}`
- **Output Length:** 1518 chars

### Output
```
# Technical Analysis for ABX
**Date:** 2026-04-20
**Current Price:** $9.26

## Price Action
- **Daily Change:** +4.51%
- **5-Day Change:** +8.18%

## RSI (14)
- **Value:** 59.6 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.039
- **Signal Line:** -0.153
- **Histogram:** 0.114 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $8.82 (+5.0% from price)
- **200 SMA:** $7.03 (+31.6% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $9.41
- **Middle:** $8.49
- **Lower:** $7.56
- **Band Position:** 92%
- **Bandwidth:** 21.7% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $0.47 (5.1% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $8.55 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.7
- **%D:** 76.7
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 14.7 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $8.66 (+7.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 2.7M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $8.44
- **Current vs VWAP:** +9.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $10.53
- **Recent Low:** $7.18
- **Next Resistance:** $9.74 (23.6%)
- **Next Support:** $9.25 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 7. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:09.973518
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ABX'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for ABX in the last 3 months.
```

---

## 8. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:10.158087
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ABX'}`
- **Output Length:** 706 chars

### Output
```
## Analyst Recommendation Trends for ABX

### 2026-04-01
- **Strong Buy**: 1
- **Buy**: 6
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 77.8% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 1
- **Buy**: 7
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 1
- **Buy**: 7
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 1
- **Buy**: 7
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 9

**Sentiment**: 88.9% Bullish, 0.0% Bearish


```

---

## 9. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:10.605253
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ABX', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 626 chars

### Output
```
## Options Activity for ABX

**Available Expirations:** 4 dates
**Analyzing:** 2026-05-15, 2026-08-21, 2026-11-20

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 490 | 42 | 0.086 |
| Open Interest | 4,628 | 1,860 | 0.402 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-08-21 | $12.5 | 350 | 423 | 0.83x | 60.8% |

```

---

## 10. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:20.856526
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TXN', 'curr_date': '2026-04-20'}`
- **Output Length:** 1589 chars

### Output
```
# Technical Analysis for TXN
**Date:** 2026-04-20
**Current Price:** $229.82

## Price Action
- **Daily Change:** +3.01%
- **5-Day Change:** +6.05%

## RSI (14)
- **Value:** 74.2 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 6.459
- **Signal Line:** 3.000
- **Histogram:** 3.459 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $206.63 (+11.2% from price)
- **200 SMA:** $189.00 (+21.6% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $229.86
- **Middle:** $203.26
- **Lower:** $176.67
- **Band Position:** 100%
- **Bandwidth:** 26.2% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $6.12 (2.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $220.64 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 95.4
- **%D:** 92.8
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 25.5 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $208.75 (+10.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 29.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $201.58
- **Current vs VWAP:** +14.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $231.32
- **Recent Low:** $184.90
- **Next Resistance:** $231.32 (0.0% (High))
- **Next Support:** $220.36 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 11. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:21.320309
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TXN'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for TXN in the last 3 months.
```

---

## 12. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:21.533654
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TXN'}`
- **Output Length:** 722 chars

### Output
```
## Analyst Recommendation Trends for TXN

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 22
- **Sell**: 5
- **Strong Sell**: 1
- **Total Analysts**: 44

**Sentiment**: 36.4% Bullish, 13.6% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 23
- **Sell**: 5
- **Strong Sell**: 1
- **Total Analysts**: 45

**Sentiment**: 35.6% Bullish, 13.3% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 23
- **Sell**: 5
- **Strong Sell**: 1
- **Total Analysts**: 45

**Sentiment**: 35.6% Bullish, 13.3% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 21
- **Sell**: 6
- **Strong Sell**: 1
- **Total Analysts**: 44

**Sentiment**: 36.4% Bullish, 15.9% Bearish


```

---

## 13. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:21.957863
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TXN', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 861 chars

### Output
```
## Options Activity for TXN

**Available Expirations:** 15 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 999 | 1,586 | 1.588 |
| Open Interest | 9,074 | 7,568 | 0.834 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $240.0 | 102 | 65 | 1.57x | 50.8% |
| CALL | 2026-04-24 | $247.5 | 144 | 147 | 0.98x | 68.6% |
| PUT | 2026-05-01 | $230.0 | 103 | 105 | 0.98x | 50.9% |
| PUT | 2026-04-24 | $140.0 | 105 | 117 | 0.9x | 184.4% |
| PUT | 2026-04-24 | $205.0 | 555 | 630 | 0.88x | 77.6% |

```

---

## 14. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:24.443784
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ALV', 'curr_date': '2026-04-20'}`
- **Output Length:** 1560 chars

### Output
```
# Technical Analysis for ALV
**Date:** 2026-04-20
**Current Price:** $118.92

## Price Action
- **Daily Change:** +6.82%
- **5-Day Change:** +7.74%

## RSI (14)
- **Value:** 69.2 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.879
- **Signal Line:** -0.692
- **Histogram:** 1.571 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $112.19 (+6.0% from price)
- **200 SMA:** $117.53 (+1.2% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $115.73
- **Middle:** $107.06
- **Lower:** $98.40
- **Band Position:** 118%
- **Bandwidth:** 16.2% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $3.63 (3.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $113.47 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.2
- **%D:** 70.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 18.4 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $109.23 (+8.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -1.7M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $107.99
- **Current vs VWAP:** +10.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $127.38
- **Recent Low:** $99.16
- **Next Resistance:** $120.72 (23.6%)
- **Next Support:** $116.60 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 15. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:24.837181
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ALV'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for ALV in the last 3 months.
```

---

## 16. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:25.003387
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ALV'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for ALV

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 69.6% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 69.6% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 12
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 73.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 12
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 73.9% Bullish, 0.0% Bearish


```

---

## 17. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:25.504436
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ALV', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 437 chars

### Output
```
## Options Activity for ALV

**Available Expirations:** 5 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-09-18

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 156 | 42 | 0.269 |
| Open Interest | 464 | 860 | 1.853 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bearish positioning

*No unusual options activity detected.*

```

---

## 18. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:27.860969
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CDNS', 'curr_date': '2026-04-20'}`
- **Output Length:** 1565 chars

### Output
```
# Technical Analysis for CDNS
**Date:** 2026-04-20
**Current Price:** $311.03

## Price Action
- **Daily Change:** +1.33%
- **5-Day Change:** +7.92%

## RSI (14)
- **Value:** 62.8 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 2.849
- **Signal Line:** -1.299
- **Histogram:** 4.148 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $290.32 (+7.1% from price)
- **200 SMA:** $321.23 (-3.2% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $308.57
- **Middle:** $284.99
- **Lower:** $261.42
- **Band Position:** 105%
- **Bandwidth:** 16.5% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $11.15 (3.6% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $294.30 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 81.9
- **%D:** 72.0
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 15.4 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $289.92 (+7.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -6.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $285.18
- **Current vs VWAP:** +9.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $318.00
- **Recent Low:** $263.38
- **Next Resistance:** $318.00 (0.0% (High))
- **Next Support:** $305.11 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 19. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:28.370516
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CDNS'}`
- **Output Length:** 478 chars

### Output
```
# Insider Transactions for CDNS
# Data retrieved on: 2026-04-20 14:28:28

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Total Sales Value:** $280,190
- **Largest Sale (3mo):** 2026-04-01 | CUNNINGHAM PAUL ALEXANDER (Officer) | 1000 shares | $280190 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-01,CUNNINGHAM PAUL ALEXANDER,Officer,Sale,1000,280190,D

```

---

## 20. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:28.529660
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CDNS'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for CDNS

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 19
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 87.5% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 19
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 87.5% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 81.2% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 81.2% Bullish, 0.0% Bearish


```

---

## 21. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:29.012094
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CDNS', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 442 chars

### Output
```
## Options Activity for CDNS

**Available Expirations:** 15 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 548 | 208 | 0.38 |
| Open Interest | 1,962 | 1,471 | 0.75 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

*No unusual options activity detected.*

```

---

## 22. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:32.155816
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'QCOM', 'curr_date': '2026-04-20'}`
- **Output Length:** 1559 chars

### Output
```
# Technical Analysis for QCOM
**Date:** 2026-04-20
**Current Price:** $136.20

## Price Action
- **Daily Change:** +1.29%
- **5-Day Change:** +3.78%

## RSI (14)
- **Value:** 61.2 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.435
- **Signal Line:** -1.972
- **Histogram:** 1.537 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $134.26 (+1.4% from price)
- **200 SMA:** $155.59 (-12.5% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $135.39
- **Middle:** $129.29
- **Lower:** $123.19
- **Band Position:** 107%
- **Bandwidth:** 9.4% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $3.45 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $131.02 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.5
- **%D:** 78.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 11.8 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $131.11 (+3.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 43.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $129.66
- **Current vs VWAP:** +5.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $146.00
- **Recent Low:** $121.99
- **Next Resistance:** $136.83 (38.2%)
- **Next Support:** $134.00 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 23. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:32.551867
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'QCOM'}`
- **Output Length:** 711 chars

### Output
```
# Insider Transactions for QCOM
# Data retrieved on: 2026-04-20 14:28:32

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 2 transactions
- **Award/Grant:** 2 transactions
- **Total Sales Value:** $336,522
- **Largest Sale (3mo):** 2026-04-13 | PALKHIWALA AKASH J (Chief Operating Officer) | 2500 shares | $325854.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-13,PALKHIWALA AKASH J,Chief Operating Officer,Sale,2500,325854.0,D
2026-04-02,GRECH PATRICIA Y,Officer,Sale,85,10668.0,I
2026-03-31,MCLAUGHLIN MARK D,Director,Award/Grant,538,0.0,D
2026-03-31,TRICOIRE JEAN-PASCAL,Director,Award/Grant,262,0.0,D

```

---

## 24. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:32.733729
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'QCOM'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for QCOM

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 13
- **Hold**: 24
- **Sell**: 3
- **Strong Sell**: 0
- **Total Analysts**: 46

**Sentiment**: 41.3% Bullish, 6.5% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 24
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 46

**Sentiment**: 45.7% Bullish, 2.2% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 21
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 45

**Sentiment**: 51.1% Bullish, 2.2% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 17
- **Hold**: 19
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 44

**Sentiment**: 54.5% Bullish, 2.3% Bearish


```

---

## 25. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:33.154120
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'QCOM', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 845 chars

### Output
```
## Options Activity for QCOM

**Available Expirations:** 17 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,082 | 2,787 | 0.904 |
| Open Interest | 38,378 | 16,799 | 0.438 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $112.0 | 182 | 62 | 2.94x | 71.3% |
| PUT | 2026-05-01 | $113.0 | 235 | 88 | 2.67x | 68.8% |
| PUT | 2026-04-24 | $90.0 | 103 | 115 | 0.9x | 250.0% |
| PUT | 2026-04-24 | $112.0 | 220 | 289 | 0.76x | 87.9% |
| PUT | 2026-04-24 | $118.0 | 214 | 308 | 0.69x | 59.8% |

```

---

## 26. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:35.011811
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AAPL', 'curr_date': '2026-04-20'}`
- **Output Length:** 1573 chars

### Output
```
# Technical Analysis for AAPL
**Date:** 2026-04-20
**Current Price:** $270.23

## Price Action
- **Daily Change:** +2.59%
- **5-Day Change:** +4.26%

## RSI (14)
- **Value:** 64.2 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 2.061
- **Signal Line:** 0.237
- **Histogram:** 1.823 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $260.55 (+3.7% from price)
- **200 SMA:** $251.83 (+7.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $268.64
- **Middle:** $256.39
- **Lower:** $244.13
- **Band Position:** 106%
- **Bandwidth:** 9.6% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $6.11 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $261.06 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 87.7
- **%D:** 84.0
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 10.7 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $259.26 (+4.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1043.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $255.85
- **Current vs VWAP:** +5.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $280.65
- **Recent Low:** $245.51
- **Next Resistance:** $272.35 (23.6%)
- **Next Support:** $267.22 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 27. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:35.462656
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AAPL'}`
- **Output Length:** 884 chars

### Output
```
# Insider Transactions for AAPL
# Data retrieved on: 2026-04-20 14:28:35

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Unknown:** 5 transactions
- **Sale:** 2 transactions
- **Total Sales Value:** $24,173,073
- **Largest Sale (3mo):** 2026-04-02 | COOK TIMOTHY D (Chief Executive Officer) | 64949 shares | $16512198.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,PAREKH KEVAN,Chief Financial Officer,Unknown,10928,,D
2026-04-15,BORDERS BEN,Officer,Unknown,1717,,D
2026-04-02,O'BRIEN DEIRDRE,Officer,Sale,30002,7660875.0,D
2026-04-02,COOK TIMOTHY D,Chief Executive Officer,Sale,64949,16512198.0,D
2026-04-01,O'BRIEN DEIRDRE,Officer,Unknown,64317,,D
2026-04-01,KHAN SABIH,Chief Operating Officer,Unknown,64317,,D
2026-04-01,COOK TIMOTHY D,Chief Executive Officer,Unknown,131576,,D

```

---

## 28. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:35.605278
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AAPL'}`
- **Output Length:** 723 chars

### Output
```
## Analyst Recommendation Trends for AAPL

### 2026-04-01
- **Strong Buy**: 14
- **Buy**: 23
- **Hold**: 15
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 54

**Sentiment**: 68.5% Bullish, 3.7% Bearish

### 2026-03-01
- **Strong Buy**: 14
- **Buy**: 22
- **Hold**: 16
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 54

**Sentiment**: 66.7% Bullish, 3.7% Bearish

### 2026-02-01
- **Strong Buy**: 14
- **Buy**: 21
- **Hold**: 17
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 54

**Sentiment**: 64.8% Bullish, 3.7% Bearish

### 2026-01-01
- **Strong Buy**: 14
- **Buy**: 21
- **Hold**: 16
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 53

**Sentiment**: 66.0% Bullish, 3.8% Bearish


```

---

## 29. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:36.135135
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AAPL', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1203 chars

### Output
```
## Options Activity for AAPL

**Available Expirations:** 24 dates
**Analyzing:** 2026-04-20, 2026-04-22, 2026-04-24

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 264,978 | 56,500 | 0.213 |
| Open Interest | 203,838 | 102,523 | 0.503 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-20 | $275.0 | 95,037 | 6,632 | 14.33x | 22.3% |
| PUT | 2026-04-20 | $275.0 | 1,888 | 152 | 12.42x | 17.7% |
| CALL | 2026-04-22 | $272.5 | 12,267 | 1,164 | 10.54x | 27.1% |
| PUT | 2026-04-20 | $272.5 | 7,187 | 836 | 8.6x | 19.0% |
| PUT | 2026-04-22 | $272.5 | 784 | 103 | 7.61x | 24.6% |
| CALL | 2026-04-20 | $272.5 | 40,805 | 6,404 | 6.37x | 25.2% |
| CALL | 2026-04-22 | $280.0 | 7,224 | 1,352 | 5.34x | 26.7% |
| PUT | 2026-04-20 | $267.5 | 9,754 | 2,129 | 4.58x | 25.1% |
| CALL | 2026-04-20 | $277.5 | 13,171 | 2,935 | 4.49x | 23.6% |
| PUT | 2026-04-22 | $270.0 | 2,020 | 488 | 4.14x | 25.4% |

```

---

## 30. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:37.566778
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ABCL', 'curr_date': '2026-04-20'}`
- **Output Length:** 1526 chars

### Output
```
# Technical Analysis for ABCL
**Date:** 2026-04-20
**Current Price:** $4.01

## Price Action
- **Daily Change:** +4.70%
- **5-Day Change:** +9.56%

## RSI (14)
- **Value:** 69.1 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.102
- **Signal Line:** 0.038
- **Histogram:** 0.064 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $3.42 (+17.3% from price)
- **200 SMA:** $4.13 (-3.0% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $3.93
- **Middle:** $3.51
- **Lower:** $3.08
- **Band Position:** 110%
- **Bandwidth:** 24.2% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $0.19 (4.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $3.73 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 85.4
- **%D:** 75.1
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 29.4 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $3.60 (+11.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 27.4M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $3.54
- **Current vs VWAP:** +13.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $4.05
- **Recent Low:** $2.74
- **Next Resistance:** $4.05 (0.0% (High))
- **Next Support:** $3.74 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 31. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:38.034006
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ABCL'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for ABCL in the last 3 months.
```

---

## 32. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:38.186617
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ABCL'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for ABCL

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 85.7% Bullish, 0.0% Bearish


```

---

## 33. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:38.600282
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ABCL', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 682 chars

### Output
```
## Options Activity for ABCL

**Available Expirations:** 6 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 540 | 296 | 0.548 |
| Open Interest | 8,407 | 4,635 | 0.551 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-06-18 | $6.0 | 131 | 31 | 4.23x | 94.1% |
| PUT | 2026-07-17 | $2.0 | 200 | 248 | 0.81x | 98.4% |

```

---

## 34. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:41.753137
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'OKTA', 'curr_date': '2026-04-20'}`
- **Output Length:** 1509 chars

### Output
```
# Technical Analysis for OKTA
**Date:** 2026-04-20
**Current Price:** $72.25

## Price Action
- **Daily Change:** +0.33%
- **5-Day Change:** +10.37%

## RSI (14)
- **Value:** 46.9 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.753
- **Signal Line:** -2.353
- **Histogram:** -0.400 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $77.38 (-6.6% from price)
- **200 SMA:** $86.96 (-16.9% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $86.29
- **Middle:** $74.42
- **Lower:** $62.55
- **Band Position:** 41%
- **Bandwidth:** 31.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.06 (5.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $66.16 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 36.2
- **%D:** 31.2
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 23.3 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $73.13 (-1.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 32.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $73.92
- **Current vs VWAP:** -2.3%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $89.79
- **Recent Low:** $62.66
- **Next Resistance:** $73.02 (61.8%)
- **Next Support:** $68.47 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 35. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:42.285687
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'OKTA'}`
- **Output Length:** 768 chars

### Output
```
# Insider Transactions for OKTA
# Data retrieved on: 2026-04-20 14:28:42

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 4 transactions
- **Unknown:** 1 transactions
- **Total Sales Value:** $3,426,422
- **Largest Sale (3mo):** 2026-03-25 | ADDISON JONATHAN JAMES (Officer) | 23304 shares | $1812755.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-07,SCHWARTZ LARISSA NICOLE,Officer,Sale,6377,508583.0,D
2026-04-01,KELLEHER ERIC ROBERT,President,Sale,2409,192720.0,D
2026-04-01,KELLEHER ERIC ROBERT,President,Unknown,2409,,D
2026-03-25,ADDISON JONATHAN JAMES,Officer,Sale,23304,1812755.0,D
2026-03-23,MCKINNON TODD,Chief Executive Officer,Sale,11263,912364.0,D

```

---

## 36. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:42.439124
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'OKTA'}`
- **Output Length:** 723 chars

### Output
```
## Analyst Recommendation Trends for OKTA

### 2026-04-01
- **Strong Buy**: 12
- **Buy**: 25
- **Hold**: 14
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 53

**Sentiment**: 69.8% Bullish, 3.8% Bearish

### 2026-03-01
- **Strong Buy**: 12
- **Buy**: 23
- **Hold**: 14
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 68.6% Bullish, 3.9% Bearish

### 2026-02-01
- **Strong Buy**: 11
- **Buy**: 24
- **Hold**: 14
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 68.6% Bullish, 3.9% Bearish

### 2026-01-01
- **Strong Buy**: 11
- **Buy**: 24
- **Hold**: 14
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 68.6% Bullish, 3.9% Bearish


```

---

## 37. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:42.849273
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'OKTA', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 967 chars

### Output
```
## Options Activity for OKTA

**Available Expirations:** 16 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 1,892 | 3,219 | 1.701 |
| Open Interest | 9,557 | 8,298 | 0.868 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $72.0 | 355 | 48 | 7.4x | 62.1% |
| PUT | 2026-05-01 | $73.0 | 155 | 23 | 6.74x | 57.0% |
| PUT | 2026-04-24 | $71.0 | 243 | 37 | 6.57x | 65.0% |
| CALL | 2026-05-01 | $80.0 | 279 | 66 | 4.23x | 60.6% |
| PUT | 2026-05-01 | $62.0 | 432 | 434 | 1.0x | 80.9% |
| PUT | 2026-05-08 | $71.0 | 477 | 479 | 1.0x | 69.0% |
| PUT | 2026-05-08 | $72.0 | 349 | 443 | 0.79x | 66.7% |

```

---

## 38. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:44.688958
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TSLA', 'curr_date': '2026-04-20'}`
- **Output Length:** 1561 chars

### Output
```
# Technical Analysis for TSLA
**Date:** 2026-04-20
**Current Price:** $400.62

## Price Action
- **Daily Change:** +3.01%
- **5-Day Change:** +13.68%

## RSI (14)
- **Value:** 60.3 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -3.216
- **Signal Line:** -9.386
- **Histogram:** 6.169 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $390.33 (+2.6% from price)
- **200 SMA:** $398.87 (+0.4% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $402.34
- **Middle:** $367.80
- **Lower:** $333.26
- **Band Position:** 98%
- **Bandwidth:** 18.8% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $15.72 (3.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $377.04 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 74.2
- **%D:** 56.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 19.8 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $373.65 (+7.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 913.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $368.67
- **Current vs VWAP:** +8.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $436.35
- **Recent Low:** $337.24
- **Next Resistance:** $412.96 (23.6%)
- **Next Support:** $398.49 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 39. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:45.090664
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TSLA'}`
- **Output Length:** 675 chars

### Output
```
# Insider Transactions for TSLA
# Data retrieved on: 2026-04-20 14:28:45

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Option Exercise:** 2 transactions
- **Sale:** 1 transactions
- **Total Sales Value:** $9,273,888
- **Largest Sale (3mo):** 2026-03-30 | WILSON - THOMPSON KATHLEEN (Director) | 25809 shares | $9273888.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-03-31,ZHU XIAOTONG,Officer,Option Exercise,20000,411400.0,D
2026-03-30,WILSON - THOMPSON KATHLEEN,Director,Sale,25809,9273888.0,D
2026-03-30,WILSON - THOMPSON KATHLEEN,Director,Option Exercise,40000,599600.0,D

```

---

## 40. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:45.257316
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TSLA'}`
- **Output Length:** 723 chars

### Output
```
## Analyst Recommendation Trends for TSLA

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 20
- **Hold**: 21
- **Sell**: 8
- **Strong Sell**: 2
- **Total Analysts**: 60

**Sentiment**: 48.3% Bullish, 16.7% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 20
- **Hold**: 21
- **Sell**: 8
- **Strong Sell**: 2
- **Total Analysts**: 59

**Sentiment**: 47.5% Bullish, 16.9% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 19
- **Hold**: 21
- **Sell**: 8
- **Strong Sell**: 2
- **Total Analysts**: 58

**Sentiment**: 46.6% Bullish, 17.2% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 20
- **Hold**: 21
- **Sell**: 9
- **Strong Sell**: 2
- **Total Analysts**: 59

**Sentiment**: 45.8% Bullish, 18.6% Bearish


```

---

## 41. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:45.754135
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TSLA', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1182 chars

### Output
```
## Options Activity for TSLA

**Available Expirations:** 23 dates
**Analyzing:** 2026-04-20, 2026-04-24, 2026-04-27

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 380,102 | 278,178 | 0.732 |
| Open Interest | 348,273 | 369,955 | 1.062 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-20 | $397.5 | 15,997 | 970 | 16.49x | 51.9% |
| PUT | 2026-04-20 | $392.5 | 23,612 | 1,738 | 13.59x | 27.1% |
| PUT | 2026-04-20 | $397.5 | 21,229 | 1,626 | 13.06x | 18.7% |
| PUT | 2026-04-20 | $395.0 | 37,282 | 3,161 | 11.79x | 23.5% |
| PUT | 2026-04-20 | $390.0 | 32,744 | 3,597 | 9.1x | 29.9% |
| CALL | 2026-04-20 | $437.5 | 1,249 | 155 | 8.06x | 79.7% |
| PUT | 2026-04-20 | $402.5 | 7,881 | 1,119 | 7.04x | 0.0% |
| PUT | 2026-04-20 | $400.0 | 22,360 | 3,612 | 6.19x | 0.0% |
| PUT | 2026-04-27 | $417.5 | 330 | 61 | 5.41x | 49.0% |
| PUT | 2026-04-20 | $405.0 | 8,868 | 1,672 | 5.3x | 0.0% |

```

---

## 42. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:47.633640
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMD', 'curr_date': '2026-04-20'}`
- **Output Length:** 1602 chars

### Output
```
# Technical Analysis for AMD
**Date:** 2026-04-20
**Current Price:** $278.39

## Price Action
- **Daily Change:** +0.05%
- **5-Day Change:** +12.79%

## RSI (14)
- **Value:** 80.2 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 16.650
- **Signal Line:** 10.053
- **Histogram:** 6.598 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $212.13 (+31.2% from price)
- **200 SMA:** $202.45 (+37.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $278.53
- **Middle:** $226.72
- **Lower:** $174.92
- **Band Position:** 100%
- **Bandwidth:** 45.7% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $10.19 (3.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $263.11 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 96.3
- **%D:** 93.3
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 52.2 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $235.21 (+18.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1146.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $226.07
- **Current vs VWAP:** +23.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $281.05
- **Recent Low:** $188.22
- **Next Resistance:** $281.05 (0.0% (High))
- **Next Support:** $259.14 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 43. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:48.068444
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMD'}`
- **Output Length:** 708 chars

### Output
```
# Insider Transactions for AMD
# Data retrieved on: 2026-04-20 14:28:48

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 2 transactions
- **Option Exercise:** 1 transactions
- **Total Sales Value:** $9,729,140
- **Largest Sale (3mo):** 2026-04-16 | PAPERMASTER MARK D (Chief Technology Officer) | 33109 shares | $8988215.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,PAPERMASTER MARK D,Chief Technology Officer,Sale,33109,8988215.0,D
2026-04-15,PAPERMASTER MARK D,Chief Technology Officer,Option Exercise,6000,509100.0,D
2026-04-06,PAPERMASTER MARK D,Chief Technology Officer,Sale,3293,740925.0,D

```

---

## 44. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:48.217403
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMD'}`
- **Output Length:** 722 chars

### Output
```
## Analyst Recommendation Trends for AMD

### 2026-04-01
- **Strong Buy**: 15
- **Buy**: 30
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 59

**Sentiment**: 76.3% Bullish, 1.7% Bearish

### 2026-03-01
- **Strong Buy**: 15
- **Buy**: 32
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 61

**Sentiment**: 77.0% Bullish, 1.6% Bearish

### 2026-02-01
- **Strong Buy**: 15
- **Buy**: 31
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 60

**Sentiment**: 76.7% Bullish, 1.7% Bearish

### 2026-01-01
- **Strong Buy**: 15
- **Buy**: 29
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 58

**Sentiment**: 75.9% Bullish, 1.7% Bearish


```

---

## 45. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:48.690328
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMD', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1166 chars

### Output
```
## Options Activity for AMD

**Available Expirations:** 21 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 69,994 | 29,428 | 0.42 |
| Open Interest | 174,673 | 174,451 | 0.999 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-24 | $322.5 | 626 | 14 | 44.71x | 70.2% |
| CALL | 2026-04-24 | $312.5 | 1,402 | 39 | 35.95x | 68.1% |
| PUT | 2026-05-01 | $287.5 | 152 | 8 | 19.0x | 38.4% |
| PUT | 2026-04-24 | $287.5 | 318 | 17 | 18.71x | 23.5% |
| PUT | 2026-04-24 | $282.5 | 1,856 | 107 | 17.35x | 34.7% |
| PUT | 2026-04-24 | $285.0 | 1,486 | 120 | 12.38x | 30.4% |
| CALL | 2026-04-24 | $327.5 | 133 | 14 | 9.5x | 72.0% |
| PUT | 2026-05-01 | $285.0 | 320 | 67 | 4.78x | 38.9% |
| CALL | 2026-05-01 | $287.5 | 151 | 36 | 4.19x | 65.2% |
| CALL | 2026-05-08 | $185.0 | 215 | 55 | 3.91x | 162.3% |

```

---

## 46. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:50.586998
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MRVL', 'curr_date': '2026-04-20'}`
- **Output Length:** 1556 chars

### Output
```
# Technical Analysis for MRVL
**Date:** 2026-04-20
**Current Price:** $139.69

## Price Action
- **Daily Change:** +4.74%
- **5-Day Change:** +6.39%

## RSI (14)
- **Value:** 82.3 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 12.684
- **Signal Line:** 9.916
- **Histogram:** 2.768 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $94.00 (+48.6% from price)
- **200 SMA:** $84.04 (+66.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $145.90
- **Middle:** $110.80
- **Lower:** $75.70
- **Band Position:** 91%
- **Bandwidth:** 63.4% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $6.09 (4.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $130.56 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 93.0
- **%D:** 91.6
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 55.0 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $115.41 (+21.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 505.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $111.83
- **Current vs VWAP:** +24.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $139.91
- **Recent Low:** $70.65
- **Next Resistance:** $139.91 (0.0% (High))
- **Next Support:** $123.57 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 47. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:50.951903
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MRVL'}`
- **Output Length:** 1226 chars

### Output
```
# Insider Transactions for MRVL
# Data retrieved on: 2026-04-20 14:28:50

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 7 transactions
- **Unknown:** 5 transactions
- **Total Sales Value:** $23,785,663
- **Largest Sale (3mo):** 2026-04-16 | BHARATHI SANDEEP (Officer) | 66892 shares | $8719372.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,BHARATHI SANDEEP,Officer,Sale,66892,8719372.0,D
2026-04-15,CASPER MARK JOHN,Officer,Unknown,83018,,D
2026-04-15,MEINTJES WILLEM A,Chief Financial Officer,Sale,30000,4020300.0,D
2026-04-15,MEINTJES WILLEM A,Chief Financial Officer,Unknown,130970,,D
2026-04-15,KOOPMANS CHRISTOPHER R,President,Unknown,170022,,I
2026-04-15,MURPHY MATTHEW JOSEPH,Chief Executive Officer,Sale,7500,1008450.0,D
2026-04-15,MURPHY MATTHEW JOSEPH,Chief Executive Officer,Unknown,938324,,D
2026-04-15,BHARATHI SANDEEP,Officer,Unknown,141278,,D
2026-04-06,CASPER MARK JOHN,Officer,Sale,6900,755205.0,I
2026-04-02,CASPER MARK JOHN,Officer,Sale,17854,1897257.0,D
2026-03-26,MURPHY MATTHEW JOSEPH,Chief Executive Officer,Sale,30000,2961000.0,D
2026-03-26,BHARATHI SANDEEP,Officer,Sale,44414,4424079.0,D

```

---

## 48. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:51.154989
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MRVL'}`
- **Output Length:** 722 chars

### Output
```
## Analyst Recommendation Trends for MRVL

### 2026-04-01
- **Strong Buy**: 13
- **Buy**: 28
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 50

**Sentiment**: 82.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 11
- **Buy**: 27
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 74.5% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 11
- **Buy**: 26
- **Hold**: 14
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 72.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 11
- **Buy**: 24
- **Hold**: 15
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 50

**Sentiment**: 70.0% Bullish, 0.0% Bearish


```

---

## 49. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:51.702308
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MRVL', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1165 chars

### Output
```
## Options Activity for MRVL

**Available Expirations:** 18 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 34,378 | 12,592 | 0.366 |
| Open Interest | 73,751 | 59,480 | 0.806 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $165.0 | 634 | 8 | 79.25x | 73.5% |
| CALL | 2026-05-08 | $146.0 | 109 | 3 | 36.33x | 71.2% |
| CALL | 2026-05-08 | $155.0 | 1,151 | 38 | 30.29x | 71.1% |
| PUT | 2026-05-01 | $118.0 | 1,998 | 86 | 23.23x | 71.6% |
| CALL | 2026-05-01 | $147.0 | 198 | 12 | 16.5x | 75.3% |
| PUT | 2026-04-24 | $144.0 | 370 | 26 | 14.23x | 61.2% |
| CALL | 2026-05-01 | $170.0 | 1,260 | 135 | 9.33x | 74.4% |
| PUT | 2026-05-01 | $139.0 | 170 | 20 | 8.5x | 63.0% |
| PUT | 2026-04-24 | $140.0 | 978 | 128 | 7.64x | 62.5% |
| CALL | 2026-05-01 | $152.5 | 193 | 26 | 7.42x | 73.4% |

```

---

## 50. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:53.865883
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'WMT', 'curr_date': '2026-04-20'}`
- **Output Length:** 1546 chars

### Output
```
# Technical Analysis for WMT
**Date:** 2026-04-20
**Current Price:** $127.50

## Price Action
- **Daily Change:** +2.15%
- **5-Day Change:** +2.35%

## RSI (14)
- **Value:** 56.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.588
- **Signal Line:** 0.437
- **Histogram:** 0.151 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $125.45 (+1.6% from price)
- **200 SMA:** $110.36 (+15.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $129.31
- **Middle:** $124.37
- **Lower:** $119.43
- **Band Position:** 82%
- **Bandwidth:** 7.9% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $2.88 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $123.18 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 56.9
- **%D:** 57.1
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 8.0 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $125.10 (+1.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 460.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $123.56
- **Current vs VWAP:** +3.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $134.41
- **Recent Low:** $118.02
- **Next Resistance:** $128.15 (38.2%)
- **Next Support:** $126.22 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 51. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:54.267699
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'WMT'}`
- **Output Length:** 1562 chars

### Output
```
# Insider Transactions for WMT
# Data retrieved on: 2026-04-20 14:28:54

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 11 transactions
- **Sale:** 5 transactions
- **Total Sales Value:** $407,566,901
- **Largest Sale (3mo):** 2026-03-25 | WALTON FAMILY HOLDINGS TRUST (Beneficial Owner of more than 10% of a Class of Security) | 3279000 shares | $403009014 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,NICHOLAS CHRISTOPHER JAMES,Officer,Sale,2900,361696,D
2026-04-16,FURNER JOHN R,Chief Executive Officer,Sale,13125,1638530,D
2026-04-15,BARTLETT DANIEL J,Officer,Sale,1335,166381,D
2026-03-31,MORITZ ROBERT EDWARD JR.,Director,Award/Grant,231,28709,D
2026-03-31,HARRIS CARLA A,Director,Award/Grant,141,17523,D
2026-03-31,MEHROTRA SHISHIR,Director,Award/Grant,213,26472,D
2026-03-31,FLYNN TIMOTHY PATRICK,Director,Award/Grant,302,37533,D
2026-03-31,FRIAR SARAH J,Director,Award/Grant,282,35047,D
2026-03-31,WALTON STEUART L.,Director,Award/Grant,282,35047,D
2026-03-31,STEPHENSON RANDALL L,Director,Award/Grant,382,47475,D
2026-03-31,CONDE CESAR,Director,Award/Grant,231,28709,D
2026-03-31,NICCOL BRIAN R,Director,Award/Grant,231,28709,D
2026-03-31,PENNER GREGORY BOYD,Director,Award/Grant,458,56920,D
2026-03-31,MAYER MARISSA A,Director,Award/Grant,231,28709,D
2026-03-26,MCMILLON C DOUGLAS,Director,Sale,19416,2391280,D
2026-03-25,WALTON FAMILY HOLDINGS TRUST,Beneficial Owner of more than 10% of a Class of Security,Sale,3279000,403009014,D

```

---

## 52. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:54.417758
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'WMT'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for WMT

### 2026-04-01
- **Strong Buy**: 14
- **Buy**: 30
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 91.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 14
- **Buy**: 31
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 49

**Sentiment**: 91.8% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 14
- **Buy**: 32
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 49

**Sentiment**: 93.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 15
- **Buy**: 31
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 95.8% Bullish, 0.0% Bearish


```

---

## 53. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:54.843793
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'WMT', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1170 chars

### Output
```
## Options Activity for WMT

**Available Expirations:** 16 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 32,597 | 8,573 | 0.263 |
| Open Interest | 55,974 | 47,015 | 0.84 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-24 | $100.0 | 183 | 11 | 16.64x | 149.6% |
| CALL | 2026-04-24 | $90.0 | 279 | 43 | 6.49x | 202.1% |
| CALL | 2026-04-24 | $134.0 | 2,936 | 498 | 5.9x | 30.9% |
| PUT | 2026-05-01 | $129.0 | 352 | 60 | 5.87x | 21.7% |
| CALL | 2026-04-24 | $135.0 | 9,751 | 1,994 | 4.89x | 31.3% |
| CALL | 2026-04-24 | $118.0 | 188 | 47 | 4.0x | 64.1% |
| CALL | 2026-04-24 | $129.0 | 3,134 | 1,475 | 2.12x | 30.1% |
| PUT | 2026-04-24 | $128.0 | 286 | 151 | 1.89x | 22.7% |
| CALL | 2026-04-24 | $137.0 | 173 | 162 | 1.07x | 33.2% |
| CALL | 2026-04-24 | $132.0 | 981 | 959 | 1.02x | 29.5% |

```

---

## 54. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:57.663479
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NFLX', 'curr_date': '2026-04-20'}`
- **Output Length:** 1498 chars

### Output
```
# Technical Analysis for NFLX
**Date:** 2026-04-20
**Current Price:** $97.31

## Price Action
- **Daily Change:** -9.72%
- **5-Day Change:** -5.67%

## RSI (14)
- **Value:** 48.6 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 3.249
- **Signal Line:** 3.024
- **Histogram:** 0.225 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $91.95 (+5.8% from price)
- **200 SMA:** $105.88 (-8.1% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $108.91
- **Middle:** $98.15
- **Lower:** $87.38
- **Band Position:** 46%
- **Bandwidth:** 21.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $3.48 (3.6% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $92.09 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 67.1
- **%D:** 81.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 41.4 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $99.47 (-2.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -665.2M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $97.83
- **Current vs VWAP:** -0.5%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $108.95
- **Recent Low:** $75.01
- **Next Resistance:** $100.94 (23.6%)
- **Next Support:** $95.98 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 55. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:58.031512
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NFLX'}`
- **Output Length:** 747 chars

### Output
```
# Insider Transactions for NFLX
# Data retrieved on: 2026-04-20 14:28:58

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 2 transactions
- **Option Exercise:** 2 transactions
- **Total Sales Value:** $42,962,205
- **Largest Sale (3mo):** 2026-04-01 | HASTINGS REED (Director) | 420550 shares | $40156465.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-02,NEUMANN SPENCER ADAM,Chief Financial Officer,Sale,28630,2805740.0,D
2026-04-02,NEUMANN SPENCER ADAM,Chief Financial Officer,Option Exercise,28630,1055547.0,D
2026-04-01,HASTINGS REED,Director,Sale,420550,40156465.0,D
2026-04-01,HASTINGS REED,Director,Option Exercise,420550,3968730.0,D

```

---

## 56. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:58.241721
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NFLX'}`
- **Output Length:** 723 chars

### Output
```
## Analyst Recommendation Trends for NFLX

### 2026-04-01
- **Strong Buy**: 16
- **Buy**: 29
- **Hold**: 14
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 59

**Sentiment**: 76.3% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 15
- **Buy**: 26
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 54

**Sentiment**: 75.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 14
- **Buy**: 23
- **Hold**: 14
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 52

**Sentiment**: 71.2% Bullish, 1.9% Bearish

### 2026-01-01
- **Strong Buy**: 14
- **Buy**: 23
- **Hold**: 13
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 72.5% Bullish, 2.0% Bearish


```

---

## 57. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:28:58.701327
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NFLX', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1172 chars

### Output
```
## Options Activity for NFLX

**Available Expirations:** 19 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 71,095 | 32,801 | 0.461 |
| Open Interest | 327,863 | 122,257 | 0.373 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-24 | $85.0 | 1,058 | 353 | 3.0x | 102.3% |
| CALL | 2026-05-01 | $85.0 | 136 | 97 | 1.4x | 59.2% |
| CALL | 2026-04-24 | $96.0 | 2,954 | 2,144 | 1.38x | 41.2% |
| PUT | 2026-04-24 | $97.0 | 7,096 | 5,594 | 1.27x | 24.1% |
| CALL | 2026-04-24 | $123.0 | 731 | 585 | 1.25x | 75.0% |
| PUT | 2026-05-08 | $50.0 | 120 | 100 | 1.2x | 103.1% |
| PUT | 2026-04-24 | $94.0 | 3,186 | 2,688 | 1.19x | 28.8% |
| PUT | 2026-04-24 | $96.0 | 4,111 | 3,832 | 1.07x | 26.0% |
| PUT | 2026-04-24 | $81.0 | 167 | 157 | 1.06x | 55.5% |
| PUT | 2026-04-24 | $108.0 | 821 | 772 | 1.06x | 0.0% |

```

---

## 58. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:00.532356
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'APP', 'curr_date': '2026-04-20'}`
- **Output Length:** 1544 chars

### Output
```
# Technical Analysis for APP
**Date:** 2026-04-20
**Current Price:** $477.20

## Price Action
- **Daily Change:** +2.38%
- **5-Day Change:** +14.31%

## RSI (14)
- **Value:** 61.4 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 2.173
- **Signal Line:** -8.932
- **Histogram:** 11.106 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $429.50 (+11.1% from price)
- **200 SMA:** $520.02 (-8.2% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $482.66
- **Middle:** $416.72
- **Lower:** $350.78
- **Band Position:** 96%
- **Bandwidth:** 31.6% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $28.30 (5.9% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $434.75 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 79.5
- **%D:** 65.0
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 23.0 (Strengthening ↑)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $428.73 (+11.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 170.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $415.22
- **Current vs VWAP:** +14.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $520.36
- **Recent Low:** $359.00
- **Next Resistance:** $482.28 (23.6%)
- **Next Support:** $458.72 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 59. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:01.118421
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'APP'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for APP in the last 3 months.
```

---

## 60. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:01.289589
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'APP'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for APP

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 21
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 34

**Sentiment**: 88.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 20
- **Hold**: 4
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 34

**Sentiment**: 85.3% Bullish, 2.9% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 20
- **Hold**: 4
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 34

**Sentiment**: 85.3% Bullish, 2.9% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 18
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 81.8% Bullish, 3.0% Bearish


```

---

## 61. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:01.879237
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'APP', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 992 chars

### Output
```
## Options Activity for APP

**Available Expirations:** 17 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 9,976 | 3,255 | 0.326 |
| Open Interest | 23,441 | 14,754 | 0.629 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-24 | $520.0 | 3,248 | 423 | 7.68x | 89.5% |
| PUT | 2026-05-01 | $580.0 | 137 | 64 | 2.14x | 0.0% |
| CALL | 2026-04-24 | $480.0 | 291 | 259 | 1.12x | 113.6% |
| CALL | 2026-04-24 | $550.0 | 2,841 | 2,977 | 0.95x | 85.7% |
| PUT | 2026-04-24 | $300.0 | 850 | 937 | 0.91x | 156.4% |
| PUT | 2026-04-24 | $390.0 | 157 | 189 | 0.83x | 91.2% |
| CALL | 2026-04-24 | $530.0 | 152 | 221 | 0.69x | 88.9% |

```

---

## 62. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:03.827661
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BSX', 'curr_date': '2026-04-20'}`
- **Output Length:** 1518 chars

### Output
```
# Technical Analysis for BSX
**Date:** 2026-04-20
**Current Price:** $64.23

## Price Action
- **Daily Change:** +1.28%
- **5-Day Change:** +1.66%

## RSI (14)
- **Value:** 42.1 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.293
- **Signal Line:** -2.838
- **Histogram:** 0.545 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $69.94 (-8.2% from price)
- **200 SMA:** $92.16 (-30.3% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $71.19
- **Middle:** $64.81
- **Lower:** $58.42
- **Band Position:** 45%
- **Bandwidth:** 19.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.93 (3.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $61.34 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 65.1
- **%D:** 54.0
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 23.2 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $65.08 (-1.3% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -270.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $64.57
- **Current vs VWAP:** -0.5%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $79.17
- **Recent Low:** $60.59
- **Next Resistance:** $64.57 (78.6%)
- **Next Support:** $60.59 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 63. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:04.229968
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BSX'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for BSX in the last 3 months.
```

---

## 64. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:04.401413
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'BSX'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for BSX

### 2026-04-01
- **Strong Buy**: 13
- **Buy**: 25
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 41

**Sentiment**: 92.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 14
- **Buy**: 23
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 92.5% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 15
- **Buy**: 22
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 92.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 15
- **Buy**: 21
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 90.0% Bullish, 0.0% Bearish


```

---

## 65. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:04.933420
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BSX', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 750 chars

### Output
```
## Options Activity for BSX

**Available Expirations:** 16 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,773 | 1,317 | 0.475 |
| Open Interest | 6,974 | 19,009 | 2.726 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-24 | $64.0 | 1,966 | 136 | 14.46x | 84.6% |
| PUT | 2026-04-24 | $59.0 | 460 | 205 | 2.24x | 81.5% |
| PUT | 2026-04-24 | $64.0 | 235 | 324 | 0.73x | 88.8% |

```

---

## 66. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:07.046684
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ORCL', 'curr_date': '2026-04-20'}`
- **Output Length:** 1556 chars

### Output
```
# Technical Analysis for ORCL
**Date:** 2026-04-20
**Current Price:** $175.06

## Price Action
- **Daily Change:** -1.84%
- **5-Day Change:** +12.49%

## RSI (14)
- **Value:** 67.3 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 4.348
- **Signal Line:** 0.036
- **Histogram:** 4.312 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $151.12 (+15.8% from price)
- **200 SMA:** $214.05 (-18.2% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $174.55
- **Middle:** $150.03
- **Lower:** $125.52
- **Band Position:** 101%
- **Bandwidth:** 32.7% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $8.38 (4.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $162.50 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.5
- **%D:** 69.6
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 31.9 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $154.84 (+13.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -275.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $154.21
- **Current vs VWAP:** +13.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $184.50
- **Recent Low:** $134.57
- **Next Resistance:** $184.50 (0.0% (High))
- **Next Support:** $172.72 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 67. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:07.409659
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ORCL'}`
- **Output Length:** 299 chars

### Output
```
# Insider Transactions for ORCL
# Data retrieved on: 2026-04-20 14:29:07

## Summary
- **Insider Sentiment:** Neutral
- **Gift:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-03-30,HENLEY JEFFREY O,Officer and Director,Gift,16810,0.0,I

```

---

## 68. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:07.559841
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ORCL'}`
- **Output Length:** 723 chars

### Output
```
## Analyst Recommendation Trends for ORCL

### 2026-04-01
- **Strong Buy**: 16
- **Buy**: 24
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 78.4% Bullish, 2.0% Bearish

### 2026-03-01
- **Strong Buy**: 16
- **Buy**: 22
- **Hold**: 11
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 50

**Sentiment**: 76.0% Bullish, 2.0% Bearish

### 2026-02-01
- **Strong Buy**: 15
- **Buy**: 23
- **Hold**: 11
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 74.5% Bullish, 3.9% Bearish

### 2026-01-01
- **Strong Buy**: 15
- **Buy**: 22
- **Hold**: 12
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 51

**Sentiment**: 72.5% Bullish, 3.9% Bearish


```

---

## 69. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:08.065562
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ORCL', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1166 chars

### Output
```
## Options Activity for ORCL

**Available Expirations:** 20 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 27,938 | 18,719 | 0.67 |
| Open Interest | 229,034 | 103,042 | 0.45 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-08 | $110.0 | 119 | 7 | 17.0x | 148.2% |
| CALL | 2026-05-08 | $115.0 | 118 | 7 | 16.86x | 137.6% |
| CALL | 2026-05-08 | $125.0 | 138 | 17 | 8.12x | 116.0% |
| PUT | 2026-04-24 | $172.5 | 5,235 | 887 | 5.9x | 50.4% |
| PUT | 2026-05-01 | $144.0 | 290 | 73 | 3.97x | 64.7% |
| PUT | 2026-04-24 | $132.0 | 501 | 270 | 1.86x | 105.9% |
| CALL | 2026-04-24 | $192.5 | 1,450 | 976 | 1.49x | 66.2% |
| PUT | 2026-04-24 | $190.0 | 196 | 168 | 1.17x | 57.9% |
| PUT | 2026-05-01 | $185.0 | 462 | 432 | 1.07x | 48.0% |
| CALL | 2026-05-01 | $182.5 | 196 | 202 | 0.97x | 61.2% |

```

---

## 70. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:10.144133
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'IBM', 'curr_date': '2026-04-20'}`
- **Output Length:** 1557 chars

### Output
```
# Technical Analysis for IBM
**Date:** 2026-04-20
**Current Price:** $253.47

## Price Action
- **Daily Change:** +0.98%
- **5-Day Change:** +6.58%

## RSI (14)
- **Value:** 56.7 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -1.508
- **Signal Line:** -3.492
- **Histogram:** 1.983 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $250.51 (+1.2% from price)
- **200 SMA:** $273.13 (-7.2% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $253.31
- **Middle:** $242.50
- **Lower:** $231.69
- **Band Position:** 101%
- **Bandwidth:** 8.9% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $6.97 (2.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $243.01 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 74.9
- **%D:** 59.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 13.0 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $244.69 (+3.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 32.8M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $242.60
- **Current vs VWAP:** +4.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $298.19
- **Recent Low:** $220.72
- **Next Resistance:** $259.45 (50.0%)
- **Next Support:** $250.31 (61.8%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 71. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:10.576975
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'IBM'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for IBM in the last 3 months.
```

---

## 72. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:10.767715
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'IBM'}`
- **Output Length:** 716 chars

### Output
```
## Analyst Recommendation Trends for IBM

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 11
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 27

**Sentiment**: 51.9% Bullish, 7.4% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 11
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 27

**Sentiment**: 51.9% Bullish, 7.4% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 10
- **Sell**: 3
- **Strong Sell**: 0
- **Total Analysts**: 27

**Sentiment**: 51.9% Bullish, 11.1% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 7
- **Hold**: 11
- **Sell**: 3
- **Strong Sell**: 0
- **Total Analysts**: 27

**Sentiment**: 48.1% Bullish, 11.1% Bearish


```

---

## 73. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:11.310672
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'IBM', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 732 chars

### Output
```
## Options Activity for IBM

**Available Expirations:** 19 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,552 | 2,041 | 0.8 |
| Open Interest | 10,595 | 8,520 | 0.804 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $300.0 | 522 | 33 | 15.82x | 53.7% |
| PUT | 2026-04-24 | $240.0 | 1,059 | 311 | 3.41x | 68.5% |
| CALL | 2026-04-24 | $265.0 | 317 | 465 | 0.68x | 83.6% |

```

---

## 74. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:13.711247
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'UBER', 'curr_date': '2026-04-20'}`
- **Output Length:** 1557 chars

### Output
```
# Technical Analysis for UBER
**Date:** 2026-04-20
**Current Price:** $77.12

## Price Action
- **Daily Change:** +0.84%
- **5-Day Change:** +6.61%

## RSI (14)
- **Value:** 60.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.353
- **Signal Line:** -0.402
- **Histogram:** 0.755 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $73.41 (+5.1% from price)
- **200 SMA:** $86.15 (-10.5% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $77.22
- **Middle:** $72.71
- **Lower:** $68.20
- **Band Position:** 99%
- **Bandwidth:** 12.4% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $2.48 (3.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $73.39 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.9
- **%D:** 61.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 18.0 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $73.63 (+4.7% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -129.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $72.94
- **Current vs VWAP:** +5.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $79.23
- **Recent Low:** $68.46
- **Next Resistance:** $79.23 (0.0% (High))
- **Next Support:** $76.69 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 75. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:14.151663
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'UBER'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for UBER in the last 3 months.
```

---

## 76. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:14.324656
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'UBER'}`
- **Output Length:** 720 chars

### Output
```
## Analyst Recommendation Trends for UBER

### 2026-04-01
- **Strong Buy**: 16
- **Buy**: 35
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 61

**Sentiment**: 83.6% Bullish, 1.6% Bearish

### 2026-03-01
- **Strong Buy**: 16
- **Buy**: 35
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 61

**Sentiment**: 83.6% Bullish, 1.6% Bearish

### 2026-02-01
- **Strong Buy**: 16
- **Buy**: 35
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 61

**Sentiment**: 83.6% Bullish, 1.6% Bearish

### 2026-01-01
- **Strong Buy**: 16
- **Buy**: 33
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 60

**Sentiment**: 81.7% Bullish, 0.0% Bearish


```

---

## 77. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:14.772043
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'UBER', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 863 chars

### Output
```
## Options Activity for UBER

**Available Expirations:** 19 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,409 | 5,262 | 2.184 |
| Open Interest | 41,839 | 48,423 | 1.157 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $67.0 | 400 | 130 | 3.08x | 52.7% |
| CALL | 2026-04-24 | $68.0 | 211 | 71 | 2.97x | 56.3% |
| PUT | 2026-04-24 | $75.0 | 2,460 | 829 | 2.97x | 37.0% |
| PUT | 2026-04-24 | $80.0 | 357 | 130 | 2.75x | 52.5% |
| PUT | 2026-05-01 | $79.0 | 113 | 105 | 1.08x | 42.7% |

```

---

## 78. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:18.698028
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'EA', 'curr_date': '2026-04-20'}`
- **Output Length:** 1528 chars

### Output
```
# Technical Analysis for EA
**Date:** 2026-04-20
**Current Price:** $203.83

## Price Action
- **Daily Change:** +0.23%
- **5-Day Change:** +0.63%

## RSI (14)
- **Value:** 60.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.590
- **Signal Line:** 0.589
- **Histogram:** 0.001 (Weakening ↓)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $201.31 (+1.3% from price)
- **200 SMA:** $190.26 (+7.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $204.86
- **Middle:** $202.80
- **Lower:** $200.74
- **Band Position:** 75%
- **Bandwidth:** 2.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.98 (0.5% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $202.37 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 61.8
- **%D:** 56.3
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 22.7 (Strengthening ↑)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $202.71 (+0.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 46.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $202.48
- **Current vs VWAP:** +0.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $204.22
- **Recent Low:** $196.22
- **Next Resistance:** $204.22 (0.0% (High))
- **Next Support:** $202.33 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 79. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:19.062997
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'EA'}`
- **Output Length:** 611 chars

### Output
```
# Insider Transactions for EA
# Data retrieved on: 2026-04-20 14:29:19

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 3 transactions
- **Total Sales Value:** $1,767,984
- **Largest Sale (3mo):** 2026-04-15 | WILSON ANDREW (Chief Executive Officer) | 5000 shares | $1016058.0 | Ownership: I

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,WILSON ANDREW,Chief Executive Officer,Sale,5000,1016058.0,I
2026-04-15,MIELE LAURA KAREN,Officer,Sale,2500,508026.0,D
2026-04-15,SINGH VIJAYANTHIMALA,Officer,Sale,1200,243900.0,I

```

---

## 80. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:19.231110
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

## 81. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:19.644104
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'EA', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 435 chars

### Output
```
## Options Activity for EA

**Available Expirations:** 20 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 26 | 67 | 2.577 |
| Open Interest | 752 | 133 | 0.177 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bullish positioning

*No unusual options activity detected.*

```

---

## 82. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:21.773286
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CTAS', 'curr_date': '2026-04-20'}`
- **Output Length:** 1549 chars

### Output
```
# Technical Analysis for CTAS
**Date:** 2026-04-20
**Current Price:** $179.17

## Price Action
- **Daily Change:** +0.93%
- **5-Day Change:** +1.99%

## RSI (14)
- **Value:** 49.9 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.877
- **Signal Line:** -4.372
- **Histogram:** 1.494 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $187.71 (-4.6% from price)
- **200 SMA:** $195.80 (-8.5% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $182.56
- **Middle:** $174.24
- **Lower:** $165.92
- **Band Position:** 80%
- **Bandwidth:** 9.5% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.17 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $172.92 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 85.3
- **%D:** 78.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 13.9 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $177.53 (+0.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -15.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $174.19
- **Current vs VWAP:** +2.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $209.08
- **Recent Low:** $165.60
- **Next Resistance:** $182.21 (61.8%)
- **Next Support:** $174.90 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 83. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:22.080068
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CTAS'}`
- **Output Length:** 312 chars

### Output
```
# Insider Transactions for CTAS
# Data retrieved on: 2026-04-20 14:29:22

## Summary
- **Insider Sentiment:** Neutral
- **Option Exercise:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-09,COLETTI ROBERT E,Director,Option Exercise,12544,339463,D

```

---

## 84. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:22.255037
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CTAS'}`
- **Output Length:** 717 chars

### Output
```
## Analyst Recommendation Trends for CTAS

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 27

**Sentiment**: 48.1% Bullish, 3.7% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 10
- **Hold**: 14
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 29

**Sentiment**: 48.3% Bullish, 3.4% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 10
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 28

**Sentiment**: 50.0% Bullish, 3.6% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 8
- **Hold**: 14
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 27

**Sentiment**: 44.4% Bullish, 3.7% Bearish


```

---

## 85. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:22.638421
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CTAS', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 626 chars

### Output
```
## Options Activity for CTAS

**Available Expirations:** 14 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 162 | 357 | 2.204 |
| Open Interest | 374 | 1,034 | 2.765 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-04-24 | $150.0 | 250 | 250 | 1.0x | 105.1% |

```

---

## 86. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:26.439740
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DGX', 'curr_date': '2026-04-20'}`
- **Output Length:** 1522 chars

### Output
```
# Technical Analysis for DGX
**Date:** 2026-04-20
**Current Price:** $195.02

## Price Action
- **Daily Change:** +1.86%
- **5-Day Change:** +1.09%

## RSI (14)
- **Value:** 48.4 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -1.261
- **Signal Line:** -0.968
- **Histogram:** -0.293 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $199.32 (-2.2% from price)
- **200 SMA:** $183.58 (+6.2% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $199.85
- **Middle:** $195.31
- **Lower:** $190.77
- **Band Position:** 47%
- **Bandwidth:** 4.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.33 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $188.53 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 33.3
- **%D:** 35.6
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 13.7 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $195.39 (-0.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -2.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $194.95
- **Current vs VWAP:** +0.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $212.57
- **Recent Low:** $186.77
- **Next Resistance:** $196.62 (61.8%)
- **Next Support:** $192.29 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 87. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:26.653383
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DGX'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for DGX in the last 3 months.
```

---

## 88. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:26.803047
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DGX'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for DGX

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 7
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 54.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 7
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 54.2% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 7
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 54.2% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 6
- **Hold**: 12
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 50.0% Bullish, 0.0% Bearish


```

---

## 89. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:27.083938
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DGX', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 627 chars

### Output
```
## Options Activity for DGX

**Available Expirations:** 4 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 269 | 652 | 2.424 |
| Open Interest | 5,989 | 2,811 | 0.469 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $195.0 | 124 | 121 | 1.02x | 33.4% |

```

---

## 90. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:29.772151
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DDOG', 'curr_date': '2026-04-20'}`
- **Output Length:** 1543 chars

### Output
```
# Technical Analysis for DDOG
**Date:** 2026-04-20
**Current Price:** $126.61

## Price Action
- **Daily Change:** +2.54%
- **5-Day Change:** +15.02%

## RSI (14)
- **Value:** 58.3 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.859
- **Signal Line:** -1.907
- **Histogram:** 1.048 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $119.43 (+6.0% from price)
- **200 SMA:** $138.60 (-8.7% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $131.06
- **Middle:** $118.38
- **Lower:** $105.69
- **Band Position:** 82%
- **Bandwidth:** 21.4% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $7.05 (5.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $116.04 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.9
- **%D:** 53.7
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 15.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $118.48 (+6.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -156.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $118.27
- **Current vs VWAP:** +7.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $133.60
- **Recent Low:** $98.01
- **Next Resistance:** $133.60 (0.0% (High))
- **Next Support:** $125.20 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 91. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:30.039249
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DDOG'}`
- **Output Length:** 2000 chars

### Output
```
# Insider Transactions for DDOG
# Data retrieved on: 2026-04-20 14:29:30

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 9 transactions
- **Sale:** 8 transactions
- **Unknown:** 7 transactions
- **Total Sales Value:** $23,373,502
- **Largest Sale (3mo):** 2026-04-09 | POMEL OLIVIER (Chief Executive Officer) | 42443 shares | $4717074.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,SHAH SHARDUL,Director,Sale,7916,949920.0,I
2026-04-15,AGARWAL AMIT,Director,Unknown,20000,,I
2026-04-15,AGARWAL AMIT,Director,Sale,20000,2400000.0,I
2026-04-09,POMEL OLIVIER,Chief Executive Officer,Sale,42443,4717074.0,D
2026-04-09,POMEL OLIVIER,Chief Executive Officer,Unknown,42443,,D
2026-04-08,AGARWAL AMIT,Director,Sale,20000,2445600.0,I
2026-04-08,AGARWAL AMIT,Director,Unknown,20000,,I
2026-04-06,LE-QUOC ALEXIS T,Chief Technology Officer,Sale,32300,3761653.0,D
2026-04-06,LE-QUOC ALEXIS T,Chief Technology Officer,Unknown,32300,,D
2026-04-01,AGARWAL AMIT,Director,Sale,20000,2410600.0,I
2026-04-01,POMEL OLIVIER,Chief Executive Officer,Award/Grant,130858,0.0,D
2026-04-01,LI YANBING,Officer,Award/Grant,51116,0.0,D
2026-04-01,GALLOREESE DAVID,Officer,Award/Grant,25354,0.0,D
2026-04-01,AGARWAL AMIT,Director,Unknown,20000,,I
2026-04-01,ACOCELLA KERRY,General Counsel,Award/Grant,28625,0.0,D
2026-04-01,OBSTLER DAVID M,Chief Financial Officer,Award/Grant,59295,0.0,D
2026-04-01,LE-QUOC ALEXIS T,Chief Technology Officer,Award/Grant,94054,0.0,D
2026-04-01,WALTERS SEAN MICHAEL,Officer,Award/Grant,54797,0.0,D
2026-04-01,PHILLIPS DOMINIC,Director,Award/Grant,40,4722.0,D
2026-04-01,BLITZER ADAM,Chief Operating Officer,Award/Grant,59295,0.0,D
2026-03-23,LE-QUOC ALEXIS T,Chief Technology Officer,Sale,32418,4141418.0,D
2026-03-23,LE-QUOC ALEXIS T,Chief Technology Officer,Unknown,32418,,D
2026-03-23,AGARWAL AMIT,Director,Sale,20000,2547237.0,I
2026-03-23,AGARWAL AMIT,Director,Unknown,20000,,I

```

---

## 92. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:30.229642
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DDOG'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for DDOG

### 2026-04-01
- **Strong Buy**: 17
- **Buy**: 31
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 54

**Sentiment**: 88.9% Bullish, 1.9% Bearish

### 2026-03-01
- **Strong Buy**: 17
- **Buy**: 31
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 54

**Sentiment**: 88.9% Bullish, 1.9% Bearish

### 2026-02-01
- **Strong Buy**: 15
- **Buy**: 32
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 53

**Sentiment**: 88.7% Bullish, 1.9% Bearish

### 2026-01-01
- **Strong Buy**: 13
- **Buy**: 33
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 52

**Sentiment**: 88.5% Bullish, 0.0% Bearish


```

---

## 93. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:30.618548
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DDOG', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 801 chars

### Output
```
## Options Activity for DDOG

**Available Expirations:** 16 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,408 | 2,390 | 0.701 |
| Open Interest | 22,981 | 15,749 | 0.685 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $128.0 | 146 | 134 | 1.09x | 66.7% |
| CALL | 2026-05-01 | $124.0 | 151 | 145 | 1.04x | 72.4% |
| CALL | 2026-04-24 | $143.0 | 1,190 | 1,174 | 1.01x | 77.1% |
| PUT | 2026-05-08 | $115.0 | 1,009 | 1,951 | 0.52x | 89.7% |

```

---

## 94. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:32.142669
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'SNX', 'curr_date': '2026-04-20'}`
- **Output Length:** 1559 chars

### Output
```
# Technical Analysis for SNX
**Date:** 2026-04-20
**Current Price:** $214.32

## Price Action
- **Daily Change:** +2.64%
- **5-Day Change:** +3.98%

## RSI (14)
- **Value:** 83.7 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 13.527
- **Signal Line:** 10.870
- **Histogram:** 2.657 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $168.93 (+26.9% from price)
- **200 SMA:** $154.78 (+38.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $226.43
- **Middle:** $183.81
- **Lower:** $141.18
- **Band Position:** 86%
- **Bandwidth:** 46.4% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $6.77 (3.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $204.16 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 94.9
- **%D:** 93.3
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 62.1 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $189.23 (+13.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 32.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $181.28
- **Current vs VWAP:** +18.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $215.01
- **Recent Low:** $147.84
- **Next Resistance:** $215.01 (0.0% (High))
- **Next Support:** $199.16 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 95. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:32.438738
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'SNX'}`
- **Output Length:** 1110 chars

### Output
```
# Insider Transactions for SNX
# Data retrieved on: 2026-04-20 14:29:32

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 8 transactions
- **Sale:** 3 transactions
- **Total Sales Value:** $1,743,182
- **Largest Sale (3mo):** 2026-04-06 | JORDAN DAVID GREGORY (Chief Financial Officer) | 3225 shares | $611008 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-17,POLK DENNIS J,Officer and Director,Sale,2500,526750,I
2026-04-15,VEZINA ANN F,Director,Award/Grant,1022,0,D
2026-04-15,PUMILIA CLAUDE J,Director,Award/Grant,1022,0,D
2026-04-15,CRUSCO KATHLEEN M,Director,Award/Grant,1022,0,D
2026-04-15,LAMNECK KENNETH T,Director,Award/Grant,1022,0,D
2026-04-15,HUME RICHARD T,Director,Award/Grant,1022,0,D
2026-04-15,SAINTIL MERLINE A,Director,Award/Grant,1022,0,D
2026-04-15,NAYYAR NAYAKI R,Director,Award/Grant,1022,0,D
2026-04-15,HERH TING,Director,Award/Grant,1022,0,D
2026-04-09,MURPHY MIRIAM ANNE,Officer,Sale,3025,605424,D
2026-04-06,JORDAN DAVID GREGORY,Chief Financial Officer,Sale,3225,611008,D

```

---

## 96. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:32.608807
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'SNX'}`
- **Output Length:** 710 chars

### Output
```
## Analyst Recommendation Trends for SNX

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 82.4% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 82.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 9
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

## 97. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:32.954188
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'SNX', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 626 chars

### Output
```
## Options Activity for SNX

**Available Expirations:** 4 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-09-18

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 413 | 113 | 0.274 |
| Open Interest | 1,300 | 772 | 0.594 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $220.0 | 224 | 238 | 0.94x | 36.3% |

```

---

## 98. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:35.874846
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PWR', 'curr_date': '2026-04-20'}`
- **Output Length:** 1553 chars

### Output
```
# Technical Analysis for PWR
**Date:** 2026-04-20
**Current Price:** $601.88

## Price Action
- **Daily Change:** +2.46%
- **5-Day Change:** +1.01%

## RSI (14)
- **Value:** 64.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 11.858
- **Signal Line:** 9.292
- **Histogram:** 2.565 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $556.94 (+8.1% from price)
- **200 SMA:** $454.74 (+32.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $608.81
- **Middle:** $569.90
- **Lower:** $530.99
- **Band Position:** 91%
- **Bandwidth:** 13.7% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $18.84 (3.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $573.61 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.1
- **%D:** 83.7
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 10.4 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $575.29 (+4.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 19.4M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $568.21
- **Current vs VWAP:** +5.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $603.39
- **Recent Low:** $456.16
- **Next Resistance:** $603.39 (0.0% (High))
- **Next Support:** $568.64 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 99. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:36.162434
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PWR'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for PWR in the last 3 months.
```

---

## 100. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:36.303986
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PWR'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for PWR

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 72.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 16
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 69.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 15
- **Hold**: 12
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 66.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 14
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 63.9% Bullish, 0.0% Bearish


```

---

## 101. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:36.666429
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PWR', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 691 chars

### Output
```
## Options Activity for PWR

**Available Expirations:** 10 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 540 | 1,844 | 3.415 |
| Open Interest | 7,640 | 6,857 | 0.898 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-15 | $550.0 | 1,177 | 1,340 | 0.88x | 53.3% |
| PUT | 2026-05-15 | $590.0 | 177 | 204 | 0.87x | 49.9% |

```

---

## 102. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:38.148484
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BURL', 'curr_date': '2026-04-20'}`
- **Output Length:** 1590 chars

### Output
```
# Technical Analysis for BURL
**Date:** 2026-04-20
**Current Price:** $347.26

## Price Action
- **Daily Change:** +2.96%
- **5-Day Change:** +4.47%

## RSI (14)
- **Value:** 64.7 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 8.133
- **Signal Line:** 7.171
- **Histogram:** 0.963 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $315.36 (+10.1% from price)
- **200 SMA:** $286.07 (+21.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $348.74
- **Middle:** $328.86
- **Lower:** $308.98
- **Band Position:** 96%
- **Bandwidth:** 12.1% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $10.74 (3.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $331.14 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.9
- **%D:** 75.9
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 16.3 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $329.37 (+5.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 15.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $327.75
- **Current vs VWAP:** +6.0%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $348.08
- **Recent Low:** $288.16
- **Next Resistance:** $348.08 (0.0% (High))
- **Next Support:** $333.94 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 103. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:38.447595
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BURL'}`
- **Output Length:** 579 chars

### Output
```
# Insider Transactions for BURL
# Data retrieved on: 2026-04-20 14:29:38

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 3 transactions
- **Total Sales Value:** $1,474,688
- **Largest Sale (3mo):** 2026-03-24 | MARQUETTE TRAVIS (President) | 3759 shares | $1232007 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-01,VECCHIO JENNIFER,Officer,Sale,419,138654,D
2026-03-24,MARQUETTE TRAVIS,President,Sale,3759,1232007,D
2026-03-24,GOODWIN MICHAEL W.,Director,Sale,321,104027,D

```

---

## 104. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:38.610861
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'BURL'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for BURL

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 11
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 79.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 11
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 79.2% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 11
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 79.2% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 11
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 86.4% Bullish, 0.0% Bearish


```

---

## 105. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:38.974844
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BURL', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 423 chars

### Output
```
## Options Activity for BURL

**Available Expirations:** 12 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 266 | 218 | 0.82 |
| Open Interest | 994 | 13,538 | 13.62 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bearish positioning

*No unusual options activity detected.*

```

---

## 106. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:40.644596
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'Q', 'curr_date': '2026-04-20'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for Q
**Date:** 2026-04-20
**Current Price:** $136.29

## Price Action
- **Daily Change:** +3.80%
- **5-Day Change:** +0.94%

## RSI (14)
- **Value:** 67.0 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 5.742
- **Signal Line:** 4.397
- **Histogram:** 1.345 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $116.58 (+16.9% from price)
- **200 SMA:** $100.06 (+36.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $140.56
- **Middle:** $121.91
- **Lower:** $103.25
- **Band Position:** 89%
- **Bandwidth:** 30.6% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $5.54 (4.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $127.97 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 86.0
- **%D:** 84.2
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 26.9 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $124.56 (+9.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 47.9M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $120.01
- **Current vs VWAP:** +13.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $140.51
- **Recent Low:** $92.61
- **Next Resistance:** $140.51 (0.0% (High))
- **Next Support:** $129.21 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 107. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:40.919333
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'Q'}`
- **Output Length:** 57 chars

### Output
```
No insider transactions found for Q in the last 3 months.
```

---

## 108. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:41.089375
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'Q'}`
- **Output Length:** 708 chars

### Output
```
## Analyst Recommendation Trends for Q

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 92.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 92.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 92.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 9
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 14

**Sentiment**: 92.9% Bullish, 0.0% Bearish


```

---

## 109. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:41.444174
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'Q', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 688 chars

### Output
```
## Options Activity for Q

**Available Expirations:** 8 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 1,013 | 508 | 0.501 |
| Open Interest | 19,974 | 8,532 | 0.427 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-06-18 | $90.0 | 200 | 216 | 0.93x | 81.0% |
| CALL | 2026-05-15 | $105.0 | 605 | 1,128 | 0.54x | 64.7% |

```

---

## 110. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:43.156095
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MTSI', 'curr_date': '2026-04-20'}`
- **Output Length:** 1587 chars

### Output
```
# Technical Analysis for MTSI
**Date:** 2026-04-20
**Current Price:** $276.97

## Price Action
- **Daily Change:** +5.95%
- **5-Day Change:** +5.06%

## RSI (14)
- **Value:** 69.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 10.430
- **Signal Line:** 6.516
- **Histogram:** 3.914 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $237.30 (+16.7% from price)
- **200 SMA:** $175.50 (+57.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $277.65
- **Middle:** $241.68
- **Lower:** $205.71
- **Band Position:** 99%
- **Bandwidth:** 29.8% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $12.21 (4.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $258.65 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 92.6
- **%D:** 88.5
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 17.8 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $247.13 (+12.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 39.8M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $238.43
- **Current vs VWAP:** +16.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $277.37
- **Recent Low:** $197.63
- **Next Resistance:** $277.37 (0.0% (High))
- **Next Support:** $258.55 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 111. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:43.480229
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MTSI'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for MTSI in the last 3 months.
```

---

## 112. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:43.647087
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MTSI'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for MTSI

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 13
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 81.8% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 13
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 78.3% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 13
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 81.8% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 13
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 81.8% Bullish, 0.0% Bearish


```

---

## 113. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:43.996731
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MTSI', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 627 chars

### Output
```
## Options Activity for MTSI

**Available Expirations:** 5 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 404 | 143 | 0.354 |
| Open Interest | 1,474 | 406 | 0.275 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $270.0 | 105 | 100 | 1.05x | 59.9% |

```

---

## 114. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:45.707870
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ASTS', 'curr_date': '2026-04-20'}`
- **Output Length:** 1530 chars

### Output
```
# Technical Analysis for ASTS
**Date:** 2026-04-20
**Current Price:** $85.53

## Price Action
- **Daily Change:** -5.95%
- **5-Day Change:** -13.58%

## RSI (14)
- **Value:** 46.4 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 0.046
- **Signal Line:** 0.253
- **Histogram:** -0.207 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $89.30 (-4.2% from price)
- **200 SMA:** $72.31 (+18.3% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $101.54
- **Middle:** $89.12
- **Lower:** $76.71
- **Band Position:** 36%
- **Bandwidth:** 27.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $8.28 (9.7% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $73.11 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 33.9
- **%D:** 49.7
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 7.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $89.95 (-4.9% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 227.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $89.58
- **Current vs VWAP:** -4.5%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $108.34
- **Recent Low:** $71.85
- **Next Resistance:** $85.79 (61.8%)
- **Next Support:** $79.66 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 115. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:46.037531
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ASTS'}`
- **Output Length:** 969 chars

### Output
```
# Insider Transactions for ASTS
# Data retrieved on: 2026-04-20 14:29:46

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 4 transactions
- **Sale:** 2 transactions
- **Total Sales Value:** $274,452,000
- **Largest Sale (3mo):** 2026-04-15 | MIKITANI HIROSHI (Beneficial Owner of more than 10% of a Class of Security) | 3040000 shares | $270896800 | Ownership: I

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,MIKITANI HIROSHI,Beneficial Owner of more than 10% of a Class of Security,Sale,3040000,270896800,I
2026-03-24,JOHNSON ANDREW MARTIN,Chief Financial Officer,Award/Grant,100000,0,D
2026-03-24,AVELLAN ABEL ANTONIO,Chief Executive Officer,Award/Grant,184375,0,D
2026-03-24,GUPTA SHANTI B.,Chief Operating Officer,Award/Grant,54687,0,D
2026-03-24,WISNIEWSKI SCOTT,President,Award/Grant,123437,0,D
2026-03-23,YAO HUIWEN,Chief Technology Officer,Sale,40000,3555200,D

```

---

## 116. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:46.207425
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ASTS'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for ASTS

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 7
- **Hold**: 7
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 50.0% Bullish, 11.1% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 7
- **Hold**: 7
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 50.0% Bullish, 11.1% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 7
- **Hold**: 7
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 50.0% Bullish, 11.1% Bearish

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

## 117. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-20T14:29:46.549824
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ASTS', 'num_expirations': 3, 'curr_date': '2026-04-20'}`
- **Output Length:** 1162 chars

### Output
```
## Options Activity for ASTS

**Available Expirations:** 16 dates
**Analyzing:** 2026-04-24, 2026-05-01, 2026-05-08

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 37,850 | 19,316 | 0.51 |
| Open Interest | 91,357 | 54,117 | 0.592 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $60.0 | 529 | 3 | 176.33x | 112.4% |
| CALL | 2026-05-08 | $76.0 | 102 | 1 | 102.0x | 95.5% |
| CALL | 2026-04-24 | $74.0 | 1,317 | 67 | 19.66x | 92.3% |
| CALL | 2026-04-24 | $77.0 | 668 | 52 | 12.85x | 94.9% |
| PUT | 2026-04-24 | $62.0 | 858 | 94 | 9.13x | 122.7% |
| CALL | 2026-04-24 | $78.0 | 563 | 87 | 6.47x | 91.3% |
| CALL | 2026-04-24 | $75.0 | 1,930 | 329 | 5.87x | 88.8% |
| CALL | 2026-05-01 | $77.0 | 109 | 22 | 4.95x | 88.7% |
| CALL | 2026-04-24 | $80.0 | 6,250 | 1,321 | 4.73x | 99.6% |
| CALL | 2026-04-24 | $81.0 | 292 | 62 | 4.71x | 95.9% |

```

---

## 118. Rank candidates

- **Type:** `llm`
- **Node:** ranker
- **Timestamp:** 2026-04-20T14:31:49.708356
- **Model:** `gemini-3-pro-preview`
- **Prompt Length:** 189304 chars
- **Output Length:** 2000 chars

### Prompt
```
You are a professional stock analyst selecting the best short-term trading opportunities from a pre-filtered candidate list.

CURRENT DATE: 2026-04-20

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
"Historical 1-day win rate: 39.5% (618 tracked)\nHistorical 7-day win rate: 44.4% (603 tracked)\nHistorical 30-day win rate: 38.2% (529 tracked)\n\nBest performing strategies (7-day):\n  - short_squeeze: 60.0% win rate, avg +2.1% return (10 samples)\n  - analyst_upgrade: 55.9% win rate, avg +0.2% return (34 samples)\n  - insider_buying: 47.7% win rate, avg -0.1% return (176 samples)\n\nWORST performing strategies (7-day) \u2014 penalize these heavily in scoring:\n  - pre_earnings_accumulation: 28.6% win rate, avg -1.9% return (7 samples)\n  - social_hype: 18.2% win rate, avg -4.1% return (22 samples)\n  - news_catalyst: 0.0% win rate, avg -9.1% return (8 samples)"

CANDIDATES FOR REVIEW:
### NKE (Priority: CRITICAL)
- **Strategy Match**: insider_buying
- **Sources**: insider_buying | **Confluence**: 1 source(s)
- **Quant Pre-Score**: 20/100 | **ML Win Probability**: N/A | **Altman Z-Score**: 3.74 | **Piotroski F-Score**: 4
- **Price**: $45.89 | **Current Price (numeric)**: 45.88999938964844 | **Intraday**: -0.3% | **Avg Volume**: 24,950,915
- **Short Interest**: 4.8%
- **Discovery Context**: Cluster: 2 insiders buying NKE. Largest: Dir Cook Timothy D purchased +$1,060,750
- **Business**: NIKE, Inc., together with its subsidiaries, designs, develops, markets, and sells athletic and casual footwear, apparel, equipment, accessories, and services for men, women, and kids in North America, Europe, the Middle East, Africa, Greater China, the Asia Pacific, and Latin America. The company offers its products under the NIKE, Jordan, Jumpman, Converse, Chuck Taylor, All Star, One Star, Star Chevron, and Jack Purcell trademarks. It also provides a line of performance equipment and accessories, including bags, socks, sport balls, eyewear, timepieces, digital devices, bats, gloves, protective equipment, and other equipment for sports activities, as well as various plastic products to other manufacturers; distributes and licenses casual sneakers, apparel, and accessories; and markets apparel with licensed college and professional team and league logos. In addition, the company offers consumer services and experiences, including sport focused events and activations; fitness and activity apps; sport, fitness, and wellness content; and digital services and features in retail stores. It sells its products to footwear stores; sporting goods stores; athletic specialty stores; department stores; skate, tennis, and golf shops; and other wholesale accounts through NIKE-owned retail stores, independent distributors, licensees, sales representatives, and digital platforms. The company was formerly known as Blue Ribbon Sports, Inc. and changed its name to NIKE, Inc. in May 1971. NIKE, Inc. was founded in 1964 and is headquartered in Beaverton, Oregon.
- **News**: Nike shares recovered slightly on April 20 as CEO Elliott Hill and Director Tim Cook engaged in insider buying near 52-week lows, signaling confidence in the company's Win Now turnaround strategy ahead of the 2026 World Cup.

**Technical Analysis**:
# Technical Analysis for NKE
**Date:** 2026-04-20
**Current Price:** $46.03

## Price Action
- **Daily Change:** +0.72%
- **5-Day Change:** +7.27%

## RSI (14)
- **Value:** 38.8 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -3.045
- **Signal Line:** -3.464
- **Histogram:** 0.419 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $54.92 (-16.2% from price)
- **200 SMA:** $64.79 (-29.0% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $55.97
- **Middle:** $47.43
- **Lower:** $38.90
- **Band Position:** 42%
- **Bandwidth:** 36.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.65 (3.6% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $43.55 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 65.2
- **%D:** 44.3
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 37.2 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $47.43 (-2.9% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -246.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $46.33
- **Current vs VWAP:** -0.6%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $68.04
- **Recent Low:** $42.09
- **Next Resistance:** $47.64 (78.6%)
- **Next Support:** $42.09 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA


**Fundamentals**: Sector: Consumer Cyclical | Industry: Footwear & Accessories | Valuation: Cap: $67.95B, P/E: 30.19, Fwd P/E: 24.42, PEG: 1.42, P/B: 4.82, P/S: 1.46 | Growth: Rev Growth: 0.1%, Earnings Growth: -34.8% | Profitability: Profit Margin: 4.8%, Oper Margin: 6.9%, ROE: 16.0%, ROA: 5.2% | Dividend: 356.0% yield | Financial Health: Current Ratio: 2.14, D/E: 79.33 | Analyst Target: $62 | EPS: $1.52 | Beta: 1.32 | 52W Range: $42.09 - $80.17

**Insider Transactions**:
# Insider Transactions for NKE
# Data retrieved on: 2026-04-20 14:28:06

## Summary
- **Insider Sentiment:** BULLISH ⚡ (more buying than selling)
- **Purchase:** 4 transactions
- **Total Purchases Value:** $2,734,112
- **Largest Purchase (3mo):** 2026-04-10 | COOK TIMOTHY D (Director) | 25000 shares | $1060750.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-13,HILL ELLIOTT,Chief Executive Officer,Purchase,23660,1000000.0,D
2026-04-10,COOK TIMOTHY D,Director,Purchase,25000,1060750.0,D
2026-04-09,ROGERS JOHN W JR,Director,Purchase,4000,173360.0,D
2026-04-07,SWAN ROBERT HOLMES,Director,Purchase,11781,500002.0,D


**Analyst Recommendations**:
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


... [truncated]
```

### Output
```
{
  "rankings": [
    {
      "rank": 1,
      "ticker": "ABX",
      "company_name": "Abacus Global Management, Inc.",
      "current_price": 9.649999618530273,
      "strategy_match": "short_squeeze",
      "final_score": 88,
      "confidence": 9,
      "risk_level": "high",
      "reason": "This setup boasts an extreme 50.0% short interest with 7.7 days to cover, making it a prime candidate for a short squeeze. Timing is urgent as an unusually bullish options flow was detected with a put/call volume ratio of 0.086. The technical setup supports the move, showing a MACD bullish crossover and price trading 5.0% above its 50 SMA. The primary risk is that short sellers double down, leading to increased volatility without a forced squeeze.",
      "description": "Abacus Global Management, Inc. operates as an alternative asset manager and market maker in the life insurance market."
    },
    {
      "rank": 2,
      "ticker": "NKE",
      "company_name": "NIKE, Inc.",
      "current_price": 45.88999938964844,
      "strategy_match": "insider_buying",
      "final_score": 82,
      "confidence": 8,
      "risk_level": "moderate",
      "reason": "NKE features a massive $2.7M cluster of insider buying, highlighting strong fundamental conviction near the 52-week low of $42.09. Timing is critical as CEO Elliott Hill purchased $1M in shares, followed by Director Tim Cook's $1.06M buy, aligning with a fresh MACD bullish crossover. The risk/reward is highly favorable for a contrarian bounce off heavy historical support. The main risk is the elevated put/call ratio of 2.47, indicating persistent institutional bearishness that could cap the upside.",
      "description": "NIKE, Inc. designs, develops, markets, and sells athletic footwear, apparel, equipment, and accessories globally."
    },
    {
      "rank": 3,
      "ticker": "APP",
      "company_name": "AppLovin Corporation",
      "current_price": 478.94000244140625,
      "strategy_match": "options_flow",
      "final_
```

---

