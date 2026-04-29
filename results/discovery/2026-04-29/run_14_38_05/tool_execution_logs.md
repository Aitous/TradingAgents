# Tool Execution Logs - 2026-04-29

## 1. Check average volume (batch)

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:39:35.224442
- **Tool:** `get_average_volume_batch`
- **Parameters:** `{'symbols': ['PRLD', 'TORO', 'COE', 'ALMR', 'RZLV', 'HOMB', 'CHTR', 'ABT', 'WASH', 'INTC', 'PLTR', 'NUE', 'KO', 'CNC', 'APLS', 'SNDK', 'LITE', 'WDC', 'CIEN', 'TER', 'COHR', 'FIX', 'AMKR', 'IVT', 'CTO', 'ARE', 'LKNCY', 'LGIH', 'VNCE', 'DEA', 'VTR', 'ABNB', 'ADI', 'AES', 'ALAB', 'ALGM', 'ALGN', 'ALSN', 'AMAT', 'AMD', 'AMGN', 'LNBIX', 'THM', 'AVEX', 'HYMC', 'BMI', 'BORR', 'V', 'MSFT', 'GOOGL', 'ADP', 'WFC', 'NVDA', 'EQIX', 'VRSN', 'NFLX', 'BEN', 'OGN', 'AIT', 'GTLB', 'OVV', 'OXY', 'EOG', 'FANG', 'ACGL', 'CRL', 'CPRT', 'AMRZ', 'JMSB', 'ASIC', 'FNMA', 'SOMC', 'LXP', 'JOE', 'CLMB', 'CMPR', 'JBSS', 'ET', 'KLRA', 'ARXS', 'TSLA', 'AAPL', 'QCOM', 'AVGO', 'GOOG', 'MA', 'CVX', 'TXN', 'ISRG', 'GS', 'BAC', 'GEV', 'ORCL', 'TMO', 'AVB', 'CHRD', 'MPC', 'OKE', 'DVN', 'DINO', 'SJM', 'HOLX', 'EXAS', 'SEE', 'PGR', 'CTAS', 'ULTA', 'TECH', 'IV', 'SOFI', 'SEGG', 'BTBD'], 'lookback_days': 10, 'curr_date': '2026-04-29', 'cache_key': 'default'}`
- **Output Length:** 18835 chars

### Output
```
{"HOMB": {"symbol": "HOMB", "average_volume": 1976597.9, "latest_volume": 177079.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "CHTR": {"symbol": "CHTR", "average_volume": 3849435.3, "latest_volume": 1015553.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "ABT": {"symbol": "ABT", "average_volume": 13778493.8, "latest_volume": 1317038.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "INTC": {"symbol": "INTC", "average_volume": 133648812.2, "latest_volume": 68421622.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "NUE": {"symbol": "NUE", "average_volume": 1702578.9, "latest_volume": 293589.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "KO": {"symbol": "KO", "average_volume": 13203240.2, "latest_volume": 3620502.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "CNC": {"symbol": "CNC", "average_volume": 6742100.1, "latest_volume": 5801401.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "ARE": {"symbol": "ARE", "average_volume": 2299380.1, "latest_volume": 695101.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AES": {"symbol": "AES", "average_volume": 8923861.6, "latest_volume": 1754416.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "ALGN": {"symbol": "ALGN", "average_volume": 1057816.6, "latest_volume": 201566.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AMAT": {"symbol": "AMAT", "average_volume": 5319389.4, "latest_volume": 1002394.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AMD": {"symbol": "AMD", "average_volume": 45616373.8, "latest_volume": 12221838.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AMGN": {"symbol": "AMGN", "average_volume": 2155363.3, "latest_volume": 311533.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "MSFT": {"symbol": "MSFT", "average_volume": 31391086.0, "latest_volume": 7685460.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "GOOGL": {"symbol": "GOOGL", "average_volume": 21774909.1, "latest_volume": 7994891.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "ADP": {"symbol": "ADP", "average_volume": 2773017.0, "latest_volume": 1064370.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "NVDA": {"symbol": "NVDA", "average_volume": 135603963.2, "latest_volume": 32177232.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "EQIX": {"symbol": "EQIX", "average_volume": 496295.0, "latest_volume": 61450.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "NFLX": {"symbol": "NFLX", "average_volume": 48877594.4, "latest_volume": 6454544.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "BEN": {"symbol": "BEN", "average_volume": 5514468.6, "latest_volume": 2436586.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "OGN": {"symbol": "OGN", "average_volume": 24587004.5, "latest_volume": 2441345.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "EOG": {"symbol": "EOG", "average_volume": 3980141.0, "latest_volume": 423710.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "FANG": {"symbol": "FANG", "average_volume": 2470776.5, "latest_volume": 352565.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "ACGL": {"symbol": "ACGL", "average_volume": 1937199.1, "latest_volume": 409391.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "CRL": {"symbol": "CRL", "average_volume": 696905.0, "latest_volume": 83350.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "CPRT": {"symbol": "CPRT", "average_volume": 8083281.5, "latest_volume": 622415.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AAPL": {"symbol": "AAPL", "average_volume": 39380085.7, "latest_volume": 5990257.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AVGO": {"symbol": "AVGO", "average_volume": 18439721.0, "latest_volume": 3105610.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "GOOG": {"symbol": "GOOG", "average_volume": 14635043.6, "latest_volume": 4635336.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "MA": {"symbol": "MA", "average_volume": 2816491.8, "latest_volume": 1789918.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "CVX": {"symbol": "CVX", "average_volume": 9168665.6, "latest_volume": 1897656.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "ISRG": {"symbol": "ISRG", "average_volume": 2497905.0, "latest_volume": 379350.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "GS": {"symbol": "GS", "average_volume": 1608960.9, "latest_volume": 378109.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "BAC": {"symbol": "BAC", "average_volume": 29785502.6, "latest_volume": 5382926.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "AVB": {"symbol": "AVB", "average_volume": 842393.3, "latest_volume": 363433.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "MPC": {"symbol": "MPC", "average_volume": 1901109.7, "latest_volume": 437797.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "DVN": {"symbol": "DVN", "average_volume": 11189738.0, "latest_volume": 2729780.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "DINO": {"symbol": "DINO", "average_volume": 1763517.3, "latest_volume": 437873.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "HOLX": {"symbol": "HOLX", "average_volume": 12273089.2, "latest_volume": 0.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "EXAS": {"symbol": "EXAS", "average_volume": 3743537.9, "latest_volume": 0.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "CTAS": {"symbol": "CTAS", "average_volume": 1609295.4, "latest_volume": 171654.0, "lookback_days": 10, "source": "cache", "cache_date": "2026-04-29", "error": null}, "PRLD": {"symbol": "PRLD", "average_volume": 552078.3, "latest_volume": 54683.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "TORO": {"symbol": "TORO", "average_volume": 2541895.9, "latest_volume": 35559.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "COE": {"symbol": "COE", "average_volume": 9944.0, "latest_volume": 640.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "ALMR": {"symbol": "ALMR", "average_volume": 929267.2222222222, "latest_volume": 47505.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "RZLV": {"symbol": "RZLV", "average_volume": 11475416.5, "latest_volume": 3154165.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "WASH": {"symbol": "WASH", "average_volume": 273773.2, "latest_volume": 17832.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "PLTR": {"symbol": "PLTR", "average_volume": 38301215.6, "latest_volume": 13300256.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "APLS": {"symbol": "APLS", "average_volume": 3409206.1, "latest_volume": 852661.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "SNDK": {"symbol": "SNDK", "average_volume": 13221957.1, "latest_volume": 4757671.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "LITE": {"symbol": "LITE", "average_volume": 4578361.3, "latest_volume": 2849413.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "WDC": {"symbol": "WDC", "average_volume": 5881777.8, "latest_volume": 3408778.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "CIEN": {"symbol": "CIEN", "average_volume": 1968937.4, "latest_volume": 433474.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "TER": {"symbol": "TER", "average_volume": 3012266.2, "latest_volume": 5694762.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "COHR": {"symbol": "COHR", "average_volume": 5310084.1, "latest_volume": 2515441.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "FIX": {"symbol": "FIX", "average_volume": 409387.6, "latest_volume": 63276.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "AMKR": {"symbol": "AMKR", "average_volume": 4689319.5, "latest_volume": 801995.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "IVT": {"symbol": "IVT", "average_volume": 428479.5, "latest_volume": 120295.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "CTO": {"symbol": "CTO", "average_volume": 151510.3, "latest_volume": 89803.0, "lookback_days": 10, "source": "download", "cache_date": "2026-04-29", "error": null}, "LKNCY": {"symbol": "LKNCY", "average_volume": 971582.3, "latest_volume": 301923.0, "lookback_days": 10, "source": "d... [truncated]
```

---

## 2. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:49.080904
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ALMR', 'curr_date': '2026-04-29'}`
- **Output Length:** 1522 chars

### Output
```
# Technical Analysis for ALMR
**Date:** 2026-04-29
**Current Price:** $26.11

## Price Action
- **Daily Change:** -2.17%
- **5-Day Change:** +7.58%

## RSI (14)
- **Value:** 86.9 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.303
- **Signal Line:** 0.175
- **Histogram:** 0.128 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $24.18 (+8.0% from price)
- **200 SMA:** $24.18 (+8.0% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $27.63
- **Middle:** $24.18
- **Lower:** $20.72
- **Band Position:** 78%
- **Bandwidth:** 28.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.00 (7.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $23.12 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.2
- **%D:** 70.1
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 89.9 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $24.53 (+6.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1.7M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $23.22
- **Current vs VWAP:** +12.4%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $27.20
- **Recent Low:** $21.50
- **Next Resistance:** $27.20 (0.0% (High))
- **Next Support:** $25.85 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 3. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:49.267266
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ALMR'}`
- **Output Length:** 52 chars

### Output
```
No insider transactions data found for symbol 'ALMR'
```

---

## 4. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:49.382571
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ALMR'}`
- **Output Length:** 44 chars

### Output
```
No recommendation trends data found for ALMR
```

---

## 5. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:49.435333
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ALMR', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 34 chars

### Output
```
No options data available for ALMR
```

---

## 6. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:54.304421
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ABT', 'curr_date': '2026-04-29'}`
- **Output Length:** 1531 chars

### Output
```
# Technical Analysis for ABT
**Date:** 2026-04-29
**Current Price:** $93.86

## Price Action
- **Daily Change:** +1.14%
- **5-Day Change:** +2.36%

## RSI (14)
- **Value:** 35.9 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -3.605
- **Signal Line:** -3.344
- **Histogram:** -0.261 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $104.84 (-10.5% from price)
- **200 SMA:** $119.83 (-21.7% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $106.38
- **Middle:** $97.93
- **Lower:** $89.48
- **Band Position:** 26%
- **Bandwidth:** 17.3% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.39 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $90.27 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 21.3
- **%D:** 17.7
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 52.6 (Weakening ↓)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $97.32 (-3.6% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -226.3M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $97.28
- **Current vs VWAP:** -3.5%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $116.09
- **Recent Low:** $90.72
- **Next Resistance:** $96.15 (78.6%)
- **Next Support:** $90.72 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 7. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:54.486070
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ABT'}`
- **Output Length:** 1291 chars

### Output
```
# Insider Transactions for ABT
# Data retrieved on: 2026-04-29 14:48:54

## Summary
- **Insider Sentiment:** BULLISH ⚡ (more buying than selling)
- **Award/Grant:** 12 transactions
- **Purchase:** 2 transactions
- **Total Purchases Value:** $1,127,837
- **Largest Purchase (3mo):** 2026-04-27 | STARKS DANIEL J (Director) | 10000 shares | $926537 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-27,STARKS DANIEL J,Director,Purchase,10000,926537,D
2026-04-24,BABINEAUX CLAIRE FONTENOT,Director,Award/Grant,2286,0,D
2026-04-24,MCDEW DARREN W,Director,Award/Grant,2286,0,D
2026-04-24,GONZALEZ PATRICIA PAOLA,Director,Award/Grant,2286,0,D
2026-04-24,AHUJA NITA M.D.,Director,Award/Grant,2286,0,D
2026-04-24,STARKS DANIEL J,Director,Award/Grant,2286,0,D
2026-04-24,KUMBIER MICHELLE A,Director,Award/Grant,2286,0,D
2026-04-24,O GRADY MICHAEL G,Director,Award/Grant,2286,0,D
2026-04-24,MCKINSTRY NANCY,Director,Award/Grant,2286,0,D
2026-04-24,CONROY KEVIN T,Director,Award/Grant,2286,0,D
2026-04-24,BLOUNT SALLY E,Director,Award/Grant,2286,0,D
2026-04-24,ROMAN MICHAEL F,Director,Award/Grant,2286,0,D
2026-04-24,STRATTON JOHN G.,Director,Award/Grant,2286,0,D
2026-04-23,BOUDREAU PHILIP P,Chief Financial Officer,Purchase,2200,201300,I

```

---

## 8. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:54.618659
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ABT'}`
- **Output Length:** 721 chars

### Output
```
## Analyst Recommendation Trends for ABT

### 2026-04-01
- **Strong Buy**: 10
- **Buy**: 16
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 74.3% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 10
- **Buy**: 15
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 10
- **Buy**: 15
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 10
- **Buy**: 14
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 68.6% Bullish, 0.0% Bearish


```

---

## 9. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:54.897215
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ABT', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 864 chars

### Output
```
## Options Activity for ABT

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,101 | 1,235 | 0.588 |
| Open Interest | 33,250 | 32,630 | 0.981 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $91.0 | 178 | 112 | 1.59x | 31.7% |
| PUT | 2026-05-15 | $86.0 | 201 | 215 | 0.93x | 29.4% |
| PUT | 2026-05-15 | $97.0 | 274 | 301 | 0.91x | 22.1% |
| CALL | 2026-05-15 | $98.0 | 235 | 275 | 0.85x | 31.7% |
| CALL | 2026-05-15 | $90.0 | 199 | 237 | 0.84x | 41.2% |

```

---

## 10. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:56.813703
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'INTC', 'curr_date': '2026-04-29'}`
- **Output Length:** 1580 chars

### Output
```
# Technical Analysis for INTC
**Date:** 2026-04-29
**Current Price:** $84.52

## Price Action
- **Daily Change:** -0.55%
- **5-Day Change:** +29.49%

## RSI (14)
- **Value:** 82.2 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 8.279
- **Signal Line:** 6.245
- **Histogram:** 2.034 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $52.56 (+60.8% from price)
- **200 SMA:** $39.06 (+116.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $86.44
- **Middle:** $63.81
- **Lower:** $41.19
- **Band Position:** 96%
- **Bandwidth:** 70.9% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $4.30 (5.1% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $78.07 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 82.1
- **%D:** 78.0
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 63.8 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $66.09 (+27.9% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 3568.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $65.38
- **Current vs VWAP:** +29.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $87.10
- **Recent Low:** $40.63
- **Next Resistance:** $87.10 (0.0% (High))
- **Next Support:** $76.13 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 11. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:56.982784
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'INTC'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for INTC in the last 3 months.
```

---

## 12. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:57.119642
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

## 13. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:57.490860
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'INTC', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1162 chars

### Output
```
## Options Activity for INTC

**Available Expirations:** 17 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 131,747 | 140,805 | 1.069 |
| Open Interest | 549,415 | 618,660 | 1.126 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $87.0 | 4,684 | 110 | 42.58x | 92.8% |
| PUT | 2026-05-01 | $87.0 | 10,359 | 380 | 27.26x | 112.8% |
| PUT | 2026-05-08 | $85.0 | 7,653 | 559 | 13.69x | 91.8% |
| PUT | 2026-05-01 | $88.0 | 4,375 | 369 | 11.86x | 116.2% |
| PUT | 2026-05-01 | $90.0 | 5,234 | 494 | 10.6x | 119.2% |
| PUT | 2026-05-08 | $84.0 | 7,437 | 749 | 9.93x | 91.3% |
| PUT | 2026-05-01 | $86.0 | 7,589 | 846 | 8.97x | 111.3% |
| PUT | 2026-05-15 | $87.0 | 899 | 112 | 8.03x | 85.6% |
| CALL | 2026-05-01 | $48.5 | 485 | 61 | 7.95x | 0.0% |
| CALL | 2026-05-01 | $88.0 | 26,412 | 5,756 | 4.59x | 52.2% |

```

---

## 14. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:59.251178
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PLTR', 'curr_date': '2026-04-29'}`
- **Output Length:** 1547 chars

### Output
```
# Technical Analysis for PLTR
**Date:** 2026-04-29
**Current Price:** $141.18

## Price Action
- **Daily Change:** -1.34%
- **5-Day Change:** -7.50%

## RSI (14)
- **Value:** 46.8 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.887
- **Signal Line:** -1.250
- **Histogram:** 0.364 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $144.83 (-2.5% from price)
- **200 SMA:** $164.43 (-14.1% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $155.56
- **Middle:** $142.57
- **Lower:** $129.58
- **Band Position:** 45%
- **Bandwidth:** 18.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $6.72 (4.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $131.09 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 43.6
- **%D:** 55.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 3.8 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $143.68 (-1.7% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 431.7M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $140.23
- **Current vs VWAP:** +0.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $162.40
- **Recent Low:** $122.68
- **Next Resistance:** $142.54 (50.0%)
- **Next Support:** $137.85 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 15. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:59.462161
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PLTR'}`
- **Output Length:** 577 chars

### Output
```
# Insider Transactions for PLTR
# Data retrieved on: 2026-04-29 14:48:59

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 1 transactions
- **Sale:** 1 transactions
- **Total Sales Value:** $2,230,580
- **Largest Sale (3mo):** 2026-04-15 | MOORE ALEXANDER D (Director) | 16000 shares | $2230580.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-24,BUCKLEY JEFFREY JOHANSING,Officer,Award/Grant,21141,0.0,D
2026-04-15,MOORE ALEXANDER D,Director,Sale,16000,2230580.0,D

```

---

## 16. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:59.573451
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PLTR'}`
- **Output Length:** 717 chars

### Output
```
## Analyst Recommendation Trends for PLTR

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 15
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 1
- **Total Analysts**: 36

**Sentiment**: 66.7% Bullish, 5.6% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 14
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 1
- **Total Analysts**: 34

**Sentiment**: 64.7% Bullish, 5.9% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 8
- **Hold**: 17
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 33

**Sentiment**: 39.4% Bullish, 9.1% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 7
- **Hold**: 18
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 32

**Sentiment**: 34.4% Bullish, 9.4% Bearish


```

---

## 17. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:48:59.861538
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PLTR', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1177 chars

### Output
```
## Options Activity for PLTR

**Available Expirations:** 19 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 83,910 | 78,651 | 0.937 |
| Open Interest | 452,021 | 359,683 | 0.796 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $136.0 | 2,234 | 198 | 11.28x | 57.9% |
| CALL | 2026-05-08 | $149.0 | 3,325 | 518 | 6.42x | 81.9% |
| CALL | 2026-05-01 | $137.0 | 4,090 | 1,016 | 4.03x | 58.4% |
| CALL | 2026-05-08 | $142.0 | 3,730 | 949 | 3.93x | 81.2% |
| CALL | 2026-05-01 | $138.0 | 2,674 | 694 | 3.85x | 56.9% |
| CALL | 2026-05-01 | $141.0 | 4,059 | 1,456 | 2.79x | 57.6% |
| CALL | 2026-05-01 | $135.0 | 2,919 | 1,064 | 2.74x | 58.7% |
| PUT | 2026-05-01 | $134.0 | 4,008 | 1,495 | 2.68x | 56.1% |
| CALL | 2026-05-01 | $139.0 | 2,197 | 863 | 2.55x | 57.6% |
| CALL | 2026-05-08 | $137.0 | 391 | 163 | 2.4x | 82.0% |

```

---

## 18. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:01.658900
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NUE', 'curr_date': '2026-04-29'}`
- **Output Length:** 1596 chars

### Output
```
# Technical Analysis for NUE
**Date:** 2026-04-29
**Current Price:** $225.11

## Price Action
- **Daily Change:** +4.70%
- **5-Day Change:** +5.06%

## RSI (14)
- **Value:** 86.9 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 12.394
- **Signal Line:** 9.197
- **Histogram:** 3.196 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $179.14 (+25.7% from price)
- **200 SMA:** $158.50 (+42.0% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $227.67
- **Middle:** $192.97
- **Lower:** $158.27
- **Band Position:** 96%
- **Bandwidth:** 36.0% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $5.73 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $216.52 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 92.8
- **%D:** 92.8
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 65.5 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $197.29 (+14.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 23.8M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $195.61
- **Current vs VWAP:** +15.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $227.48
- **Recent Low:** $154.61
- **Next Resistance:** $227.48 (0.0% (High))
- **Next Support:** $210.28 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 19. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:01.844733
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NUE'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for NUE in the last 3 months.
```

---

## 20. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:01.962259
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NUE'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for NUE

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 72.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 68.2% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 68.2% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 12
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 21

**Sentiment**: 81.0% Bullish, 0.0% Bearish


```

---

## 21. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:02.273640
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NUE', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 690 chars

### Output
```
## Options Activity for NUE

**Available Expirations:** 10 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,003 | 874 | 0.436 |
| Open Interest | 19,689 | 13,807 | 0.701 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-06-18 | $250.0 | 832 | 996 | 0.84x | 37.6% |
| PUT | 2026-06-18 | $65.0 | 240 | 300 | 0.8x | 181.7% |

```

---

## 22. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:04.282821
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'KO', 'curr_date': '2026-04-29'}`
- **Output Length:** 1550 chars

### Output
```
# Technical Analysis for KO
**Date:** 2026-04-29
**Current Price:** $78.35

## Price Action
- **Daily Change:** +3.86%
- **5-Day Change:** +4.98%

## RSI (14)
- **Value:** 59.8 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.016
- **Signal Line:** -0.186
- **Histogram:** 0.169 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $77.00 (+1.7% from price)
- **200 SMA:** $71.06 (+10.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $78.35
- **Middle:** $76.25
- **Lower:** $74.15
- **Band Position:** 100%
- **Bandwidth:** 5.5% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $1.50 (1.9% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $76.11 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 51.0
- **%D:** 41.5
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 16.8 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $76.19 (+2.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -142.4M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $76.44
- **Current vs VWAP:** +2.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $81.44
- **Recent Low:** $74.07
- **Next Resistance:** $78.62 (38.2%)
- **Next Support:** $77.76 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 23. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:04.545959
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'KO'}`
- **Output Length:** 58 chars

### Output
```
No insider transactions found for KO in the last 3 months.
```

---

## 24. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:04.675697
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'KO'}`
- **Output Length:** 713 chars

### Output
```
## Analyst Recommendation Trends for KO

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 20
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 82.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 20
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 82.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 21
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 85.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 22
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 88.6% Bullish, 0.0% Bearish


```

---

## 25. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:05.008870
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'KO', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1165 chars

### Output
```
## Options Activity for KO

**Available Expirations:** 17 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 12,665 | 5,504 | 0.435 |
| Open Interest | 113,005 | 107,452 | 0.951 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $86.0 | 286 | 169 | 1.69x | 77.7% |
| CALL | 2026-05-08 | $76.0 | 1,634 | 969 | 1.69x | 39.3% |
| CALL | 2026-05-15 | $83.0 | 166 | 141 | 1.18x | 23.4% |
| CALL | 2026-05-01 | $88.0 | 183 | 169 | 1.08x | 52.3% |
| CALL | 2026-05-08 | $90.0 | 314 | 321 | 0.98x | 41.8% |
| CALL | 2026-05-01 | $80.0 | 4,421 | 4,670 | 0.95x | 25.1% |
| CALL | 2026-05-15 | $60.0 | 412 | 450 | 0.92x | 77.7% |
| PUT | 2026-05-01 | $76.0 | 1,098 | 1,248 | 0.88x | 23.4% |
| CALL | 2026-05-15 | $81.0 | 118 | 148 | 0.8x | 22.4% |
| PUT | 2026-05-01 | $68.0 | 134 | 196 | 0.68x | 64.1% |

```

---

## 26. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:06.809713
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'APLS', 'curr_date': '2026-04-29'}`
- **Output Length:** 1549 chars

### Output
```
# Technical Analysis for APLS
**Date:** 2026-04-29
**Current Price:** $40.99

## Price Action
- **Daily Change:** +0.10%
- **5-Day Change:** +0.24%

## RSI (14)
- **Value:** 87.3 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 3.928
- **Signal Line:** 4.532
- **Histogram:** -0.603 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $28.10 (+45.9% from price)
- **200 SMA:** $24.68 (+66.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $41.20
- **Middle:** $40.77
- **Lower:** $40.34
- **Band Position:** 76%
- **Bandwidth:** 2.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.66 (1.6% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $40.00 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 93.7
- **%D:** 92.9
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 75.7 (Weakening ↓)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $37.85 (+8.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 148.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $40.50
- **Current vs VWAP:** +1.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $41.00
- **Recent Low:** $16.83
- **Next Resistance:** $41.00 (0.0% (High))
- **Next Support:** $35.30 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 27. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:06.991391
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'APLS'}`
- **Output Length:** 307 chars

### Output
```
# Insider Transactions for APLS
# Data retrieved on: 2026-04-29 14:49:06

## Summary
- **Insider Sentiment:** Neutral
- **Gift:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-17,SULLIVAN TIMOTHY EUGENE,Chief Financial Officer,Gift,11096,0,D

```

---

## 28. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:07.096843
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'APLS'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for APLS

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 9
- **Hold**: 15
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 27

**Sentiment**: 44.4% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 12
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 69.0% Bullish, 3.4% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 12
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 28

**Sentiment**: 71.4% Bullish, 3.6% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 12
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 28

**Sentiment**: 67.9% Bullish, 3.6% Bearish


```

---

## 29. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:07.476567
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'APLS', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 628 chars

### Output
```
## Options Activity for APLS

**Available Expirations:** 6 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 274 | 964 | 3.518 |
| Open Interest | 10,492 | 11,806 | 1.125 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-15 | $15.0 | 246 | 66 | 3.73x | 190.6% |

```

---

## 30. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:09.465402
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'SNDK', 'curr_date': '2026-04-29'}`
- **Output Length:** 1537 chars

### Output
```
# Technical Analysis for SNDK
**Date:** 2026-04-29
**Current Price:** $1002.35

## Price Action
- **Daily Change:** -6.34%
- **5-Day Change:** +2.38%

## RSI (14)
- **Value:** 64.1 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 86.357
- **Signal Line:** 76.869
- **Histogram:** 9.488 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $732.18 (+36.9% from price)
- **200 SMA:** $336.89 (+197.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $1110.59
- **Middle:** $868.45
- **Lower:** $626.31
- **Band Position:** 78%
- **Bandwidth:** 55.8% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $63.58 (6.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $906.98 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 78.6
- **%D:** 80.8
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 47.9 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $886.94 (+13.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 459.4M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $852.43
- **Current vs VWAP:** +17.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $1070.66
- **Recent Low:** $517.00
- **Next Resistance:** $1070.66 (0.0% (High))
- **Next Support:** $940.00 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 31. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:09.703082
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'SNDK'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for SNDK in the last 3 months.
```

---

## 32. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:09.826834
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'SNDK'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for SNDK

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 13
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 80.8% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 13
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 76.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 13
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 28

**Sentiment**: 71.4% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 12
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 73.1% Bullish, 0.0% Bearish


```

---

## 33. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:10.258758
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'SNDK', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1157 chars

### Output
```
## Options Activity for SNDK

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 18,136 | 16,827 | 0.928 |
| Open Interest | 88,502 | 135,358 | 1.529 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $1090.0 | 160 | 8 | 20.0x | 191.3% |
| CALL | 2026-05-01 | $1145.0 | 445 | 50 | 8.9x | 172.1% |
| PUT | 2026-05-01 | $1070.0 | 199 | 39 | 5.1x | 189.8% |
| PUT | 2026-05-01 | $1080.0 | 184 | 37 | 4.97x | 192.1% |
| CALL | 2026-05-01 | $1340.0 | 559 | 162 | 3.45x | 160.7% |
| PUT | 2026-05-08 | $340.0 | 235 | 74 | 3.18x | 336.5% |
| PUT | 2026-05-08 | $315.0 | 220 | 84 | 2.62x | 357.4% |
| CALL | 2026-05-01 | $1090.0 | 250 | 108 | 2.31x | 176.6% |
| CALL | 2026-05-01 | $1390.0 | 526 | 231 | 2.28x | 159.9% |
| CALL | 2026-05-01 | $1400.0 | 1,443 | 782 | 1.85x | 159.4% |

```

---

## 34. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:12.060631
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'LITE', 'curr_date': '2026-04-29'}`
- **Output Length:** 1566 chars

### Output
```
# Technical Analysis for LITE
**Date:** 2026-04-29
**Current Price:** $791.37

## Price Action
- **Daily Change:** -7.95%
- **5-Day Change:** -9.41%

## RSI (14)
- **Value:** 48.2 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 34.630
- **Signal Line:** 47.072
- **Histogram:** -12.443 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $745.00 (+6.2% from price)
- **200 SMA:** $366.90 (+115.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $951.50
- **Middle:** $844.41
- **Lower:** $737.33
- **Band Position:** 25%
- **Bandwidth:** 25.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $68.65 (8.7% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $688.40 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 36.4
- **%D:** 47.2
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 16.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $829.33 (-4.6% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 223.7M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $838.59
- **Current vs VWAP:** -5.6%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $960.00
- **Recent Low:** $528.01
- **Next Resistance:** $794.98 (38.2%)
- **Next Support:** $744.01 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 35. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:12.262627
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'LITE'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for LITE in the last 3 months.
```

---

## 36. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:12.366061
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'LITE'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for LITE

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 17
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 30

**Sentiment**: 80.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 17
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 82.8% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 79.3% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 79.3% Bullish, 0.0% Bearish


```

---

## 37. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:12.713661
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'LITE', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1139 chars

### Output
```
## Options Activity for LITE

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 7,617 | 7,844 | 1.03 |
| Open Interest | 24,022 | 38,259 | 1.593 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $510.0 | 357 | 32 | 11.16x | 180.8% |
| PUT | 2026-05-08 | $505.0 | 208 | 37 | 5.62x | 181.5% |
| CALL | 2026-05-08 | $810.0 | 225 | 45 | 5.0x | 124.3% |
| PUT | 2026-05-08 | $490.0 | 153 | 37 | 4.14x | 189.5% |
| PUT | 2026-05-08 | $365.0 | 162 | 52 | 3.12x | 266.5% |
| PUT | 2026-05-01 | $622.5 | 201 | 65 | 3.09x | 215.5% |
| PUT | 2026-05-01 | $835.0 | 187 | 64 | 2.92x | 129.1% |
| CALL | 2026-05-01 | $845.0 | 181 | 70 | 2.59x | 97.5% |
| PUT | 2026-05-01 | $555.0 | 262 | 104 | 2.52x | 235.4% |
| PUT | 2026-05-01 | $602.5 | 174 | 88 | 1.98x | 229.9% |

```

---

## 38. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:14.262880
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'WDC', 'curr_date': '2026-04-29'}`
- **Output Length:** 1527 chars

### Output
```
# Technical Analysis for WDC
**Date:** 2026-04-29
**Current Price:** $390.99

## Price Action
- **Daily Change:** -2.43%
- **5-Day Change:** +0.49%

## RSI (14)
- **Value:** 69.1 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 28.693
- **Signal Line:** 25.956
- **Histogram:** 2.737 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $308.76 (+26.6% from price)
- **200 SMA:** $185.52 (+110.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $432.70
- **Middle:** $353.04
- **Lower:** $273.39
- **Band Position:** 74%
- **Bandwidth:** 45.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $20.68 (5.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $359.98 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 74.7
- **%D:** 82.1
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 49.1 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $359.88 (+8.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 446.0M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $345.37
- **Current vs VWAP:** +13.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $416.37
- **Recent Low:** $238.00
- **Next Resistance:** $416.37 (0.0% (High))
- **Next Support:** $374.27 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 39. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:14.509636
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'WDC'}`
- **Output Length:** 611 chars

### Output
```
# Insider Transactions for WDC
# Data retrieved on: 2026-04-29 14:49:14

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Unknown:** 2 transactions
- **Sale:** 1 transactions
- **Total Sales Value:** $136,884
- **Largest Sale (3mo):** 2026-04-21 | TREGILLIS CYNTHIA LOCK (Officer) | 363 shares | $136884.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-21,TREGILLIS CYNTHIA LOCK,Officer,Sale,363,136884.0,D
2026-04-20,GUBBI VIDYADHARA K,Officer,Unknown,34,,D
2026-04-20,TREGILLIS CYNTHIA LOCK,Officer,Unknown,12,,D

```

---

## 40. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:14.611373
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'WDC'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for WDC

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 19
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 30

**Sentiment**: 86.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 19
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 31

**Sentiment**: 80.6% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 20
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 81.2% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 20
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 81.2% Bullish, 0.0% Bearish


```

---

## 41. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:14.971633
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'WDC', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1144 chars

### Output
```
## Options Activity for WDC

**Available Expirations:** 19 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 6,705 | 7,537 | 1.124 |
| Open Interest | 36,110 | 41,250 | 1.142 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $175.0 | 228 | 12 | 19.0x | 195.3% |
| PUT | 2026-05-08 | $170.0 | 175 | 16 | 10.94x | 233.8% |
| PUT | 2026-05-01 | $427.5 | 104 | 11 | 9.45x | 169.3% |
| PUT | 2026-05-08 | $160.0 | 327 | 45 | 7.27x | 203.9% |
| CALL | 2026-05-01 | $475.0 | 273 | 57 | 4.79x | 128.6% |
| PUT | 2026-05-08 | $145.0 | 1,088 | 227 | 4.79x | 300.8% |
| PUT | 2026-05-08 | $150.0 | 522 | 133 | 3.92x | 358.6% |
| PUT | 2026-05-01 | $265.0 | 206 | 60 | 3.43x | 193.8% |
| PUT | 2026-05-08 | $165.0 | 202 | 68 | 2.97x | 241.8% |
| CALL | 2026-05-08 | $410.0 | 330 | 119 | 2.77x | 90.0% |

```

---

## 42. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:16.587937
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CIEN', 'curr_date': '2026-04-29'}`
- **Output Length:** 1571 chars

### Output
```
# Technical Analysis for CIEN
**Date:** 2026-04-29
**Current Price:** $473.69

## Price Action
- **Daily Change:** -6.45%
- **5-Day Change:** -5.07%

## RSI (14)
- **Value:** 53.6 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 30.191
- **Signal Line:** 34.421
- **Histogram:** -4.230 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $403.95 (+17.3% from price)
- **200 SMA:** $230.64 (+105.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $548.57
- **Middle:** $478.06
- **Lower:** $407.55
- **Band Position:** 47%
- **Bandwidth:** 29.5% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $32.36 (6.8% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $425.16 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 58.3
- **%D:** 71.6
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 18.9 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $474.53 (-0.2% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 159.3M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $472.41
- **Current vs VWAP:** +0.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $527.86
- **Recent Low:** $278.39
- **Next Resistance:** $527.86 (0.0% (High))
- **Next Support:** $468.99 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 43. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:16.772544
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CIEN'}`
- **Output Length:** 827 chars

### Output
```
# Insider Transactions for CIEN
# Data retrieved on: 2026-04-29 14:49:16

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 7 transactions
- **Total Sales Value:** $15,506,055
- **Largest Sale (3mo):** 2026-04-01 | PHIPPS JASON (Officer) | 22014 shares | $9021272 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,GAGE BRODIE,Officer,Sale,1200,566148,D
2026-04-15,DIPERNA DINO,Officer,Sale,2829,1334694,D
2026-04-15,ROTHENSTEIN DAVID M.,Officer,Sale,2500,1169134,D
2026-04-15,SMITH GARY BRIAN,Chief Executive Officer,Sale,2952,1379976,D
2026-04-01,PHIPPS JASON,Officer,Sale,22014,9021272,D
2026-04-01,KOSARAJU SHEELA,General Counsel,Sale,2012,808241,D
2026-04-01,SMITH GARY BRIAN,Chief Executive Officer,Sale,2952,1226590,D

```

---

## 44. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:16.875996
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CIEN'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for CIEN

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 11
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 72.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 10
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 66.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 10
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 66.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 10
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 70.8% Bullish, 0.0% Bearish


```

---

## 45. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:17.250141
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CIEN', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1141 chars

### Output
```
## Options Activity for CIEN

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 4,593 | 4,840 | 1.054 |
| Open Interest | 16,069 | 20,114 | 1.252 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $640.0 | 386 | 57 | 6.77x | 89.4% |
| CALL | 2026-05-15 | $460.0 | 126 | 35 | 3.6x | 87.5% |
| CALL | 2026-05-08 | $470.0 | 147 | 68 | 2.16x | 91.8% |
| PUT | 2026-05-08 | $370.0 | 159 | 75 | 2.12x | 132.3% |
| CALL | 2026-05-01 | $517.5 | 168 | 81 | 2.07x | 92.7% |
| PUT | 2026-05-01 | $285.0 | 120 | 61 | 1.97x | 280.1% |
| CALL | 2026-05-01 | $520.0 | 471 | 288 | 1.64x | 97.6% |
| CALL | 2026-05-15 | $550.0 | 309 | 240 | 1.29x | 85.3% |
| PUT | 2026-05-01 | $275.0 | 170 | 160 | 1.06x | 304.9% |
| CALL | 2026-05-01 | $502.5 | 810 | 835 | 0.97x | 95.0% |

```

---

## 46. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:18.943436
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TER', 'curr_date': '2026-04-29'}`
- **Output Length:** 1532 chars

### Output
```
# Technical Analysis for TER
**Date:** 2026-04-29
**Current Price:** $380.13

## Price Action
- **Daily Change:** -5.44%
- **5-Day Change:** -1.31%

## RSI (14)
- **Value:** 58.6 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 23.089
- **Signal Line:** 21.367
- **Histogram:** 1.722 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $328.62 (+15.7% from price)
- **200 SMA:** $205.72 (+84.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $428.98
- **Middle:** $361.65
- **Lower:** $294.33
- **Band Position:** 64%
- **Bandwidth:** 37.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $18.20 (4.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $352.84 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 67.2
- **%D:** 80.2
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 30.2 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $366.33 (+3.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 155.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $361.19
- **Current vs VWAP:** +5.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $422.11
- **Recent Low:** $263.02
- **Next Resistance:** $384.56 (23.6%)
- **Next Support:** $361.34 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 47. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:19.216127
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TER'}`
- **Output Length:** 553 chars

### Output
```
# Insider Transactions for TER
# Data retrieved on: 2026-04-29 14:49:19

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 3 transactions
- **Total Sales Value:** $273,902
- **Largest Sale (3mo):** 2026-04-06 | JOHNSON MERCEDES (Director) | 625 shares | $195125 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-08,MILLS REGAN,Officer,Sale,16,5475,D
2026-04-06,JOHNSON MERCEDES,Director,Sale,625,195125,D
2026-04-02,MILLS REGAN,Officer,Sale,252,73302,D

```

---

## 48. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:19.326837
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TER'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for TER

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 65.2% Bullish, 4.3% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 65.2% Bullish, 4.3% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 68.2% Bullish, 4.5% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 11
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 72.7% Bullish, 4.5% Bearish


```

---

## 49. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:19.677587
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TER', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1144 chars

### Output
```
## Options Activity for TER

**Available Expirations:** 14 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 4,437 | 3,699 | 0.834 |
| Open Interest | 21,805 | 15,126 | 0.694 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-08 | $320.0 | 170 | 7 | 24.29x | 97.0% |
| CALL | 2026-05-01 | $415.0 | 133 | 36 | 3.69x | 114.3% |
| CALL | 2026-05-15 | $377.5 | 296 | 162 | 1.83x | 81.5% |
| CALL | 2026-05-01 | $500.0 | 328 | 201 | 1.63x | 165.6% |
| CALL | 2026-05-01 | $435.0 | 354 | 266 | 1.33x | 118.0% |
| CALL | 2026-05-01 | $417.5 | 546 | 502 | 1.09x | 103.9% |
| PUT | 2026-05-15 | $410.0 | 176 | 172 | 1.02x | 0.0% |
| PUT | 2026-05-01 | $290.0 | 154 | 166 | 0.93x | 95.5% |
| PUT | 2026-05-01 | $265.0 | 114 | 127 | 0.9x | 114.3% |
| PUT | 2026-05-01 | $305.0 | 116 | 138 | 0.84x | 99.2% |

```

---

## 50. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:21.328494
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'COHR', 'curr_date': '2026-04-29'}`
- **Output Length:** 1530 chars

### Output
```
# Technical Analysis for COHR
**Date:** 2026-04-29
**Current Price:** $303.97

## Price Action
- **Daily Change:** -5.46%
- **5-Day Change:** -13.27%

## RSI (14)
- **Value:** 51.6 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 19.407
- **Signal Line:** 20.865
- **Histogram:** -1.458 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $273.06 (+11.3% from price)
- **200 SMA:** $175.15 (+73.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $377.38
- **Middle:** $303.48
- **Lower:** $229.58
- **Band Position:** 50%
- **Bandwidth:** 48.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $22.84 (7.5% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $269.71 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 43.7
- **%D:** 62.9
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 23.6 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $308.64 (-1.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 130.2M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $301.58
- **Current vs VWAP:** +0.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $364.80
- **Recent Low:** $204.57
- **Next Resistance:** $326.99 (23.6%)
- **Next Support:** $303.59 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 51. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:21.556596
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'COHR'}`
- **Output Length:** 494 chars

### Output
```
# Insider Transactions for COHR
# Data retrieved on: 2026-04-29 14:49:21

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Total Sales Value:** $702,000
- **Largest Sale (3mo):** 2026-04-22 | LUTHER SHERRI R (Chief Financial Officer) | 2000 shares | $702000.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-22,LUTHER SHERRI R,Chief Financial Officer,Sale,2000,702000.0,D

```

---

## 52. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:21.657010
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'COHR'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for COHR

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 14
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 75.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 13
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 28

**Sentiment**: 75.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 14
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 75.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 14
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 75.9% Bullish, 0.0% Bearish


```

---

## 53. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:21.946655
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'COHR', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1156 chars

### Output
```
## Options Activity for COHR

**Available Expirations:** 14 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,887 | 4,946 | 1.272 |
| Open Interest | 26,473 | 39,145 | 1.479 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-08 | $305.0 | 152 | 28 | 5.43x | 123.4% |
| CALL | 2026-05-08 | $355.0 | 209 | 137 | 1.53x | 122.0% |
| CALL | 2026-05-01 | $370.0 | 407 | 313 | 1.3x | 99.6% |
| PUT | 2026-05-01 | $340.0 | 502 | 436 | 1.15x | 117.5% |
| CALL | 2026-05-01 | $365.0 | 256 | 225 | 1.14x | 97.9% |
| CALL | 2026-05-01 | $347.5 | 1,255 | 1,274 | 0.99x | 89.3% |
| PUT | 2026-05-01 | $237.5 | 227 | 234 | 0.97x | 154.0% |
| PUT | 2026-05-01 | $255.0 | 640 | 689 | 0.93x | 116.3% |
| PUT | 2026-05-01 | $302.5 | 1,224 | 1,326 | 0.92x | 112.6% |
| PUT | 2026-05-08 | $297.5 | 103 | 113 | 0.91x | 133.8% |

```

---

## 54. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:23.615967
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMKR', 'curr_date': '2026-04-29'}`
- **Output Length:** 1519 chars

### Output
```
# Technical Analysis for AMKR
**Date:** 2026-04-29
**Current Price:** $71.36

## Price Action
- **Daily Change:** -5.63%
- **5-Day Change:** -1.23%

## RSI (14)
- **Value:** 67.1 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 7.109
- **Signal Line:** 5.885
- **Histogram:** 1.224 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $52.00 (+37.2% from price)
- **200 SMA:** $38.60 (+84.9% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $82.74
- **Middle:** $61.00
- **Lower:** $39.26
- **Band Position:** 74%
- **Bandwidth:** 71.3% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.19 (5.9% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $65.08 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 79.6
- **%D:** 87.0
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 41.7 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $63.69 (+12.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 96.7M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $63.58
- **Current vs VWAP:** +12.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $79.23
- **Recent Low:** $40.14
- **Next Resistance:** $79.23 (0.0% (High))
- **Next Support:** $70.01 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 55. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:23.805668
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMKR'}`
- **Output Length:** 1010 chars

### Output
```
# Insider Transactions for AMKR
# Data retrieved on: 2026-04-29 14:49:23

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Option Exercise:** 3 transactions
- **Unknown:** 3 transactions
- **Sale:** 1 transactions
- **Total Sales Value:** $297,150
- **Largest Sale (3mo):** 2026-04-16 | ROGERS MARK N. (General Counsel) | 5000 shares | $297150.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,ROGERS MARK N.,General Counsel,Sale,5000,297150.0,D
2026-04-16,ROGERS MARK N.,General Counsel,Option Exercise,5000,37000.0,D
2026-04-07,KIM SUSAN Y,Director and Beneficial Owner of more than 10% of a Class of Security,Option Exercise,20000,113200.0,D
2026-04-07,MORSE ROBERT RANDOLPH,Director,Option Exercise,20000,113200.0,D
2026-03-31,RUTTEN GUILLAUME MARIE JEAN,Director,Unknown,8692,,D
2026-03-31,FAUST MEGAN,Chief Financial Officer,Unknown,8692,,D
2026-03-31,ENGEL KEVIN K,Chief Executive Officer,Unknown,8692,,D

```

---

## 56. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:23.912145
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMKR'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for AMKR

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 7
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 50.0% Bullish, 5.6% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 7
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 50.0% Bullish, 5.6% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 7
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 50.0% Bullish, 5.6% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 9
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 19

**Sentiment**: 57.9% Bullish, 5.3% Bearish


```

---

## 57. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:24.185990
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMKR', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 692 chars

### Output
```
## Options Activity for AMKR

**Available Expirations:** 8 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,384 | 1,542 | 0.647 |
| Open Interest | 47,854 | 64,036 | 1.338 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-15 | $36.0 | 468 | 491 | 0.95x | 144.5% |
| PUT | 2026-05-15 | $49.0 | 119 | 155 | 0.77x | 116.4% |

```

---

## 58. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:26.797927
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'LKNCY', 'curr_date': '2026-04-29'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for LKNCY
**Date:** 2026-04-29
**Current Price:** $31.06

## Price Action
- **Daily Change:** +0.13%
- **5-Day Change:** -4.37%

## RSI (14)
- **Value:** 39.8 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.564
- **Signal Line:** -0.439
- **Histogram:** -0.125 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $34.02 (-8.7% from price)
- **200 SMA:** $36.32 (-14.5% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $34.48
- **Middle:** $32.39
- **Lower:** $30.30
- **Band Position:** 18%
- **Bandwidth:** 12.9% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $1.45 (4.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $28.88 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 20.5
- **%D:** 36.2
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 34.1 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $32.49 (-4.4% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -19.4M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $32.39
- **Current vs VWAP:** -4.1%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $39.20
- **Recent Low:** $30.00
- **Next Resistance:** $31.97 (78.6%)
- **Next Support:** $30.00 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 59. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:27.031951
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'LKNCY'}`
- **Output Length:** 53 chars

### Output
```
No insider transactions data found for symbol 'LKNCY'
```

---

## 60. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:27.152948
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'LKNCY'}`
- **Output Length:** 716 chars

### Output
```
## Analyst Recommendation Trends for LKNCY

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 12
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 12
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 88.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 12
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 94.1% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 12
- **Hold**: 1
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 94.1% Bullish, 0.0% Bearish


```

---

## 61. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:27.202188
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'LKNCY', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 35 chars

### Output
```
No options data available for LKNCY
```

---

## 62. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:28.968994
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'VTR', 'curr_date': '2026-04-29'}`
- **Output Length:** 1558 chars

### Output
```
# Technical Analysis for VTR
**Date:** 2026-04-29
**Current Price:** $87.57

## Price Action
- **Daily Change:** +3.39%
- **5-Day Change:** +9.33%

## RSI (14)
- **Value:** 62.7 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.272
- **Signal Line:** 0.063
- **Histogram:** 0.210 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $84.46 (+3.7% from price)
- **200 SMA:** $75.25 (+16.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $87.24
- **Middle:** $83.93
- **Lower:** $80.61
- **Band Position:** 105%
- **Bandwidth:** 7.9% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $2.00 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $84.57 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.7
- **%D:** 60.0
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 17.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $84.06 (+4.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 36.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $83.94
- **Current vs VWAP:** +4.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $88.00
- **Recent Low:** $79.37
- **Next Resistance:** $88.00 (0.0% (High))
- **Next Support:** $85.96 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 63. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:29.178629
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'VTR'}`
- **Output Length:** 961 chars

### Output
```
# Insider Transactions for VTR
# Data retrieved on: 2026-04-29 14:49:29

## Summary
- **Insider Sentiment:** Neutral
- **Award/Grant:** 12 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,RODRIGUEZ JOE VASQUEZ JR.,Director,Award/Grant,41,3525,D
2026-04-16,BARNES MELODY C,Director,Award/Grant,66,5707,D
2026-04-16,MARTINO ROXANNE M.,Director,Award/Grant,188,16103,D
2026-04-16,SMITH MAURICE S,Director,Award/Grant,126,10779,D
2026-04-16,LUSTIG MATTHEW J,Director,Award/Grant,295,25309,D
2026-04-16,EMBLER MICHAEL J,Director,Award/Grant,66,5707,D
2026-04-16,NADER MARGUERITE M,Director,Award/Grant,66,5707,D
2026-04-16,ROY SUMIT,Director,Award/Grant,116,9958,D
2026-04-01,MARTINO ROXANNE M.,Director,Award/Grant,624,51250,D
2026-04-01,SMITH MAURICE S,Director,Award/Grant,457,37500,D
2026-04-01,LUSTIG MATTHEW J,Director,Award/Grant,396,32500,D
2026-04-01,ROY SUMIT,Director,Award/Grant,381,31250,D

```

---

## 64. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:29.306425
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'VTR'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for VTR

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 84.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 84.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 84.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 84.0% Bullish, 0.0% Bearish


```

---

## 65. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:29.570260
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'VTR', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 420 chars

### Output
```
## Options Activity for VTR

**Available Expirations:** 6 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 201 | 155 | 0.771 |
| Open Interest | 3,524 | 741 | 0.21 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

*No unusual options activity detected.*

```

---

## 66. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:31.799018
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ABNB', 'curr_date': '2026-04-29'}`
- **Output Length:** 1519 chars

### Output
```
# Technical Analysis for ABNB
**Date:** 2026-04-29
**Current Price:** $139.04

## Price Action
- **Daily Change:** -1.43%
- **5-Day Change:** -3.56%

## RSI (14)
- **Value:** 56.6 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 3.434
- **Signal Line:** 2.896
- **Histogram:** 0.537 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $131.92 (+5.4% from price)
- **200 SMA:** $128.30 (+8.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $149.07
- **Middle:** $134.70
- **Lower:** $120.33
- **Band Position:** 65%
- **Bandwidth:** 21.3% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.37 (3.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $132.48 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 54.2
- **%D:** 70.6
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 21.2 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $136.80 (+1.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 65.6M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $134.57
- **Current vs VWAP:** +3.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $147.25
- **Recent Low:** $119.94
- **Next Resistance:** $140.80 (23.6%)
- **Next Support:** $136.82 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 67. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:32.002624
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ABNB'}`
- **Output Length:** 1189 chars

### Output
```
# Insider Transactions for ABNB
# Data retrieved on: 2026-04-29 14:49:32

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 4 transactions
- **Award/Grant:** 2 transactions
- **Unknown:** 1 transactions
- **Total Sales Value:** $21,259,851
- **Largest Sale (3mo):** 2026-04-20 | GEBBIA JOSEPH (Director and Beneficial Owner of more than 10% of a Class of Security) | 58000 shares | $8270238.0 | Ownership: I

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-23,MERTZ ELINOR,Chief Financial Officer,Award/Grant,72806,0.0,D
2026-04-23,BLECHARCZYK NATHAN,"Officer, Director and Beneficial Owner",Award/Grant,39204,0.0,D
2026-04-22,BLECHARCZYK NATHAN,"Officer, Director and Beneficial Owner",Sale,35565,5180945.0,I
2026-04-22,BLECHARCZYK NATHAN,"Officer, Director and Beneficial Owner",Unknown,35565,,I
2026-04-20,GEBBIA JOSEPH,Director and Beneficial Owner of more than 10% of a Class of Security,Sale,58000,8270238.0,I
2026-04-08,MERTZ ELINOR,Chief Financial Officer,Sale,3750,491212.0,D
2026-04-06,GEBBIA JOSEPH,Director and Beneficial Owner of more than 10% of a Class of Security,Sale,58000,7317456.0,I

```

---

## 68. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:32.133920
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ABNB'}`
- **Output Length:** 720 chars

### Output
```
## Analyst Recommendation Trends for ABNB

### 2026-04-01
- **Strong Buy**: 8
- **Buy**: 16
- **Hold**: 22
- **Sell**: 2
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 50.0% Bullish, 4.2% Bearish

### 2026-03-01
- **Strong Buy**: 8
- **Buy**: 16
- **Hold**: 21
- **Sell**: 3
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 50.0% Bullish, 6.2% Bearish

### 2026-02-01
- **Strong Buy**: 8
- **Buy**: 13
- **Hold**: 24
- **Sell**: 3
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 43.8% Bullish, 6.2% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 13
- **Hold**: 21
- **Sell**: 6
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 43.8% Bullish, 12.5% Bearish


```

---

## 69. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:32.484041
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ABNB', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 692 chars

### Output
```
## Options Activity for ABNB

**Available Expirations:** 16 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 1,424 | 981 | 0.689 |
| Open Interest | 28,706 | 17,198 | 0.599 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $141.0 | 112 | 99 | 1.13x | 52.2% |
| CALL | 2026-05-08 | $129.0 | 602 | 776 | 0.78x | 60.7% |

```

---

## 70. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:34.085181
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ADI', 'curr_date': '2026-04-29'}`
- **Output Length:** 1529 chars

### Output
```
# Technical Analysis for ADI
**Date:** 2026-04-29
**Current Price:** $383.26

## Price Action
- **Daily Change:** -2.38%
- **5-Day Change:** +0.48%

## RSI (14)
- **Value:** 64.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 18.115
- **Signal Line:** 14.610
- **Histogram:** 3.505 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $339.73 (+12.8% from price)
- **200 SMA:** $277.52 (+38.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $412.04
- **Middle:** $357.42
- **Lower:** $302.80
- **Band Position:** 74%
- **Bandwidth:** 30.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $11.31 (3.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $366.29 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 76.3
- **%D:** 83.9
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 39.7 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $363.74 (+5.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 74.1M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $359.77
- **Current vs VWAP:** +6.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $408.37
- **Recent Low:** $300.56
- **Next Resistance:** $408.37 (0.0% (High))
- **Next Support:** $382.93 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 71. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:34.315801
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ADI'}`
- **Output Length:** 1037 chars

### Output
```
# Insider Transactions for ADI
# Data retrieved on: 2026-04-29 14:49:34

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 6 transactions
- **Sale:** 1 transactions
- **Option Exercise:** 1 transactions
- **Total Sales Value:** $3,181,400
- **Largest Sale (3mo):** 2026-04-01 | ROCHE VINCENT T (Chief Executive Officer) | 10000 shares | $3181400.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-07,SONDEL MICHAEL,Officer,Award/Grant,1341,0.0,D
2026-04-07,PUCCIO RICHARD C. JR.,Chief Financial Officer,Award/Grant,6513,0.0,D
2026-04-07,NAKAMURA KATSUFUMI,Officer,Award/Grant,4085,0.0,D
2026-04-07,COTTER MARTIN,Officer,Award/Grant,5807,0.0,D
2026-04-07,JAIN VIVEK,Officer,Award/Grant,6734,0.0,D
2026-04-07,ROCHE VINCENT T,Chief Executive Officer,Award/Grant,19712,0.0,D
2026-04-01,ROCHE VINCENT T,Chief Executive Officer,Sale,10000,3181400.0,D
2026-04-01,ROCHE VINCENT T,Chief Executive Officer,Option Exercise,10000,944100.0,D

```

---

## 72. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:34.432041
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

## 73. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:34.736654
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ADI', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 984 chars

### Output
```
## Options Activity for ADI

**Available Expirations:** 13 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,765 | 1,797 | 0.65 |
| Open Interest | 11,864 | 11,270 | 0.95 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $347.5 | 172 | 82 | 2.1x | 76.6% |
| CALL | 2026-05-01 | $420.0 | 1,197 | 716 | 1.67x | 50.0% |
| PUT | 2026-05-01 | $377.5 | 630 | 377 | 1.67x | 53.7% |
| CALL | 2026-05-01 | $390.0 | 441 | 341 | 1.29x | 50.8% |
| PUT | 2026-05-01 | $385.0 | 177 | 158 | 1.12x | 50.9% |
| CALL | 2026-05-15 | $350.0 | 110 | 112 | 0.98x | 54.1% |
| CALL | 2026-05-01 | $360.0 | 130 | 148 | 0.88x | 72.1% |

```

---

## 74. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:37.114623
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ALAB', 'curr_date': '2026-04-29'}`
- **Output Length:** 1507 chars

### Output
```
# Technical Analysis for ALAB
**Date:** 2026-04-29
**Current Price:** $183.31

## Price Action
- **Daily Change:** -6.78%
- **5-Day Change:** -5.54%

## RSI (14)
- **Value:** 61.4 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 18.937
- **Signal Line:** 15.543
- **Histogram:** 3.394 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $136.55 (+34.2% from price)
- **200 SMA:** $161.24 (+13.7% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $227.19
- **Middle:** $159.03
- **Lower:** $90.86
- **Band Position:** 68%
- **Bandwidth:** 85.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $12.77 (7.0% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $164.16 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 70.3
- **%D:** 83.4
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 46.1 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $167.93 (+9.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 72.1M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $162.03
- **Current vs VWAP:** +13.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $214.24
- **Recent Low:** $97.89
- **Next Resistance:** $186.78 (23.6%)
- **Next Support:** $169.79 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 75. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:37.340332
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ALAB'}`
- **Output Length:** 823 chars

### Output
```
# Insider Transactions for ALAB
# Data retrieved on: 2026-04-29 14:49:37

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 6 transactions
- **Total Sales Value:** $102,040,584
- **Largest Sale (3mo):** 2026-04-20 | MOHAN JITENDRA (Chief Executive Officer) | 335881 shares | $57852785 | Ownership: I

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-20,DYCKERHOFF STEFAN A,Director,Sale,24998,4376258,D/I
2026-04-20,MOHAN JITENDRA,Chief Executive Officer,Sale,335881,57852785,I
2026-04-15,MOHAN JITENDRA,Chief Executive Officer,Sale,2967,518036,I
2026-04-10,MOHAN JITENDRA,Chief Executive Officer,Sale,123750,18522445,I
2026-04-01,MAZZARA PHILIP,General Counsel,Sale,10000,1077663,D
2026-04-01,ALBA MANUEL,Director,Sale,183000,19693397,I

```

---

## 76. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:37.489052
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ALAB'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for ALAB

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 79.3% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 15
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 28

**Sentiment**: 78.6% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 15
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 28

**Sentiment**: 78.6% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 27

**Sentiment**: 77.8% Bullish, 0.0% Bearish


```

---

## 77. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:37.812192
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ALAB', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1107 chars

### Output
```
## Options Activity for ALAB

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 4,165 | 2,618 | 0.629 |
| Open Interest | 41,468 | 33,761 | 0.814 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $98.0 | 200 | 3 | 66.67x | 168.8% |
| CALL | 2026-05-01 | $195.0 | 695 | 333 | 2.09x | 110.4% |
| CALL | 2026-05-15 | $80.0 | 120 | 61 | 1.97x | 212.0% |
| PUT | 2026-05-15 | $290.0 | 187 | 139 | 1.35x | 136.6% |
| PUT | 2026-05-08 | $202.5 | 118 | 119 | 0.99x | 142.1% |
| CALL | 2026-05-08 | $185.0 | 332 | 404 | 0.82x | 128.4% |
| PUT | 2026-05-08 | $167.5 | 101 | 128 | 0.79x | 138.0% |
| CALL | 2026-05-01 | $190.0 | 315 | 448 | 0.7x | 109.3% |
| CALL | 2026-05-15 | $270.0 | 269 | 446 | 0.6x | 118.6% |

```

---

## 78. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:40.536866
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ALGN', 'curr_date': '2026-04-29'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for ALGN
**Date:** 2026-04-29
**Current Price:** $177.28

## Price Action
- **Daily Change:** -4.02%
- **5-Day Change:** -9.57%

## RSI (14)
- **Value:** 45.1 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 3.052
- **Signal Line:** 3.289
- **Histogram:** -0.237 (Weakening ↓)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $179.87 (-1.4% from price)
- **200 SMA:** $158.78 (+11.7% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $199.55
- **Middle:** $181.51
- **Lower:** $163.46
- **Band Position:** 38%
- **Bandwidth:** 19.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $7.54 (4.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $165.97 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 33.6
- **%D:** 54.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 14.6 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $183.37 (-3.3% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 6.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $182.55
- **Current vs VWAP:** -2.9%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $200.44
- **Recent Low:** $162.27
- **Next Resistance:** $181.36 (50.0%)
- **Next Support:** $176.85 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 79. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:40.748256
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ALGN'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for ALGN in the last 3 months.
```

---

## 80. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:40.850358
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ALGN'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for ALGN

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 9
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 69.6% Bullish, 4.3% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 9
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 66.7% Bullish, 4.2% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 8
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 62.5% Bullish, 4.2% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 4.3% Bearish


```

---

## 81. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:41.080987
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ALGN', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 689 chars

### Output
```
## Options Activity for ALGN

**Available Expirations:** 14 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 363 | 1,024 | 2.821 |
| Open Interest | 1,482 | 2,895 | 1.953 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $157.5 | 341 | 175 | 1.95x | 150.5% |
| PUT | 2026-05-15 | $160.0 | 122 | 126 | 0.97x | 76.0% |

```

---

## 82. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:42.956991
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ALSN', 'curr_date': '2026-04-29'}`
- **Output Length:** 1526 chars

### Output
```
# Technical Analysis for ALSN
**Date:** 2026-04-29
**Current Price:** $130.02

## Price Action
- **Daily Change:** -2.75%
- **5-Day Change:** -2.27%

## RSI (14)
- **Value:** 58.2 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 4.161
- **Signal Line:** 3.812
- **Histogram:** 0.349 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $121.23 (+7.3% from price)
- **200 SMA:** $99.09 (+31.2% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $139.27
- **Middle:** $126.80
- **Lower:** $114.32
- **Band Position:** 63%
- **Bandwidth:** 19.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $3.56 (2.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $124.67 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 69.9
- **%D:** 80.6
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 31.4 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $127.66 (+1.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 5.7M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $127.77
- **Current vs VWAP:** +1.8%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $137.62
- **Recent Low:** $109.42
- **Next Resistance:** $130.96 (23.6%)
- **Next Support:** $126.85 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 83. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:43.161742
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ALSN'}`
- **Output Length:** 301 chars

### Output
```
# Insider Transactions for ALSN
# Data retrieved on: 2026-04-29 14:49:43

## Summary
- **Insider Sentiment:** Neutral
- **Unknown:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-14,MELL SCOTT A.,Chief Financial Officer,Unknown,2865,,D

```

---

## 84. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:43.283122
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ALSN'}`
- **Output Length:** 711 chars

### Output
```
## Analyst Recommendation Trends for ALSN

### 2026-04-01
- **Strong Buy**: 4
- **Buy**: 5
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 52.9% Bullish, 5.9% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 5
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 52.9% Bullish, 5.9% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 5
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 52.9% Bullish, 5.9% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 5
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 52.9% Bullish, 5.9% Bearish


```

---

## 85. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:43.550161
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ALSN', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 436 chars

### Output
```
## Options Activity for ALSN

**Available Expirations:** 5 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 69 | 162 | 2.348 |
| Open Interest | 474 | 488 | 1.03 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

*No unusual options activity detected.*

```

---

## 86. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:45.211463
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMAT', 'curr_date': '2026-04-29'}`
- **Output Length:** 1561 chars

### Output
```
# Technical Analysis for AMAT
**Date:** 2026-04-29
**Current Price:** $381.11

## Price Action
- **Daily Change:** -5.87%
- **5-Day Change:** -5.54%

## RSI (14)
- **Value:** 49.9 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 11.852
- **Signal Line:** 12.378
- **Histogram:** -0.526 (Weakening ↓)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $368.07 (+3.5% from price)
- **200 SMA:** $265.99 (+43.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $429.36
- **Middle:** $385.14
- **Lower:** $340.93
- **Band Position:** 45%
- **Bandwidth:** 23.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $15.52 (4.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $357.83 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 50.9
- **%D:** 65.8
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 17.8 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $386.98 (-1.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 207.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $384.37
- **Current vs VWAP:** -0.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $420.50
- **Recent Low:** $316.66
- **Next Resistance:** $395.99 (23.6%)
- **Next Support:** $380.83 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 87. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:45.491695
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMAT'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for AMAT in the last 3 months.
```

---

## 88. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:45.592744
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMAT'}`
- **Output Length:** 721 chars

### Output
```
## Analyst Recommendation Trends for AMAT

### 2026-04-01
- **Strong Buy**: 11
- **Buy**: 24
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 81.4% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 11
- **Buy**: 24
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 81.4% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 10
- **Buy**: 23
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 43

**Sentiment**: 76.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 10
- **Buy**: 17
- **Hold**: 13
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 67.5% Bullish, 0.0% Bearish


```

---

## 89. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:45.928340
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMAT', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1158 chars

### Output
```
## Options Activity for AMAT

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,054 | 7,841 | 2.567 |
| Open Interest | 50,207 | 56,953 | 1.134 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $240.0 | 106 | 26 | 4.08x | 126.8% |
| PUT | 2026-05-08 | $210.0 | 238 | 59 | 4.03x | 144.7% |
| PUT | 2026-05-08 | $230.0 | 138 | 39 | 3.54x | 102.3% |
| PUT | 2026-05-08 | $220.0 | 254 | 76 | 3.34x | 114.8% |
| PUT | 2026-05-08 | $235.0 | 104 | 35 | 2.97x | 102.3% |
| PUT | 2026-05-08 | $250.0 | 143 | 75 | 1.91x | 117.9% |
| PUT | 2026-05-15 | $135.0 | 254 | 252 | 1.01x | 244.4% |
| PUT | 2026-05-08 | $215.0 | 188 | 189 | 0.99x | 136.9% |
| PUT | 2026-05-08 | $345.0 | 200 | 211 | 0.95x | 64.8% |
| PUT | 2026-05-01 | $195.0 | 500 | 557 | 0.9x | 218.8% |

```

---

## 90. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:47.655554
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMD', 'curr_date': '2026-04-29'}`
- **Output Length:** 1562 chars

### Output
```
# Technical Analysis for AMD
**Date:** 2026-04-29
**Current Price:** $323.21

## Price Action
- **Daily Change:** -3.41%
- **5-Day Change:** +6.51%

## RSI (14)
- **Value:** 73.5 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 30.229
- **Signal Line:** 22.750
- **Histogram:** 7.480 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $226.47 (+42.7% from price)
- **200 SMA:** $208.43 (+55.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $349.36
- **Middle:** $263.84
- **Lower:** $178.33
- **Band Position:** 85%
- **Bandwidth:** 64.8% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $15.03 (4.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $300.67 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 82.5
- **%D:** 89.1
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 65.8 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $275.16 (+17.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1126.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $271.09
- **Current vs VWAP:** +19.2%
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

## 91. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:47.810309
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMD'}`
- **Output Length:** 789 chars

### Output
```
# Insider Transactions for AMD
# Data retrieved on: 2026-04-29 14:49:47

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 3 transactions
- **Option Exercise:** 1 transactions
- **Total Sales Value:** $20,691,140
- **Largest Sale (3mo):** 2026-04-24 | PAPERMASTER MARK D (Chief Technology Officer) | 31320 shares | $10962000.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-24,PAPERMASTER MARK D,Chief Technology Officer,Sale,31320,10962000.0,D
2026-04-16,PAPERMASTER MARK D,Chief Technology Officer,Sale,33109,8988215.0,D
2026-04-15,PAPERMASTER MARK D,Chief Technology Officer,Option Exercise,6000,509100.0,D
2026-04-06,PAPERMASTER MARK D,Chief Technology Officer,Sale,3293,740925.0,D

```

---

## 92. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:47.912662
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

## 93. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:48.322948
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMD', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1193 chars

### Output
```
## Options Activity for AMD

**Available Expirations:** 21 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 65,899 | 43,234 | 0.656 |
| Open Interest | 420,958 | 430,895 | 1.024 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $320.0 | 5,406 | 2,269 | 2.38x | 60.9% |
| CALL | 2026-05-01 | $322.5 | 2,567 | 1,121 | 2.29x | 44.3% |
| CALL | 2026-05-01 | $335.0 | 7,204 | 3,174 | 2.27x | 54.3% |
| PUT | 2026-05-15 | $342.5 | 124 | 62 | 2.0x | 82.2% |
| CALL | 2026-05-01 | $490.0 | 212 | 128 | 1.66x | 137.5% |
| CALL | 2026-05-01 | $330.0 | 4,926 | 3,055 | 1.61x | 51.6% |
| CALL | 2026-05-01 | $325.0 | 4,838 | 3,641 | 1.33x | 47.8% |
| PUT | 2026-05-08 | $262.5 | 176 | 141 | 1.25x | 95.2% |
| CALL | 2026-05-01 | $340.0 | 3,082 | 2,747 | 1.12x | 57.5% |
| CALL | 2026-05-01 | $332.5 | 1,061 | 959 | 1.11x | 53.7% |

```

---

## 94. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:50.036907
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMGN', 'curr_date': '2026-04-29'}`
- **Output Length:** 1552 chars

### Output
```
# Technical Analysis for AMGN
**Date:** 2026-04-29
**Current Price:** $339.57

## Price Action
- **Daily Change:** -0.18%
- **5-Day Change:** -1.84%

## RSI (14)
- **Value:** 38.7 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -3.672
- **Signal Line:** -3.187
- **Histogram:** -0.485 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $360.00 (-5.7% from price)
- **200 SMA:** $321.33 (+5.7% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $357.57
- **Middle:** $347.98
- **Lower:** $338.39
- **Band Position:** 6%
- **Bandwidth:** 5.5% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $7.56 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $328.23 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 20.8
- **%D:** 33.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 15.6 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $348.27 (-2.5% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 48.9M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $348.26
- **Current vs VWAP:** -2.5%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $391.29
- **Recent Low:** $335.06
- **Next Resistance:** $347.09 (78.6%)
- **Next Support:** $335.06 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 95. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:50.296270
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMGN'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for AMGN in the last 3 months.
```

---

## 96. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:50.402443
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMGN'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for AMGN

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 19
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 43

**Sentiment**: 48.8% Bullish, 7.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 18
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 42

**Sentiment**: 50.0% Bullish, 7.1% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 16
- **Hold**: 16
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 41

**Sentiment**: 53.7% Bullish, 7.3% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 15
- **Hold**: 16
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 41

**Sentiment**: 53.7% Bullish, 7.3% Bearish


```

---

## 97. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:50.664319
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMGN', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 934 chars

### Output
```
## Options Activity for AMGN

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 7,246 | 691 | 0.095 |
| Open Interest | 14,154 | 5,636 | 0.398 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $295.0 | 3,330 | 2,031 | 1.64x | 52.3% |
| CALL | 2026-05-15 | $305.0 | 2,467 | 1,613 | 1.53x | 54.2% |
| CALL | 2026-05-08 | $385.0 | 247 | 241 | 1.02x | 62.7% |
| CALL | 2026-05-01 | $310.0 | 139 | 139 | 1.0x | 80.9% |
| CALL | 2026-05-01 | $400.0 | 309 | 331 | 0.93x | 84.0% |
| PUT | 2026-05-15 | $265.0 | 107 | 137 | 0.78x | 53.0% |

```

---

## 98. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:55.040202
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'BMI', 'curr_date': '2026-04-29'}`
- **Output Length:** 1531 chars

### Output
```
# Technical Analysis for BMI
**Date:** 2026-04-29
**Current Price:** $118.42

## Price Action
- **Daily Change:** -3.69%
- **5-Day Change:** -4.16%

## RSI (14)
- **Value:** 31.3 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -8.803
- **Signal Line:** -6.082
- **Histogram:** -2.721 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $147.39 (-19.7% from price)
- **200 SMA:** $171.58 (-31.0% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $177.77
- **Middle:** $141.34
- **Lower:** $104.91
- **Band Position:** 19%
- **Bandwidth:** 51.6% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $6.61 (5.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $108.50 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 18.8
- **%D:** 22.9
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 39.3 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $135.54 (-12.6% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -14.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $128.05
- **Current vs VWAP:** -7.5%
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

## 99. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:55.223893
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'BMI'}`
- **Output Length:** 747 chars

### Output
```
# Insider Transactions for BMI
# Data retrieved on: 2026-04-29 14:49:55

## Summary
- **Insider Sentiment:** BULLISH ⚡ (more buying than selling)
- **Purchase:** 5 transactions
- **Total Purchases Value:** $777,135
- **Largest Purchase (3mo):** 2026-04-21 | BOCKHORST KENNETH C (Chief Executive Officer) | 2200 shares | $258573 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-23,WROCKLAGE ROBERT A.,Officer,Purchase,1000,122350,D
2026-04-21,BOCKHORST KENNETH C,Chief Executive Officer,Purchase,2200,258573,D
2026-04-21,HTWE RICHARD,Officer,Purchase,1700,197421,D
2026-04-21,TARANTINO CHRISTINA M.,Officer,Purchase,870,99006,D
2026-04-21,CALLAHAN EDWARD F,Officer,Purchase,858,99785,D

```

---

## 100. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:55.320838
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

## 101. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:55.652679
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'BMI', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 423 chars

### Output
```
## Options Activity for BMI

**Available Expirations:** 4 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 231 | 278 | 1.203 |
| Open Interest | 4,560 | 1,148 | 0.252 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

*No unusual options activity detected.*

```

---

## 102. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:58.423659
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'V', 'curr_date': '2026-04-29'}`
- **Output Length:** 1527 chars

### Output
```
# Technical Analysis for V
**Date:** 2026-04-29
**Current Price:** $309.30

## Price Action
- **Daily Change:** -0.11%
- **5-Day Change:** -0.64%

## RSI (14)
- **Value:** 49.5 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 0.444
- **Signal Line:** 0.261
- **Histogram:** 0.184 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $309.96 (-0.2% from price)
- **200 SMA:** $332.11 (-6.9% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $318.78
- **Middle:** $308.51
- **Lower:** $298.25
- **Band Position:** 54%
- **Bandwidth:** 6.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $5.85 (1.9% of price)
- **Level:** Low volatility
- **Suggested Stop-Loss:** $300.53 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 39.5
- **%D:** 49.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 10.8 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $309.49 (-0.1% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -68.7M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $308.22
- **Current vs VWAP:** +0.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $325.77
- **Recent Low:** $293.89
- **Next Resistance:** $309.83 (50.0%)
- **Next Support:** $306.07 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 103. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:58.617667
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'V'}`
- **Output Length:** 57 chars

### Output
```
No insider transactions found for V in the last 3 months.
```

---

## 104. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:58.723764
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'V'}`
- **Output Length:** 716 chars

### Output
```
## Analyst Recommendation Trends for V

### 2026-04-01
- **Strong Buy**: 11
- **Buy**: 31
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 46

**Sentiment**: 91.3% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 13
- **Buy**: 31
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 91.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 13
- **Buy**: 29
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 87.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 13
- **Buy**: 28
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 47

**Sentiment**: 87.2% Bullish, 0.0% Bearish


```

---

## 105. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:49:59.061447
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'V', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1156 chars

### Output
```
## Options Activity for V

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 22,535 | 13,449 | 0.597 |
| Open Interest | 73,334 | 47,910 | 0.653 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $340.0 | 237 | 1 | 237.0x | 48.4% |
| PUT | 2026-05-15 | $327.5 | 113 | 1 | 113.0x | 28.9% |
| PUT | 2026-05-01 | $335.0 | 656 | 7 | 93.71x | 42.5% |
| PUT | 2026-05-01 | $325.0 | 997 | 42 | 23.74x | 40.2% |
| PUT | 2026-05-01 | $330.0 | 726 | 33 | 22.0x | 41.1% |
| CALL | 2026-05-01 | $342.5 | 2,892 | 184 | 15.72x | 28.2% |
| CALL | 2026-05-08 | $360.0 | 147 | 12 | 12.25x | 25.2% |
| CALL | 2026-05-01 | $360.0 | 863 | 108 | 7.99x | 37.3% |
| PUT | 2026-05-01 | $317.5 | 230 | 35 | 6.57x | 46.1% |
| PUT | 2026-05-01 | $327.5 | 206 | 40 | 5.15x | 41.1% |

```

---

## 106. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:00.700045
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MSFT', 'curr_date': '2026-04-29'}`
- **Output Length:** 1510 chars

### Output
```
# Technical Analysis for MSFT
**Date:** 2026-04-29
**Current Price:** $429.25

## Price Action
- **Daily Change:** +1.04%
- **5-Day Change:** -0.85%

## RSI (14)
- **Value:** 65.4 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 10.819
- **Signal Line:** 6.778
- **Histogram:** 4.041 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $395.04 (+8.7% from price)
- **200 SMA:** $467.93 (-8.3% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $448.96
- **Middle:** $398.89
- **Lower:** $348.81
- **Band Position:** 80%
- **Bandwidth:** 25.1% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $10.29 (2.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $413.82 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.4
- **%D:** 82.2
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 31.9 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $407.39 (+5.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -218.5M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $399.50
- **Current vs VWAP:** +7.4%
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

## 107. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:00.890080
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MSFT'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for MSFT in the last 3 months.
```

---

## 108. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:00.990829
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MSFT'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for MSFT

### 2026-04-01
- **Strong Buy**: 23
- **Buy**: 36
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 65

**Sentiment**: 90.8% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 24
- **Buy**: 36
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 66

**Sentiment**: 90.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 25
- **Buy**: 37
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 66

**Sentiment**: 93.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 25
- **Buy**: 37
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 67

**Sentiment**: 92.5% Bullish, 0.0% Bearish


```

---

## 109. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:01.335239
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MSFT', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1152 chars

### Output
```
## Options Activity for MSFT

**Available Expirations:** 22 dates
**Analyzing:** 2026-05-01, 2026-05-04, 2026-05-06

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 30,080 | 21,484 | 0.714 |
| Open Interest | 286,260 | 124,408 | 0.435 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $377.5 | 226 | 49 | 4.61x | 87.7% |
| CALL | 2026-05-01 | $407.5 | 155 | 83 | 1.87x | 91.4% |
| CALL | 2026-05-06 | $425.0 | 207 | 173 | 1.2x | 63.8% |
| PUT | 2026-05-04 | $417.5 | 244 | 262 | 0.93x | 72.2% |
| PUT | 2026-05-01 | $460.0 | 431 | 513 | 0.84x | 105.4% |
| PUT | 2026-05-01 | $470.0 | 410 | 565 | 0.73x | 112.7% |
| CALL | 2026-05-01 | $565.0 | 119 | 168 | 0.71x | 110.9% |
| CALL | 2026-05-01 | $580.0 | 141 | 210 | 0.67x | 127.0% |
| PUT | 2026-05-01 | $397.5 | 487 | 818 | 0.6x | 95.0% |
| PUT | 2026-05-01 | $345.0 | 1,598 | 2,786 | 0.57x | 107.9% |

```

---

## 110. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:03.177269
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'GOOGL', 'curr_date': '2026-04-29'}`
- **Output Length:** 1552 chars

### Output
```
# Technical Analysis for GOOGL
**Date:** 2026-04-29
**Current Price:** $349.78

## Price Action
- **Daily Change:** -0.16%
- **5-Day Change:** +3.08%

## RSI (14)
- **Value:** 71.1 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 11.536
- **Signal Line:** 9.089
- **Histogram:** 2.447 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $311.21 (+12.4% from price)
- **200 SMA:** $277.99 (+25.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $363.55
- **Middle:** $325.03
- **Lower:** $286.52
- **Band Position:** 82%
- **Bandwidth:** 23.7% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $7.96 (2.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $337.84 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 87.3
- **%D:** 87.9
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 39.0 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $329.14 (+6.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 837.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $322.31
- **Current vs VWAP:** +8.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $353.18
- **Recent Low:** $272.11
- **Next Resistance:** $353.18 (0.0% (High))
- **Next Support:** $334.05 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 111. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:03.363930
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'GOOGL'}`
- **Output Length:** 61 chars

### Output
```
No insider transactions found for GOOGL in the last 3 months.
```

---

## 112. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:03.469104
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'GOOGL'}`
- **Output Length:** 722 chars

### Output
```
## Analyst Recommendation Trends for GOOGL

### 2026-04-01
- **Strong Buy**: 19
- **Buy**: 41
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 68

**Sentiment**: 88.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 19
- **Buy**: 41
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 69

**Sentiment**: 87.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 19
- **Buy**: 40
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 69

**Sentiment**: 85.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 19
- **Buy**: 41
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 70

**Sentiment**: 85.7% Bullish, 0.0% Bearish


```

---

## 113. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:03.824287
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'GOOGL', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1173 chars

### Output
```
## Options Activity for GOOGL

**Available Expirations:** 23 dates
**Analyzing:** 2026-05-01, 2026-05-04, 2026-05-06

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 35,916 | 13,664 | 0.38 |
| Open Interest | 126,459 | 77,121 | 0.61 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-06 | $302.5 | 146 | 15 | 9.73x | 63.4% |
| CALL | 2026-05-06 | $300.0 | 147 | 16 | 9.19x | 59.9% |
| PUT | 2026-05-06 | $322.5 | 469 | 235 | 2.0x | 54.6% |
| CALL | 2026-05-01 | $420.0 | 1,304 | 871 | 1.5x | 82.6% |
| CALL | 2026-05-01 | $370.0 | 12,163 | 8,699 | 1.4x | 73.3% |
| CALL | 2026-05-04 | $330.0 | 224 | 178 | 1.26x | 48.1% |
| CALL | 2026-05-01 | $362.5 | 836 | 727 | 1.15x | 71.2% |
| PUT | 2026-05-01 | $170.0 | 168 | 150 | 1.12x | 428.0% |
| PUT | 2026-05-01 | $210.0 | 371 | 383 | 0.97x | 189.1% |
| PUT | 2026-05-01 | $320.0 | 1,572 | 2,112 | 0.74x | 80.0% |

```

---

## 114. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:05.530641
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ADP', 'curr_date': '2026-04-29'}`
- **Output Length:** 1528 chars

### Output
```
# Technical Analysis for ADP
**Date:** 2026-04-29
**Current Price:** $199.17

## Price Action
- **Daily Change:** +0.98%
- **5-Day Change:** -1.25%

## RSI (14)
- **Value:** 45.2 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.594
- **Signal Line:** -3.226
- **Histogram:** 0.632 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $206.34 (-3.5% from price)
- **200 SMA:** $254.93 (-21.9% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $207.25
- **Middle:** $199.43
- **Lower:** $191.62
- **Band Position:** 48%
- **Bandwidth:** 7.8% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.94 (2.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $191.76 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 40.5
- **%D:** 48.7
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 21.2 (Weakening ↓)
- **Interpretation:** Trending

## 20 EMA
- **Value:** $200.48 (-0.7% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -44.6M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $199.24
- **Current vs VWAP:** -0.0%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $224.64
- **Recent Low:** $188.16
- **Next Resistance:** $202.10 (61.8%)
- **Next Support:** $195.97 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 115. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:05.794151
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ADP'}`
- **Output Length:** 469 chars

### Output
```
# Insider Transactions for ADP
# Data retrieved on: 2026-04-29 14:50:05

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Total Sales Value:** $106,287
- **Largest Sale (3mo):** 2026-04-14 | D'AMBROSIO CHRISTOPHER (Officer) | 543 shares | $106287 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-14,D'AMBROSIO CHRISTOPHER,Officer,Sale,543,106287,D

```

---

## 116. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:05.899659
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ADP'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for ADP

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 5
- **Hold**: 12
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 23

**Sentiment**: 34.8% Bullish, 13.0% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 5
- **Hold**: 12
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 22

**Sentiment**: 31.8% Bullish, 13.6% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 5
- **Hold**: 12
- **Sell**: 2
- **Strong Sell**: 1
- **Total Analysts**: 22

**Sentiment**: 31.8% Bullish, 13.6% Bearish

### 2026-01-01
- **Strong Buy**: 1
- **Buy**: 2
- **Hold**: 13
- **Sell**: 6
- **Strong Sell**: 2
- **Total Analysts**: 24

**Sentiment**: 12.5% Bullish, 33.3% Bearish


```

---

## 117. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:06.166227
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ADP', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 629 chars

### Output
```
## Options Activity for ADP

**Available Expirations:** 14 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 875 | 466 | 0.533 |
| Open Interest | 8,139 | 4,894 | 0.601 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $205.0 | 135 | 151 | 0.89x | 67.5% |

```

---

## 118. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:08.813628
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NVDA', 'curr_date': '2026-04-29'}`
- **Output Length:** 1552 chars

### Output
```
# Technical Analysis for NVDA
**Date:** 2026-04-29
**Current Price:** $213.17

## Price Action
- **Daily Change:** -1.59%
- **5-Day Change:** +5.27%

## RSI (14)
- **Value:** 71.0 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 8.039
- **Signal Line:** 5.749
- **Histogram:** 2.291 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $186.22 (+14.5% from price)
- **200 SMA:** $183.34 (+16.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $219.15
- **Middle:** $193.24
- **Lower:** $167.32
- **Band Position:** 88%
- **Bandwidth:** 26.8% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $5.93 (2.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $204.28 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 88.8
- **%D:** 90.4
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 37.2 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $196.80 (+8.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1650.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $192.51
- **Current vs VWAP:** +10.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $216.83
- **Recent Low:** $164.27
- **Next Resistance:** $216.83 (0.0% (High))
- **Next Support:** $204.43 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 119. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:08.976483
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NVDA'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for NVDA in the last 3 months.
```

---

## 120. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:09.080345
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'NVDA'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for NVDA

### 2026-04-01
- **Strong Buy**: 24
- **Buy**: 42
- **Hold**: 4
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 71

**Sentiment**: 93.0% Bullish, 1.4% Bearish

### 2026-03-01
- **Strong Buy**: 25
- **Buy**: 42
- **Hold**: 5
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 73

**Sentiment**: 91.8% Bullish, 1.4% Bearish

### 2026-02-01
- **Strong Buy**: 25
- **Buy**: 40
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 72

**Sentiment**: 90.3% Bullish, 1.4% Bearish

### 2026-01-01
- **Strong Buy**: 25
- **Buy**: 39
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 72

**Sentiment**: 88.9% Bullish, 1.4% Bearish


```

---

## 121. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:09.369784
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NVDA', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1186 chars

### Output
```
## Options Activity for NVDA

**Available Expirations:** 24 dates
**Analyzing:** 2026-04-29, 2026-05-01, 2026-05-04

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 339,479 | 266,826 | 0.786 |
| Open Interest | 864,055 | 651,910 | 0.754 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-04 | $167.5 | 1,260 | 17 | 74.12x | 75.0% |
| CALL | 2026-04-29 | $212.5 | 60,237 | 16,972 | 3.55x | 42.0% |
| PUT | 2026-04-29 | $207.5 | 24,556 | 8,738 | 2.81x | 43.4% |
| PUT | 2026-05-04 | $152.5 | 1,618 | 590 | 2.74x | 94.5% |
| PUT | 2026-04-29 | $210.0 | 55,620 | 20,536 | 2.71x | 41.7% |
| PUT | 2026-05-04 | $195.0 | 2,138 | 896 | 2.39x | 44.6% |
| PUT | 2026-04-29 | $205.0 | 26,745 | 11,787 | 2.27x | 46.1% |
| PUT | 2026-04-29 | $167.5 | 541 | 262 | 2.06x | 143.8% |
| CALL | 2026-04-29 | $215.0 | 50,814 | 26,850 | 1.89x | 42.7% |
| PUT | 2026-04-29 | $212.5 | 8,244 | 4,482 | 1.84x | 41.3% |

```

---

## 122. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:11.507585
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'NFLX', 'curr_date': '2026-04-29'}`
- **Output Length:** 1529 chars

### Output
```
# Technical Analysis for NFLX
**Date:** 2026-04-29
**Current Price:** $92.27

## Price Action
- **Daily Change:** +0.98%
- **5-Day Change:** -1.04%

## RSI (14)
- **Value:** 41.3 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.590
- **Signal Line:** 0.860
- **Histogram:** -1.450 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $93.76 (-1.6% from price)
- **200 SMA:** $104.66 (-11.8% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $108.84
- **Middle:** $98.22
- **Lower:** $87.59
- **Band Position:** 22%
- **Bandwidth:** 21.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.95 (3.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $87.84 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 9.6
- **%D:** 17.7
- **Signal:** OVERSOLD ⚡

## ADX (Trend Strength)
- **ADX:** 13.4 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $96.03 (-3.9% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -840.5M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $98.01
- **Current vs VWAP:** -5.9%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $108.95
- **Recent Low:** $75.01
- **Next Resistance:** $95.98 (38.2%)
- **Next Support:** $91.98 (50.0%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 123. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:11.671088
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'NFLX'}`
- **Output Length:** 747 chars

### Output
```
# Insider Transactions for NFLX
# Data retrieved on: 2026-04-29 14:50:11

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

## 124. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:11.776935
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

## 125. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:12.118274
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'NFLX', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1163 chars

### Output
```
## Options Activity for NFLX

**Available Expirations:** 20 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 44,727 | 31,940 | 0.714 |
| Open Interest | 754,837 | 343,378 | 0.455 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-08 | $115.0 | 640 | 18 | 35.56x | 101.9% |
| PUT | 2026-05-15 | $111.0 | 220 | 19 | 11.58x | 70.3% |
| CALL | 2026-05-08 | $145.0 | 154 | 113 | 1.36x | 100.8% |
| PUT | 2026-05-08 | $50.0 | 120 | 100 | 1.2x | 135.9% |
| CALL | 2026-05-15 | $65.0 | 158 | 132 | 1.2x | 103.8% |
| PUT | 2026-05-01 | $82.0 | 1,183 | 1,639 | 0.72x | 53.9% |
| CALL | 2026-05-08 | $120.0 | 2,029 | 2,831 | 0.72x | 65.6% |
| CALL | 2026-05-01 | $92.0 | 6,235 | 9,138 | 0.68x | 25.0% |
| CALL | 2026-05-01 | $91.0 | 2,714 | 4,883 | 0.56x | 18.6% |
| CALL | 2026-05-15 | $1500.0 | 105 | 196 | 0.54x | 952.5% |

```

---

## 126. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:15.527063
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'GTLB', 'curr_date': '2026-04-29'}`
- **Output Length:** 1545 chars

### Output
```
# Technical Analysis for GTLB
**Date:** 2026-04-29
**Current Price:** $22.44

## Price Action
- **Daily Change:** +4.62%
- **5-Day Change:** +0.36%

## RSI (14)
- **Value:** 52.1 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.418
- **Signal Line:** -0.713
- **Histogram:** 0.296 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $23.11 (-2.9% from price)
- **200 SMA:** $37.62 (-40.4% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $23.57
- **Middle:** $21.49
- **Lower:** $19.41
- **Band Position:** 73%
- **Bandwidth:** 19.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.30 (5.8% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $20.49 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 56.7
- **%D:** 53.3
- **Signal:** Bullish crossover ⚡

## ADX (Trend Strength)
- **ADX:** 8.4 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $21.73 (+3.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -128.8M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $21.44
- **Current vs VWAP:** +4.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $29.64
- **Recent Low:** $18.73
- **Next Resistance:** $22.90 (61.8%)
- **Next Support:** $21.06 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 127. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:15.705480
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'GTLB'}`
- **Output Length:** 797 chars

### Output
```
# Insider Transactions for GTLB
# Data retrieved on: 2026-04-29 14:50:15

## Summary
- **Insider Sentiment:** Neutral
- **Sale:** 1 transactions
- **Unknown:** 1 transactions
- **Purchase:** 1 transactions
- **Total Sales Value:** $2,413,474
- **Total Purchases Value:** $128,342
- **Largest Purchase (3mo):** 2026-03-31 | STAPLES WILLIAM (Chief Executive Officer) | 6010 shares | $128342.0 | Ownership: D
- **Largest Sale (3mo):** 2026-04-15 | SIJBRANDIJ SYTSE (Director) | 116200 shares | $2413474.0 | Ownership: I

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,SIJBRANDIJ SYTSE,Director,Sale,116200,2413474.0,I
2026-04-15,SIJBRANDIJ SYTSE,Director,Unknown,116200,,I
2026-03-31,STAPLES WILLIAM,Chief Executive Officer,Purchase,6010,128342.0,D

```

---

## 128. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:15.809334
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'GTLB'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for GTLB

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 10
- **Hold**: 15
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 54.3% Bullish, 2.9% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 12
- **Hold**: 13
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 60.0% Bullish, 2.9% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 12
- **Hold**: 13
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 60.0% Bullish, 2.9% Bearish

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

## 129. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:16.051747
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'GTLB', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 924 chars

### Output
```
## Options Activity for GTLB

**Available Expirations:** 16 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,178 | 632 | 0.199 |
| Open Interest | 19,903 | 6,853 | 0.344 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $22.0 | 263 | 207 | 1.27x | 82.7% |
| CALL | 2026-05-08 | $22.5 | 288 | 284 | 1.01x | 79.5% |
| PUT | 2026-05-15 | $21.0 | 157 | 161 | 0.98x | 80.9% |
| CALL | 2026-05-15 | $23.0 | 1,188 | 1,675 | 0.71x | 81.8% |
| PUT | 2026-05-01 | $22.0 | 105 | 157 | 0.67x | 86.5% |
| CALL | 2026-05-01 | $23.5 | 390 | 592 | 0.66x | 84.8% |

```

---

## 130. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:19.229756
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'EOG', 'curr_date': '2026-04-29'}`
- **Output Length:** 1525 chars

### Output
```
# Technical Analysis for EOG
**Date:** 2026-04-29
**Current Price:** $135.78

## Price Action
- **Daily Change:** +1.92%
- **5-Day Change:** +2.08%

## RSI (14)
- **Value:** 53.2 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.312
- **Signal Line:** 0.093
- **Histogram:** -0.405 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $132.93 (+2.1% from price)
- **200 SMA:** $115.62 (+17.4% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $144.35
- **Middle:** $135.55
- **Lower:** $126.75
- **Band Position:** 51%
- **Bandwidth:** 13.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $3.79 (2.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $130.09 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 77.3
- **%D:** 60.8
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 4.4 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $134.47 (+1.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 36.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $135.48
- **Current vs VWAP:** +0.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $150.70
- **Recent Low:** $116.88
- **Next Resistance:** $137.78 (38.2%)
- **Next Support:** $133.79 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 131. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:19.430288
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'EOG'}`
- **Output Length:** 618 chars

### Output
```
# Insider Transactions for EOG
# Data retrieved on: 2026-04-29 14:50:19

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Option Exercise:** 1 transactions
- **Total Sales Value:** $856,523
- **Largest Sale (3mo):** 2026-03-31 | LEITZELL JEFFREY R (Chief Operating Officer) | 5698 shares | $856523 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-03-31,LEITZELL JEFFREY R,Chief Operating Officer,Sale,5698,856523,D
2026-03-31,LEITZELL JEFFREY R,Chief Operating Officer,Option Exercise,15002,843957,D

```

---

## 132. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:19.555985
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'EOG'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for EOG

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 18
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 39

**Sentiment**: 53.8% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 17
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 38

**Sentiment**: 55.3% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 18
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 39

**Sentiment**: 53.8% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 17
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 38

**Sentiment**: 55.3% Bullish, 0.0% Bearish


```

---

## 133. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:19.803353
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'EOG', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 929 chars

### Output
```
## Options Activity for EOG

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,104 | 1,337 | 0.635 |
| Open Interest | 16,523 | 11,267 | 0.682 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $129.0 | 149 | 42 | 3.55x | 58.1% |
| CALL | 2026-05-08 | $160.0 | 216 | 202 | 1.07x | 59.4% |
| CALL | 2026-05-01 | $160.0 | 216 | 214 | 1.01x | 70.3% |
| CALL | 2026-05-08 | $132.0 | 104 | 105 | 0.99x | 45.1% |
| CALL | 2026-05-08 | $146.0 | 150 | 152 | 0.99x | 42.5% |
| CALL | 2026-05-01 | $139.0 | 170 | 182 | 0.93x | 43.0% |

```

---

## 134. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:23.124582
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CRL', 'curr_date': '2026-04-29'}`
- **Output Length:** 1555 chars

### Output
```
# Technical Analysis for CRL
**Date:** 2026-04-29
**Current Price:** $166.79

## Price Action
- **Daily Change:** -2.59%
- **5-Day Change:** -9.75%

## RSI (14)
- **Value:** 43.4 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 0.739
- **Signal Line:** 2.036
- **Histogram:** -1.297 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $169.23 (-1.4% from price)
- **200 SMA:** $175.81 (-5.1% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $187.39
- **Middle:** $175.67
- **Lower:** $163.95
- **Band Position:** 12%
- **Bandwidth:** 13.3% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $7.56 (4.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $155.44 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 25.7
- **%D:** 43.1
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 15.5 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $173.74 (-4.0% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 5.2M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $175.19
- **Current vs VWAP:** -4.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $190.14
- **Recent Low:** $146.33
- **Next Resistance:** $168.24 (50.0%)
- **Next Support:** $163.07 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 135. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:23.312362
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CRL'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for CRL in the last 3 months.
```

---

## 136. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:23.420875
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CRL'}`
- **Output Length:** 710 chars

### Output
```
## Analyst Recommendation Trends for CRL

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 7
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 22

**Sentiment**: 63.6% Bullish, 0.0% Bearish


```

---

## 137. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:23.675873
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CRL', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 439 chars

### Output
```
## Options Activity for CRL

**Available Expirations:** 7 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 154 | 84 | 0.545 |
| Open Interest | 903 | 1,142 | 1.265 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

*No unusual options activity detected.*

```

---

## 138. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:25.533742
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CPRT', 'curr_date': '2026-04-29'}`
- **Output Length:** 1521 chars

### Output
```
# Technical Analysis for CPRT
**Date:** 2026-04-29
**Current Price:** $33.34

## Price Action
- **Daily Change:** +0.45%
- **5-Day Change:** -1.01%

## RSI (14)
- **Value:** 45.2 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.275
- **Signal Line:** -0.409
- **Histogram:** 0.133 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $34.57 (-3.6% from price)
- **200 SMA:** $41.03 (-18.7% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $33.89
- **Middle:** $33.29
- **Lower:** $32.68
- **Band Position:** 54%
- **Bandwidth:** 3.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $0.80 (2.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $32.14 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 38.6
- **%D:** 47.2
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 13.2 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $33.49 (-0.4% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -144.9M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $33.32
- **Current vs VWAP:** +0.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $38.84
- **Recent Low:** $32.20
- **Next Resistance:** $33.62 (78.6%)
- **Next Support:** $32.20 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 139. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:25.723387
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CPRT'}`
- **Output Length:** 603 chars

### Output
```
# Insider Transactions for CPRT
# Data retrieved on: 2026-04-29 14:50:25

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Option Exercise:** 1 transactions
- **Total Sales Value:** $869,724
- **Largest Sale (3mo):** 2026-04-15 | LIAW JEFFREY (Chief Executive Officer) | 26213 shares | $869724 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,LIAW JEFFREY,Chief Executive Officer,Sale,26213,869724,D
2026-04-15,LIAW JEFFREY,Chief Executive Officer,Option Exercise,49775,422805,D

```

---

## 140. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:25.823227
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

## 141. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:26.111390
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CPRT', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 445 chars

### Output
```
## Options Activity for CPRT

**Available Expirations:** 9 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 260 | 123 | 0.473 |
| Open Interest | 22,007 | 11,320 | 0.514 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

*No unusual options activity detected.*

```

---

## 142. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:27.518098
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AMRZ', 'curr_date': '2026-04-29'}`
- **Output Length:** 1508 chars

### Output
```
# Technical Analysis for AMRZ
**Date:** 2026-04-29
**Current Price:** $57.47

## Price Action
- **Daily Change:** -1.19%
- **5-Day Change:** +2.36%

## RSI (14)
- **Value:** 52.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.280
- **Signal Line:** 0.147
- **Histogram:** 0.133 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $57.83 (-0.6% from price)
- **200 SMA:** $53.14 (+8.2% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $59.96
- **Middle:** $56.67
- **Lower:** $53.38
- **Band Position:** 62%
- **Bandwidth:** 11.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.62 (2.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $55.04 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 50.6
- **%D:** 47.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 6.1 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $56.92 (+1.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -23.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $56.67
- **Current vs VWAP:** +1.4%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $65.43
- **Recent Low:** $51.88
- **Next Resistance:** $58.66 (50.0%)
- **Next Support:** $57.06 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 143. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:27.680016
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AMRZ'}`
- **Output Length:** 868 chars

### Output
```
# Insider Transactions for AMRZ
# Data retrieved on: 2026-04-29 14:50:27

## Summary
- **Insider Sentiment:** Neutral
- **Award/Grant:** 10 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-21,LADHANI HOLLI C,Director,Award/Grant,2976,0,D
2026-04-21,NEWMAN DONALD P,Director,Award/Grant,2976,0,D
2026-04-21,GANGESTAD NICHOLAS C.,Director,Award/Grant,2976,0,D
2026-04-21,GIBSON DWIGHT AUDLEY KONRAD,Director,Award/Grant,2976,0,D
2026-04-21,RIVKIN ROBERT S,Director,Award/Grant,2976,0,D
2026-04-21,WILBUR MARIA CRISTINA ALAPAG,Director,Award/Grant,2976,0,D
2026-04-21,OLEAS JURG AMADEO,Director,Award/Grant,2976,0,D
2026-04-21,MCKELVY MICHAEL E.,Director,Award/Grant,2976,0,D
2026-04-21,ROTH PELLANDA KATJA NICOLE,Director,Award/Grant,2976,0,D
2026-04-21,SANCHE JACQUES WOLF PH.D.,Director,Award/Grant,2976,0,D

```

---

## 144. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:27.794916
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AMRZ'}`
- **Output Length:** 712 chars

### Output
```
## Analyst Recommendation Trends for AMRZ

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 11
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 26

**Sentiment**: 65.4% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 4
- **Buy**: 8
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 66.7% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 4
- **Buy**: 8
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 18

**Sentiment**: 66.7% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 3
- **Buy**: 8
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 17

**Sentiment**: 64.7% Bullish, 0.0% Bearish


```

---

## 145. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:28.075890
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AMRZ', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 684 chars

### Output
```
## Options Activity for AMRZ

**Available Expirations:** 6 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 454 | 227 | 0.5 |
| Open Interest | 16,815 | 6,581 | 0.391 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-07-17 | $50.0 | 133 | 107 | 1.24x | 0.0% |
| PUT | 2026-07-17 | $57.06 | 126 | 135 | 0.93x | 41.0% |

```

---

## 146. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:32.638540
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ARXS', 'curr_date': '2026-04-29'}`
- **Output Length:** 1536 chars

### Output
```
# Technical Analysis for ARXS
**Date:** 2026-04-29
**Current Price:** $35.04

## Price Action
- **Daily Change:** -0.54%
- **5-Day Change:** -3.36%

## RSI (14)
- **Value:** 21.6 ↑
- **Signal:** OVERSOLD ⚡

## MACD
- **MACD Line:** -0.273
- **Signal Line:** -0.157
- **Histogram:** -0.115 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $36.77 (-4.7% from price)
- **200 SMA:** $36.77 (-4.7% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $39.61
- **Middle:** $36.77
- **Lower:** $33.93
- **Band Position:** 20%
- **Bandwidth:** 15.5% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $1.94 (5.5% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $32.13 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 21.9
- **%D:** 34.2
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 50.2 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $36.45 (-3.9% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -8.0M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $37.42
- **Current vs VWAP:** -6.4%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $39.45
- **Recent Low:** $34.45
- **Next Resistance:** $35.52 (78.6%)
- **Next Support:** $34.45 (100% (Low))

## Summary
- **Key Signals:** RSI oversold, Below 50 SMA

```

---

## 147. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:32.783539
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ARXS'}`
- **Output Length:** 52 chars

### Output
```
No insider transactions data found for symbol 'ARXS'
```

---

## 148. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:32.895138
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ARXS'}`
- **Output Length:** 44 chars

### Output
```
No recommendation trends data found for ARXS
```

---

## 149. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:33.213112
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ARXS', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 433 chars

### Output
```
## Options Activity for ARXS

**Available Expirations:** 4 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-08-21

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 44 | 9 | 0.205 |
| Open Interest | 52 | 9 | 0.173 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

*No unusual options activity detected.*

```

---

## 150. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:34.918680
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TSLA', 'curr_date': '2026-04-29'}`
- **Output Length:** 1535 chars

### Output
```
# Technical Analysis for TSLA
**Date:** 2026-04-29
**Current Price:** $376.02

## Price Action
- **Daily Change:** -0.70%
- **5-Day Change:** -2.97%

## RSI (14)
- **Value:** 48.8 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.353
- **Signal Line:** -2.199
- **Histogram:** 1.845 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $385.48 (-2.5% from price)
- **200 SMA:** $401.51 (-6.3% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $407.01
- **Middle:** $371.01
- **Lower:** $335.00
- **Band Position:** 57%
- **Bandwidth:** 19.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $14.69 (3.9% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $353.99 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 40.5
- **%D:** 51.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 7.3 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $377.13 (-0.3% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 1295.4M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $371.31
- **Current vs VWAP:** +1.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $420.34
- **Recent Low:** $337.24
- **Next Resistance:** $378.79 (50.0%)
- **Next Support:** $368.98 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 151. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:35.103554
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TSLA'}`
- **Output Length:** 675 chars

### Output
```
# Insider Transactions for TSLA
# Data retrieved on: 2026-04-29 14:50:35

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

## 152. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:35.207691
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

## 153. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:35.532534
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TSLA', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1178 chars

### Output
```
## Options Activity for TSLA

**Available Expirations:** 23 dates
**Analyzing:** 2026-04-29, 2026-05-01, 2026-05-04

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 398,070 | 300,967 | 0.756 |
| Open Interest | 682,599 | 422,080 | 0.618 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-04-29 | $422.5 | 322 | 2 | 161.0x | 163.3% |
| CALL | 2026-04-29 | $355.0 | 3,027 | 264 | 11.47x | 0.0% |
| CALL | 2026-05-01 | $327.5 | 140 | 13 | 10.77x | 90.0% |
| PUT | 2026-04-29 | $372.5 | 32,037 | 3,026 | 10.59x | 43.4% |
| CALL | 2026-04-29 | $372.5 | 17,622 | 1,787 | 9.86x | 24.9% |
| PUT | 2026-05-04 | $387.5 | 272 | 33 | 8.24x | 43.6% |
| CALL | 2026-04-29 | $375.0 | 44,168 | 5,801 | 7.61x | 28.2% |
| PUT | 2026-04-29 | $367.5 | 33,040 | 4,580 | 7.21x | 41.0% |
| CALL | 2026-04-29 | $370.0 | 12,287 | 1,741 | 7.06x | 19.5% |
| PUT | 2026-05-01 | $392.5 | 4,878 | 696 | 7.01x | 61.5% |

```

---

## 154. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:37.463870
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AAPL', 'curr_date': '2026-04-29'}`
- **Output Length:** 1534 chars

### Output
```
# Technical Analysis for AAPL
**Date:** 2026-04-29
**Current Price:** $270.71

## Price Action
- **Daily Change:** +1.16%
- **5-Day Change:** -0.90%

## RSI (14)
- **Value:** 58.3 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 3.791
- **Signal Line:** 2.957
- **Histogram:** 0.834 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $260.56 (+3.9% from price)
- **200 SMA:** $253.92 (+6.6% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $277.19
- **Middle:** $263.54
- **Lower:** $249.89
- **Band Position:** 76%
- **Bandwidth:** 10.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $5.87 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $261.91 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 70.0
- **%D:** 76.3
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 9.0 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $265.00 (+2.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 731.6M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $262.94
- **Current vs VWAP:** +3.0%
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

## 155. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:37.680111
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AAPL'}`
- **Output Length:** 953 chars

### Output
```
# Insider Transactions for AAPL
# Data retrieved on: 2026-04-29 14:50:37

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Unknown:** 5 transactions
- **Sale:** 3 transactions
- **Total Sales Value:** $24,594,923
- **Largest Sale (3mo):** 2026-04-02 | COOK TIMOTHY D (Chief Executive Officer) | 64949 shares | $16512198.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-23,PAREKH KEVAN,Chief Financial Officer,Sale,1534,421850.0,D
2026-04-15,PAREKH KEVAN,Chief Financial Officer,Unknown,10928,,D
2026-04-15,BORDERS BEN,Officer,Unknown,1717,,D
2026-04-02,O'BRIEN DEIRDRE,Officer,Sale,30002,7660875.0,D
2026-04-02,COOK TIMOTHY D,Chief Executive Officer,Sale,64949,16512198.0,D
2026-04-01,O'BRIEN DEIRDRE,Officer,Unknown,64317,,D
2026-04-01,KHAN SABIH,Chief Operating Officer,Unknown,64317,,D
2026-04-01,COOK TIMOTHY D,Chief Executive Officer,Unknown,131576,,D

```

---

## 156. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:37.778389
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

## 157. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:38.044120
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AAPL', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1191 chars

### Output
```
## Options Activity for AAPL

**Available Expirations:** 23 dates
**Analyzing:** 2026-04-29, 2026-05-01, 2026-05-04

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 146,420 | 79,851 | 0.545 |
| Open Interest | 278,122 | 192,353 | 0.692 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $155.0 | 7,505 | 3 | 2501.67x | 187.5% |
| PUT | 2026-05-01 | $160.0 | 7,500 | 57 | 131.58x | 175.0% |
| PUT | 2026-05-01 | $170.0 | 7,503 | 297 | 25.26x | 176.6% |
| PUT | 2026-05-01 | $165.0 | 7,500 | 491 | 15.27x | 192.2% |
| CALL | 2026-04-29 | $270.0 | 49,900 | 8,536 | 5.85x | 18.7% |
| CALL | 2026-04-29 | $230.0 | 359 | 62 | 5.79x | 123.4% |
| PUT | 2026-04-29 | $267.5 | 13,121 | 3,657 | 3.59x | 24.3% |
| CALL | 2026-05-01 | $180.0 | 107 | 36 | 2.97x | 289.6% |
| CALL | 2026-05-01 | $175.0 | 108 | 37 | 2.92x | 299.6% |
| PUT | 2026-05-01 | $290.0 | 143 | 56 | 2.55x | 63.2% |

```

---

## 158. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:39.747062
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'QCOM', 'curr_date': '2026-04-29'}`
- **Output Length:** 1567 chars

### Output
```
# Technical Analysis for QCOM
**Date:** 2026-04-29
**Current Price:** $150.00

## Price Action
- **Daily Change:** -0.17%
- **5-Day Change:** +10.24%

## RSI (14)
- **Value:** 73.9 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 3.562
- **Signal Line:** 1.115
- **Histogram:** 2.447 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $134.77 (+11.3% from price)
- **200 SMA:** $155.06 (-3.3% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $149.57
- **Middle:** $133.80
- **Lower:** $118.03
- **Band Position:** 101%
- **Bandwidth:** 23.6% (volatility indicator)
- **Signal:** AT UPPER BAND - Potential reversal ⚠️

## ATR (Volatility)
- **ATR:** $5.25 (3.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $142.12 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 70.3
- **%D:** 75.7
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 33.5 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $137.05 (+9.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 31.5M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $138.00
- **Current vs VWAP:** +8.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $161.00
- **Recent Low:** $121.99
- **Next Resistance:** $151.79 (23.6%)
- **Next Support:** $146.10 (38.2%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 159. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:39.942381
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'QCOM'}`
- **Output Length:** 711 chars

### Output
```
# Insider Transactions for QCOM
# Data retrieved on: 2026-04-29 14:50:39

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
2026-03-31,TRICOIRE JEAN-PASCAL,Director,Award/Grant,262,0.0,D
2026-03-31,MCLAUGHLIN MARK D,Director,Award/Grant,538,0.0,D

```

---

## 160. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:40.051668
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

## 161. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:40.375115
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'QCOM', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1170 chars

### Output
```
## Options Activity for QCOM

**Available Expirations:** 17 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 26,641 | 8,699 | 0.327 |
| Open Interest | 135,497 | 65,201 | 0.481 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $108.0 | 163 | 36 | 4.53x | 176.6% |
| CALL | 2026-05-08 | $157.5 | 1,204 | 281 | 4.28x | 64.7% |
| CALL | 2026-05-08 | $162.5 | 1,149 | 302 | 3.8x | 67.4% |
| PUT | 2026-05-08 | $75.0 | 435 | 148 | 2.94x | 271.7% |
| CALL | 2026-05-01 | $114.0 | 166 | 65 | 2.55x | 195.3% |
| CALL | 2026-05-01 | $115.0 | 166 | 95 | 1.75x | 0.0% |
| PUT | 2026-05-15 | $143.0 | 870 | 634 | 1.37x | 59.0% |
| PUT | 2026-05-01 | $117.0 | 2,030 | 2,018 | 1.01x | 227.9% |
| CALL | 2026-05-01 | $123.0 | 135 | 139 | 0.97x | 0.0% |
| CALL | 2026-05-08 | $190.0 | 187 | 251 | 0.75x | 80.4% |

```

---

## 162. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:42.286251
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'AVGO', 'curr_date': '2026-04-29'}`
- **Output Length:** 1535 chars

### Output
```
# Technical Analysis for AVGO
**Date:** 2026-04-29
**Current Price:** $399.83

## Price Action
- **Daily Change:** -4.39%
- **5-Day Change:** -5.40%

## RSI (14)
- **Value:** 62.3 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 23.524
- **Signal Line:** 21.075
- **Histogram:** 2.449 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $344.26 (+16.1% from price)
- **200 SMA:** $336.95 (+18.7% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $455.73
- **Middle:** $375.52
- **Lower:** $295.32
- **Band Position:** 65%
- **Bandwidth:** 42.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $13.24 (3.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $379.97 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 63.5
- **%D:** 81.2
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 37.9 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $383.66 (+4.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 398.9M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $372.99
- **Current vs VWAP:** +7.2%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $429.31
- **Recent Low:** $289.96
- **Next Resistance:** $429.31 (0.0% (High))
- **Next Support:** $396.42 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 163. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:42.493339
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'AVGO'}`
- **Output Length:** 1183 chars

### Output
```
# Insider Transactions for AVGO
# Data retrieved on: 2026-04-29 14:50:42

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 7 transactions
- **Sale:** 5 transactions
- **Gift:** 1 transactions
- **Total Sales Value:** $18,125,124
- **Largest Sale (3mo):** 2026-04-09 | VELAGA S. RAM (Officer) | 30215 shares | $10638012 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-21,SAMUELI HENRY,Director,Award/Grant,864,0,D
2026-04-20,BRYANT DIANE M,Director,Award/Grant,864,0,D
2026-04-20,LOW CHECK KIAN,Director,Award/Grant,864,0,D
2026-04-20,PAGE JUSTINE F.,Director,Award/Grant,864,0,D
2026-04-20,YOU HARRY L,Director,Award/Grant,864,0,D
2026-04-20,DELLY GAYLA J,Director,Award/Grant,864,0,D
2026-04-20,HAO KENNETH Y,Director,Award/Grant,864,0,D
2026-04-10,VELAGA S. RAM,Officer,Sale,8000,2964178,D
2026-04-09,VELAGA S. RAM,Officer,Sale,30215,10638012,D
2026-04-09,DELLY GAYLA J,Director,Sale,1000,358310,D
2026-04-08,PAGE JUSTINE F.,Director,Sale,2018,712354,D
2026-04-08,KAWWAS CHARLIE B,Officer,Sale,10000,3452270,I
2026-04-08,TAN HOCK E,Chief Executive Officer,Gift,22000,0,D

```

---

## 164. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:42.600152
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'AVGO'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for AVGO

### 2026-04-01
- **Strong Buy**: 17
- **Buy**: 37
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 57

**Sentiment**: 94.7% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 17
- **Buy**: 38
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 58

**Sentiment**: 94.8% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 17
- **Buy**: 37
- **Hold**: 2
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 56

**Sentiment**: 96.4% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 17
- **Buy**: 36
- **Hold**: 3
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 56

**Sentiment**: 94.6% Bullish, 0.0% Bearish


```

---

## 165. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:42.953464
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'AVGO', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1149 chars

### Output
```
## Options Activity for AVGO

**Available Expirations:** 24 dates
**Analyzing:** 2026-04-29, 2026-05-01, 2026-05-04

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 19,496 | 14,105 | 0.723 |
| Open Interest | 91,494 | 90,824 | 0.993 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-04-29 | $452.5 | 439 | 27 | 16.26x | 81.3% |
| PUT | 2026-04-29 | $305.0 | 163 | 50 | 3.26x | 162.5% |
| CALL | 2026-05-04 | $427.5 | 228 | 81 | 2.81x | 52.1% |
| PUT | 2026-04-29 | $335.0 | 109 | 41 | 2.66x | 109.4% |
| PUT | 2026-04-29 | $390.0 | 2,210 | 872 | 2.53x | 49.3% |
| CALL | 2026-04-29 | $405.0 | 2,130 | 891 | 2.39x | 40.2% |
| CALL | 2026-04-29 | $402.5 | 876 | 400 | 2.19x | 40.2% |
| CALL | 2026-05-04 | $487.5 | 903 | 448 | 2.02x | 71.7% |
| PUT | 2026-05-04 | $415.0 | 192 | 127 | 1.51x | 54.4% |
| CALL | 2026-04-29 | $400.0 | 656 | 554 | 1.18x | 39.9% |

```

---

## 166. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:44.769207
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'GOOG', 'curr_date': '2026-04-29'}`
- **Output Length:** 1551 chars

### Output
```
# Technical Analysis for GOOG
**Date:** 2026-04-29
**Current Price:** $347.50

## Price Action
- **Daily Change:** -0.29%
- **5-Day Change:** +2.89%

## RSI (14)
- **Value:** 71.3 ↑
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 11.308
- **Signal Line:** 8.781
- **Histogram:** 2.527 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $310.12 (+12.1% from price)
- **200 SMA:** $278.21 (+24.9% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $361.20
- **Middle:** $323.04
- **Lower:** $284.88
- **Band Position:** 82%
- **Bandwidth:** 23.6% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $7.65 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $336.03 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 87.8
- **%D:** 88.7
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 39.4 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $327.29 (+6.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 626.2M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $320.26
- **Current vs VWAP:** +8.5%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $350.90
- **Recent Low:** $271.54
- **Next Resistance:** $350.90 (0.0% (High))
- **Next Support:** $332.17 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 167. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:45.020323
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'GOOG'}`
- **Output Length:** 665 chars

### Output
```
# Insider Transactions for GOOG
# Data retrieved on: 2026-04-29 14:50:45

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 3 transactions
- **Gift:** 1 transactions
- **Total Sales Value:** $555,003
- **Largest Sale (3mo):** 2026-04-15 | HENNESSY JOHN L (Director) | 1050 shares | $348232.0 | Ownership: I

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-15,HENNESSY JOHN L,Director,Sale,1050,348232.0,I
2026-04-01,O TOOLE AMIE THUENER,Officer,Sale,617,178702.0,D
2026-03-31,WALKER JOHN KENT,Officer,Gift,8993,0.0,D
2026-03-30,ARNOLD FRANCES H PH.D.,Director,Sale,102,28069.0,D

```

---

## 168. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:45.119978
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'GOOG'}`
- **Output Length:** 721 chars

### Output
```
## Analyst Recommendation Trends for GOOG

### 2026-04-01
- **Strong Buy**: 19
- **Buy**: 41
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 68

**Sentiment**: 88.2% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 19
- **Buy**: 41
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 69

**Sentiment**: 87.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 19
- **Buy**: 40
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 69

**Sentiment**: 85.5% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 19
- **Buy**: 41
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 70

**Sentiment**: 85.7% Bullish, 0.0% Bearish


```

---

## 169. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:45.502133
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'GOOG', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1153 chars

### Output
```
## Options Activity for GOOG

**Available Expirations:** 19 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 25,539 | 19,307 | 0.756 |
| Open Interest | 248,883 | 196,828 | 0.791 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-08 | $297.5 | 507 | 17 | 29.82x | 58.6% |
| CALL | 2026-05-08 | $307.5 | 107 | 23 | 4.65x | 47.0% |
| CALL | 2026-05-01 | $402.5 | 347 | 96 | 3.61x | 77.9% |
| PUT | 2026-05-01 | $375.0 | 346 | 102 | 3.39x | 96.6% |
| CALL | 2026-05-15 | $430.0 | 653 | 194 | 3.37x | 45.9% |
| PUT | 2026-05-08 | $352.5 | 105 | 36 | 2.92x | 50.7% |
| PUT | 2026-05-01 | $355.0 | 1,306 | 513 | 2.55x | 82.5% |
| PUT | 2026-05-01 | $350.0 | 2,673 | 1,158 | 2.31x | 81.9% |
| CALL | 2026-05-08 | $280.0 | 149 | 91 | 1.64x | 70.1% |
| PUT | 2026-05-08 | $310.0 | 1,972 | 1,364 | 1.45x | 51.2% |

```

---

## 170. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:47.254395
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MA', 'curr_date': '2026-04-29'}`
- **Output Length:** 1529 chars

### Output
```
# Technical Analysis for MA
**Date:** 2026-04-29
**Current Price:** $507.62

## Price Action
- **Daily Change:** +0.23%
- **5-Day Change:** -0.50%

## RSI (14)
- **Value:** 50.0 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.597
- **Signal Line:** 0.660
- **Histogram:** -0.063 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $507.08 (+0.1% from price)
- **200 SMA:** $547.31 (-7.3% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $523.88
- **Middle:** $506.45
- **Lower:** $489.02
- **Band Position:** 53%
- **Bandwidth:** 6.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $10.40 (2.0% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $492.02 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 38.5
- **%D:** 46.6
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 9.0 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $507.44 (+0.0% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -18.1M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $505.96
- **Current vs VWAP:** +0.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $529.85
- **Recent Low:** $479.68
- **Next Resistance:** $510.68 (38.2%)
- **Next Support:** $504.76 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 171. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:47.426302
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MA'}`
- **Output Length:** 58 chars

### Output
```
No insider transactions found for MA in the last 3 months.
```

---

## 172. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:47.532165
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MA'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for MA

### 2026-04-01
- **Strong Buy**: 13
- **Buy**: 30
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 48

**Sentiment**: 89.6% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 14
- **Buy**: 29
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 50

**Sentiment**: 86.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 14
- **Buy**: 27
- **Hold**: 9
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 50

**Sentiment**: 82.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 13
- **Buy**: 26
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 49

**Sentiment**: 79.6% Bullish, 0.0% Bearish


```

---

## 173. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:47.839468
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MA', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1098 chars

### Output
```
## Options Activity for MA

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 4,303 | 2,457 | 0.571 |
| Open Interest | 15,850 | 16,694 | 1.053 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $550.0 | 2,235 | 234 | 9.55x | 51.7% |
| PUT | 2026-05-01 | $480.0 | 321 | 105 | 3.06x | 60.9% |
| PUT | 2026-05-08 | $490.0 | 125 | 58 | 2.16x | 43.1% |
| PUT | 2026-05-08 | $415.0 | 197 | 97 | 2.03x | 78.7% |
| CALL | 2026-05-01 | $560.0 | 295 | 173 | 1.71x | 52.9% |
| PUT | 2026-05-08 | $410.0 | 192 | 120 | 1.6x | 82.1% |
| PUT | 2026-05-01 | $515.0 | 119 | 78 | 1.53x | 61.2% |
| CALL | 2026-05-01 | $502.5 | 142 | 118 | 1.2x | 43.7% |
| PUT | 2026-05-15 | $320.0 | 400 | 404 | 0.99x | 139.2% |

```

---

## 174. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:49.584823
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CVX', 'curr_date': '2026-04-29'}`
- **Output Length:** 1518 chars

### Output
```
# Technical Analysis for CVX
**Date:** 2026-04-29
**Current Price:** $188.36

## Price Action
- **Daily Change:** +1.94%
- **5-Day Change:** +1.09%

## RSI (14)
- **Value:** 46.9 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -2.499
- **Signal Line:** -1.969
- **Histogram:** -0.530 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $192.41 (-2.1% from price)
- **200 SMA:** $163.80 (+15.0% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $203.99
- **Middle:** $190.64
- **Lower:** $177.29
- **Band Position:** 41%
- **Bandwidth:** 14.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $5.14 (2.7% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $180.66 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 66.0
- **%D:** 52.3
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 16.9 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $189.65 (-0.7% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** 239.6M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $191.73
- **Current vs VWAP:** -1.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $214.71
- **Recent Low:** $177.74
- **Next Resistance:** $191.86 (61.8%)
- **Next Support:** $185.65 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 175. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:49.743503
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CVX'}`
- **Output Length:** 583 chars

### Output
```
# Insider Transactions for CVX
# Data retrieved on: 2026-04-29 14:50:49

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Option Exercise:** 1 transactions
- **Total Sales Value:** $8,574,857
- **Largest Sale (3mo):** 2026-03-30 | PATE ROBERT HEWITT (Officer) | 40200 shares | $8574857.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-03-30,PATE ROBERT HEWITT,Officer,Sale,40200,8574857.0,D
2026-03-30,PATE ROBERT HEWITT,Officer,Option Exercise,40200,5039070.0,D

```

---

## 176. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:49.843831
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'CVX'}`
- **Output Length:** 717 chars

### Output
```
## Analyst Recommendation Trends for CVX

### 2026-04-01
- **Strong Buy**: 6
- **Buy**: 16
- **Hold**: 9
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 68.8% Bullish, 3.1% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 15
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 65.6% Bullish, 3.1% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 14
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 65.6% Bullish, 3.1% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 13
- **Hold**: 11
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 62.5% Bullish, 3.1% Bearish


```

---

## 177. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:50.177115
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CVX', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1085 chars

### Output
```
## Options Activity for CVX

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 6,423 | 5,446 | 0.848 |
| Open Interest | 95,919 | 71,159 | 0.742 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $192.5 | 1,824 | 293 | 6.23x | 35.0% |
| PUT | 2026-05-08 | $155.0 | 244 | 85 | 2.87x | 92.7% |
| PUT | 2026-05-15 | $230.0 | 200 | 114 | 1.75x | 65.3% |
| PUT | 2026-05-01 | $177.5 | 1,076 | 812 | 1.33x | 50.0% |
| PUT | 2026-05-08 | $182.5 | 346 | 282 | 1.23x | 37.4% |
| CALL | 2026-05-08 | $192.5 | 145 | 189 | 0.77x | 36.0% |
| CALL | 2026-05-15 | $270.0 | 112 | 147 | 0.76x | 72.9% |
| PUT | 2026-05-01 | $180.0 | 542 | 813 | 0.67x | 48.9% |
| PUT | 2026-05-08 | $172.5 | 121 | 197 | 0.61x | 44.2% |

```

---

## 178. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:52.217259
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TXN', 'curr_date': '2026-04-29'}`
- **Output Length:** 1555 chars

### Output
```
# Technical Analysis for TXN
**Date:** 2026-04-29
**Current Price:** $265.00

## Price Action
- **Daily Change:** -1.67%
- **5-Day Change:** +12.14%

## RSI (14)
- **Value:** 72.6 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 17.745
- **Signal Line:** 12.031
- **Histogram:** 5.713 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $211.36 (+25.4% from price)
- **200 SMA:** $190.56 (+39.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $281.43
- **Middle:** $226.25
- **Lower:** $171.06
- **Band Position:** 85%
- **Bandwidth:** 48.8% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $9.20 (3.5% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $251.21 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.2
- **%D:** 87.3
- **Signal:** OVERBOUGHT ⚠️

## ADX (Trend Strength)
- **ADX:** 58.5 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $234.33 (+13.1% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 5.5M
- **5-Day Trend:** Falling ↓
- **Signal:** BEARISH DIVERGENCE ⚠️ (distribution)

## VWAP (20-period)
- **VWAP:** $238.03
- **Current vs VWAP:** +11.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $287.83
- **Recent Low:** $184.90
- **Next Resistance:** $287.83 (0.0% (High))
- **Next Support:** $263.54 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA

```

---

## 179. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:52.412928
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TXN'}`
- **Output Length:** 1625 chars

### Output
```
# Insider Transactions for TXN
# Data retrieved on: 2026-04-29 14:50:52

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 9 transactions
- **Option Exercise:** 8 transactions
- **Award/Grant:** 1 transactions
- **Total Sales Value:** $31,346,427
- **Largest Sale (3mo):** 2026-04-24 | LIZARDI RAFAEL R (Chief Financial Officer) | 40541 shares | $11117556 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-27,KANE KATHARINE,General Counsel,Sale,6125,1656283,D
2026-04-27,BAHAI AHMAD,Officer,Sale,3660,982359,D
2026-04-27,BAHAI AHMAD,Officer,Option Exercise,3660,290092,D
2026-04-27,KANE KATHARINE,General Counsel,Option Exercise,6125,1021992,D
2026-04-24,LIZARDI RAFAEL R,Chief Financial Officer,Sale,40541,11117556,D
2026-04-24,PATSLEY PAMELA H,Director,Sale,4306,1188918,D
2026-04-24,BLINN MARK A,Director,Sale,15000,4162689,I
2026-04-24,BLUEDORN TODD M,Director,Option Exercise,4306,474306,D
2026-04-24,BLUEDORN TODD M,Director,Sale,4306,1180930,D
2026-04-24,LIZARDI RAFAEL R,Chief Financial Officer,Option Exercise,40541,6860753,D
2026-04-24,LEONARD SHANON J,Officer,Option Exercise,18189,2877483,D
2026-04-24,ABRAHAM TSEDENIYA,Officer,Award/Grant,7217,0,D
2026-04-24,LEONARD SHANON J,Officer,Sale,18189,5055707,D
2026-04-24,WITZSCHE CHRISTINE,Officer,Option Exercise,3625,389261,D
2026-04-24,WITZSCHE CHRISTINE,Officer,Sale,3625,998771,D
2026-04-24,RON AMICHAI,Officer,Option Exercise,15865,1656465,D
2026-04-24,RON AMICHAI,Officer,Sale,18365,5003214,D
2026-04-24,PATSLEY PAMELA H,Director,Option Exercise,4306,474306,D

```

---

## 180. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:52.526636
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

## 181. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:52.890080
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TXN', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1163 chars

### Output
```
## Options Activity for TXN

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 5,508 | 3,783 | 0.687 |
| Open Interest | 45,770 | 44,508 | 0.972 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-08 | $272.5 | 255 | 22 | 11.59x | 36.9% |
| CALL | 2026-05-08 | $277.5 | 126 | 31 | 4.06x | 36.8% |
| CALL | 2026-05-01 | $245.0 | 1,000 | 406 | 2.46x | 61.2% |
| PUT | 2026-05-01 | $160.0 | 123 | 55 | 2.24x | 336.7% |
| CALL | 2026-05-08 | $310.0 | 185 | 125 | 1.48x | 53.5% |
| CALL | 2026-05-08 | $267.5 | 122 | 86 | 1.42x | 38.8% |
| PUT | 2026-05-08 | $267.5 | 746 | 552 | 1.35x | 51.0% |
| PUT | 2026-05-01 | $140.0 | 158 | 120 | 1.32x | 409.8% |
| PUT | 2026-05-08 | $300.0 | 124 | 102 | 1.22x | 64.5% |
| CALL | 2026-05-08 | $312.5 | 209 | 199 | 1.05x | 55.8% |

```

---

## 182. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:54.575715
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ISRG', 'curr_date': '2026-04-29'}`
- **Output Length:** 1545 chars

### Output
```
# Technical Analysis for ISRG
**Date:** 2026-04-29
**Current Price:** $466.64

## Price Action
- **Daily Change:** -0.92%
- **5-Day Change:** -3.51%

## RSI (14)
- **Value:** 48.4 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -0.592
- **Signal Line:** -3.058
- **Histogram:** 2.465 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $477.47 (-2.3% from price)
- **200 SMA:** $501.02 (-6.9% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $483.80
- **Middle:** $463.59
- **Lower:** $443.39
- **Band Position:** 58%
- **Bandwidth:** 8.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $14.27 (3.1% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $445.24 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 53.9
- **%D:** 58.5
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 5.3 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $468.49 (-0.4% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -7.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $466.06
- **Current vs VWAP:** +0.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $511.88
- **Recent Low:** $447.78
- **Next Resistance:** $472.27 (61.8%)
- **Next Support:** $461.50 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 183. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:54.742055
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ISRG'}`
- **Output Length:** 60 chars

### Output
```
No insider transactions found for ISRG in the last 3 months.
```

---

## 184. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:54.846440
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ISRG'}`
- **Output Length:** 723 chars

### Output
```
## Analyst Recommendation Trends for ISRG

### 2026-04-01
- **Strong Buy**: 11
- **Buy**: 18
- **Hold**: 11
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 41

**Sentiment**: 70.7% Bullish, 2.4% Bearish

### 2026-03-01
- **Strong Buy**: 10
- **Buy**: 19
- **Hold**: 12
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 42

**Sentiment**: 69.0% Bullish, 2.4% Bearish

### 2026-02-01
- **Strong Buy**: 10
- **Buy**: 19
- **Hold**: 12
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 42

**Sentiment**: 69.0% Bullish, 2.4% Bearish

### 2026-01-01
- **Strong Buy**: 10
- **Buy**: 15
- **Hold**: 14
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 62.5% Bullish, 2.5% Bearish


```

---

## 185. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:55.148082
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ISRG', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 810 chars

### Output
```
## Options Activity for ISRG

**Available Expirations:** 18 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,302 | 925 | 0.28 |
| Open Interest | 13,393 | 13,485 | 1.007 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-08 | $560.0 | 272 | 37 | 7.35x | 87.9% |
| CALL | 2026-05-08 | $515.0 | 1,421 | 433 | 3.28x | 61.8% |
| CALL | 2026-05-01 | $502.5 | 103 | 65 | 1.58x | 97.1% |
| CALL | 2026-05-15 | $530.0 | 220 | 312 | 0.71x | 47.1% |

```

---

## 186. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:58.282709
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'GEV', 'curr_date': '2026-04-29'}`
- **Output Length:** 1546 chars

### Output
```
# Technical Analysis for GEV
**Date:** 2026-04-29
**Current Price:** $1088.93

## Price Action
- **Daily Change:** -2.79%
- **5-Day Change:** -3.43%

## RSI (14)
- **Value:** 66.5 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 65.132
- **Signal Line:** 54.220
- **Histogram:** 10.912 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $908.04 (+19.9% from price)
- **200 SMA:** $700.31 (+55.5% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $1171.52
- **Middle:** $996.62
- **Lower:** $821.71
- **Band Position:** 76%
- **Bandwidth:** 35.1% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $47.05 (4.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $1018.36 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 71.7
- **%D:** 78.3
- **Signal:** Bearish crossover ⚠️

## ADX (Trend Strength)
- **ADX:** 36.9 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $1014.32 (+7.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 12.6M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $1013.62
- **Current vs VWAP:** +7.4%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $1181.95
- **Recent Low:** $776.53
- **Next Resistance:** $1181.95 (0.0% (High))
- **Next Support:** $1086.27 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 187. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:58.467906
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'GEV'}`
- **Output Length:** 283 chars

### Output
```
# Insider Transactions for GEV
# Data retrieved on: 2026-04-29 14:50:58

## Summary
- **Insider Sentiment:** Neutral
- **Unknown:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-02,BAERT STEVEN,Officer,Unknown,8505,,D

```

---

## 188. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:58.580237
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'GEV'}`
- **Output Length:** 714 chars

### Output
```
## Analyst Recommendation Trends for GEV

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 25
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 85.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 24
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 82.5% Bullish, 2.5% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 23
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 40

**Sentiment**: 80.0% Bullish, 2.5% Bearish

### 2026-01-01
- **Strong Buy**: 8
- **Buy**: 22
- **Hold**: 8
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 39

**Sentiment**: 76.9% Bullish, 2.6% Bearish


```

---

## 189. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:50:58.965451
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'GEV', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1158 chars

### Output
```
## Options Activity for GEV

**Available Expirations:** 17 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 3,018 | 6,594 | 2.185 |
| Open Interest | 24,128 | 38,525 | 1.597 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-01 | $1275.0 | 164 | 46 | 3.57x | 82.8% |
| PUT | 2026-05-01 | $885.0 | 101 | 32 | 3.16x | 101.1% |
| PUT | 2026-05-15 | $730.0 | 295 | 139 | 2.12x | 82.2% |
| CALL | 2026-05-01 | $1070.0 | 135 | 67 | 2.01x | 74.5% |
| PUT | 2026-05-01 | $940.0 | 145 | 80 | 1.81x | 72.1% |
| PUT | 2026-05-01 | $920.0 | 151 | 110 | 1.37x | 82.6% |
| PUT | 2026-05-01 | $890.0 | 280 | 224 | 1.25x | 98.4% |
| CALL | 2026-05-08 | $1220.0 | 106 | 86 | 1.23x | 58.5% |
| PUT | 2026-05-01 | $945.0 | 106 | 93 | 1.14x | 65.0% |
| PUT | 2026-05-15 | $590.0 | 1,065 | 973 | 1.09x | 89.6% |

```

---

## 190. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:01.185475
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ORCL', 'curr_date': '2026-04-29'}`
- **Output Length:** 1504 chars

### Output
```
# Technical Analysis for ORCL
**Date:** 2026-04-29
**Current Price:** $165.96

## Price Action
- **Daily Change:** -4.05%
- **5-Day Change:** -11.49%

## RSI (14)
- **Value:** 52.6 ↓
- **Signal:** Bullish

## MACD
- **MACD Line:** 6.613
- **Signal Line:** 5.764
- **Histogram:** 0.849 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $154.50 (+7.4% from price)
- **200 SMA:** $212.13 (-21.8% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $194.16
- **Middle:** $161.03
- **Lower:** $127.89
- **Band Position:** 57%
- **Bandwidth:** 41.2% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $8.72 (5.3% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $152.88 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 42.6
- **%D:** 61.6
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 26.8 (Weakening ↓)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $165.20 (+0.5% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -380.6M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $163.89
- **Current vs VWAP:** +1.3%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $189.18
- **Recent Low:** $134.57
- **Next Resistance:** $168.32 (38.2%)
- **Next Support:** $161.88 (50.0%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 191. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:01.376847
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ORCL'}`
- **Output Length:** 559 chars

### Output
```
# Insider Transactions for ORCL
# Data retrieved on: 2026-04-29 14:51:01

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Gift:** 1 transactions
- **Total Sales Value:** $2,642,850
- **Largest Sale (3mo):** 2026-04-16 | LEVEY STUART A (Officer) | 15000 shares | $2642850.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,LEVEY STUART A,Officer,Sale,15000,2642850.0,D
2026-03-30,HENLEY JEFFREY O,Officer and Director,Gift,16810,0.0,I

```

---

## 192. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:01.485947
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

## 193. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:01.964547
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ORCL', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 1185 chars

### Output
```
## Options Activity for ORCL

**Available Expirations:** 20 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 36,300 | 23,036 | 0.635 |
| Open Interest | 456,121 | 236,873 | 0.519 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-15 | $280.0 | 267 | 50 | 5.34x | 155.9% |
| PUT | 2026-05-01 | $132.0 | 356 | 157 | 2.27x | 100.8% |
| CALL | 2026-05-01 | $172.5 | 1,924 | 1,567 | 1.23x | 63.1% |
| CALL | 2026-05-01 | $162.5 | 761 | 683 | 1.11x | 59.0% |
| CALL | 2026-05-08 | $240.0 | 153 | 150 | 1.02x | 111.7% |
| CALL | 2026-05-01 | $265.0 | 125 | 126 | 0.99x | 162.5% |
| CALL | 2026-05-08 | $105.0 | 110 | 120 | 0.92x | 111.3% |
| CALL | 2026-05-01 | $167.5 | 954 | 1,366 | 0.7x | 60.7% |
| CALL | 2026-05-01 | $170.0 | 2,842 | 4,433 | 0.64x | 61.8% |
| CALL | 2026-05-01 | $230.0 | 745 | 1,188 | 0.63x | 118.8% |

```

---

## 194. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:03.619431
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TMO', 'curr_date': '2026-04-29'}`
- **Output Length:** 1547 chars

### Output
```
# Technical Analysis for TMO
**Date:** 2026-04-29
**Current Price:** $470.22

## Price Action
- **Daily Change:** +0.47%
- **5-Day Change:** -8.51%

## RSI (14)
- **Value:** 37.8 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -5.233
- **Signal Line:** -0.087
- **Histogram:** -5.146 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $497.58 (-5.5% from price)
- **200 SMA:** $525.24 (-10.5% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $543.47
- **Middle:** $500.51
- **Lower:** $457.55
- **Band Position:** 15%
- **Bandwidth:** 17.2% (volatility indicator)
- **Signal:** Near lower band

## ATR (Volatility)
- **ATR:** $16.10 (3.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $446.07 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 26.7
- **%D:** 41.9
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 30.1 (Strengthening ↑)
- **Interpretation:** Strong trend

## 20 EMA
- **Value:** $495.66 (-5.1% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -4.2M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $496.35
- **Current vs VWAP:** -5.3%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $538.41
- **Recent Low:** $455.60
- **Next Resistance:** $473.32 (78.6%)
- **Next Support:** $455.60 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 195. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:03.896036
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TMO'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for TMO in the last 3 months.
```

---

## 196. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:04.003162
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TMO'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for TMO

### 2026-04-01
- **Strong Buy**: 11
- **Buy**: 20
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 88.6% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 20
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 87.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 18
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 81.8% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 19
- **Hold**: 4
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 32

**Sentiment**: 87.5% Bullish, 0.0% Bearish


```

---

## 197. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:04.299916
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TMO', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 628 chars

### Output
```
## Options Activity for TMO

**Available Expirations:** 12 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 674 | 328 | 0.487 |
| Open Interest | 3,494 | 7,443 | 2.13 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $510.0 | 156 | 251 | 0.62x | 31.1% |

```

---

## 198. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:07.870986
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'MPC', 'curr_date': '2026-04-29'}`
- **Output Length:** 1530 chars

### Output
```
# Technical Analysis for MPC
**Date:** 2026-04-29
**Current Price:** $232.59

## Price Action
- **Daily Change:** +2.37%
- **5-Day Change:** +4.74%

## RSI (14)
- **Value:** 56.4 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** -0.527
- **Signal Line:** -0.325
- **Histogram:** -0.202 (Weakening ↓)
- **Signal:** Bearish

## Moving Averages
- **50 SMA:** $223.26 (+4.2% from price)
- **200 SMA:** $190.96 (+21.8% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $247.21
- **Middle:** $228.14
- **Lower:** $209.06
- **Band Position:** 62%
- **Bandwidth:** 16.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $7.59 (3.3% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $221.20 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 72.7
- **%D:** 55.2
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 7.9 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $226.36 (+2.8% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 47.6M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $228.27
- **Current vs VWAP:** +1.9%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $255.77
- **Recent Low:** $190.32
- **Next Resistance:** $240.32 (23.6%)
- **Next Support:** $230.77 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 199. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:08.111921
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'MPC'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for MPC in the last 3 months.
```

---

## 200. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:08.216926
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'MPC'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for MPC

### 2026-04-01
- **Strong Buy**: 2
- **Buy**: 12
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 56.0% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 2
- **Buy**: 12
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 56.0% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 2
- **Buy**: 12
- **Hold**: 11
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 25

**Sentiment**: 56.0% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 2
- **Buy**: 12
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 24

**Sentiment**: 58.3% Bullish, 0.0% Bearish


```

---

## 201. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:08.443684
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'MPC', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 757 chars

### Output
```
## Options Activity for MPC

**Available Expirations:** 10 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 5,871 | 442 | 0.075 |
| Open Interest | 25,403 | 11,216 | 0.442 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bullish (more call volume)
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-06-18 | $90.0 | 311 | 309 | 1.01x | 0.0% |
| CALL | 2026-05-15 | $250.0 | 2,879 | 3,077 | 0.94x | 45.0% |
| CALL | 2026-05-15 | $240.0 | 2,016 | 2,640 | 0.76x | 45.6% |

```

---

## 202. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:11.287285
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DVN', 'curr_date': '2026-04-29'}`
- **Output Length:** 1527 chars

### Output
```
# Technical Analysis for DVN
**Date:** 2026-04-29
**Current Price:** $49.48

## Price Action
- **Daily Change:** +2.66%
- **5-Day Change:** +5.01%

## RSI (14)
- **Value:** 60.8 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.212
- **Signal Line:** 0.065
- **Histogram:** 0.147 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $46.63 (+6.1% from price)
- **200 SMA:** $37.97 (+30.3% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $51.19
- **Middle:** $47.54
- **Lower:** $43.89
- **Band Position:** 77%
- **Bandwidth:** 15.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $1.60 (3.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $47.07 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 84.4
- **%D:** 69.9
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 9.1 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $47.43 (+4.3% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 195.9M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $47.53
- **Current vs VWAP:** +4.1%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $52.71
- **Recent Low:** $41.52
- **Next Resistance:** $50.07 (23.6%)
- **Next Support:** $48.44 (38.2%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 203. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:11.437082
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DVN'}`
- **Output Length:** 59 chars

### Output
```
No insider transactions found for DVN in the last 3 months.
```

---

## 204. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:11.537179
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DVN'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for DVN

### 2026-04-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 6
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 78.8% Bullish, 3.0% Bearish

### 2026-03-01
- **Strong Buy**: 9
- **Buy**: 16
- **Hold**: 7
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 75.8% Bullish, 3.0% Bearish

### 2026-02-01
- **Strong Buy**: 9
- **Buy**: 17
- **Hold**: 10
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 36

**Sentiment**: 72.2% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 9
- **Buy**: 18
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 77.1% Bullish, 0.0% Bearish


```

---

## 205. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:11.801510
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DVN', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 789 chars

### Output
```
## Options Activity for DVN

**Available Expirations:** 17 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 2,978 | 2,730 | 0.917 |
| Open Interest | 60,589 | 21,859 | 0.361 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Bullish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $51.0 | 1,000 | 86 | 11.63x | 49.9% |
| PUT | 2026-05-08 | $46.5 | 120 | 120 | 1.0x | 50.0% |
| CALL | 2026-05-08 | $65.0 | 120 | 139 | 0.86x | 87.5% |
| PUT | 2026-05-15 | $35.0 | 862 | 1,631 | 0.53x | 92.4% |

```

---

## 206. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:13.628757
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'DINO', 'curr_date': '2026-04-29'}`
- **Output Length:** 1540 chars

### Output
```
# Technical Analysis for DINO
**Date:** 2026-04-29
**Current Price:** $62.73

## Price Action
- **Daily Change:** +1.50%
- **5-Day Change:** +5.50%

## RSI (14)
- **Value:** 61.7 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.769
- **Signal Line:** 0.537
- **Histogram:** 0.232 (Strengthening ↑)
- **Signal:** BULLISH CROSSOVER ⚡

## Moving Averages
- **50 SMA:** $57.34 (+9.4% from price)
- **200 SMA:** $51.39 (+22.1% from price)
- **Trend:** STRONG UPTREND ⚡

## Bollinger Bands (20,2)
- **Upper:** $63.21
- **Middle:** $59.53
- **Lower:** $55.86
- **Band Position:** 94%
- **Bandwidth:** 12.3% (volatility indicator)
- **Signal:** Near upper band

## ATR (Volatility)
- **ATR:** $2.27 (3.6% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $59.33 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 83.8
- **%D:** 72.4
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 13.2 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $59.63 (+5.2% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 8.0M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $59.36
- **Current vs VWAP:** +5.7%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $64.70
- **Recent Low:** $46.53
- **Next Resistance:** $64.70 (0.0% (High))
- **Next Support:** $60.41 (23.6%)

## Summary
- **Key Signals:** Above 50 SMA

```

---

## 207. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:13.796769
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'DINO'}`
- **Output Length:** 311 chars

### Output
```
# Insider Transactions for DINO
# Data retrieved on: 2026-04-29 14:51:13

## Summary
- **Insider Sentiment:** Neutral
- **Award/Grant:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-17,MYERS FRANKLIN,Chief Executive Officer,Award/Grant,1747,0,D

```

---

## 208. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:13.899143
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'DINO'}`
- **Output Length:** 712 chars

### Output
```
## Analyst Recommendation Trends for DINO

### 2026-04-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 4.3% Bearish

### 2026-03-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 4.3% Bearish

### 2026-02-01
- **Strong Buy**: 5
- **Buy**: 10
- **Hold**: 7
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 23

**Sentiment**: 65.2% Bullish, 4.3% Bearish

### 2026-01-01
- **Strong Buy**: 5
- **Buy**: 9
- **Hold**: 8
- **Sell**: 0
- **Strong Sell**: 1
- **Total Analysts**: 23

**Sentiment**: 60.9% Bullish, 4.3% Bearish


```

---

## 209. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:14.169868
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'DINO', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 756 chars

### Output
```
## Options Activity for DINO

**Available Expirations:** 6 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-09-18

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 1,358 | 2,968 | 2.186 |
| Open Interest | 21,626 | 15,307 | 0.708 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| CALL | 2026-05-15 | $67.5 | 819 | 257 | 3.19x | 51.9% |
| PUT | 2026-05-15 | $30.0 | 1,440 | 1,500 | 0.96x | 229.5% |
| PUT | 2026-06-18 | $27.5 | 1,300 | 1,850 | 0.7x | 135.8% |

```

---

## 210. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:17.399390
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'PGR', 'curr_date': '2026-04-29'}`
- **Output Length:** 1526 chars

### Output
```
# Technical Analysis for PGR
**Date:** 2026-04-29
**Current Price:** $203.03

## Price Action
- **Daily Change:** +0.52%
- **5-Day Change:** +0.16%

## RSI (14)
- **Value:** 51.5 ↑
- **Signal:** Bullish

## MACD
- **MACD Line:** 0.239
- **Signal Line:** -0.282
- **Histogram:** 0.521 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $203.49 (-0.2% from price)
- **200 SMA:** $214.40 (-5.3% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $207.78
- **Middle:** $200.03
- **Lower:** $192.29
- **Band Position:** 69%
- **Bandwidth:** 7.7% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $4.83 (2.4% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $195.79 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 53.2
- **%D:** 62.3
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 8.4 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $201.90 (+0.6% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** -33.1M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $199.93
- **Current vs VWAP:** +1.6%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $214.86
- **Recent Low:** $191.92
- **Next Resistance:** $203.39 (50.0%)
- **Next Support:** $200.68 (61.8%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 211. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:17.744486
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'PGR'}`
- **Output Length:** 461 chars

### Output
```
# Insider Transactions for PGR
# Data retrieved on: 2026-04-29 14:51:17

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Sale:** 1 transactions
- **Total Sales Value:** $199,038
- **Largest Sale (3mo):** 2026-04-16 | JOHNSON DEVIN C (Director) | 980 shares | $199038.0 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-16,JOHNSON DEVIN C,Director,Sale,980,199038.0,D

```

---

## 212. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:17.882199
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'PGR'}`
- **Output Length:** 718 chars

### Output
```
## Analyst Recommendation Trends for PGR

### 2026-04-01
- **Strong Buy**: 3
- **Buy**: 10
- **Hold**: 16
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 30

**Sentiment**: 43.3% Bullish, 3.3% Bearish

### 2026-03-01
- **Strong Buy**: 3
- **Buy**: 10
- **Hold**: 15
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 44.8% Bullish, 3.4% Bearish

### 2026-02-01
- **Strong Buy**: 3
- **Buy**: 10
- **Hold**: 15
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 44.8% Bullish, 3.4% Bearish

### 2026-01-01
- **Strong Buy**: 4
- **Buy**: 10
- **Hold**: 14
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 29

**Sentiment**: 48.3% Bullish, 3.4% Bearish


```

---

## 213. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:18.183176
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'PGR', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 608 chars

### Output
```
## Options Activity for PGR

**Available Expirations:** 15 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 334 | 411 | 1.231 |
| Open Interest | 7,236 | 5,520 | 0.763 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $175.0 | 200 | 200 | 1.0x | 85.6% |

```

---

## 214. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:19.981647
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'CTAS', 'curr_date': '2026-04-29'}`
- **Output Length:** 1529 chars

### Output
```
# Technical Analysis for CTAS
**Date:** 2026-04-29
**Current Price:** $174.22

## Price Action
- **Daily Change:** +0.10%
- **5-Day Change:** -0.25%

## RSI (14)
- **Value:** 42.2 ↑
- **Signal:** Bearish

## MACD
- **MACD Line:** -1.928
- **Signal Line:** -2.522
- **Histogram:** 0.595 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $184.92 (-5.8% from price)
- **200 SMA:** $194.42 (-10.4% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $180.10
- **Middle:** $174.88
- **Lower:** $169.65
- **Band Position:** 44%
- **Bandwidth:** 6.0% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $3.91 (2.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $168.36 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 31.1
- **%D:** 44.4
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 6.0 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $176.60 (-1.3% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -21.8M
- **5-Day Trend:** Rising ↑
- **Signal:** BULLISH DIVERGENCE ⚡ (accumulation)

## VWAP (20-period)
- **VWAP:** $174.23
- **Current vs VWAP:** -0.0%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $209.08
- **Recent Low:** $165.60
- **Next Resistance:** $174.90 (78.6%)
- **Next Support:** $165.60 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 215. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:20.203869
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'CTAS'}`
- **Output Length:** 647 chars

### Output
```
# Insider Transactions for CTAS
# Data retrieved on: 2026-04-29 14:51:20

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Option Exercise:** 2 transactions
- **Sale:** 1 transactions
- **Total Sales Value:** $834,607
- **Largest Sale (3mo):** 2026-04-20 | TYSOE RONALD WILLIAM (Director) | 4666 shares | $834607 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-20,TYSOE RONALD WILLIAM,Director,Sale,4666,834607,D
2026-04-20,TYSOE RONALD WILLIAM,Director,Option Exercise,5500,149050,D
2026-04-09,COLETTI ROBERT E,Director,Option Exercise,12544,339463,D

```

---

## 216. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:20.300652
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

## 217. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:20.721790
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'CTAS', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 424 chars

### Output
```
## Options Activity for CTAS

**Available Expirations:** 14 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 152 | 155 | 1.02 |
| Open Interest | 2,298 | 1,826 | 0.795 |

### Sentiment Analysis
- **Volume P/C Ratio:** Neutral
- **OI P/C Ratio:** Neutral positioning

*No unusual options activity detected.*

```

---

## 218. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:22.516662
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'ULTA', 'curr_date': '2026-04-29'}`
- **Output Length:** 1534 chars

### Output
```
# Technical Analysis for ULTA
**Date:** 2026-04-29
**Current Price:** $538.15

## Price Action
- **Daily Change:** -0.28%
- **5-Day Change:** -4.57%

## RSI (14)
- **Value:** 42.5 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** -4.250
- **Signal Line:** -7.874
- **Histogram:** 3.624 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $583.67 (-7.8% from price)
- **200 SMA:** $566.47 (-5.0% from price)
- **Trend:** Downtrend

## Bollinger Bands (20,2)
- **Upper:** $573.88
- **Middle:** $542.82
- **Lower:** $511.77
- **Band Position:** 42%
- **Bandwidth:** 11.4% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $17.38 (3.2% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $512.08 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 41.7
- **%D:** 60.1
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 15.0 (Strengthening ↑)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $549.51 (-2.1% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -6.0M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $542.63
- **Current vs VWAP:** -0.8%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $714.97
- **Recent Low:** $499.32
- **Next Resistance:** $545.47 (78.6%)
- **Next Support:** $499.32 (100% (Low))

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 219. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:22.858493
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'ULTA'}`
- **Output Length:** 827 chars

### Output
```
# Insider Transactions for ULTA
# Data retrieved on: 2026-04-29 14:51:22

## Summary
- **Insider Sentiment:** BEARISH ⚠️ (significant insider selling)
- **Award/Grant:** 5 transactions
- **Sale:** 1 transactions
- **Total Sales Value:** $243,759
- **Largest Sale (3mo):** 2026-04-07 | MRKONIC GEORGE R JR (Director) | 452 shares | $243759 | Ownership: D

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-07,MRKONIC GEORGE R JR,Director,Sale,452,243759,D
2026-03-31,CASARES RENE GABRIEL,Officer,Award/Grant,1263,0,D
2026-03-31,STEELMAN KECIA,Chief Executive Officer,Award/Grant,10889,0,D
2026-03-31,DELOREFICE CHRISTOPHER,Chief Financial Officer,Award/Grant,4219,0,D
2026-03-31,RYAN ANITA JANE,Officer,Award/Grant,1443,0,D
2026-03-31,LIALIOS CHRISTOPHER,Officer,Award/Grant,519,0,D

```

---

## 220. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:22.971298
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'ULTA'}`
- **Output Length:** 719 chars

### Output
```
## Analyst Recommendation Trends for ULTA

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 16
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 34

**Sentiment**: 67.6% Bullish, 2.9% Bearish

### 2026-03-01
- **Strong Buy**: 6
- **Buy**: 17
- **Hold**: 11
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 65.7% Bullish, 2.9% Bearish

### 2026-02-01
- **Strong Buy**: 6
- **Buy**: 17
- **Hold**: 11
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 35

**Sentiment**: 65.7% Bullish, 2.9% Bearish

### 2026-01-01
- **Strong Buy**: 6
- **Buy**: 16
- **Hold**: 10
- **Sell**: 1
- **Strong Sell**: 0
- **Total Analysts**: 33

**Sentiment**: 66.7% Bullish, 3.0% Bearish


```

---

## 221. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:23.351234
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'ULTA', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 804 chars

### Output
```
## Options Activity for ULTA

**Available Expirations:** 14 dates
**Analyzing:** 2026-05-01, 2026-05-08, 2026-05-15

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 1,061 | 1,505 | 1.418 |
| Open Interest | 3,825 | 4,202 | 1.099 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Neutral positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-05-01 | $492.5 | 191 | 34 | 5.62x | 56.1% |
| PUT | 2026-05-01 | $490.0 | 139 | 40 | 3.48x | 59.0% |
| PUT | 2026-05-01 | $560.0 | 153 | 46 | 3.33x | 51.4% |
| CALL | 2026-05-08 | $540.0 | 134 | 140 | 0.96x | 41.1% |

```

---

## 222. Get technical indicators

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:25.842352
- **Tool:** `get_indicators`
- **Parameters:** `{'symbol': 'TECH', 'curr_date': '2026-04-29'}`
- **Output Length:** 1537 chars

### Output
```
# Technical Analysis for TECH
**Date:** 2026-04-29
**Current Price:** $53.51

## Price Action
- **Daily Change:** -2.90%
- **5-Day Change:** -8.65%

## RSI (14)
- **Value:** 44.0 ↓
- **Signal:** Bearish

## MACD
- **MACD Line:** 0.117
- **Signal Line:** 0.488
- **Histogram:** -0.372 (Strengthening ↑)
- **Signal:** BEARISH CROSSOVER ⚠️

## Moving Averages
- **50 SMA:** $55.03 (-2.8% from price)
- **200 SMA:** $58.00 (-7.7% from price)
- **Trend:** STRONG DOWNTREND ⚠️

## Bollinger Bands (20,2)
- **Upper:** $61.18
- **Middle:** $55.90
- **Lower:** $50.61
- **Band Position:** 27%
- **Bandwidth:** 18.9% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.59 (4.8% of price)
- **Level:** Moderate volatility
- **Suggested Stop-Loss:** $49.62 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 33.3
- **%D:** 45.8
- **Signal:** Bearish

## ADX (Trend Strength)
- **ADX:** 18.5 (Weakening ↓)
- **Interpretation:** WEAK/NO TREND (range-bound) ⚠️

## 20 EMA
- **Value:** $55.70 (-3.9% from price)
- **Signal:** Price BELOW 20 EMA (short-term bearish)

## OBV (On-Balance Volume)
- **Value:** -8.8M
- **5-Day Trend:** Falling ↓
- **Signal:** Confirmed downtrend (price & volume falling)

## VWAP (20-period)
- **VWAP:** $55.72
- **Current vs VWAP:** -4.0%
- **Signal:** Price BELOW VWAP (institutional selling)

## Fibonacci Levels (50-period)
- **Recent High:** $62.47
- **Recent Low:** $48.25
- **Next Resistance:** $53.68 (61.8%)
- **Next Support:** $51.29 (78.6%)

## Summary
- **Key Signals:** Below 50 SMA

```

---

## 223. Get insider transactions

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:26.056651
- **Tool:** `get_insider_transactions`
- **Parameters:** `{'ticker': 'TECH'}`
- **Output Length:** 544 chars

### Output
```
# Insider Transactions for TECH
# Data retrieved on: 2026-04-29 14:51:26

## Summary
- **Insider Sentiment:** Neutral
- **Unknown:** 3 transactions
- **Option Exercise:** 1 transactions

## Transaction Details

Start Date,Insider,Position,Transaction,Shares,Value,Ownership
2026-04-22,HIPPEL JAMES,Chief Financial Officer,Unknown,62000,,D
2026-04-22,KELDERMAN KIM,Chief Executive Officer,Unknown,11344,,D
2026-04-14,KELDERMAN KIM,Chief Executive Officer,Option Exercise,36500,1737400.0,D
2026-04-02,BOHNEN SHANE,General Counsel,Unknown,2259,,D

```

---

## 224. Get recommendations

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:26.178357
- **Tool:** `get_recommendation_trends`
- **Parameters:** `{'ticker': 'TECH'}`
- **Output Length:** 715 chars

### Output
```
## Analyst Recommendation Trends for TECH

### 2026-04-01
- **Strong Buy**: 7
- **Buy**: 10
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 73.9% Bullish, 0.0% Bearish

### 2026-03-01
- **Strong Buy**: 7
- **Buy**: 10
- **Hold**: 6
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 73.9% Bullish, 0.0% Bearish

### 2026-02-01
- **Strong Buy**: 7
- **Buy**: 11
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 78.3% Bullish, 0.0% Bearish

### 2026-01-01
- **Strong Buy**: 7
- **Buy**: 11
- **Hold**: 5
- **Sell**: 0
- **Strong Sell**: 0
- **Total Analysts**: 23

**Sentiment**: 78.3% Bullish, 0.0% Bearish


```

---

## 225. Get options activity

- **Type:** `tool`
- **Node:** filter
- **Timestamp:** 2026-04-29T14:51:26.428221
- **Tool:** `get_options_activity`
- **Parameters:** `{'ticker': 'TECH', 'num_expirations': 3, 'curr_date': '2026-04-29'}`
- **Output Length:** 683 chars

### Output
```
## Options Activity for TECH

**Available Expirations:** 5 dates
**Analyzing:** 2026-05-15, 2026-06-18, 2026-07-17

### Summary
| Metric | Calls | Puts | Put/Call Ratio |
|--------|-------|------|----------------|
| Volume | 37 | 862 | 23.297 |
| Open Interest | 1,205 | 11,942 | 9.91 |

### Sentiment Analysis
- **Volume P/C Ratio:** Bearish (more put volume)
- **OI P/C Ratio:** Bearish positioning

### Unusual Activity (High Volume vs Open Interest)
| Type | Expiry | Strike | Volume | OI | Vol/OI | IV |
|------|--------|--------|--------|----|---------|----|---|
| PUT | 2026-07-17 | $50.0 | 607 | 657 | 0.92x | 66.5% |
| PUT | 2026-07-17 | $55.0 | 136 | 156 | 0.87x | 64.2% |

```

---

## 226. Rank candidates

- **Type:** `llm`
- **Node:** ranker
- **Timestamp:** 2026-04-29T14:53:26.234912
- **Model:** `gemini-3-pro-preview`
- **Prompt Length:** 366656 chars
- **Output Length:** 2000 chars

### Prompt
```
You are a professional stock analyst selecting the best short-term trading opportunities from a pre-filtered candidate list.

CURRENT DATE: 2026-04-29

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
"Historical 1-day win rate: 41.2% (695 tracked)\nHistorical 7-day win rate: 44.8% (645 tracked)\nHistorical 30-day win rate: 39.7% (554 tracked)\n\nBest performing strategies (7-day):\n  - volatility_contraction_breakout: 77.8% win rate, avg +2.3% return (9 samples)\n  - short_squeeze: 64.3% win rate, avg +2.2% return (14 samples)\n  - ml_signal: 57.1% win rate, avg +2.0% return (7 samples)\n\nWORST performing strategies (7-day) \u2014 penalize these heavily in scoring:\n  - pre_earnings_accumulation: 28.6% win rate, avg -1.9% return (7 samples)\n  - social_hype: 18.2% win rate, avg -4.1% return (22 samples)\n  - news_catalyst: 0.0% win rate, avg -9.1% return (8 samples)"

CANDIDATES FOR REVIEW:
### ALMR (Priority: LOW)
- **Strategy Match**: insider_buying
- **Sources**: insider_buying | **Confluence**: 1 source(s)
- **Quant Pre-Score**: 10/100 | **ML Win Probability**: N/A | **Altman Z-Score**: 2.15 | **Piotroski F-Score**: 4
- **Price**: $24.65 | **Current Price (numeric)**: 24.645000457763672 | **Intraday**: -5.6% | **Avg Volume**: 929,267
- **Short Interest**: N/A
- **Discovery Context**: 2 insiders buying ALMR. Largest: 10% Owner Illumina Innovation Fund  purchased +$3,999,998 | ⚠️ Recent move: +12.0% over 7d
- **Business**: Alamar Biosciences, Inc., a proteomics company, develops a proteomic liquid biopsy platform and sells instruments, consumables, and services to early detection of diseases. The company develops NULISA technology, a proteomics platform that detects and quantifies protein biomarkers. It offers ARGO high throughput (HT) System, a fully automated precision proteomics platform, performs ultra-high sensitivity and multiplexed analysis to support biomarker analysis across the continuum of discovery, translational research and ultimately, and diagnostics. The company's offering is utilized in research areas including immunology, neurology, and oncology. It serves global research and academic institutions, biopharmaceutical companies, contract research organizations, and service labs. The company operates in the United States, rest of Europe, Middle East and Africa, Germany, Asia Pacific, and internationally. The company was incorporated in 2018 and is based in Fremont, California.
- **News**: Significant insider buying from the Illumina Innovation Fund was reported, signaling strong institutional support for the precision proteomics firm. The stock has gained approximately 50% from its recent $17.00 IPO price.

**Technical Analysis**:
# Technical Analysis for ALMR
**Date:** 2026-04-29
**Current Price:** $26.11

## Price Action
- **Daily Change:** -2.17%
- **5-Day Change:** +7.58%

## RSI (14)
- **Value:** 86.9 ↓
- **Signal:** OVERBOUGHT ⚠️

## MACD
- **MACD Line:** 0.303
- **Signal Line:** 0.175
- **Histogram:** 0.128 (Weakening ↓)
- **Signal:** Bullish

## Moving Averages
- **50 SMA:** $24.18 (+8.0% from price)
- **200 SMA:** $24.18 (+8.0% from price)
- **Trend:** Uptrend

## Bollinger Bands (20,2)
- **Upper:** $27.63
- **Middle:** $24.18
- **Lower:** $20.72
- **Band Position:** 78%
- **Bandwidth:** 28.6% (volatility indicator)
- **Signal:** Within bands

## ATR (Volatility)
- **ATR:** $2.00 (7.6% of price)
- **Level:** HIGH VOLATILITY ⚠️
- **Suggested Stop-Loss:** $23.12 (1.5x ATR)

## Stochastic (14,3,3)
- **%K:** 80.2
- **%D:** 70.1
- **Signal:** Bullish

## ADX (Trend Strength)
- **ADX:** 89.9 (Strengthening ↑)
- **Interpretation:** VERY STRONG TREND ⚡

## 20 EMA
- **Value:** $24.53 (+6.4% from price)
- **Signal:** Price ABOVE 20 EMA (short-term bullish)

## OBV (On-Balance Volume)
- **Value:** 1.7M
- **5-Day Trend:** Rising ↑
- **Signal:** Confirmed uptrend (price & volume rising)

## VWAP (20-period)
- **VWAP:** $23.22
- **Current vs VWAP:** +12.4%
- **Signal:** Price ABOVE VWAP (institutional buying)

## Fibonacci Levels (50-period)
- **Recent High:** $27.20
- **Recent Low:** $21.50
- **Next Resistance:** $27.20 (0.0% (High))
- **Next Support:** $25.85 (23.6%)

## Summary
- **Key Signals:** RSI overbought, Above 50 SMA


**Fundamentals**: Sector: Healthcare | Industry: Medical Devices | Valuation: Cap: $1.68B, P/B: -4.52, P/S: 22.65 | Profitability: Profit Margin: -40.2%, Oper Margin: -42.2%, ROE: -33.6%, ROA: -13.5% | Financial Health: Current Ratio: 3.99 | EPS: $-0.45 | 52W Range: $21.50 - $27.20

**Insider Transactions**:
No insider transactions data found for symbol 'ALMR'

**Analyst Recommendations**:
No recommendation trends data found for ALMR

**Options Activity**:
No options data available for ALMR

**Upcoming Earnings**: No upcoming earnings within 30 days

### ABT (Priority: CRITICAL)
- **Strategy Match**: insider_buying
- **Sources**: insider_buying | **Confluence**: 1 source(s)
- **Quant Pre-Score**: 35/100 | **ML Win Probability**: 41.0% (Predicted: NOT-WIN) | **Altman Z-Score**: 4.62 | **Piotroski F-Score**: 5
- **Price**: $93.86 | **Current Price (numeric)**: 93.86000061035156 | **Intraday**: +1.1% | **Avg Volume**: 13,778,494
- **Short Interest**: 1.4%
- **Discovery Context**: 2 insiders buying ABT. Largest: Dir Starks Daniel J purchased +$926,537
- **Business**: Abbott Laboratories, together with its subsidiaries, discovers, develops, manufactures, and sells health care products worldwide. It operates in four segments: Established Pharmaceutical Products, Diagnostic Products, Nutritional Products, and Medical Devices. The company offers generic pharmaceuticals for the treatment of pancreatic exocrine insufficiency, irritable bowel syndrome or biliary spasm, intrahepatic cholestasis or depressive symptoms, gynecological disorder, hormone replacement therapy, dyslipidemia, hypertension, hypothyroidism, hypertriglyceridemia, Ménière's disease and vestibular vertigo, pain, fever, inflammation, and migraine, as well as provides anti-infective clarithromycin, influenza vaccine, and products to regulate physiological rhythm of the colon. It also provides laboratory and transfusion medicine systems in the areas of immunoassay, clinical chemistry, hematology, and transfusion serology testing; molecular diagnostics polymerase chain reaction instrument systems that automate the extraction, purification, and preparation of DNA and RNA from patient samples, and detect and measure infectious agents; point of care systems; cartridges for testing blood gas, chemistry, electrolytes, coagulation, and immunoassay; rapid diagnostics lateral flow testing products; molecular point-of-care testing for HIV, SARS-CoV-2, influenza A an... [truncated]
```

### Output
```
```json
{
  "rankings": [
    {
      "rank": 1,
      "ticker": "ARXS",
      "company_name": "Arxis, Inc.",
      "current_price": 34.974998474121094,
      "strategy_match": "insider_buying",
      "final_score": 85,
      "confidence": 8,
      "risk_level": "moderate",
      "reason": "A massive cluster of 7 executives purchased $5.64M worth of stock over the last 14 days, signaling extreme internal confidence. The stock is technically oversold with a 14-day RSI of 21.6, and the On-Balance Volume (OBV) shows a bullish divergence by rising despite recent price weakness. Buying near the $34.45 52-week low provides a tight stop-loss level, offering an asymmetric risk/reward profile. The primary risk is that the stock remains in a technical downtrend below its 50-day SMA, so continued broad market weakness could delay the reversal.",
      "description": "Arxis, Inc. designs and manufactures electronic and mechanical components for mission critical applications in the defense, aerospace, and industrial sectors."
    },
    {
      "rank": 2,
      "ticker": "MPC",
      "company_name": "Marathon Petroleum Corporation",
      "current_price": 232.58999633789062,
      "strategy_match": "volatility_contraction_breakout",
      "final_score": 82,
      "confidence": 8,
      "risk_level": "low",
      "reason": "Marathon Petroleum displays a textbook volatility contraction breakout with the ATR ratio compressed to 0.70 and the price pushing above its 10-day high. Options flow is exceptionally bullish, highlighted by a staggering 0.075 P/C ratio and heavy volume of 2,879 contracts on the $250.0 call strike. The stock is in a strong uptrend above its 50-day SMA ($223.26), providing solid structural support for a continuation move. Earnings are scheduled in 6 days, introducing binary event risk that could gap the stock against the technical setup.",
      "description": "Marathon Petroleum Corporation operates as an integrated downstream energy company, refining crude oi
```

---

