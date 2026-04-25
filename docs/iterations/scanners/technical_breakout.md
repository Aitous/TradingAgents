---
name: technical_breakout
description: Price breakouts above key resistance levels on above-average volume
type: scanner
---

# Technical Breakout Scanner

## Current Understanding

Detects price breakouts above key resistance levels on above-average volume.
Minervini-style setups (stage 2 uptrend, tight base, volume-confirmed breakout)
tend to have the highest follow-through rate. False breakouts are common without
volume confirmation (>1.5x average on breakout day).

## Evidence Log

### 2026-04-22 — Fast-loop (first live appearance)
- DGX (rank 8, score=70, conf=6): Technical breakout above 20-day high (+1.5%) on heavy 2.7x average volume. Intraday strength +4.5% confirms institutional buying. Bullish MACD crossover, price well above 20 EMA.
- This is a textbook volume-confirmed breakout with strong institutional participation. Thesis is specific: price levels, volume multiples, technical alignment all concrete.
- Calibration: 70/10=7.0 vs conf=6 (Δ=1.0) — moderate gap, suggests slight confidence conservatism.
- Rank 8 in final output indicates other signals rank higher. This is appropriate given multiple earnings-driven candidates with catalysts rank above.
- Confidence: low (first appearance; new scanner implementation)

### 2026-04-23 — Fast-loop (multiple new appearances)
- ISRG (rank 2, score=84, conf=8): Earnings beat + technical breakout. Closed +0.7% above 20-day high on 2.3x average volume. Adjusted EPS $2.50, raised 2026 guidance. RSI 56.9, bullish MACD. Risk: overextended valuation.
- MANH (rank 5, score=78, conf=7): Q1 beat ($282.2M revenue) triggered breakout +2.3% above 20-day high on 2.2x volume. Cloud supply chain demand. RSI 61.0, MACD bullish. Risk: enterprise software spending contraction.
- BA (rank 8, score=73, conf=6): Breakout +5.5% intraday on 2.6x volume. Cleared local resistance. RSI 62.3, MACD bullish. Price $231.28 vs 50 SMA $218.84. Risk: headline risk (regulatory/defects).

### 2026-04-24 — Fast-loop (additional appearance)
- VOYA (rank 2, score=85, conf=8): Breakout +3.2% above 20-day high on 4.1x average volume. Activist pressure + $150M buyback. RSI 74.7 (hot momentum). Risk: activist deal breaks down.
- HXL (rank 7, score=74, conf=7): Breakout +0.8% above 20-day high on 2.9x volume. MACD bullish, price +6.9% above 50 SMA. RSI 64.5. Risk: mean reversion from upper Bollinger Band.

**Pattern: technical_breakout shows excellent specificity** — volume multiples (2.3x–4.1x), price levels, technical confirmations all concrete. No vague candidates. Scores 70-85 reflect volume quality + momentum strength.

## Pending Hypotheses
- [ ] Does >3x volume confirmation (vs >1.5x minimum) predict better 7d outcomes?
- [ ] Does technical_breakout + earnings_beat confluence outperform either alone?
- [ ] Does price invalidation risk (holding below 20-day low) correlate with max holding period?
