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

## Pending Hypotheses
- [ ] Does requiring volume confirmation on the breakout day reduce false positives?
- [ ] What volume multiple (1.5x vs 2.0x vs 2.5x) best identifies high-probability breakouts?
- [ ] Does breakout height relative to base size predict hold duration or upside potential?
