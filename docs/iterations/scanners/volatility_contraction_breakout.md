---
name: volatility_contraction_breakout
description: ATR compression breakout scanner — volatility squeeze resolving upward through 10-day high
type: project
---

# Volatility Contraction Breakout Scanner

## Current Understanding
Identifies stocks where ATR(5)/ATR(20) ≤ 0.75 (short-term volatility compressed vs 4-week baseline) AND price has broken above its 10-day high (upward resolution). Trend filter: price above SMA(50). Implemented as `atr_compression.py`. Expected holding period: 5–15 days.

Scanner has shown high output volume in early runs: 4 picks in a single day (Apr 19) out of 8 total — 50% of the final ranking. This may reflect that the ATR threshold (0.75) is generous enough to catch many setups simultaneously during low-volatility market regimes. Quality of individual theses looks good; the scanner produces specific ATR ratios and MACD crossovers.

No mature outcome data yet (count=5 in statistics.json, no 7d wins/losses measured — all too recent).

## Evidence Log

### 2026-04-18 — Fast-loop (2026-04-18 run)
- 1 appearance: PSTG (rank 6, score=77, conf=7). ATR compression ratio 0.74, +2.8% above 10-day high, MACD bullish crossover, OBV 51.2M rising. Thesis is specific and well-calibrated (score/10=7.7 vs conf=7, Δ=0.7).
- Confidence: low (single observation)

### 2026-04-19 — Fast-loop (2026-04-19 run)
- 4 appearances: CAVA (rank 1, score=82), LSCC (rank 3, score=78), PSTG (rank 4, score=76), QRVO (rank 7, score=70). Scanners producing 4 of 8 final picks (50%) in a single run.
- CAVA: ATR ratio 0.73, includes unusual options flow (8 calls vs 1 put, P/C=0.396) — multi-signal confluence. Strongest thesis.
- LSCC: ATR ratio 0.74, already +15.5% over 7 days — this is **post-breakout exhaustion risk**, not pre-breakout compression. Ranker flagged this risk in the reason. Worth monitoring.
- PSTG: Second consecutive day (also appeared Apr 18). Identical ATR ratio (0.74). The breakout did not resolve overnight — either price stalled at the 10-day high or the threshold keeps catching it. Cross-day persistence without thesis change = possible stale signal.
- QRVO: Score=70, conf=6 (lowest in run). Weak ADX (10.8) flagged in reason — "requires immediate volume confirmation." This is a marginal setup.
- Pattern: In low-volatility regimes, many stocks simultaneously satisfy ATR < 0.75 → scanner produces high-volume output. Need to watch whether this continues.
- Confidence: low (no outcome data; pattern observed over 2 runs only)

### 2026-04-20 — Fast-loop (4 runs)
- Appeared across all 4 runs: CORT (run_10, score=72), MTSI (run_14, score=68), Q (run_15_19, score=68), FDS (run_15_27, score=81), PWR (run_15_27, score=80).
- FDS (ratio=0.67) and PWR (ratio=0.70) in run_15_27: specific ATR ratios, appropriate scores (80-81) — quality setups.
- Q (score=68, conf=6) and MTSI (score=68, conf=7): both borderline — consistent with prior observation that score ≥ 75 separates quality from marginal.
- Scanner pick rate is healthy (1-2 per run), not dominating like ml_signal. Volume pattern has stabilized.
- Confidence: medium

## Pending Hypotheses
- [ ] Does PSTG cross-day persistence (same ATR ratio day 2) represent a valid continuing signal or stale detection? Check if breakout price was actually broken on Apr 18.
- [ ] Does LSCC-style "already-rallied 15%+ before detection" produce worse outcomes than stocks at compression inflection point?
- [ ] Does ATR < 0.70 (tighter threshold) produce better outcomes than 0.70–0.75?
- [ ] Does score ≥ 75 filter out marginal setups (e.g., QRVO score=70, weak ADX)?
- [ ] Does volatility_contraction_breakout + options_flow confluence (like CAVA) produce better 7d returns?
