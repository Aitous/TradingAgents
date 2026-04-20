---
name: high_52w_breakout
description: 52-week high breakout scanner — momentum continuation after price crosses annual high with volume confirmation
type: project
---

# High 52-Week Breakout Scanner

## Current Understanding
Identifies stocks crossing their 52-week high on ≥1.5x average volume. Based on George & Hwang (2004) anchor-price framework: 52w high acts as a resistance anchor; once crossed with volume, institutions that held back (waiting for confirmation) pile in, creating momentum continuation. Expected holding period: 10–31 days.

Scanner appears to be producing high-quality, high-specificity picks: both Apr 18 appearances came with concrete earnings-beat catalysts (not just technical), suggesting the ranker correctly upgrades breakout stories when fundamental support is present.

No outcome data yet (count=4 in statistics.json, 0 7d wins/losses measured).

## Evidence Log

### 2026-04-18 — Fast-loop (2026-04-18 run)
- 2 appearances: JBHT (rank 1, score=92, conf=9) and BK (rank 5, score=80, conf=7).
- JBHT: Q1 EPS beat +27% YoY to $1.49, stock +3.5% above 20-day high on 2.9x volume, OBV 9.5M rising. Highly specific thesis with dual catalyst (earnings + technical breakout). Calibration: score/10=9.2 vs conf=9 (Δ=0.2) — excellent.
- BK: Q1 $2.24 EPS beat + new $10B buyback + RSI 79.5 (slightly extended). Score/10=8.0 vs conf=7 (Δ=1.0) — slight overcalibration; RSI extension acknowledged in reason.
- Both picks have concrete fundamental catalysts alongside the technical breakout — the scanner may be systematically co-selecting with earnings_play during earnings season.
- Confidence: low (n=2 fast-loop observations, no outcome data)

## Pending Hypotheses
- [ ] Does high_52w_breakout consistently appear during earnings season (April, July) because EPS beats trigger the cross? If so, performance may be seasonally concentrated.
- [ ] Does high_52w_breakout + earnings_beat (dual signal) outperform pure 52w-high crosses?
- [ ] Is RSI >75 at breakout a reliable overextension filter to skip or reduce size?
