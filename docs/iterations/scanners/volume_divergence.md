---
name: volume_divergence
description: OBV divergence scanner — multi-week institutional accumulation detected by OBV rising while price is flat/falling
type: project
---

# Volume Divergence Scanner (OBV Divergence)

## Current Understanding
Identifies stocks where On-Balance Volume (OBV) has risen ≥8% over a 20-day lookback while price change is ≤2% — signaling covert multi-week institutional accumulation. Implemented as `obv_divergence.py`. Distinct from `volume_accumulation` (single-day spike) — this catches multi-week sustained patterns.

Academic basis: OBV divergence (rising OBV during price consolidation) qualitative win rate ~68% per research. Expected holding period: multi-week (10–30d). Pipeline registered as `volume_divergence` in statistics.json.

Statistics (count=10): 60% 1d win rate (3/5 obs), no 7d/30d data yet — scanner is too new for 7d outcomes to have settled.

## Evidence Log

### 2026-04-18 — Fast-loop (2026-04-18 run)
- 1 appearance: EA (rank 7, score=74, conf=7). OBV +24.2% above average, price flat (+1.6% over 20 days). MACD bullish crossover, ATR $0.98 (very low — low-risk setup). Thesis is highly specific with concrete OBV percentage. Score/10=7.4 vs conf=7 (Δ=0.4) — excellent calibration.
- EA is a "slow-grind" accumulation candidate. 7d return likely to be small-positive if it works. The very low ATR suggests this is a quiet institutional play, not a momentum burst.
- Note: EA also appeared in Apr 15 recs (EQR, not EA — EQR was volume_divergence Apr 15, scored=75). The scanner is finding these regularly.
- Confidence: low (fast-loop quality looks good; outcome data pending)

## Pending Hypotheses
- [ ] Does volume_divergence produce better 30d returns than 7d (slow-grind nature)?
- [ ] Is a low ATR a reliable filter to identify the best volume_divergence setups (indicating quiet accumulation rather than noise)?
- [ ] Does OBV divergence + MACD crossover (simultaneous) produce better outcomes than either alone?
