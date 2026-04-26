---
name: earnings_beat
description: Post-Earnings Announcement Drift (PEAD) — Recent EPS beats that exhibit multi-day drift upward
type: scanner
---

# Earnings Beat Scanner (PEAD)

## Current Understanding

Identifies stocks that have recently beaten earnings estimates and exhibit post-earnings announcement drift (PEAD). Academic backing (Bernard & Thomas 1989, QuantPedia backtests) shows 15% annualized returns on PEAD plays, particularly for small-to-mid caps with >10% surprise. Data source: Finnhub earnings calendar API with real-time lookback (past 14 days by default).

Unlike `earnings_calendar` (which surfaces *upcoming* earnings events), `earnings_beat` captures the drift *after* a beat has occurred, making it a distinct signal with a different holding horizon (7-14d vs event-day trading).

## Evidence Log

### 2026-04-22 — Fast-loop (first live appearance)
- NTST (rank 4, score=76, conf=7): EPS surprise +376.2% (massive); post-beat price momentum at ADX=22.8, price +5.7% above VWAP, analyst sentiment 80% bullish.
- This is an extreme surprise case — >20% threshold, should be CRITICAL priority.
- Thesis is specific and quantifiable: surprise % is measurable and concrete. Score/10=7.6 vs conf=7 (Δ=0.6) — acceptable calibration.
- Earnings momentum is strong; early indicators suggest the PEAD thesis is active (price drifting upward post-earnings).
- Confidence: low (single data point; outcome tracking needed)

### 2026-04-24 — Fast-loop (multiple new appearances, mature observation)
- CCI (rank 1, score=88, conf=9): Q1 EPS +161.5% surprise, $1B buyback announced, MACD bullish crossover, ATR $2.49 (2.8% low volatility). Risk floor established by buyback.
- DLR (rank 3, score=82, conf=8): Core FFO +325.3% surprise, 200-MW AI inference lease (largest ever). Early PEAD stage, price +9.8% above 50 SMA, RSI 72.2 (slight consolidation risk).
- PECO (rank 4, score=78, conf=8): EPS +303.7% surprise, 2026 guidance raised. PEAD window open (low market reaction), RSI 59.0, ATR $0.69 (1.8% low volatility).
- TAL (rank 4 in Apr 23, score=80, conf=8): EPS +185.2% surprise, 31.5% revenue surge on AI demand. OBV bullish divergence, price +0.9% above 50 SMA.
- Pattern: PEAD scanner is showing **extreme specificity** — massive surprises (>160%), concrete catalysts (buybacks, AI demand), low-noise scoring. No false positives detected.
- All candidates have surprise >150% — far above the 10% academic threshold. These are CRITICAL situations, not marginal beats.
- Confidence: medium (new scanner, outcomes not yet measured, but thesis quality and surprise magnitude are exceptional)

## Pending Hypotheses
- [ ] Does >300% surprise vs <200% surprise predict better 7d outcomes?
- [ ] Does PEAD drift extend beyond 7 days, or does momentum fade by day 7?
- [ ] Does combining earnings_beat with analyst_upgrades (same-day upgrade) amplify outcomes?
- [ ] Does earnings_beat outperform earnings_calendar on 7d horizon (drift vs event-day binary)?
