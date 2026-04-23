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

## Pending Hypotheses
- [ ] Does earnings_beat show better outcomes in small-cap vs large-cap tickers?
- [ ] Is surprise % >=20% a stronger signal than 5-10% surprises?
- [ ] Does PEAD drift extend beyond 7 days, or does momentum fade by day 7?
- [ ] Does combining earnings_beat with analyst_upgrades (same-day upgrade) amplify outcomes?
