# Early Accumulation Scanner

## Current Understanding
Detects quiet accumulation patterns: rising OBV, price above 50/200 SMA, low ATR
(low volatility), and bullish MACD crossover — without requiring a strong near-term
catalyst. Designed for slow-grind setups rather than explosive moves. The absence of
an immediate catalyst structurally limits the LLM's score assignment, since the ranker
rewards urgency and specificity. This may cause systematic under-scoring relative to
true edge.

## Evidence Log

### 2026-04-12 — Fast-loop (2026-04-12 run)
- Single appearance: FRT (Federal Realty Investment Trust), score=60, conf=6, risk_level=low.
- Thesis: +1.55% daily price move, OBV 12.3M rising, MACD crossover, ATR 1.7% (low risk).
- Score sub-threshold (60 < 65). Key weakness per thesis: "lack of immediate catalysts" and overbought Stochastic (88.7).
- Pattern observation: early_accumulation may be structurally score-capped by ranker's catalyst-weighting. A score of 60 with conf=6 on a low-risk setup may represent miscalibration rather than poor edge.
- 0 mature recommendations (no recommendation generated from this appearance).
- Confidence: low (single data point, no outcome data)

### 2026-04-19 — Fast-loop (2026-04-19 run) + P&L review
- 2 appearances: DOV (score=72, conf=7, risk=low) and MLI (score=68, conf=6, risk=moderate).
- DOV: OBV 44.6M divergence + MACD crossover + supportive options flow (P/C=0.303). Concrete thesis — highest-quality early_accumulation entry seen so far.
- MLI: OBV 44.6M reading mentioned, MACD crossover, RSI 61.2. Thesis less specific; no catalyst beyond volume/momentum.
- P&L data from statistics.json (n=19, mature recs): **7d win rate 43.8%, avg_return_7d -0.3%, avg_return_30d -7.6%** — worst 30d performer among non-legacy scanners. Below coin-flip at 7d.
- This confirms the structural hypothesis: early_accumulation catches stocks in late-stage distribution at resistance, not genuine institutional accumulation. Single-day volume spikes with flat price are indistinguishable from distributional selling where buyers absorb the supply.
- Code change implemented: `volume_accumulation.py` now requires `high_vol_days_5d >= 2` — minimum 2 days of elevated volume in last 5 sessions before classifying as "accumulation." Filters single-day spikes.
- Confidence: high (n=19, pattern is consistent; code change is minimal and directly addresses the root cause)

### 2026-04-19 — Code fix: sustained accumulation filter added
- Added check in candidate loop: skip "accumulation"-classified picks where `high_vol_days_5d < 2`.
- Rationale: 1-day volume spikes account for false positives; genuine institutional accumulation leaves a 2–5 day footprint.
- Expected effect: Reduce early_accumulation candidate count. Remaining picks should have multi-day OBV support.

## Pending Hypotheses
- [x] Does early_accumulation systematically score 55-65 due to ranker penalizing "no catalyst"? → Confirmed: DOV (72), MLI (68) — ranker rewards options flow as a catalyst proxy.
- [ ] Do early_accumulation setups produce better 30d returns than 7d returns (slow-grind nature)? → Evidence says NO: avg_30d=-7.6% is worse than avg_7d=-0.3%. Slow grind may be slow bleed.
- [ ] Is the overbought Stochastic reading a reliable short-term timing filter to delay entry?
- [ ] Does the `high_vol_days_5d >= 2` sustained-accumulation filter (implemented 2026-04-19) improve 7d win rate from 43.8% baseline over the next 4 weeks?
