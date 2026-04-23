# Analyst Upgrades Scanner

## Current Understanding
Detects analyst upgrades/price target increases. Most reliable when upgrade comes
from a top-tier firm (Goldman, Morgan Stanley, JPMorgan) and represents a meaningful
target increase (>15%). Short squeeze potential (high short interest) combined with
an upgrade is a historically strong setup.

## Evidence Log

### 2026-04-12 — P&L review + fast-loop
- 36 tracked recommendations (mature). Win rates: 38.2% 1d, 50.0% 7d, 30.4% 30d. Avg returns: +0.13% 1d, -0.75% 7d, -3.64% 30d.
- 7d win rate of 50% is close to coin-flip; 30d degrades sharply.
- Recent runs (Apr 6-12): 7 candidates — LRN, SEZL, NTWK, CSCO, NFLX, DLR, INTC. INTC Apr 12 (score=85) had a strong catalyst (Terafab + Apple rumor), which is a genuine material catalyst, fitting the "already priced in" concern.
- CSCO appeared in analyst_upgrade (Apr 8) AND options_flow (Apr 6, Apr 9) — cross-scanner confluence is a positive quality signal.
- Confidence calibration: Good (cal_diff ≤ 0.5 across all recent instances).
- Confidence: medium (36 samples, 7d win rate at breakeven)

### 2026-04-17 — P&L update (n=39 total, statistics.json)
- 7d win rate improved from 50.0% → 55.9%, avg_return_7d improved from -0.75% → +0.18%. First time this scanner shows positive average return at any horizon.
- 30d win rate still poor: 30.4%, avg_return_30d -3.64%. Edge is firmly short-term only.
- New picks: DLR (Apr 10, score=75, pre-earnings institutional accumulation), INTC (Apr 12, score=85, Terafab + Apple rumor catalyst). Both high-conviction entries.
- INTC Apr 12 (score=85) is a strong-catalyst case — "buy the rumor" risk noted in reason. Will be a key data point.
- The 7d improvement is meaningful: now above coin-flip with positive avg return. Previously marginal (50%, negative avg).
- Confidence: medium (39 samples; 7d improvement may partly reflect April market recovery, not scanner quality alone)

### 2026-04-18 — Fast-loop (2026-04-18 run)
- 2 appearances: NI (rank 3, score=85, conf=8) and PLD (rank 4, score=83, conf=8).
- NI: Google data center energy agreement catalyst — fundamentally reprices utility as AI infrastructure play. Specific, mechanistic catalyst with clear price event (+1.24% intraday, MACD bullish crossover). Excellent quality.
- PLD: Core FFO beat + record 64M sq ft leasing + $10B buyback. RSI=74.6 noted as slight extension. Thesis is specific but RSI proximity to overbought reduces near-term upside.
- Both picks are high-quality with concrete catalysts. Calibration: NI 8.5/8 (Δ=0.5), PLD 8.3/8 (Δ=0.3) — good.
- Overall statistics (n=44): 55.9% 7d win rate, avg_return_7d=+0.18%. Scanner continues its upward trend.
- Pattern confirmed: NI is a strong example of analyst_upgrade's thesis — a non-obvious re-rating event (utility → AI play) that institutional algos may not fully price within 1-3 days.
- Confidence: medium (Apr 18 picks look high quality; outcomes will confirm or deny)

### 2026-04-22 — Fast-loop (2026-04-22 run)
- TFC (rank 5, score=75, conf=7): Analyst upgrade to Buy; stock 5.4% above 50 SMA, ADX=30.7 (strong trend), RSI=67.6 (not extended). OBV +207.8M (heavy institutional buying).
- Catalyst is explicit and recent; technical position is solid without overextension.
- Calibration: 75/10=7.5 vs conf=7 (Δ=0.5) — good.
- This is a clean, moderate-conviction play in established uptrend with fresh catalyst.
- Confidence: low (single data point; fits expected high-quality profile for analyst_upgrade)

## Pending Hypotheses
- [ ] Does analyst tier (BB firm vs boutique) predict upgrade quality?
- [ ] Does short interest >20% combined with an upgrade produce outsized moves?
- [ ] Does cross-scanner confluence (analyst_upgrade + options_flow on same ticker) predict higher 7d returns?
