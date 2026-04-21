# Momentum Scanner

> ⚠️ AUTOPSY TRIGGERED: WR < 45% over 30+ picks. Must revise thresholds or disable within 14 days (deadline: 2026-05-05).

## Current Understanding
Large-volume scanner (136 recommendations with 7d outcome data) with a 7d win rate of **39.7%** and
avg 7d return of **-0.80%** — well below random. 30d performance is worse: 35.3% WR, -2.49% avg.
With n=136 this is statistically robust, not noise.

The scanner is actively degrading portfolio performance. Immediate action needed: either identify
a structural fix (add a selectivity filter) or disable.

No domain file existed before this run — the scanner has been operating without documented
understanding or hypothesis tracking.

## Evidence Log

### 2026-04-21 — P&L autopsy (n=136 7d outcomes, n=136 30d outcomes)
- 7d win rate: 39.7%, avg return: -0.80%. 30d win rate: 35.3%, avg return: -2.49%.
- **AUTOPSY TRIGGERED**: WR-7d < 45% with n=136 (statistically robust — not noise).
- This scanner is the worst large-sample performer in the pipeline.
- No prior domain file existed — operating entirely without learning documentation.
- Confidence: high (n=136 is largest outcome sample in the pipeline)

### 2026-04-21 — Confluence analysis
- **insider_buying + momentum confluence: 74.3% WR (n=35)** vs insider_buying alone 47.7% (+26.6 pts lift).
- **momentum + options_flow confluence: 59.1% WR (n=22)** vs options_flow alone 46.8% (+12.3 pts lift).
- Implication: momentum may have no standalone edge but acts as a strong *confirming* signal. Disabling it entirely would destroy the confluence lift. A better approach may be to use momentum only in confluence mode — filter momentum picks to those that also appear in another scanner within 3 days.
- Confidence: medium (n=35 and n=22 are reasonable samples but not large; could be noise)

## Pending Hypotheses
- [ ] What is the momentum scanner's signal logic? Read `tradingagents/dataflows/discovery/scanners/` to understand what it's computing.
- [ ] Is the 39.7% WR driven by a specific sub-period (e.g., March 2026 correction) or consistent throughout?
- [ ] Would adding a trend filter (price > SMA200) reduce the pick rate and improve precision?
- [ ] Should the scanner be disabled until a root cause is identified? Compare runs with vs without it enabled.
