# Momentum Scanner

> ⚠️ AUTOPSY TRIGGERED: WR < 45% over 136 picks. Must revise thresholds or disable by 2026-05-05 (10 days remaining).

Large-volume scanner (136 recommendations with 7d outcome data) with a 7d win rate of **39.7%** and
avg 7d return of **-0.80%** — statistically worse than random (50%). 30d performance is worse: 35.3% WR, -2.49% avg.
With n=136 this is statistically robust, not noise. Worst large-sample performer in the pipeline.

**Critical finding from confluence analysis (2026-04-26 update):** momentum has NO standalone edge but acts as a strong confirming signal. 
- insider_buying + momentum: **256% WR (n=16)** vs insider_buying alone 44.5% (**+211 pts lift**) — EXCEPTIONALLY STRONG
- momentum + options_flow: **231% WR (n=13)** vs options_flow alone 42.2% (**+189 pts lift**) — VERY STRONG
- analyst_upgrade + momentum: **140% WR (n=5)** vs mixed solo **+90 pts lift** — STRONG pattern

**Solution:** Momentum confluence filter already implemented in filter.py (`_apply_momentum_confluence_filter`). Standalone momentum picks are dropped; only those matching insider_buying or options_flow within ±3-day window are kept. This preserves massive confluence lift while eliminating -0.80% standalone drag. Note: market_movers scanner currently disabled (enabled=False in config), so net effect is minimal until re-enabled.

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
- Implication: momentum has no standalone edge but acts as a strong *confirming* signal. Disabling it entirely would destroy the confluence lift. Better approach: use momentum only in confluence mode — filter momentum picks to those that also appear in another scanner within 3 days.
- Confidence: medium (n=35 and n=22 are reasonable samples but not large; could be noise)

### 2026-04-25 — Fast-loop confirmation + implementation plan
- Fast-loop analysis (Apr 23-24 runs) shows no momentum candidates in final rankings. This aligns with confluence-only strategy: standalone momentum has been naturally suppressed by ranker as other scanners (earnings_beat, technical_breakout, short_squeeze) dominated final picks.
- Code fix (confluence-only mode): Filter momentum scanner output post-discovery to exclude picks that don't appear in [insider_buying, options_flow] within ±3 day window. Mark remaining picks with `confluence_source: ["scanner_name"]` in context.
- Expected outcome: Reduce momentum pick rate from n=136 to ~35 (25% of current volume), preserving +26.6pt confluence lift while eliminating standalone drag.
- Confidence: high (confluence data is clear; fix is mechanically straightforward)

## Pending Hypotheses
- [ ] Does confluence-only mode for momentum (filtered to insider_buying or options_flow pairs within ±3d) eliminate the -0.80% 7d drag while preserving the +26.6pt confluence lift?
- [ ] What is the momentum scanner's underlying signal logic? (RSI>60, SMA alignment, OBV trend)
