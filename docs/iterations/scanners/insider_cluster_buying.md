---
name: insider_cluster_buying
description: Insider cluster buying — 3+ executives buying within 14-day window (2× returns vs solo buys)
type: scanner
---

# Insider Cluster Buying Scanner

## Current Understanding

Identifies insider buying clusters where 3 or more executives purchase shares within a rolling 14-day window. Academic evidence (Alldredge & Blank 2019, Kang et al. 2018) shows cluster buys generate 2.1–3.8% abnormal returns over 1-3 months — nearly **2× the returns of solo insider purchases**. This scanner leverages existing SEC Form 4 data infrastructure (reuses insider_buying parser) and adds cluster aggregation logic to detect board-level conviction signals.

Distinction: `insider_buying` surfaces individual transactions; `insider_cluster_buying` surfaces clusters (3+ within 14d), providing a higher-conviction filter that eliminates noise from routine buyback programs.

## Evidence Log

### 2026-04-22 — Fast-loop (first live appearance)
- CGTX (rank 3, score=78, conf=8): Insider cluster with CEO, CFO, and one director; 29,175 shares purchased at avg $1.11; stock subsequently rallied with MACD bullish crossover.
- Thesis is highly specific: executive roster is identified, share count/price are concrete, board-level consensus is clear.
- Calibration: score=78/10=7.8 vs conf=8 (Δ=0.2) — excellent calibration.
- As a micro-cap (9.1% ATR), volatility is high; early technical momentum (bullish MACD, rising OBV) is encouraging.
- Confidence: low (first appearance; outcome data needed; micro-cap volatility is high)

## Pending Hypotheses
- [ ] Does CEO+CFO cluster outperform director-only clusters?
- [ ] Does cluster size (3 vs 4+ insiders) predict stronger outcomes?
- [ ] Is the 14-day clustering window optimal, or does a shorter window (7d) improve precision?
- [ ] Does clustering predict better long-term (30d) outcomes than insider_buying alone?
