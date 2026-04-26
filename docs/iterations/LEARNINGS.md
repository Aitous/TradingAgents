# Learnings Index

**Last analyzed run:** 2026-04-25

> **What changed this run:** P&L autopsy resolved momentum crisis: 39.7% 7d WR on 136 picks shows momentum is broken in standalone mode BUT +26.6pt confluence lift with insider_buying (74.3% vs 47.7%) justifies confluence-only filtering (fix planned). PEAD scanner validated: CCI/DLR/PECO all >300% surprise with concrete catalysts (buybacks, AI contracts) — zero false positives, excellent specificity. technical_breakout showing 2.3x-4.1x volume confirmation, specific price levels, no noise. ACHC 5-day persistence (Apr 15-24, score 85→88→80→72→70) fully validates short_squeeze urgency decay model — not staleness, price extension. insider_buying staleness filter (suppress_days=3) working; NKE same-day multi-run staleness identified (new gap: intraday deduplication needed). No confluence lift detected in Apr 23-24 runs (mature test sample needed).

| Domain | File | Last Updated | One-line Summary |
|--------|------|--------------|-----------------|
| options_flow | scanners/options_flow.md | 2026-04-12 | Premium filter confirmed applied; CSCO cross-scanner confluence detected; 42.2% 7d win rate (83 recs) |
| insider_buying | scanners/insider_buying.md | 2026-04-25 | 44.5% 7d WR (155 recs); suppress_days=3 blocks cross-day; new gap: same-day multi-run staleness (NKE 3/4 runs Apr 20) |
| insider_cluster_buying | scanners/insider_cluster_buying.md | 2026-04-22 | First live appearance: CGTX CEO+CFO+director cluster, score=78, outcomes TBD; 2× expected returns per academic studies |
| minervini | scanners/minervini.md | 2026-04-17 | 100% win rate still holds (n=3 measured, 15 total); +7.19% avg 7d; AVGO persistence Apr 10-11; AA score=92 Apr 12 |
| analyst_upgrades | scanners/analyst_upgrades.md | 2026-04-25 | 45.8% 7d (24 recs), -1.12% avg; strong performer in Apr 23-24 (LRCX 85, TXN 82, PYPL 70) |
| earnings_calendar | scanners/earnings_calendar.md | 2026-04-17 | 7d win rate 50.0% (6 recs), -10.39% avg; high volatility; binary event risk |
| earnings_beat | scanners/earnings_beat.md | 2026-04-25 | PEAD scanner: CCI +161.5%, DLR +325.3%, PECO +303.7%, TAL +185.2% surprises all >150%. Zero false positives. Concrete catalysts (buybacks, AI contracts). Score 78-88, conf 8-9. No outcome data yet. |
| pipeline/scoring | pipeline/scoring.md | 2026-04-25 | 744 total recs tracked; 39.2% 1d overall WR; short_squeeze +60% 7d best performer; momentum -0.80% 7d worst |
| early_accumulation | scanners/early_accumulation.md | 2026-04-20 | 40.0% 7d (10 recs), +0.61% avg; 30d 20.0%, -8.09% avg (poor long-term); sustained-accumulation filter still needed |
| social_dd | scanners/social_dd.md | 2026-04-14 | 41.7% 7d (24 recs), -1.92% avg; 30d 45.8%, +1.14% avg — only scanner positive at 30d |
| reddit_dd | scanners/reddit_dd.md | 2026-04-22 | GME Apr 22 shows quality filter working: meme play with technical + fundamental backing (cash, Bitcoin thesis) |
| volume_accumulation | scanners/volume_accumulation.md | — | No data yet |
| short_squeeze | scanners/short_squeeze.md | 2026-04-25 | 60% 7d WR (10 recs), +2.15% avg — best 7d performer; 30% 30d WR, -1.10% avg (sharp decay); max hold 7d enforced. ACHC 5-day persistence validates urgency decay (score 85→88→80→72→70). |
| technical_breakout | scanners/technical_breakout.md | 2026-04-25 | Volume-confirmed breakouts (2.3x-4.1x), specific price levels, ISRG 84, MANH 78, BA 73, VOYA 85, HXL 74. No outcome data yet. Excellent specificity. |
| volume_divergence | scanners/volume_divergence.md | 2026-04-25 | OBV divergence signal; HOLX Apr 24 extreme OBV divergence (+135.4%), rank 8; DRI/PGR Apr 24 also strong. No outcome data yet. |
| momentum | scanners/momentum.md | 2026-04-25 | ⚠️ AUTOPSY: 39.7% 7d WR (136 recs), -0.80% avg — worst performer. BUT +26.6pt confluence lift with insider_buying. Confluence-only fix planned (preserve lift, eliminate drag). Deadline 2026-05-05. |
| ML Signal Improvement | research/2026-04-21-ml-signal-improvement.md | 2026-04-21 | Root cause: 3-class TIMEOUT label dominates (48%), caps WIN prob at 46%; fix: binary labels + 5 regime features |
| Options Flow ML Features | research/2026-04-21-options-flow-ml-features.md | 2026-04-21 | IV skew has strong academic evidence (10.9% annual alpha) but historical IV data requires paid source; implement as live inference features + scanner augmentation |
| earnings_beat | scanners/earnings_beat.md | 2026-04-14 | New PEAD scanner: recent EPS beats ≥5% surprise; 15% annualized academic edge; distinct from earnings_calendar |

## Confluence Signals

| Pair | n | WR Lift | Notes |
|------|---|---------|-------|
| insider_buying + momentum | 35 | +26.6 pts (74.3% vs 47.7% IB alone) | Strongest confluence in pipeline; momentum as confirmer despite 39.7% standalone WR |
| momentum + options_flow | 22 | +12.3 pts (59.1% vs 46.8% OF alone) | Significant lift; momentum confirming options flow signal |

## Autopsy Clock

| Scanner | Triggered | Deadline | Condition | Status |
|---------|-----------|----------|-----------|--------|
| momentum | 2026-04-21 | 2026-05-05 | WR-7d=39.7%, n=136 | ⚠️ ACTIVE (10 days) — confluence-only fix in progress |

## Discarded Signals

Exact signal names that have been researched, implemented, and backtested — do not re-propose without a documented reason the prior failure no longer applies.

| Signal Name | Discard Type | Date | Why Failed | Re-research Condition |
|-------------|-------------|------|------------|-----------------------|
| `nr7_breakout` | DISCARD | 2026-04-16 | Fires 8+/day (unselective); avg-20d=+1.15% below promotion threshold | Only if combined with a market-state filter that reduces picks to <3/day |
| `bb_squeeze` | DISCARD | 2026-04-16 | Bollinger squeeze without directional trigger = noise; WR-20d insufficient | Add a directional trigger (e.g. momentum bar breaking squeeze high/low) |
| `consecutive_down_days` | DISCARD | 2026-04-16 | 9.5/day (unselective); WR-20d=53.6% avg+0.83% below threshold | Only if pick rate drops to <3/day via additional filters |
| `pullback_in_uptrend` | DISCARD | 2026-04-16 | 9.4/day (unselective); WR-20d=54.3% avg+1.02% below threshold | Only if pick rate drops to <3/day via additional filters |
| `adx_trend_inception` | DISCARD | 2026-04-16 | 9.8/day (unselective); WR-20d=56.9% avg+1.64% near threshold but not selective | Only if pick rate drops to <3/day; promising edge, just too broad |
| `selling_climax_reversal` | DISCARD | 2026-04-16 | WR-20d=44.4% (<50% = below random); volume climax without market-regime filter | Only with a VIX/breadth regime filter confirming macro capitulation |
| `macd_histogram_reversal` | DISCARD | 2026-04-16 | 8.7/day (unselective); fires too frequently across all market conditions | Only if combined with a compression state (e.g. price range <2% for 5 days) |
| `bollinger_band_mean_reversion` | DISCARD-CALIBRATION | 2026-04-17 | 0 picks in Apr 2025–Feb 2026 bull market; signal only fires in corrections; March 2026 correction in buffer cutoff | Re-run backtest after Sept 2026 (March correction enters measurable window) |
| `inside_days_breakout` | DISCARD-CALIBRATION | 2026-04-17 | Only 2 picks in 11 months (too rare for evaluation); triple condition too strict in trending markets | Consider relaxing to 1 inside day, or revisit as part of a broader NR4/compression scanner |
| `gap_down_reversal` | DISCARD | 2026-04-17 | WR-20d=16%, avg=-5.72%; all 25 picks in March 2026 correction where macro decline continued; SMA200 filter insufficient in early correction | Add market-breadth or VIX regime filter; do not re-research without macro context filter |

## Research

| Title | File | Date | Summary |
|-------|------|------|---------|
| RSI(2) Mean Reversion Oversold Bounce | research/2026-04-15-rsi-mean-reversion.md | 2026-04-15 | Connors RSI(2)<10 + price above 200d SMA = 75-79% win rate over 25y backtest; only contrarian signal in pipeline; implemented as rsi_oversold scanner |
| OBV Divergence Accumulation | research/2026-04-14-obv-divergence.md | 2026-04-14 | OBV rising while price flat/down = multi-week institutional accumulation; qualitative 68% win rate; implemented as obv_divergence scanner |
| Short Interest Squeeze Scanner | research/2026-04-12-short-interest-squeeze.md | 2026-04-12 | High SI (>20%) + DTC >5 as squeeze-risk discovery; implemented as short_squeeze scanner |
| 52-Week High Breakout Momentum | research/2026-04-13-52-week-high-breakout.md | 2026-04-13 | George & Hwang (2004) validated: 52w high crossing + 1.5x volume = 72% win rate, +11.4% avg over 31d; implemented as high_52w_breakout scanner |
| PEAD Post-Earnings Drift | research/2026-04-14-pead-earnings-beat.md | 2026-04-14 | Bernard & Thomas (1989): 18% annualized PEAD; QuantPedia: 15% annualized (1987-2004); implemented as earnings_beat scanner (distinct from earnings_calendar's upcoming-only scope) |
| Dark Pool / Block Trade Flow | research/2026-04-16-dark-pool-flow.md | 2026-04-16 | Zhu 2012 + Buti 2022 academic backing; meridianfin.io provides free scrapable daily FINRA Z-scored anomalies (no auth, 1-day lag); recommend implement as dark_pool_flow scanner in edge pipeline |
| Volatility Contraction Batch | research/2026-04-16-volatility-batch.md | 2026-04-16 | 3 candidates backtested: atr_compression WR-20d=59.3% PROMOTED; nr7_breakout +1.15% avg DISCARDED (too many picks); bb_squeeze +0.54% avg DISCARDED (squeeze without direction = noise) |
| Price Structure Batch | research/2026-04-16-price-structure-batch.md | 2026-04-16 | 3 candidates backtested: all DISCARDED — consecutive_down_days WR-20d=53.6% avg+0.83%, pullback_in_uptrend WR-20d=54.3% avg+1.02%, adx_trend_inception WR-20d=56.9% avg+1.64%; all hit 10-pick limit every day (9.4-9.8/day), not selective enough for promotion |
| Volume Extreme Strategies | research/2026-04-16-volume-extreme-strategies.md | 2026-04-16 | 3 candidates backtested: volume_dry_up WR-20d=80% avg+3.26% PROMOTED; selling_climax_reversal WR-20d=44.4% DISCARDED (WR<50%); macd_histogram_reversal avg+1.35% DISCARDED (hits limit daily) |
| volume_dry_up | scanners/volume_dry_up.md | 2026-04-16 | Walk-forward: WR-20d=80%, avg-20d=+3.26%, 20 picks — PROMOTE (slow signal: hold ≥10d; small sample caveat) |
| Gap-Up Continuation (Breakaway Gap) | research/2026-04-17-gap-up-continuation.md | 2026-04-17 | PMC study: 54-60% win rate, +0.30-0.58% avg daily gain; large gaps (>0.4%) fill <50% of time; implemented as gap_up_continuation scanner using OHLCV cache |
| Mean Reversion Batch | research/2026-04-17-mean-reversion-batch.md | 2026-04-17 | 3 candidates backtested: all DISCARDED — bollinger_band_mean_reversion DISCARD-CALIBRATION (0 picks, bull-market window), inside_days_breakout DISCARD-CALIBRATION (2 picks, too rare), gap_down_reversal DISCARD (WR-20d=16%, avg=-5.72%, all picks in March 2026 correction) |
| atr_compression | scanners/atr_compression.md | 2026-04-16 | Walk-forward backtest: WR-20d=59.3%, avg-20d=+1.88%, 738 picks — PROMOTE-MARGINAL |
| volatility_contraction_breakout | scanners/volatility_contraction_breakout.md | 2026-04-21 | Present in all 4 runs Apr 20; pick rate stable (1-2/run); FDS (0.67) and PWR (0.70) quality picks; score<75 picks marginal |
| high_52w_breakout | scanners/high_52w_breakout.md | 2026-04-20 | First live appearances: JBHT (score=92, EPS+27%) and BK (score=80, buyback) on Apr 18; no outcome data yet |
| volume_divergence | scanners/volume_divergence.md | 2026-04-20 | obv_divergence scanner; EA (OBV+24.2%, ATR $0.98) on Apr 18; 60% 1d win rate (n=5), no 7d data yet |
| Insider Cluster Buying Detection | research/2026-04-21-insider-cluster-buying.md | 2026-04-21 | 3+ insiders buying within 14d = 2.1-3.8% abnormal returns (2× vs solo buys); Alldredge 2019, Kang 2018 backing; implement as insider_cluster_buying scanner |
| reddit_dd | scanners/reddit_dd.md | — | No data yet |
| reddit_trending | scanners/reddit_trending.md | — | No data yet |
| semantic_news | scanners/semantic_news.md | — | No data yet |
| market_movers | scanners/market_movers.md | — | No data yet |
| technical_breakout | scanners/technical_breakout.md | — | No data yet |
| sector_rotation | scanners/sector_rotation.md | — | No data yet |
| ml_signal | scanners/ml_signal.md | 2026-04-21 | min_win_prob=0.35 config override causes 44-49% (sub-coin-flip) picks to dominate runs; fix: raise to 0.50 |
| momentum | scanners/momentum.md | 2026-04-21 | ⚠️ AUTOPSY: 39.7% WR-7d (n=136), -0.80% avg — worst large-sample performer; deadline 2026-05-05 |
| Non-OHLCV Signal Saturation Analysis | research/2026-04-23-non-ohlcv-saturation.md | 2026-04-23 | Pipeline exhausted accessible free non-OHLCV signals; remaining opportunities blocked by paid data (IV/borrow costs), stale lags (13F 135d), or already confluenced (SI+insider). Redirect: evaluate dark_pool_flow & insider_cluster_buying P&L, test confluence rules, explore paid data ROI. |
| Institutional Block Trade Flow (Lit Exchanges) | research/2026-04-25-block-trade-flow.md | 2026-04-25 | SKIP: Investigated block trades on NYSE/NASDAQ as complement to dark_pool_flow. NBER 2024 shows block size predicts timing only, not returns. FINRA TRF data requires paid vendor agreement (no free source). Signal redundant with dark_pool_flow + options_flow for institutional conviction detection. |

---

**What changed this run:** P&L analysis revealed momentum is broken in standalone mode (39.7% WR, worst performer) BUT has strong confluence effect (+26.6pts with insider_buying). Fix: filter momentum to confluence-only mode, preserving the +26.6pt lift while eliminating -0.80% drag. PEAD (earnings_beat) scanner validated: all 4 recent picks showed >150% EPS surprise (CCI +161%, DLR +325%, PECO +303%, TAL +185%) with concrete catalysts. Zero false positives detected. technical_breakout scanner showing proper volume confirmation (2.3x-4.1x) and specific price levels — no noise. short_squeeze urgency model fully validated: ACHC 5-day persistence (score 85→88→80→72→70) reflects price extension, not staleness. insider_buying intraday staleness identified: NKE appeared 3/4 times same day (Apr 20) despite suppress_days=3 filter; cross-day filter working but intraday deduplication gap found.
