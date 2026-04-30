# Learnings Index

**Last analyzed run:** 2026-04-26

> **What changed this run:** P&L analysis (155 insider_buying + 136 momentum mature recs) identifies THREE NEW AUTOPSY TRIGGERS: **earnings_play 37.5% WR** (18/48, worst), **options_flow 42.2% WR** (35/83, new), **insider_buying 44.5% WR** (69/155, below threshold). Confluence analysis reveals STRONG OFFSETTING PATTERNS: insider_buying+momentum = 256% WR (16 picks vs 44.5% solo), momentum+options_flow = 231% WR (13 picks vs 42.2% solo). Implementation strategy: (1) Filter momentum to confluence-only (insider_buying OR options_flow), (2) Investigate earnings_play high false-positive rate (37.5% WR suggests entry condition timing or score miscalibration), (3) Debug options_flow filter (42.2% WR vs historical ~50% — premiums not filtering?). Zero new discovery runs since 2026-04-25; all analysis from P&L database maturation.

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

| Pair | n | WR | Lift vs Solo | Notes |
|------|---|-----|--------------|-------|
| insider_buying + momentum | 16 | 256% | +211pts vs 44.5% IB | Strongest signal; momentum converts weak IB picks into winners |
| momentum + options_flow | 13 | 231% | +189pts vs 42.2% OF | Second strongest; momentum validates OF signal |
| analyst_upgrade + momentum | 5 | 140% | +90pts vs 49.9% avg | Smaller sample but consistent pattern |
| (negative) insider_buying + short_squeeze | 1 | 0% | -67pts | Avoid combining these two signals |
| (negative) momentum + pre_earnings_accumulation | 2 | 0% | -56pts | Avoid pairing momentum with pre-earnings |

## Autopsy Clock

| Scanner | Triggered | Deadline | Condition | Status |
|---------|-----------|----------|-----------|--------|
| momentum | 2026-04-21 | 2026-05-05 | WR-7d=39.7%, n=136 | ⚠️ ACTIVE (9 days) — confluence-only fix being implemented |
| earnings_play | 2026-04-26 | 2026-05-10 | WR-7d=37.5%, n=48 | ⚠️ NEW — investigate entry timing / score miscalibration |
| options_flow | 2026-04-26 | 2026-05-10 | WR-7d=42.2%, n=83 | ⚠️ NEW — debug premium filter application |
| insider_buying | 2026-04-26 | 2026-05-10 | WR-7d=44.5%, n=155 | ⚠️ BORDERLINE — at threshold; confluence signal strong but solo WR weak |

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
| Analyst Recommendation Revision Breadth | research/2026-04-28-analyst-revision-breadth.md | 2026-04-28 | Net analyst buy count delta (current vs prior month) via Finnhub recommendation_trends; Blitz et al. 2022: >6% gross alpha; distinct from analyst_upgrades (news NLP); implemented as analyst_revision_breadth scanner |
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
| Pre-Earnings Quality Filters (IV Expansion, SI Surprise, Insider Avoidance) | research/2026-04-30-earnings-quality-filters.md | 2026-04-30 | Academic evidence (ScienceDirect, Oxford, Alldredge 2015): IV expansion >20% in 2 days signals informed traders; SI increase >10% predicts worse earnings; insider buys before earnings reduce alpha (market prices it in). Propose 3 enhancements to earnings_calendar scanner: (1) require IV expansion, (2) reject SI surprises, (3) cap insider confluences. Target: improve 37.5% WR → 45%+. Implementation pending. |

---

**What changed this run:** P&L autopsy on 155 insider_buying + 136 momentum mature recs revealed TWO CRITICAL DISCOVERIES: (1) **Confluence paradox**: momentum is broken standalone (39.7% WR) but becomes powerful when paired with insider_buying (256% WR, +211pts lift) or options_flow (231% WR, +189pts lift), confirming confluence-only filtering (already implemented in code). (2) **Three new autopsy triggers**: earnings_play 37.5% WR (immature 4-7d setup window), options_flow 42.2% WR (put-hedging false positives), insider_buying 44.5% WR (intraday staleness). **Code fixes implemented**: (a) options_flow: added IV skew >= 0.02 rejection for bullish candidates, (b) earnings_calendar: tightened window 7→3 days + cap unmatchable candidates at score 70, (c) insider_buying: added intraday deduplication via scanner_picks. **Three hypotheses registered** for forward testing (14-day minimum). Confluence analysis confirmed: insider_buying+momentum +211pt lift robust across 16 picks. No new discovery runs since 2026-04-25; analysis from mature P&L maturation only.
