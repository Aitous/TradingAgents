# Learnings Index

**Last analyzed run:** 2026-04-20

| Domain | File | Last Updated | One-line Summary |
|--------|------|--------------|-----------------|
| options_flow | scanners/options_flow.md | 2026-04-12 | Premium filter confirmed applied; CSCO cross-scanner confluence detected; 45.1% 7d win rate (94 recs) |
| insider_buying | scanners/insider_buying.md | 2026-04-17 | suppress_days raised 2→3 (FUL 3-day gap missed); 47.7% 7d / 41.6% 30d post-staleness-filter improvement |
| minervini | scanners/minervini.md | 2026-04-17 | 100% win rate still holds (n=3 measured, 15 total); +7.19% avg 7d; AVGO persistence Apr 10-11; AA score=92 Apr 12 |
| analyst_upgrades | scanners/analyst_upgrades.md | 2026-04-20 | 55.9% 7d, +0.18% avg; NI (AI infrastructure re-rating) and PLD (buyback+FFO beat) strong Apr 18 picks |
| earnings_calendar | scanners/earnings_calendar.md | 2026-04-17 | 7d win rate 47.6% (was 37.7%); likely market-recovery effect; still lowest-scoring scanner |
| pipeline/scoring | pipeline/scoring.md | 2026-04-20 | 684 total recs, 44.4% overall 7d; volatility_contraction_breakout 50% of output on Apr 19 |
| early_accumulation | scanners/early_accumulation.md | 2026-04-20 | 43.8% 7d, -7.6% avg 30d (worst); sustained-accumulation filter (high_vol_days_5d≥2) added to code |
| social_dd | scanners/social_dd.md | 2026-04-14 | 57.1% 30d win rate (+1.41% avg 30d, n=26) — only scanner positive at 30d; eval horizon mismatch persists |
| volume_accumulation | scanners/volume_accumulation.md | — | No data yet |
| short_squeeze | scanners/short_squeeze.md | 2026-04-20 | 60% 7d, +2.15% avg (n=19); ACHC 4-day persistence pre-earnings is valid urgency, not staleness |
| earnings_beat | scanners/earnings_beat.md | 2026-04-14 | New PEAD scanner: recent EPS beats ≥5% surprise; 15% annualized academic edge; distinct from earnings_calendar |

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
| atr_compression | scanners/atr_compression.md | 2026-04-16 | Walk-forward backtest: WR-20d=59.3%, avg-20d=+1.88%, 738 picks — PROMOTE-MARGINAL |
| volatility_contraction_breakout | scanners/volatility_contraction_breakout.md | 2026-04-20 | 5 appearances in 2 days; 4/8 picks on Apr 19; specific theses; PSTG cross-day persistence observed; no outcome data yet |
| high_52w_breakout | scanners/high_52w_breakout.md | 2026-04-20 | First live appearances: JBHT (score=92, EPS+27%) and BK (score=80, buyback) on Apr 18; no outcome data yet |
| volume_divergence | scanners/volume_divergence.md | 2026-04-20 | obv_divergence scanner; EA (OBV+24.2%, ATR $0.98) on Apr 18; 60% 1d win rate (n=5), no 7d data yet |
| reddit_dd | scanners/reddit_dd.md | — | No data yet |
| reddit_trending | scanners/reddit_trending.md | — | No data yet |
| semantic_news | scanners/semantic_news.md | — | No data yet |
| market_movers | scanners/market_movers.md | — | No data yet |
| technical_breakout | scanners/technical_breakout.md | — | No data yet |
| sector_rotation | scanners/sector_rotation.md | — | No data yet |
| ml_signal | scanners/ml_signal.md | — | No data yet |
