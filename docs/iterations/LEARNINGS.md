# Learnings Index

**Last analyzed run:** 2026-04-14

| Domain | File | Last Updated | One-line Summary |
|--------|------|--------------|-----------------|
| options_flow | scanners/options_flow.md | 2026-04-12 | Premium filter confirmed applied; CSCO cross-scanner confluence detected; 45.1% 7d win rate (94 recs) |
| insider_buying | scanners/insider_buying.md | 2026-04-14 | Staleness suppression filter added (PAGS/ZBIO/HMH 3-4 day repeats confirmed); 45.9% 7d, negative avg returns |
| minervini | scanners/minervini.md | 2026-04-12 | Best performer: 100% 1d win rate (n=3), +3.68% avg; 7 candidates in Apr 6-12 week |
| analyst_upgrades | scanners/analyst_upgrades.md | 2026-04-12 | 51.6% 7d win rate (marginal positive); cross-scanner confluence with options_flow is positive signal |
| earnings_calendar | scanners/earnings_calendar.md | 2026-04-12 | Appears as earnings_play; 38.1% 1d, 37.7% 7d — poor; best setups require high short interest |
| pipeline/scoring | pipeline/scoring.md | 2026-04-14 | news_catalyst 0% 7d now explicit in ranker criteria; insider staleness filter implemented; 41.9% overall 7d win rate |
| early_accumulation | scanners/early_accumulation.md | 2026-04-12 | Sub-threshold (score=60); no catalyst → structurally score-capped by ranker |
| social_dd | scanners/social_dd.md | 2026-04-14 | 57.1% 30d win rate (+1.41% avg 30d, n=26) — only scanner positive at 30d; eval horizon mismatch persists |
| volume_accumulation | scanners/volume_accumulation.md | — | No data yet |
| short_squeeze | scanners/short_squeeze.md | 2026-04-14 | 60% 7d win rate (n=11), best 7d performer; BUT 30% 30d — short-term signal only, degrades at 30d |
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
| atr_compression | scanners/atr_compression.md | 2026-04-16 | Walk-forward backtest: WR-20d=59.3%, avg-20d=+1.88%, 738 picks — PROMOTE-MARGINAL |
| reddit_dd | scanners/reddit_dd.md | — | No data yet |
| reddit_trending | scanners/reddit_trending.md | — | No data yet |
| semantic_news | scanners/semantic_news.md | — | No data yet |
| market_movers | scanners/market_movers.md | — | No data yet |
| technical_breakout | scanners/technical_breakout.md | — | No data yet |
| sector_rotation | scanners/sector_rotation.md | — | No data yet |
| ml_signal | scanners/ml_signal.md | — | No data yet |
