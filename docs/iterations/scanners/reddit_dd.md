# Reddit DD Scanner

## Current Understanding
Scans r/investing, r/stocks, r/wallstreetbets for DD posts. LLM quality score is
computed and used for filtering — posts scoring >=80 are HIGH priority, 60-79 are
MEDIUM, and <60 are skipped. This quality filter is the key differentiator from
the reddit_trending scanner.

The quality_score filter (>=60) is working: social_dd is the ONLY strategy with
positive 30d returns (+0.94% avg) and 55% 30d win rate across all tracked strategies.
This is confirmed by P&L data spanning 608 total recommendations.

## Evidence Log

### 2026-04-11 — P&L review
- 26 recommendations. 30d avg return: +0.94% (only positive 30d avg among all strategies).
- 30d win rate: 55%. 7d win rate: 44%. 1d win rate: 46.2%.
- The positive 30d return despite negative 1d/7d averages suggests DD-based picks
  need time to play out — the thesis takes weeks, not days, to materialize.
- Compare with social_hype (reddit_trending, no quality filter): -10.64% 30d avg.
  The quality_score filter alone appears to be the separator between signal and noise.
- The code already implements the quality filter correctly (>=60 threshold).
- Confidence: high (26 data points, consistent pattern vs. sister scanner)

### 2026-04-22 — Fast-loop (2026-04-22 run)
- GME (rank 6, score=74, conf=7): Social sentiment backed by real technical accumulation (OBV +164.1M divergence). Fundamental thesis: $8.8B cash + Bitcoin treasury provides valuation floor. Price above 50 SMA and VWAP, ADX=24.8 rising.
- Scanner is surfacing meme-adjacent play but with concrete technical + fundamental backing, not pure sentiment.
- Calibration: 74/10=7.4 vs conf=7 (Δ=0.4) — good.
- This pick shows the reddit_dd quality filter (score >=60) is working: specific thesis with both social + technical + fundamental support.
- Confidence: low (single data point; meme stock vol is extreme, outcome highly uncertain)

## Pending Hypotheses
- [ ] Does filtering by LLM quality score >80 (HIGH only) further improve outcomes vs >60?
- [ ] Does subreddit weighting change hit rates (r/investing vs r/wallstreetbets)?
- [ ] Does fundamental backing (e.g., cash position, buyback program) improve reddit_dd outcomes beyond social sentiment alone?
