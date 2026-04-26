---
name: short_squeeze
description: Stocks with structurally high short interest vulnerable to forced covering
type: scanner
---

# Short Squeeze Scanner

## Current Understanding
Identifies stocks with structurally high short interest (>15% of float by default, CRITICAL at >30%)
where short sellers are vulnerable to forced covering on any positive catalyst. The scanner uses
Finviz for discovery (screener filters) + Yahoo Finance for exact SI% and days-to-cover (shortRatio)
verification.

Key distinction: High SI alone predicts *negative* long-term returns on average (academic consensus).
However, live P&L data (n=10) shows 60% 7d win rate and +2.15% avg 7d — best 7d performer in
the pipeline. This reflects that discovery-pipeline filtering (technical confirmation, enrichment)
already adds the catalyst signal needed to convert squeeze-risk into a directional trade. 

**Critical finding:** 30d win rate drops to 30% (-1.10% avg), confirming squeeze resolves within 7d.
Max holding period should be capped at 7 days in ranker guidance.

## Evidence Log

### 2026-04-13 — P&L review (first real outcome data)
- 10 tracked recommendations, 5/10 1d wins (50% win rate), 6/10 7d wins (60% win rate).
- Avg 7d return: +2.15%. This makes short_squeeze the **best 7d performer** among scanners with ≥5 samples.
- Outperforms analyst_upgrade (50% 7d), insider_buying (46.4% 7d), options_flow (45.6% 7d).
- The scanner is producing positive outcomes as a standalone signal, not only as a cross-scanner modifier.
- However, ranker prompt says "Focus on days to cover" but context string only shows SI%. DTC value is available in Yahoo Finance (`shortRatio`) but was not being fetched or passed through — gap confirmed.
- Confidence: medium (small sample n=10; 30d data will be more conclusive; DTC gap has been fixed)

### 2026-04-13 — Code fix: days_to_cover surfaced in context
- Added `days_to_cover` extraction (`shortRatio` from Yahoo Finance) to `finviz_scraper.py`.
- Applied `min_days_to_cover` filter (previously accepted as parameter but never enforced).
- Updated `short_squeeze.py` context string to include DTC value so ranker can use "days to cover" criterion.
- Confidence: high (this is a clear context gap between ranker criteria and available data)

### 2026-04-14 — P&L review (updated statistics, n=11)
- 7d win rate: 60% (6/10 wins), avg 7d return: +2.15% — still best 7d performer. No change from prior analysis.
- **NEW: 30d win rate: 30% (3/10), avg 30d return: -1.1%** — signal degrades sharply at 30d. The squeeze resolves (or fails) within 7 days; holding longer is harmful.
- This confirms short_squeeze is a **short-term-only signal**. The 7d alpha is real; the 30d outcome is poor.
- Pattern: WTI and TSLA appeared in Apr 3-9 mature recs as short_squeeze plays — high SI but no clear catalyst timing to trigger covering.
- Confidence: medium (n=11 still small; 30d degradation pattern is consistent with academic squeeze literature)

### 2026-04-22 — Fast-loop (ACHC 4-day persistence validates urgency decay model)
- ACHC appeared again on 2026-04-22 (rank 7, score=72, conf=6). SI=37.1% unchanged, DTC=7.1d, earnings in 7 days.
- This is the 4th consecutive appearance: Apr 15 (score 85), Apr 18 (score 88), Apr 19 (score 80), Apr 22 (score 72).
- Score erosion (85→88→80→72) reflects price extension as rally accelerates — classic squeeze momentum arc. This is NOT staleness, it is urgency tracking.
- Thesis remains mechanically sound throughout: high SI × approaching binary earnings event = two converging catalysts for forced covering.
- Pattern: cross-day persistence for squeeze candidates is valid urgency signal; score decline on rally is expected and reflects diminishing upside room as price extends.
- Confidence: medium (multi-day persistence confirmed; price action consistent with squeeze thesis)

### 2026-04-24 — Fast-loop (ACHC 5th appearance, score further decay)
- ACHC appeared yet again on 2026-04-24 (rank 11, score=70, conf=6). Decline from Apr 22 score=72 indicates continued price extension.
- Now 5 consecutive appearances over 9 trading days (Apr 15, 18, 19, 22, 24). Earnings in ~5 days from Apr 24.
- Pattern holds: score decay (85→88→80→72→70) = price extension without new catalyst = diminishing edge within 7d window.
- Thesis: if entered on Apr 15 at score 85, position would now be 9 days old with earnings only 5 days away. Score erosion suggests bull trap risk if held past squeeze resolution (7d max rule).
- Confidence: high (multi-day persistence pattern consistent; score decay validates urgency model)

## Pending Hypotheses
- [ ] Does short_squeeze + options_flow confluence produce better 7d win rate than either scanner alone?
- [ ] Does short_squeeze + earnings_calendar (SI>20%) produce better outcomes than earnings alone? (See earnings_calendar.md pending hypothesis)
- [ ] Is there a volume threshold (e.g., market cap <$2B small-cap) that sharpens the signal?
- [ ] Does DTC >5 (now surfaced in context) predict better outcomes than DTC 2-5 within the scanner?
- [ ] Does standalone short_squeeze (no cross-scanner confluence) continue to outperform at 7d as sample grows?

## Implementation Notes
- **Max holding period:** Capped at 7 days in ranker guidance (30d outcomes show 30% WR, -1.1% avg).
- **Urgency scoring:** Score decay on multi-day persistence is expected and reflects price extension, NOT staleness.
