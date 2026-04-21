# Insider Buying Scanner

## Current Understanding
Scrapes SEC Form 4 filings. CEO/CFO purchases >$100K are the most reliable signal.
Cluster detection (2+ insiders buying within 14 days) historically a high-conviction
setup. Transaction details (name, title, value) must be preserved from scraper output
and included in candidate context — dropping them loses signal clarity.

## Evidence Log

### 2026-04-12 — P&L review (2026-02-18 to 2026-04-07)
- insider_buying produced 136 recommendations — by far the highest volume scanner.
- Score distribution is healthy and concentrated: 53 picks in 80-89, 11 in 90-99, only 1 below 60.
- Confidence calibration is tight: avg score 78.6 (score/10 = 7.9) vs avg confidence 7.5 — well aligned.
- Cluster detection (2+ insiders → CRITICAL priority) is **already implemented** in code at `insider_buying.py:73`. The hypothesis was incorrect — this is live, not pending.
- High-conviction cluster examples surfaced: HMH (appeared in 2 separate runs Apr 8-9), FUL (Apr 9 and Apr 12), both with scores 71-82.
- Confidence: high

### 2026-04-12 — Fast-loop (2026-04-08 to 2026-04-12)
- Insider_buying dominates final rankings: 3 of 6 ranked slots on Apr 9, 2 of 5 on Apr 10, contributing highest-ranked picks regularly.
- Context strings are specific and include insider name, title, dollar value — good signal clarity preserved.
- Confidence: high

### 2026-04-12 — P&L update (180 tracked recs, mature data)
- Win rates are weaker than expected given high confidence scores: 38.1% 1d, 46.4% 7d, 29.7% 30d.
- Avg returns: -0.01% 1d, -0.4% 7d, -1.98% 30d — negative at every horizon.
- **Staleness pattern confirmed**: HMH appeared 4 consecutive days (Apr 6-9) with nearly identical scores (72, 85, 71, 82) — same insider filing, no new catalyst. FUL appeared Apr 9 and Apr 12 with identical scores (75). This is redundant signal, not confluence.
- High confidence (avg 7.1) combined with poor actual win rates = miscalibration — scanner assigns scores optimistically but real outcomes are below 50%.
- Confidence: high

### 2026-04-14 — P&L review (Apr 3-9 mature recs) + staleness filter implementation
- Staleness pattern confirmed at scale: PAGS appeared 4 consecutive days (Apr 3-6, identical $10.34 entry, same Director Frias $4.96M purchase). ZBIO appeared 4 consecutive days (Apr 3-6, same $5.59M cluster buy). HMH appeared 3 consecutive days (Apr 7-9, same CFO $1M purchase).
- 11 of 22 insider_buying picks in Apr 3-9 (50%) were stale repeats — same Form 4 filing surfaced daily within the 7-day lookback window.
- Root cause: `lookback_days=7` causes any filing made on day D to appear every day from D through D+6. The deduplication is within a single fetch, not across runs.
- Code fix: Added `_load_recent_insider_tickers(suppress_days=2)` in `insider_buying.py`. Loads the past 2 days of recommendation files and filters out tickers already recommended as `insider_buying`. This directly suppresses the PAGS/ZBIO/HMH pattern.
- Updated statistics: 184 recs total (+48 since last analysis). 7d win rate 45.9% (was 46.4%), 30d win rate 32.8%. Avg returns negative at all horizons: -0.01% 1d, -0.44% 7d, -1.62% 30d.
- Confidence: high (staleness pattern now confirmed across 3 distinct tickers in a single week)

### 2026-04-17 — P&L update + staleness filter gap found
- n=190 total picks (was 184). 7d win rate 47.7% (was 45.9%), 30d win rate 41.6% (was 32.8%). Modest improvement consistent with staleness filter helping.
- Avg returns: -0.1% 7d (was -0.44%), +0.04% 30d (was -1.62%). Direction improving.
- **Staleness filter gap confirmed**: FUL appeared Apr 9 AND Apr 12 (3-day gap) with identical score (75) and same CEO purchase ($295,104 by Mastin Celeste Beeks). The suppress_days=2 window only blocks 1-2 day repeats; a 3-day gap slips through.
- **Code fix applied 2026-04-17**: suppress_days raised from 2 → 3 in `insider_buying.py:123`.
- New Apr 12 picks: FUL (score=75, staleness miss now fixed), GF (score=65, Saba Capital 10% owner $1.49M). Note: GF is an institutional activist (10% owner), not an operational insider — borderline signal quality.
- Confidence: high (staleness gap confirmed by direct observation; fix is minimal and targeted)

### 2026-04-20 — Fast-loop (same-day multi-run staleness)
- NKE appeared in 3 of 4 runs today: run_10_59 (score=75), run_14_20 (score=82), run_15_42_19 (score=85). Same insider purchase surfaced across 3 intraday runs.
- BORR appeared in 2 runs: run_15_42_19 (score=80) and run_15_42_27 (score=92). Same $2.79M Director Troim purchase.
- The 3-day suppress_days window blocks cross-day repeats but NOT same-day repeats across multiple runs. When the system runs multiple discovery passes in one day, the staleness filter doesn't deduplicate within the day.
- This is a distinct failure mode from the cross-day staleness: NKE's thesis escalated slightly (75→82→85) as the run was re-enriched, but it's still the same underlying SEC filing being re-discovered 3 times in one day.
- Confidence: high (3/4 runs = direct observation; same filing confirmed by identical context)

## Pending Hypotheses
- [x] Does cluster detection (2+ insiders in 14 days) outperform single-insider signals? → **Already implemented**: cluster detection assigns CRITICAL priority. Code verified at `insider_buying.py:73-74`. Cannot assess outcome vs single-insider yet (all statuses 'open').
- [x] Does filtering out repeat appearances of the same ticker from the same scanner within 3 days improve precision? → **Implemented 2026-04-14**: staleness suppression added; expanded to 3-day window 2026-04-17 after FUL gap found.
- [ ] Is there a minimum transaction size below which signal quality degrades sharply? (current min: $100K raised from $25K as of 2026-04-07)
- [ ] Does the staleness suppression (3-day lookback) measurably improve 7d win rate vs 2-day lookback? Track over next 2 weeks.
- [ ] Are 10%-owner purchases (activists like Saba Capital) lower quality signals than operational insiders (CEO/CFO)? GF Apr 12 is a test case.
