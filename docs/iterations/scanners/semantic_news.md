# Semantic News Scanner

## Current Understanding
Currently regex-based extraction, not semantic. Headline text IS included in
candidate context via `news_headline` field (improved from prior version).
Catalyst classification from headline keywords maps to priority:
- CRITICAL: FDA approval, acquisition, merger, breakthrough
- HIGH: upgrade, beat, contract win, patent, guidance raise
- MEDIUM: downgrade, miss, lawsuit, investigation, recall, warning

P&L data shows `news_catalyst` is the worst-performing strategy: -17.5% avg 30d
return, 0% 7d win rate, 12.5% 1d win rate. Root cause: MEDIUM-priority candidates
(negative catalysts — downgrades, lawsuits, recalls) are included in the candidate
pool and frequently get through to recommendations with a bullish framing. Scanner
now restricted to CRITICAL-only to eliminate negative-catalyst contamination.

## Evidence Log

### 2026-04-30 — P&L review (AUTOPSY TRIGGERED)
- 6 closed recommendations; 0% 7d win rate, -8.14% avg return.
- Updated baseline: 0% 1d/7d/30d win rate across all 6 picks (worse than prior 8-pick sample).
- All outcomes uniformly losing; no winners suggest semantic parsing is producing false
  catalyst signals or the NLP is extracting boilerplate metadata as "news".
- CRITICAL-only filter (applied Apr 11) did not prevent the 0% 7d WR outcome.
- Pattern: news catalyst is mechanically unsound — investors react slower than parser
  detection, or parser hallucinated catalysts entirely.
- Confidence: **high** — 6 uniformly losing picks with no confounding factors.

### 2026-04-11 — P&L review
- 8 recommendations, 1d win rate 12.5%, 7d win rate 0% (worst of all strategies).
- Avg 30d return: -17.5%. Avg 1d return: -4.19%. Avg 7d return: -8.79%.
- Sample shows WTI (W&T Offshore) appearing twice (Apr 3 and Apr 6) as news_catalyst
  based on geopolitical oil price spike — both marked as "high" risk. The spike
  reversed, consistent with the -17.5% 30d outcome.
- Root issue 1: MEDIUM-priority keywords include negative events (downgrade, miss,
  lawsuit) that generate candidates with inherently negative thesis.
- Root issue 2: CRITICAL/HIGH keywords like "upgrade" and "patent" overlap with
  noise in global news feeds that mention these terms incidentally.
- Fix applied: only emit candidates when headline matches CRITICAL-priority keywords.
  Eliminates the negative-catalyst false positives.
- Confidence: medium (8 data points; market downturn may amplify losses)

## Status
**⚠️ AUTOPSY TRIGGERED (2026-04-30): 0% WR over 6 closed picks. DISABLE or RETOOL by 2026-05-14.**

## Recommendation
**DISABLE** the semantic_news scanner (mapped to "news" pipeline alias) until root cause is identified.
The signal appears fundamentally broken — all 6 picks lost money with no directional consistency.
Options:
1. Disable entirely (remove from pipeline)
2. Retool to use third-party news API (Bloomberg, Facteus) instead of regex headline parsing
3. Pivot to causal news (post-earnings, guidance changes) instead of forward-looking speculation
