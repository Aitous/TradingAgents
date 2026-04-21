# ML Signal Scanner

## Current Understanding
Uses a trained LightGBM model to predict short-term price movement probability. The `min_win_prob`
is configured at **0.35** in `default_config.py`, which means the scanner surfaces picks with
sub-50% win probability — below a coin flip. This is the root cause: 7 of 15 picks in the Apr 20
run_15_42_27 had win probs of 44–49%, yet scored 74–89 with confidence 7–9 because the scoring
formula doesn't account for absolute win probability level.

**Immediate fix needed**: raise `min_win_prob` to 0.50 in `default_config.py`. Code-level default
is already 0.50 — the config is actively overriding it to a harmful value.

At limit=15 with a 0.35 threshold, ml_signal dominates final rankings (7/15 picks in one run) and
floods out genuinely higher-quality signals from other scanners.

## Evidence Log

### 2026-04-20 — Fast-loop (run_15_42_27)
- 7 of 15 final picks were ml_signal: FLS (48.8%), DVN (47.4%), APTV (47.0%), OXY (46.4%), PR (44.9%), ELAN (44.4%), EXE (44.3%).
- **All 7 win probabilities are below 50%** — below a coin flip. Yet scores ranged 74–89, confidence 7–9.
- Root cause confirmed: `default_config.py` sets `min_win_prob: 0.35`, overriding the correct 0.50 code default.
- Pickup volume: ml_signal produced 47% of the final ranking in one run. This drowns other signals.
- Context strings are coherent but misleading: "exceptional 48.8% win probability" presented as high-conviction.
- Code fix applied 2026-04-21: raised `min_win_prob` from 0.35 → 0.50 in `default_config.py`.
- Confidence: high (root cause directly confirmed in config)

### 2026-04-20 — Fast-loop (run_15_42_19)
- No ml_signal picks in this run (different config run, or model didn't meet even 0.35 threshold for many tickers).
- Contrast with run_15_42_27 (15 picks total, 7 ml_signal) suggests model output is highly variable between runs.
- Confidence: low (single run comparison)

## Pending Hypotheses
- [x] Does raising the threshold to 50% remove below-coin-flip picks? → Confirmed by analysis: all Apr 20 ml_signal picks were 44–49%. Fix: raise to 0.50 in config.
- [ ] Does raising to 55%+ further improve precision at meaningful cost to recall?
- [ ] Would retraining on the last 90 days of recommendations improve accuracy at the 50%+ threshold?
- [ ] Does reducing limit from 15 → 5 further reduce ml_signal output dominance?
