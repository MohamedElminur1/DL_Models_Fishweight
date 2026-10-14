# Re-review of ANALYSIS_PLAN.md (draft v0.3)

Reviewer: stats-reviewer, 2026-10-14. Read-only: the plan, `src/guard.py`, `src/scoring.py`, `results/qc/`, the clock-time audit file, eligibility flags and dates in `data/processed/` (no weights), one simulation on made-up numbers. No model run, no fish-weight metric.

## My 17 items

Resolved (15): 1 to 10 and 13 to 17. Gaps remain in item 2 (T2 part, N1), item 4 (N2), item 7 (§12 and `src/guard.py` agree; see N3) and item 13 (`mlp_static`, N6). Item 16 holds in the file: only pair 1-2 at 0 h shares sensor values at any clock shift.

Partly resolved (2): 11 lacks a rule for a non-finite prediction from ridge, boosting or a deep fit (propose: stop the run, log it) and a tie rule (N2). 12 has no mask ablation for deep models; acceptable, since §8.2 permutes masks with the values, if §10 is worded as in N5.

Not adopted: none.

Checked and fine: §2.7 as coded; §8.2 (about 10^10 distinct joint permutations in T1, 10^9 in T2); the §9.5 fixed sequence; `src/scoring.py` against §9.0.

## New problems

No MUST-FIX. N1 to N4 are SHOULD: small edits, best made before the tag.

**N1. §7.4 T2 hold-out leaks for h = 2 (my v0.2 text had the same hole).** At c = 2021-08-28, 7 of the 12 h = 2 test samples have a same-pond, same-origin h = 1 sample in training whose target (2021-08-14) lies between origin and test target. h = 1 is clean.

> 7.4 T2: train on eligible samples whose target is before c; test on those whose target is on or after c and whose origin is on or after c - 14 days (10 h = 1 and 5 h = 2 samples now).

**N2. §9.4, §7.2 selection.** (a) "T2: S over both horizons" is ambiguous; the confirmatory horizon is h = 1. (b) A tuned model's inner score is a minimum over trials, which flatters ridge and boosting against untuned baselines. (c) Inner folds validate on the ponds the outer test scores. In a null simulation (made-up numbers, 7 ponds, 4 candidates a side, nominal 3.9 %), pooled selection rejected in 3 to 7 % of runs, per-fold selection in 4 to 8 %, a fixed pair in 3.9 %. Nesting buys nothing: keep §9.4, call the level approximate.

> 9.4 T2: the mean of the h = 1 and h = 2 values of S. A tuned model enters with the mean inner S of its best trial. Ties go to the model listed first in §6.1 or §6.3. 10: if another §6.1 baseline has a lower outer S than the primary deep model, say so beside the confirmatory result.

**N3. §12 fix tags.** A fix tag can change `src/` and `configs/` after outer predictions exist (a crash on fold 5 shows folds 1 to 4). `data/processed/` and `uv.lock` are unguarded, and nothing forces the §9.4 selection to be committed before scoring.

> 12 A fix tag is allowed only while no outer prediction file exists; afterwards a change to `src/` or `configs/` makes the changed models' results descriptive. The tag message records the SHA-256 of `uv.lock` and of each `data/processed/` file; the runner checks them and refuses to score unless the §9.4 selection file is committed.

**N4. §6.1, §9.0 `linear`.** "Scored as it is" has no log score when the prediction is 0 or less. The code gives +infinity (`src/scoring.py`); the plan must say so.

> 9.0 A prediction of a weight of 0 or less has log error +infinity, so that pond's S is +infinity and the model cannot be primary. Gram metrics use it as it is. The count is reported.

**Notes.** N5 (§8.2, §10): give the permuted T2 statistic's horizon (h = 1) and say "pond-specific sensor information (readings and availability)", since masks move with the values. N6 (`mlp_static`): in T1 its only input is t, so it is a neural time-only model; it twins LSTM, GRU and CNN-LSTM, not TFT; "lower S than `mlp_static`" needs a pond count. N7 (§2.5): the flat-line test blanks a run crossing an event date using later readings (values are only removed; say so). (§7.4): T1 test dates (t >= 70) lie beyond all training dates (t <= 56), so tree models cannot extrapolate.

## Open until `[PRE]` is filled

Snapshot of the pipeline tables (10:53; still running): eligible T1 events 35 in 7 ponds (1:3, 2:7, 3:5, 4:5, 6:4, 7:6, 8:0, 9:5); T2 h = 1 29, h = 2 24, 7 ponds each. Pond 8 has none, so each confirmatory test has n = 7 and rejects only if the loss ranks sum to 3 or less. Cut-off c = 2021-08-28 (t = 70): 11 T1 test events in 7 ponds, 24 training events. Largest scored T1 t: 112 d. Recheck on the final tables; the paper should say eight cohort ponds, seven scored.

SIGN-OFF: PASS_WITH_NOTES
