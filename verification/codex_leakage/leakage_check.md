# Phase 4 leakage and statistics review

Sign-off fails check 12. The stopped ridge runs are not identified as stopped in the metrics output, and each retains five partial prediction files. The aggregate prediction files are absent, those partial files were not scored, and no main permutation result exists for a stopped run. The checked splits, preprocessing, feature visibility, selection and numerical results agree with the frozen plan.

Reviewed on 2026-10-14 against `analysis-plan-frozen`, commit `91b38b99494313ed4b1e6eb570bcdf0e41ab0639`. I read `CLAUDE.md`, the tagged plan, Gate 1 and Gate 2 in `DECISIONS.md`, `configs/phase4.yaml` and the task. The final verification started at HEAD `8dca5cffb3561778f36a1d6fe8f938bceb0282c1` and ended at `47745973ede493d782a091ae0c357757df004f54`. Other work advanced HEAD; all 3,811 files in the review's input manifest retained identical SHA-256 hashes during that run. Evidence: `audit_details.json/head_start,head_end`; `input_manifest.csv/rows 2-3812`. PASS.

All evidence paths below are relative to the repository root unless a filename alone is given, which means this review directory. CSV row numbers include the header as row 1. A prediction's `row_id` corresponds to its source table's physical row number minus 2. References to the plan use the tagged version; the current plan has identical bytes.

The independent checker is `verification/codex_leakage/verify_leakage.py`. It constructs its own reference folds, features, scores, sign-flip distributions and bootstrap draws. Calls to sequence and scaler functions compare the implementation with those independently constructed references. No model was retrained. All 31,670 numerical and membership comparisons passed. Scalar numeric comparisons use `rtol=1e-10`, `atol=1e-11`; membership, dates and identifiers were checked exactly. The separate existence and reporting failures appear below. Evidence: `numeric_comparisons.csv/rows 2-31671`; `checks.csv/rows 2+`. PASS.

Run from `/home/dev/development/publications`, one Python process at a time:

```bash
nice -n 19 /home/dev/development/publications/.venv/bin/python -I verification/codex_leakage/verify_leakage.py
```

The checker replaces only its own evidence files in this directory and uses `/tmp/leakcheck` for scratch. I changed no analysis, result or other repository file and made no commit.

**1. Freeze.** The tag is annotated and resolves to the SHA recorded in `decisions.tsv:40`. All 71 files under `results/predictions/` were first added in commit `95b15b2daba761306af2e9d652ad793fc18e818c`, a descendant of the freeze. No later commit touches the frozen plan, `src/`, `configs/`, `data/processed/` or `uv.lock`; their working-tree contents also match the tag, with no untracked files in those paths. No fix tag is present or needed. `results/dev/` has no tracked files. The scan of 823 dev CSV, compressed CSV and JSON outputs found no populated error, objective or loss value. Epoch counts and runtime seconds are not error values.

Evidence: `decisions.tsv:40`; `prediction_history.csv/rows 2-72`; `dev_scan.csv/rows 2-824`; `git_evidence.txt:1`; `final_git_check.txt:1`; frozen `ANALYSIS_PLAN.md:323`. PASS.

**2. Outer folds.** Independent reconstruction from the saved task tables gives these eligible-set counts. T2 counts combine both horizons, as required for fitting one model.

| Held-out pond | T1 train | T1 test | T2 train | T2 test | Partner dropped |
|---|---:|---:|---:|---:|---|
| 1 | 25 | 3 | 36 | 6 | 2 |
| 2 | 25 | 7 | 36 | 11 | 1 |
| 3 | 30 | 5 | 46 | 7 | none |
| 4 | 30 | 5 | 46 | 7 | none |
| 6 | 31 | 4 | 47 | 6 | none |
| 7 | 29 | 6 | 45 | 8 | none |
| 9 | 30 | 5 | 45 | 8 | none |

Every saved runtime plan agrees on training/test counts, excluded counts, test ponds and dropped ponds. Every one of the 71 aggregate prediction files has exactly the expected multiset of fold, pond and row identifiers for each required seed. Truth values and dates match the source tables. Pond 8 supplies no eligible training or test sample. It does supply samples in the secondary T1 all-events analysis, which the frozen plan explicitly requires; that analysis was checked separately against all 63 events.

Evidence: `outer_folds.csv/rows 2-25`; `numeric_comparisons.csv/check 2 and check 7 rows`; `results/logs/runtime.csv/row 2`; `src/folds.py:194`; frozen `ANALYSIS_PLAN.md:109` and `ANALYSIS_PLAN.md:118`. PASS.

**3. Inner folds.** I independently sorted eligible training stream units by smallest pond ID and assigned them round-robin to three folds. All 66 LOPO inner splits have disjoint training and validation ponds, and ponds 1 and 2 always remain in one stream unit. For example, outer fold 2 validates on ponds 1, 2 and 7 in inner fold 0, while fitting ponds 4, 6 and 9. All 903 saved inner-score row counts match the reconstructed plans. The two additional calendar inner splits separate dates, as plan 7.4 specifies; pond disjointness is not required for those splits.

Evidence: `inner_folds.csv/rows 2-69`, especially row 8; `numeric_comparisons.csv/check 3 rows`; `src/folds.py:146`; `src/folds.py:201`; `src/tuning.py:176`. PASS.

**4. Fitting only on training rows.** Ridge determines active columns, medians, indicator scaling and regression coefficients from `train`. Its prediction path reuses those fitted values. Gradient boosting drops all-missing columns using training rows, fits only their targets and disables early stopping. Growth curves choose training AIC; cohort means and cohort growth dictionaries use training events/samples only. T2 anchored growth curves extract origin and target events only from their supplied training samples.

Deep channel statistics are restricted to visible sensor-days referenced by training row IDs, deduplicated by pond, day and sensor. Static and target statistics use those same training IDs. My own calculation matched all channel, static and target scaler statistics in 60 outer-training or inner-training subsets. Validation and test batches use the fitted scalers. `make_batch` also constructs a standardized target for a prediction batch, but `predict_log` passes only channels, statics and padding to the network; the held-out target cannot affect a prediction.

Inner validation supplies the authorized tuning scores and early-stopping losses. Final fitting passes no validation frame and trains for the median best epoch of the selected inner trial. All 80 neural fold choices and all 400 final seed fits agree with that rule; every final learning-curve validation loss is blank. This conclusion combines source tracing with executed checks of saved curves and actual scaler outputs, without rerunning training.

Evidence: `src/baselines.py:569`, `src/baselines.py:617`, `src/baselines.py:236`, `src/baselines.py:270`, `src/baselines.py:388`, `src/baselines.py:463`; `src/models/common.py:259`, `src/models/common.py:299`, `src/models/common.py:341`, `src/models/common.py:399`, `src/models/common.py:419`; `src/tuning.py:174`, `src/tuning.py:415`; `final_epochs.csv/rows 2-81`; `numeric_comparisons.csv/check 4 rows`. PASS.

**5. Features before the event.** I recomputed the main window features, T1 cumulative features, coverage and eligibility for every table row: 63 T1 events and 83 T2 origin/horizon samples across all eight cohort ponds. All 4,174 comparisons agree, including missing-value patterns and the minimum of three visible days per sensor.

The adjacent-interpolation case is explicit. T1 pond 2 on 2021-07-17 excludes the interpolated temperature value on 2021-07-16 because its right anchor is 2021-07-17. T2 pond 1 at origin 2021-07-17 excludes the equivalent sensor-day for the same reason. These are not treated as observed days. Other sensors and references with adjacent interpolation are listed in the evidence.

I also independently assembled all 88 eligible sequences and compared their values, validity masks, padding and final dates with `DailyStore.visible_window` and `build_sampleset`. T1 begins at day 0 and ends on the event date minus one day; T2 contains exactly the 28 days ending on the origin date minus one day. All comparisons agree.

Evidence: `feature_references.csv/rows 2-147`; `adjacent_interpolations.csv/row 7` and `/row 60`; `data/processed/t1_table.csv/row 8`; `data/processed/t2_table.csv/row 2`; `data/processed/daily_wq.csv/row 1850` and `/row 332`; `sequence_references.csv/rows 2-89`; `src/models/common.py:144`, `src/models/common.py:211`, `src/models/common.py:228`. PASS.

**6. Inputs allowed by the plan.** All 22 configured tabular feature lists satisfy the task-specific allowlists. T1 learned predictors receive time and/or sensor features; only the named length ablations receive `log_l`, and the separate allometric reference uses length. A T1 weight is used as the fitting target, never as a predictor. T2 learned predictors receive only the two permitted log past weights, time since stocking, previous-event gap, target gap and optional sensor features. No absolute target date or target weight enters those feature matrices. Neural statics are exactly `[t]` for T1 and `[log_w_k, log_w_prev, t_k, dt_prev, dt_target]` for T2.

The cohort-growth baseline uses the origin/target date pair as a lookup key, as explicitly prescribed by plan 6.1. This does not introduce an additional learned predictor and its lookup values come only from training samples.

Evidence: `input_features.csv/rows 2-23`; `configs/phase4.yaml:91`, `configs/phase4.yaml:104`, `configs/phase4.yaml:131`; `src/baselines.py:519`, `src/baselines.py:305`, `src/baselines.py:463`; `src/models/common.py:231`. PASS.

**7. Calendar hold-out.** T1 has 24 training events dated before 2021-08-28 and 11 test events on or after that date. T2 has 19/12 training samples at h1/h2 and 10/5 test samples. Every T2 test origin is on or after 2021-08-14. The seven excluded h2 samples are precisely source-table rows 5, 13, 24, 31, 49, 71 and 81, one per scored pond, with origin 2021-07-31 and target 2021-08-28. None occurs in training or any calendar prediction file.

The latest training event/target date is 2021-08-14. Training sensor windows end before their reference dates and apply the causal anchor rule. Same-date cohort dictionaries are built only from the training frames, so late test dates use the prescribed growth-curve fallback rather than late observations. Saved runtime plans agree with these sets.

Evidence: `calendar_rows.csv/rows 2-89`; `outer_folds.csv/row 9` and `/row 17`; `numeric_comparisons.csv/check 7 rows`; `src/folds.py:231`, `src/folds.py:243`; `src/baselines.py:270`, `src/baselines.py:463`; frozen `ANALYSIS_PLAN.md:207`. PASS.

**8. Permutation test.** The code draws uniformly conditional derangements by rejecting permutations with fixed points, within each date's stream units. Ponds 1 and 2 share a unit and pond 2 is its donor. Seeds are `20261014 + r`. Whole tabular sensor columns, including NaNs, move together; sequences are rebuilt from the donor with their validity masks and causal visibility rule. Targets, eligibility and non-sensor predictors remain fixed. Saved fold hyperparameters and epoch counts are loaded once, then final fits are repeated without tuning. Observed and permuted deep fits use seed 0. The observed-repeat metadata matches independently computed seed-0 S.

| Task/model | R | Permuted S <= observed S | Recomputed p |
|---|---:|---:|---:|
| T1 ridge_time_wq | 999 | 947 | 0.948 |
| T1 gbm_time_wq | 999 | 869 | 0.870 |
| T1 lstm | 199 | 0 | 0.005 |
| T2 gbm_pw_wq | 999 | 259 | 0.260 |
| T2 cnn_lstm | 199 | 190 | 0.955 |

All permutation indices, counts, summary statistics and p-values agree. Stopped T2 ridge has no main permutation result, as assessed under check 12.

Evidence: `src/permute.py:85`, `src/permute.py:111`, `src/permute.py:138`, `src/permute.py:188`, `src/permute.py:208`, `src/permute.py:255`, `src/permute.py:273`; `configs/phase4.yaml:216`; `recomputed_permutation.csv/rows 2-6`; `results/stats/permutation.csv/rows 2-6`. PASS.

**9. Selection before scoring.** Both selection files were added in `95b15b2` at 18:21:19 UTC. The first additions under either `results/metrics/` or `results/stats/` are in descendant commit `494084e` at 18:45:39 UTC. The selection input hashes match the current inner-score files. I recomputed each model's mean over inner folds, then outer folds, and applied the first-listed-model tie rule. T2 inner S also agrees with the arithmetic mean of its two horizon-specific values.

| Task | Selected baseline | Mean inner S | Selected deep model | Mean inner S |
|---|---|---:|---|---:|
| T1 | cohort_mean | 0.250558297490 | lstm | 0.203901149179 |
| T2 | ridge_pw | 0.250957604244 | cnn_lstm | 0.221082279486 |

Every candidate and both arg-min choices agree. The scorer's entry point checks that both selection files are committed and unmodified before scoring.

Evidence: `selection_history.csv/rows 2-5`; `selection_recomputed.csv/rows 2-20`; `results/selection/primary_T1.json:119`; `results/selection/primary_T2.json:155`; `src/select_primary.py:85`, `src/select_primary.py:98`; `src/evaluate.py:255`, `src/evaluate.py:568`. PASS.

**10. Scores.** I calculated absolute log error directly from saved gram predictions and truths, averaged within pond and seed, averaged each pond over seeds, then averaged ponds. Pooled MAE, RMSE, MAPE and R-squared were calculated separately for each seed and summarized across seeds. All 133 score rows, 293 pooled seed rows, 1,953 pond/seed rows and 893 seed-averaged pond rows agree, including sample counts, nonpositive counts and reported SDs. There are no saved non-finite learned predictions.

I independently assigned midranks and enumerated all 128 sign patterns for each primary comparison. Both have seven nonzero differences and zero dropped differences.

| Comparison | S deep | S baseline | Positive rank sum | One-sided p | Two-sided p | Fixed-sequence result |
|---|---:|---:|---:|---:|---:|---|
| T1 LSTM / cohort_mean | 0.290055090312 | 0.265477372022 | 16 | 0.656250 | 0.812500 | not rejected |
| T2 h1 CNN-LSTM / ridge_pw | 0.332120920519 | 0.294720616648 | 22 | 0.921875 | 0.218750 | not tested |

The independent T1 pond bootstrap, B=2000 and seed 20261014, gives the LSTM S interval `[0.217388573786, 0.380593186223]` and the paired difference interval `[-0.072752833912, 0.131373194862]`. Both match. I additionally recomputed all saved bootstrap, pairwise, influence and merged-stream statistics, with no mismatches. The implementation rounds Wilcoxon differences to 12 decimals, a convention recorded before freeze; raw-difference enumeration gives the same reported results in the checked data.

Evidence: `recomputed_scores.csv/rows 2-134`; `recomputed_pooled_by_seed.csv/rows 2-294`; `recomputed_per_pond_seed.csv/rows 2-1954`; `recomputed_per_pond.csv/rows 2-894`; `recomputed_confirmatory.csv/rows 2-3`; `bootstrap_spotcheck.csv/rows 2-2001`; `recomputed_primary_comparisons.csv/rows 2-5`; `results/stats/confirmatory.csv/rows 2-3`; `src/evaluate.py:104`; `LOG.md:156`. PASS.

**11. Claims.** The Phase 5 entry was checked in a 39-item claim ledger. Its stated S values, primary differences and bootstrap intervals, lower-error pond counts, named better baselines, permutation results, influence results, merged-stream direction, ablations and allometric reference agree with the independently verified numbers. T1 LSTM is lower than the cohort mean in ponds 1, 2 and 4; T2 h1 CNN-LSTM is lower than ridge on past weights in two ponds. Neither task is presented as a confirmatory rejection, and the entry explicitly says T2 was not tested.

The benchmark framing follows plan 10. The T1 LSTM permutation result supports use of pond-specific sensor information without establishing superiority to the baseline. The entry keeps ablations and robustness descriptive, states that non-significance is not evidence of no effect, and excludes causal claims, forecasting claims from LOPO, extrapolation beyond scored times and general claims against deep learning. Title/abstract changes and the withdrawal of old tables/figures are stated consequences, not verified manuscript edits in this review.

Evidence: `phase5_claims.csv/rows 2-40`; `DECISIONS.md:213`, `DECISIONS.md:219`, `DECISIONS.md:235`, `DECISIONS.md:242`, `DECISIONS.md:250`, `DECISIONS.md:254`; frozen `ANALYSIS_PLAN.md:296`. PASS.

**12. Plan 9.0 stops.** The logs identify exactly three stopped LOPO runs. No aggregate prediction file exists for any of them under `results/predictions/`; no main permutation file exists for any of them under `results/permutation/`. Their partial fold files were not read by the scorer.

| Stopped run | Error lines in run_all_phase4.log | completeness.csv row | Retained partial prediction files |
|---|---|---:|---:|
| T1 ridge_time_wq_w28 | 154, 198, 204, 501 | 44 | 5 |
| T2 ridge_pw_wq | 269, 274, 523 | 62 | 5 |
| T2 ridge_pw_wq_noind | 323, 329, 540 | 76 | 5 |

Two requested conditions fail. First, `results/metrics/completeness.csv` contains only `has_predictions=False` for these runs. It supplies no stopped status or reason, and uses the same representation for unscheduled calendar sensitivity models. The score and pooled tables omit the stopped LOPO runs. Second, the task's literal requirement that no prediction file exist is false: each run retains `fold00`, `fold01`, `fold04`, `fold05` and `fold07` prediction CSVs under `results/tuning/`. Aggregate-only absence passes; absence of all prediction files does not. These are reporting and intermediate-artifact findings, not evidence that partial runs entered any score.

All repeats are logged. T1 failed four times and each T2 run three times. Completed folds were resumed, rather than overwritten, while failing folds were attempted again. This qualifies the autopilot's phrase "not re-run"; it does not conceal an alternate completed run. Evidence: `results/logs/autopilot.log:4`, `results/logs/autopilot.log:11`, `results/logs/autopilot.log:15`; `results/logs/run_all_phase4.log:193`, `results/logs/run_all_phase4.log:496`; `stopped_runs.csv/rows 2-4`. NOTE.

Diagnostic model fitting and prediction use inner-training and inner-validation indices, including the T1 and T2 overflow reproductions. One diagnostic also surveys raw eligible sensor-feature columns across all ponds outside an individual inner split. Its blanket statement "No outer-test row was used" is therefore broader than the code supports. That survey computes no outer-test prediction or target error and does not establish a fitting leak. Evidence: `verification/diag_ridge_stop/README.md:5`; `verification/diag_ridge_stop/step1_w28.py:18`; `verification/diag_ridge_stop/step3_models.py:15`; `verification/diag_ridge_stop/step4_verify.py:23`; `verification/diag_ridge_stop/step6_t2.py:34`; `verification/diag_ridge_stop/step7_t2_detail.py:22`. NOTE.

D-003 is decided and explicitly labelled post hoc in `DECISIONS.md:175`. Its code lives outside `src/`, redirects outputs outside the main result directories, checks for `DONE evaluate`, and retains the main selection guard. The post hoc root README, run/permutation metadata and every scored table identify D-003 and its supplementary status. The three missing main aggregates remain missing; no frozen output was replaced. Evidence: `posthoc/ridge_variant.py:34`, `posthoc/ridge_variant.py:56`, `posthoc/ridge_variant.py:64`; `results/logs/autopilot.log:34`; `results/posthoc/ridge_candidate_loses/README.md:3`; `posthoc_labels.csv/rows 2-10`; `input_manifest.csv/rows 2-3812`. PASS.

Check 12 overall evidence: task `tasks/phase4_leakage_check.md:70`; `results/metrics/completeness.csv/rows 44, 62, 76`; `src/evaluate.py:353`, `src/evaluate.py:195`; `results/tuning/T1/lopo/ridge_time_wq_w28/fold00/predictions.csv/row 2`; `results/tuning/T2/lopo/ridge_pw_wq/fold00/predictions.csv/row 2`; `results/tuning/T2/lopo/ridge_pw_wq_noind/fold00/predictions.csv/row 2`; `stopped_runs.csv/rows 2-4`. FAIL.

Findings ranked by severity:

1. **Medium, blocks sign-off.** The main metrics output does not identify the three stopped runs as stopped. A reader cannot distinguish a numerical stop from an unscheduled analysis without consulting separate logs. This fails the explicit reporting requirement even though `DECISIONS.md` describes the stops. Evidence: task `tasks/phase4_leakage_check.md:72`; `results/metrics/completeness.csv/rows 44, 62, 76`; `src/evaluate.py:366`. FAIL.
2. **Low, fails the literal file-absence requirement.** Fifteen partial fold prediction files survive across the three stopped runs. They are outside the scorer's prediction directory and have contributed no reported score. The aggregate-only condition passes, but the task does not limit its wording to aggregates. Evidence: task `tasks/phase4_leakage_check.md:71`; `stopped_runs.csv/rows 2-4`; the three `fold00/predictions.csv/row 2` examples above; `src/evaluate.py:195`. FAIL.
3. **Low, audit-trail wording.** Repeated attempts are fully logged, but the later "not re-run" wording needs that history, and the diagnostic's inner-only claim needs the raw-feature survey qualification. Neither changes the verified frozen results. Evidence: `results/logs/autopilot.log:4`, `results/logs/autopilot.log:15`; `results/logs/run_all_phase4.log:198`; `verification/diag_ridge_stop/README.md:5`; `verification/diag_ridge_stop/step4_verify.py:23`. NOTE.

SIGN-OFF: FAIL

## Re-check after the lead resolution

Re-checked on 2026-10-14 from the artifacts, after reading `CLAUDE.md`, check 12, this report and `LEAD_RESOLUTION.md`. The tagged plan remains the scientific specification. The verdicts below also distinguish compliance with that plan from the explicit acceptance conditions in check 12.

I ran an unchanged copy of `reporting/stopped_runs.py` against copied inputs under `/tmp/leakcheck/lead-recheck.v4fev61y/`. It exited 0 and produced a byte-for-byte match to `results/logs/stopped_runs.csv`. Independent checks against both logs, my earlier `stopped_runs.csv` and the files confirmed all three rows, their error messages, timestamps, 4/3/3 failed attempts, five retained folds per run and separate D-003 paths. I did not run the evaluate stage or any model. Evidence: `recheck_evidence.txt:17`; `recheck_evidence.txt:23`; `recheck_evidence.txt:26`.

1. Finding 1 is partly resolved. The new index explicitly identifies all three runs as `stopped (plan 9.0)` and supplies the missing reasons and attempt history. `run_all.sh:111` calls the reporter after `src/evaluate.py`; `bash -n run_all.sh` passed. The manuscript tables also retain stopped rows, through `reporting/tables/phase4.py:55` and `reporting/tables/phase4.py:102`, rather than the `shared.py` helper named in the resolution. This fixes disclosure in the reporting outputs. However, `results/metrics/completeness.csv` still contains only `has_predictions=False` at rows 44, 62 and 76, and the other metric tables still omit these runs. The requirement that `results/metrics/` identify them as stopped remains unmet. The frozen plan requires logging and reporting every model, without prescribing that directory; section 12 freezes the scorer's source, not the metrics directory itself. Evidence: `reporting/stopped_runs.py:25`; `results/logs/stopped_runs.csv/rows 2-4`; `manuscript/src/tables/main_results_full.csv/rows 21, 30, 37, 49, 56`; task `tasks/phase4_leakage_check.md:72`; frozen `ANALYSIS_PLAN.md:260`, `ANALYSIS_PLAN.md:315`, `ANALYSIS_PLAN.md:323`.

2. Finding 2 is partly resolved. The index now labels the surviving fold files as unscored, and its fold lists exactly match my earlier inventory. All 15 files retain their earlier SHA-256 values. Each run still has only `fold00`, `fold01`, `fold04`, `fold05` and `fold07` predictions under `results/tuning/`; its main aggregate and main permutation files remain absent. No metric or statistic scores these stopped LOPO runs. Keeping the files is consistent with the frozen plan: section 9.0 requires stopping and logging, section 11 saves predictions, and section 12(c) reports the first complete outer run. None requires deleting intermediate predictions. The literal check 12 condition that no prediction file exists is still false; the lead's narrower interpretation is not written into the task. Evidence: `stopped_runs.csv/rows 2-4`; `results/logs/stopped_runs.csv/rows 2-4`; `recheck_evidence.txt:26`; `recheck_evidence.txt:30`; `recheck_evidence.txt:34`; `src/evaluate.py:195`; task `tasks/phase4_leakage_check.md:71`; frozen `ANALYSIS_PLAN.md:260`, `ANALYSIS_PLAN.md:318`, `ANALYSIS_PLAN.md:332`.

3. Finding 3 is resolved. The new index makes the preceding attempts explicit, so the historical autopilot phrase "not re-run" can no longer be read as evidence that these models had no earlier failed attempts. Its counts agree with error lines 154, 198, 204 and 501 for T1, 269, 274 and 523 for T2 ridge, and 323, 329 and 540 for T2 without indicators. The revised diagnostic paragraph also matches the scripts. Every fit and prediction selects inner-training or inner-validation rows. Step 4(b) additionally surveys the three dissolved-oxygen feature columns across eligible cohort rows, including rows held out in some outer folds; it computes no weight prediction or target error. Evidence: `results/logs/autopilot.log:4`, `results/logs/autopilot.log:15`; `results/logs/stopped_runs.csv/rows 2-4`; `verification/diag_ridge_stop/README.md:5`; `verification/diag_ridge_stop/step1_w28.py:18`; `verification/diag_ridge_stop/step2_detail.py:14`; `verification/diag_ridge_stop/step3_models.py:15`; `verification/diag_ridge_stop/step4_verify.py:23`; `verification/diag_ridge_stop/step5_verify2.py:26`; `verification/diag_ridge_stop/step6_t2.py:34`; `verification/diag_ridge_stop/step7_t2_detail.py:22`.

Both `git diff HEAD -- results/predictions results/metrics results/stats src configs` and `git status --short --untracked-files=all -- results/predictions results/metrics results/stats src configs` returned empty output. All 113 tracked files in these paths also match the original review's manifest. The before/after snapshot found identical paths and hashes for all 137 files, including 24 ignored bytecode caches. Other work advanced HEAD from `fe56d051e58c0de7f7afaf9a54c61031f3d9c513` to `8090392d4b8185e4b429b39ab7da98d90e902df9` during these checks; the protected contents and all 25 resolution inputs stayed identical. Evidence: `recheck_evidence.txt:3`; `recheck_evidence.txt:6`; `recheck_evidence.txt:12`.

The remaining failure concerns check 12's written reporting-location and file-absence conditions. The new stop index is correct, retaining the 15 unscored files complies with the frozen plan, and this re-check found no change to the verified analysis. I wrote only this appended section, `recheck_evidence.txt` and scratch files under `/tmp`; I made no commit.

SIGN-OFF: FAIL
