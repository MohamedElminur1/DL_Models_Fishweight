# Pre-freeze review of ANALYSIS_PLAN.md (draft v0.2)

Reviewer: stats-reviewer, 2026-10-14. I computed only sensor-coverage counts, cross-pond sensor comparisons, combinatorics and simulations on made-up numbers. No model was run and no fish-weight metric computed.

Scale. From `results/audit/daily_coverage.csv` (days with rows in 12+ clock hours, 2-day gaps bridged, no sensor QC), at most 35 T1 events, 35 T2 h=1 origins and 31 h=2 origins qualify, in 7 ponds. 29 of the 35 T1 events are at t <= 70 d. "About 60" is the 63 events with t > 0 before the coverage rule. Fill the `[PRE]` counts before the tag.

Checked and fine: T1 inputs, fold-wise scalers, LOPO partner removal, same-date baselines in LOPO, inner-fold grouping, seeds, §4.3, the event-day exclusion (except item 5).

## MUST-FIX

**1. §8.2 The permutation null is miscalibrated.** Reassigning whole streams while the §4 evaluation set stays fixed gives many events a donor window with too few days: no derangement of the 7 streams (1,854 exist) keeps every eligible T1 event eligible, and on average 30% of events (9 to 46%) get under 7 usable days. Imputation then shifts the null for reasons unrelated to the signal. R = 50 cannot support p < 0.05 (smallest p 0.0196, Monte Carlo SD 0.03 near 0.05); R = 200 is workable. The deep model to permute is picked on test results. The p formula is correct.

> 8.2 At each grid date, the eligible water-quality rows of the streams with an eligible event that date are reassigned among those streams by a random derangement (ponds 1 and 2 are one unit; a date with one stream stays unchanged and is counted), in training and test alike. Targets and eligibility stay as observed. Statistic S (§9.0), tuned hyperparameters fixed, R = 999 for ridge and gradient boosting, p = (1 + #{S_perm <= S_obs}) / (R + 1). Only the primary deep model (§9.4) is permuted, seed 0 in both runs, R = 199.

**2. §7.4, §7.5 The hold-out does not test extrapolation and leaks the future.** Read as each pond's own last two events (the text is ambiguous), the scheme fails: all ponds share the stocking date and weighing grid, so a pond's test dates are still in other ponds' training data (pond 3's k = 4, 5 each have training events from five or six other ponds). Pond 1's test events share five sensors with pond 2's training events on the same dates, the leak §7.1 closes for LOPO. No eligible event is later than k = 8, so under §4 only about 6 T1 events in 4 ponds would be scored.

> 7.4 Calendar hold-out, descriptive only (no p-values or CIs). Cut-off c = 2021-08-28 [PRE: the latest grid date leaving at least 10 eligible T1 events in at least 5 ponds on or after c]. T1: train on eligible events before c, test on those on or after c. T2: train on eligible samples with target before c, test on those with target on or after c. Nothing dated c or later enters training, same-date baselines included. Inner validation: the latest training grid date.

**3. §9.5, §10 The test cannot reach the threshold, and the statistic is not fixed.** Exact one-sided Wilcoxon with Holm over 4 models: n = 8 ponds passes only with a loss in at most one pond, of rank 1 or 2; n = 7 only if the model wins in all 7 (smallest p 0.0078 against the 0.0125 limit); n = 6 cannot pass (0.0156). §9.5 argues from n = 8, but pond 8 has no eligible event; at n = 7 the two-sided test cannot pass at all and the one-sided still needs 7 wins. Simulated power with 7 ponds and four correlated models is 0.21 for an effect of half the SD of the per-pond differences, 0.58 for one SD. Holm runs inside three families, not across them. The rule mixes pooled MAE (§8.2, §10) with per-pond MAE (§9.5), and pooled gram MAE is likely dominated by the 6 events at t >= 98 d.

> 9.0 Primary statistic S: mean over ponds with a scored event of the per-pond mean absolute error of log W; all other metrics are descriptive. 9.5 The primary deep model is tested against the primary baseline (§9.4) by the exact one-sided Wilcoxon on per-pond differences in S; zero differences are dropped and counted, ties use all 2^n sign flips on midranks. Order: T1, then T2 h=1, each at 0.05 only if the previous passed. Everything else, and any task-horizon with fewer than 6 scored ponds, is descriptive. 10: add "A non-significant result is not evidence of no effect."

**4. §9.4 The comparator is chosen after the outer results.** The best baseline is the minimum of 4 (T1) or 7 (T2) noisy test MAEs, so it is biased low. That makes a positive claim harder, as the plan says, but it favours the default framing and biases the bootstrap difference.

> 9.4 The primary baseline is the time-only model with the lowest mean inner-validation S over outer folds (baselines are scored on the same inner folds). The primary deep model is chosen the same way. Both are fixed before any outer prediction is scored.

**5. §2.7 Interpolation uses days the model must not see.** A run of up to 2 missing days is filled from observed days on both sides, found on the whole series. When a window ends in such a run, the filled days at d-1 and d-2 come from day d (the event day) or later, which §3 excludes.

> 2.7 Interpolation for an event or origin on date d uses only observed days before d. A filled day whose later anchor is d or after counts as missing for that event.

**6. §3, §6.1 Horizon and same-date cohort growth disagree.** h is the h-th next event, but the baseline uses d_k + 14h. Missing grid dates give 6 of 35 h=1 and 11 of 31 h=2 samples a gap other than 14 or 28 days (up to 84), so the baseline forecasts the wrong span and per-horizon scores mix horizons.

> 3 h = 1 and h = 2 mean a target event dated exactly d_k + 14 and d_k + 28 days; origins without it are not samples for that horizon. Event k-1 is the previous available event.

**7. §12 The freeze covers only the plan file.** The guard hashes `ANALYSIS_PLAN.md`. §13 sends growth-curve bounds to `configs/`, and feature, QC, model and evaluation code can change after the tag without tripping it. A moved tag is not detected.

> 12 The runner requires (a) the annotated tag, its commit SHA recorded in `decisions.tsv`; (b) `src/` and `configs/` equal to the tag, or each difference listed in `LOG.md` before the run and reported with results; (c) the first complete outer run is the reported run, reruns only after a crash and logged.

## SHOULD

**8. §9.3** With 7 clusters the percentile bootstrap covered about 83% in my simulation (79 to 86%). Call it descriptive, resample ponds jointly for model and baseline, add a leave-one-pond-out table.

**9. §5.3, §7.2** Refit scalers, medians and target standardization in every inner fold; one Optuna study per task, scheme, model and outer fold, no shared storage; inner folds use only streams with an eligible training event.

**10. §2.8, §9.5** Pond 2's file has 89 usable days in the shared period, pond 1's 62, and ammonia differs in 28% of matched rows. Each pond uses its own file; when the pair is a permutation donor, use pond 2's. Also report a Wilcoxon run averaging ponds 1 and 2.

**11. Missing rules.** A sensor with under 3 days in a window is missing; an all-missing training feature is dropped for the fold; the same-date mean needs 3 training ponds, else the growth curve; a failed fit is not retried: the primary baseline's prediction is used and the failure counted; curve bounds and start values go in the frozen config (von Bertalanffy is undefined for t below t0).

**12. §5.1, §5.2** Turbidity is flat-lined in ponds 1 to 6 and sensors stop on different dates, so missing indicators and masks can encode pond and period. Report with and without them.

**13. §6.3** Deep models have no twin without water quality, so architecture and information are confounded: add a static-inputs-only twin of the primary deep model, same tuning and seeds. Expect 25 to 31 training events per T1 fold.

**14. §10 Claims.** May say: model X had lower S than the baseline in n of 7 ponds; pond-specific sensor information was or was not used. Descriptive only: ablations, T2 h=2, calendar hold-out, all-events table, allometric reference. Not claimable: causal effects; anything after t = 112 d; forecasting from LOPO (siblings' later weights are in training); that deep learning is unsuitable (about 30 events).

**15. §2.2, §12** "Rows on non-sampling dates never provide targets or inputs" should say fish values. `--dev` runs score future test ponds: print no error metrics.

## NOTE

**16. Gate 1 evidence gap.** `cross_pond_duplicates.csv` matches rows on the raw timestamp string, zone label included, so 12 pairs with pond 6 (UTC) or 8 (+01:00) were never compared (`src/audit.py:748-790`). Comparing sensor values at equal clock times (plus 1 h shifts for ponds 6 and 8) shows no sharing beyond ponds 1 and 2. Re-run the audit on clock time.

**17.** There are 1,854 derangements of 7 streams, 265 of the 6 that contribute events. Set `early_stopping=False` in `HistGradientBoostingRegressor`; fix torch seed, shuffle generator and thread count.

SIGN-OFF: FAIL
