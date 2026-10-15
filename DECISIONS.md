# DECISIONS (excerpt)

The analysis decisions D-001 to D-003 of the revision, copied word for word from the
decision record of the working repository. The gate verdicts are reported in the paper;
references to `tasks/`, `LOG.md` and `decisions.tsv` rows point to the working repository,
which also holds the manuscript drafts and is not public.

## D-001 (2026-10-14) The original study analysed pond 2 only

Evidence: `original_code/model-fishweight (1).ipynb` cell 1 reads `IoTPond2.csv`; its first
row and its row count (172,249) match `data/raw/pond02.csv`; manuscript Table 2 (population
50, ammonia with 90 missing values) matches pond 2.
Consequence: Phase 2 reproduces the original design on pond 2.

## D-002 (2026-10-14) Phase 2 protocol fixed before any run

The Phase 2 diagnostic grid, models, seeds and evaluation sets are fixed in
`tasks/phase2_diagnostics.md` (committed before the first run). Every listed configuration is
reported. This keeps the diagnostics free of selection even though they precede the frozen
analysis plan of Phase 3.

## D-003 (2026-10-14) Post hoc ridge variant for the plan 9.0 stops, fixed before any test score

Decided at 17:37, before `src/evaluate.py` has run: no outer prediction has been scored
(`src/train.py` computes no outer error), and the variant is fixed here so that it cannot be
chosen for how it scores.

Facts (LOG.md 17:27, `verification/diag_ridge_stop/README.md`): T1 `ridge_time_wq_w28` and T2
`ridge_pw_wq` stopped under plan 9.0 because small-alpha tuning candidates overflowed on one
inner-validation row (pond 9, 2021-09-25), whose window minimum of dissolved oxygen (1.201) lies
far outside the near-constant values of the inner-training ponds (0.007 and 0.008). The same
check predicts the same stop for T2 `ridge_pw_wq_noind`. The frozen result stands: these
models are reported as stopped, and T2 `ridge_pw_wq` has no permutation test in the main
results. Plan 12 allows no fix tag once outer predictions exist, so any change is post hoc.

Decision: one post hoc variant, reported in the Supplementary Materials as post hoc whatever
it shows, never replacing a frozen result.

1. Rule: in ridge tuning, an alpha whose inner-validation prediction fails the plan 9.0 check
   gets an infinite inner score, so it loses its trial. The run still stops if every alpha of
   a fold fails, or if a final-fit or permuted prediction is not finite. Everything else is the
   frozen procedure (data, folds, features, alpha grid, tie rule, seeds).
2. Why this rule: it is the smallest change that removes the stop. It changes which trials
   count, not the model, and it adds no threshold. Clipping z-scores or dropping
   near-constant columns would change the model and need a new tuning constant.
3. Scope: every ridge run that the Phase 4 runs stopped under plan 9.0, as listed in
   `results/logs/autopilot.log` when the Phase 4 pass ends; for a model of
   `permutation.tabular_models` (T2 `ridge_pw_wq`), the plan 8.2 permutation test with the
   variant's chosen alphas (R = 999). For a ridge run that did not stop the rule changes
   nothing, which the build checks on T1 `ridge_time_wq`. A stop of another learner is
   recorded and decided separately.
4. Code outside `src/` (`posthoc/`, spec `tasks/posthoc_ridge_variant.md`, built by Codex),
   outputs under `results/posthoc/ridge_candidate_loses/`; run only after "DONE evaluate" in
   the autopilot log.
5. Cross-check: the Codex rebuild of Gate 4 part 1 scores candidates on the log scale, which
   is expected to select the same alphas as this rule; its T2 `ridge_pw_wq` alphas and
   predictions (`verification/codex_p4models/codex_inner.csv`, `codex_predictions.csv`) are
   compared with the variant's.
