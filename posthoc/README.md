# Post hoc ridge variant

DECISIONS.md D-003 fixes this rule before the main test scores are available. A ridge alpha
whose inner-validation prediction raises `NonFinitePredictionError` receives an infinite
trial score and loses. Tuning still stops if every alpha has an infinite score. Final-fit
and permuted predictions retain the frozen non-finite check.

Small-alpha candidates stopped the frozen runs after converting large log predictions to
grams. This variant changes which tuning candidates count, without adding a clipping
threshold or changing the model. It reuses the frozen functions in `src/` for tables,
folds, tuning, final fits, permutation and scoring. The alpha grid, larger-alpha tie rule,
seeds, features and sample sets remain frozen. Every output is post hoc and Supplementary
only. The stopped main results stand.

Run from `/home/dev/development/publications`. While Phase 4 has the CPU, use this prefix
for each command, one Python process at a time:

```bash
nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -B
```

`-I` ignores Python environment variables, so `-B` also disables bytecode writes. The
entry point sets the same flag before importing `src`.

The four commands, in order, are:

```bash
.venv/bin/python -I posthoc/ridge_variant.py inner --task T2 --scheme lopo --model ridge_pw_wq --out /tmp/ridge_inner.json
.venv/bin/python -I posthoc/ridge_variant.py run --task T2 --scheme lopo --model ridge_pw_wq --sample-set eligible --workers 4
.venv/bin/python -I posthoc/ridge_variant.py permute --task T2 --model ridge_pw_wq --workers 4
.venv/bin/python -I posthoc/ridge_variant.py score
```

`inner` fits and predicts inner splits only. Its JSON report lists every alpha's trial
score and split scores, the chosen alpha, and each non-finite candidate's outer fold,
inner split, alpha and prediction count. It prints to stdout and writes the same report
to `--out` if supplied. JSON uses `Infinity` for infinite scores, matching the repository's
Python JSON convention. Skipped folds are named. There is no outer fit or prediction.

The lead runs the remaining commands only after `DONE evaluate` appears in
`results/logs/autopilot.log`. They require the freeze guard and both main selection files
to be committed and unmodified. `score` also requires the main `results/metrics/scores.csv`
to exist and be committed and unmodified; it checks that prerequisite without reading its
contents. The build and its tests do not run outer fits, permutations or scoring.

All later output goes under `results/posthoc/ridge_candidate_loses/`:

- `run` writes the frozen prediction and tuning formats, including fold parts,
  `chosen.csv`, `inner_scores.csv`, trials, timings, `run_meta.json` and the runtime log.
  It also writes `nonfinite_candidates.csv` in each fold and the run's tuning directory,
  with columns `outer_fold,inner_split,alpha,count`, and a three-sentence root README.
- `permute` accepts only `permutation.tabular_models`. It requires a complete observed
  post hoc LOPO run, loads its chosen alphas, repeats the frozen observed-run check and
  runs the frozen `permutation.r_tabular` derangements with the frozen seeds. It writes
  `permutation/{task}/{model}.csv`, its JSON metadata and the runtime log.
- `score` imports the frozen evaluation functions. It writes `metrics/scores.csv`,
  `per_pond.csv` and `per_pond_seed.csv` for all saved post hoc runs, including separate
  T2 horizons and the pooled `all` horizon. It writes permutation p-values to
  `stats/permutation.csv` and T2 `ridge_pw_wq` LOPO alpha and prediction differences
  against the independent rebuild to `metrics/codex_agreement.csv`. If that run is
  absent, the agreement table has headers only. It never reads `codex_scores.csv`.

Every JSON carries `posthoc: ridge_candidate_loses` and `decision: DECISIONS.md D-003`.
Metric and agreement CSVs carry the same provenance columns. Prediction and tuning CSVs
retain the frozen schemas for compatibility with the imported functions.

The later runner inherits the frozen resume behavior. A second identical complete run
is a no-op; changed config or inputs are refused. After a crash, completed folds are
reused and incomplete folds restart. Candidate records are written before fold completion
markers. Permutations resume their finished parts and repeat the observed check before
continuing. Scoring replaces derived tables atomically on rerun.

The tests use real tables and only inner validation rows. All test outputs are under
pytest's temporary directory. Run them from the repository root:

```bash
nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -B -m pytest -p no:cacheprovider posthoc/tests/test_ridge_variant.py -v -s
```

`DONE.md` records the executed checks, selected alphas and limits of verification.
