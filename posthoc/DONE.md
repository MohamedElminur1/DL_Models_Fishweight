# D-003 ridge variant built

Built for `tasks/posthoc_ridge_variant.md` on 2026-10-14. Changes are uncommitted. The
ticket outside `posthoc/` was left unchanged to respect ownership. Every result of this
procedure is post hoc and Supplementary only.

## Files

- `ridge_variant.py` implements `inner`, `run`, `permute` and `score`. The ridge rule is
  at line 81, inner-only reporting at line 133, worker installation at line 163,
  full-run reuse at line 199, permutation reuse at line 222, agreement at line 249
  and scoring at line 284. It imports the frozen functions rather than copying their
  modelling, fold, permutation or scoring logic.
- `tests/test_ridge_variant.py` contains the real-table inner checks and refusal tests.
  `tests/conftest.py` enables imports under isolated Python and disables bytecode writes.
- `README.md` explains D-003, the four commands in order, outputs, prerequisites,
  provenance and resume behavior.
- `test_run.txt` contains the final pytest output and CPU measurements.
- `test_red.txt` contains the baseline check against the frozen tuner before the rule
  was enabled. It matched the successful run and reproduced both stopped runs.
- `DONE.md` records the result and verification limits. The pre-existing
  `codex_run.log` was not edited.

## Executed checks

Final result: **21 passed in 33.04 seconds**. Process CPU time was 29.01 seconds user
plus 1.26 seconds system, **30.27 CPU seconds**, below the five-minute limit. Total
process wall time was 36.02 seconds. The command ran from the repository root:

```bash
/usr/bin/time -f '\nCPU seconds: user=%U system=%S; wall=%e; max_rss_kb=%M' nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -B -m pytest -p no:cacheprovider posthoc/tests/test_ridge_variant.py -v -s > posthoc/test_run.txt 2>&1
```

`-B` supplements the required environment flag because `-I` ignores Python environment
variables. The entry point also disables bytecode writes before importing `src`.

The baseline run recorded one pass and two `NonFinitePredictionError` errors in
`test_red.txt`. With the wrapper enabled:

- T1 `ridge_time_wq` LOPO matched every saved chosen alpha exactly and every saved
  best-trial inner score within an absolute tolerance of `1e-12`.
- T2 `ridge_pw_wq` had infinite trial scores only for alpha `0.001` in outer folds
  2 and 3. Each was inner split 1 with one non-finite prediction.
- T1 `ridge_time_wq_w28` had infinite trial scores only for the seven frozen alphas
  from `0.001` through `1` in outer folds 2 and 3. Each was inner split 1 with one
  non-finite prediction. No other candidate failed in either stopped run.
- An all-infinite stub learner still raised `NonFinitePredictionError` after tuning.
  A non-finite error during fitting was also preserved.
- The test asserted the rows of every `predict_log` call for both stopped variants.
  All 1,092 calls stayed inside their current outer fold's training rows and excluded
  its `plan.test_idx`.
- `run`, `permute` and `score` refused failed freeze, selection and main-completion
  prerequisites in temporary results roots. `score` also refused missing or
  uncommitted main scores. No commit was created for these checks.
- The worker wrapper was exercised with a real T2 inner fold and a stub outer runner
  that performed no outer work. It installed the rule, wrote the candidate counts
  and JSON provenance, and restored the frozen function bindings.

All test artifact writes used pytest `tmp_path`. Only the test transcripts were written
under `posthoc/`. The tests prohibit outer fitting, observed permutation checks and
prediction loading for scoring.

Both actual `inner` CLI commands also completed with exit code 0 under the required
nice level, thread limits and isolated Python. Their `--out` files matched stdout byte
for byte with `cmp`. Reports are `/tmp/posthoc-ridge-T2-inner.json` and
`/tmp/posthoc-ridge-T1-inner.json`; stdout copies have `stdout` in place of `inner`.

## Chosen alphas from inner tuning

Both runs used LOPO and the eligible sample set. Outer fold numbers are zero-based.

| Outer fold | Held-out pond | T2 ridge_pw_wq | T1 ridge_time_wq_w28 |
|---|---|---|---|
| 0 | 1 | 1000 | 1000 |
| 1 | 2 | 1000 | 1000 |
| 2 | 3 | 1000 | 1000 |
| 3 | 4 | 1000 | 1000 |
| 4 | 6 | 1000 | 1000 |
| 5 | 7 | 1000 | 1000 |
| 6 | 8 | skipped, no test samples | skipped, no test samples |
| 7 | 9 | 1000 | 1000 |

## Verification limits and ownership

The proof reached executed real-table inner tuning, CLI output and refusal checks.
Full outer runs, actual permutations, scoring, prediction agreement and multiple worker
processes were not executed. Those paths are implemented and reuse the frozen functions,
but their end-to-end behavior remains for the lead to verify after the main run. This
deferral follows the task's explicit ban on predicting or scoring outer-test rows and
its one-process build limit. No implementation requirement was omitted.

Writes were confined to `posthoc/` and `/tmp`. No forbidden entry script was run, no
outer-test row was predicted or scored, and no contents of `results/metrics/`,
`results/stats/` or `verification/codex_p4models/codex_scores.csv` were read. A final
`git diff HEAD` of the frozen paths and an untracked-file check under `src/`, `configs/`
and `data/processed/` returned no changes. No package was added and no commit was made.
