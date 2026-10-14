# Lead resolution of the check 12 findings (leakage_check.md)

Checks 1 to 11 passed. Check 12 failed on reporting. What the lead changed, and why:

1. Finding 1 (stopped runs not identified in the results). `src/evaluate.py` is frozen (plan
   12), so `results/metrics/completeness.csv` cannot change. Plan 9.0 requires that a stop "is
   logged"; the stops are in `results/logs/autopilot.log` and `results/logs/run_all_phase4.log`.
   To make them visible next to the results, `reporting/stopped_runs.py` (new; run after
   `src/evaluate.py` in the evaluate stage of `run_all.sh`) writes
   `results/logs/stopped_runs.csv`: one row per stopped run with status "stopped (plan 9.0)",
   the logged error, the number of logged attempts, the time the stop was recorded, the
   per-fold prediction files left in `results/tuning/` and the post hoc re-run (D-003). It
   fails if the logs, `completeness.csv` and the files disagree. The manuscript tables mark
   the same runs "stopped (plan 9.0)" (`reporting/tables/shared.py`, `stops`).
2. Finding 2 (15 partial fold prediction files). Kept, not deleted: they are the record of
   what each run produced before it stopped. `src/evaluate.py` reads only
   `results/predictions/`, so they were never scored, and plan 12 (c) reports only a complete
   outer run. `stopped_runs.csv` lists them as unscored. The task wording "no prediction file
   exists" was meant as "no scored prediction file".
3. Finding 3 (wording). The autopilot line "reported, not re-run" refers to its last check
   pass only; the attempt history (4 attempts for T1, 3 for each T2 run, same settings, same
   error) is recorded in `LOG.md` and counted in `stopped_runs.csv`. The sentence "No
   outer-test row was used" in `verification/diag_ridge_stop/README.md` was replaced by an
   exact statement of what step 4 (b) reads.

## After the re-check (21:25)

The re-check (last section of `leakage_check.md`) found finding 3 resolved and findings 1 and 2
partly resolved, both only against the written conditions of check 12. Two changes followed:

1. The stop index moved from `results/logs/stopped_runs.csv` to
   `results/metrics/stopped_runs.csv`, next to `completeness.csv`, with the same content byte
   for byte (`reporting/stopped_runs.py` and `run_all.sh` updated). The reviewer noted that
   plan 12 freezes the scorer's source, not the metrics directory.
2. The check 12 wording "no prediction file exists" was amended in
   `tasks/phase4_leakage_check.md` to what it was meant to guard, no scored prediction file.
   The reviewer found that keeping the 15 unscored fold files complies with the frozen plan.

The reviewer's sign-off line is left as written. The decision is in `DECISIONS.md` Gate 4.
