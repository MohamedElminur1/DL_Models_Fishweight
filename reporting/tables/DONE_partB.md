Part B complete. [Scripts and checks](/home/dev/development/publications/reporting/tables), [CSVs and source sidecars](/home/dev/development/publications/manuscript/src/tables).
Saved as [DONE_partB.md](/home/dev/development/publications/reporting/tables/DONE_partB.md). No commit.

Rows × columns, excluding headers:

- 4: main_results 59 × 16.
- 5: confirmatory 2 × 10; permutation 6 × 8.
- 6: inner_selection 40 × 6.
- 7: chosen_hyperparameters 984 × 10.
- 8: calendar_holdout 44 × 16.
- 9: all_events 4 × 16.
- 10: sensitivity 15 × 16; sensitivity_merged_units 3 × 10; sensitivity_influence 602 × 8.
- 11: per_pond 676 × 11.
- 12: runtimes 74 × 11.
- 13: posthoc_ridge 5 × 8; posthoc_ridge_alphas 21 × 5; posthoc_ridge_agreement 7 × 10.

Verification passed: independent cell checks, 45 corruptions rejected, byte-identical regeneration. Accepted part A files are unchanged.
Unavailable: three stopped runs' complete metrics, inner scores, chosen trials and fit durations; frozen T2 ridge permutation test; agreement for two post hoc variants; GPU, RAM, thread count and OS fields.
The prohibited scores file was not read.

## Revision 1

Applied the five items in the lead review, with no other table changes.

1. `main_results.csv` is now 44 rows × 10 columns, with sensitivity rows and the six specified columns removed. The renamed comparison column reads "Ponds below the primary baseline (of 7)". Its footer gives the primary baseline, seven ponds and sample counts for T1 (35), T2 at 14 days (29) and T2 at 28 days (24), and the five-seed mean ± SD rule. The complete 59-row × 16-column version is `main_results_full.csv`, labelled supplementary, with a source sidecar.
2. R2, gram and percentage values with absolute size at least 10,000 use two significant digits in builder markup, including Unicode minus for negative values. Every affected table has a precision note. Smaller values retain their previous formatting.
3. Every affected footer now says "listed without predictions in results/metrics/completeness.csv".
4. Completed permutation rows have "–" in Status; the stopped row retains its explanation. The footer contains the requested plan 8.2 / plan 9.5 distinction, no multiplicity adjustment and p-value formula verbatim.
5. Extended the independent Part B checks invoked by `check_tables.py` to cover the compact and full tables, large values, baseline/sample/seed notes, completeness wording and permutation status/footer. Extended `verify_part_b.py` with corruption probes for these revisions and `--regenerate` for byte-identical regeneration.

Changed scripts in `reporting/tables/`: `main_results.py`, `phase4.py`, `shared.py`, `confirmatory.py`, `check_part_b.py`, `verify_part_b.py`. This section is appended to `DONE_partB.md`.

Changed existing CSVs in `manuscript/src/tables/`: `main_results.csv`, `calendar_holdout.csv`, `sensitivity.csv`, `per_pond.csv`, `permutation.csv`. Changed source sidecars: `main_results.source.txt`, `calendar_holdout.source.txt`, `sensitivity.source.txt`, `per_pond.source.txt`, `permutation.source.txt`, `all_events.source.txt`, `chosen_hyperparameters.source.txt`, `inner_selection.source.txt`, `runtimes.source.txt`. Added `main_results_full.csv` and `main_results_full.source.txt`.

Checks were run from `/home/dev/development/publications`. The original checker passed before changes. After extending the checker but before regenerating, the old artifacts failed with exit 1:

```text
FAIL: [Errno 2] No such file or directory: '/home/dev/development/publications/manuscript/src/tables/main_results_full.csv'
```

After regeneration, this command exited 0:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I reporting/tables/check_tables.py
PASS: independent Part B cell recomputation, complete rows, stops, seed precision and provenance for 16 panels
PASS: all data cells, structure, row order and provenance for all five revised Part A tables
```

The corruption and regeneration command exited 0. Key output:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I reporting/tables/verify_part_b.py --regenerate
PASS: intact copies pass the full checker
PASS: 48 mismatches rejected; restored copies pass
PASS: 60 total mismatches rejected, including revision cells and footers; restored copies pass
PASS: byte-identical regeneration; all 42 CSVs and sidecars unchanged
```

Regenerated Part B dimensions, excluding headers:

```text
main_results_full: 59 rows x 16 columns
main_results: 44 rows x 10 columns
confirmatory: 2 rows x 10 columns
permutation: 6 rows x 8 columns
inner_selection: 40 rows x 6 columns
chosen_hyperparameters: 984 rows x 10 columns
calendar_holdout: 44 rows x 16 columns
all_events: 4 rows x 16 columns
sensitivity: 15 rows x 16 columns
sensitivity_merged_units: 3 rows x 10 columns
sensitivity_influence: 602 rows x 8 columns
per_pond: 676 rows x 11 columns
runtimes: 74 rows x 11 columns
posthoc_ridge: 5 rows x 8 columns
posthoc_ridge_alphas: 21 rows x 5 columns
posthoc_ridge_agreement: 7 rows x 10 columns
```

A separate snapshot comparison, run as `env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I /tmp/audit_tables_revision1.py`, exited 0:

```text
PASS: full supplement retains all 59 original rows and 16 columns; compact table is exactly its requested projection, 44 x 10
PASS: accepted Part A artifacts unchanged; other CSV changes are only large-value formatting and permutation status
```

`git diff --check -- reporting/tables manuscript/src/tables` also exited 0. All five revisions were completed; no requested item was blocked. The saved-data limitations recorded above still apply. No new packages, no commit, and no writes outside the two allowed table directories and `/tmp`. The prohibited verification scores file was not read.

## Revision 2

Applied exactly the four items in Part B revision 2. No headers, row counts, row order or saved metric values changed.

1. Each stopped row now has exactly one `stopped (plan 9.0)` marker. It appears in S for the result panels, Observed S for permutation, Mean inner-validation S for inner selection, Outer fold for chosen hyperparameters, and Tuning time for runtimes. Other statistic cells are `–`; repeated markers in selection, parameter, value and status cells are also replaced with `–`. Runtime hardware and software cells retain the recorded values.
2. Stop notes now correspond to stopped runs represented in each table. Removed the two unrelated notes from permutation, the unrelated frozen T2 ridge note from sensitivity, and all stop notes from all_events and per_pond. Adjusted the per_pond sentence that referred to the removed notes and the runtime sentence that described the first time cell.
3. Negative numbers use ASCII hyphen-minus, including large-value builder markup. Every affected precision footer now gives `-4.3 × 10^{5}` as its example. Accepted Part A artifacts remain byte-identical.
4. Extended the independent checker and corruption probes for marker placement and repetition, represented stop notes, ordinary and large negative values, negative exponents, and the footer example. Re-ran both requested commands and audited the artifacts against a snapshot made before this revision.

Changed scripts in `reporting/tables/`: `shared.py`, `phase4.py`, `confirmatory.py`, `inner_selection.py`, `chosen_hyperparameters.py`, `runtimes.py`, `per_pond.py`, `check_part_b.py`, `verify_part_b.py`. This section is appended to `DONE_partB.md`.

Changed CSVs in `manuscript/src/tables/`: `main_results.csv`, `main_results_full.csv`, `calendar_holdout.csv`, `sensitivity.csv`, `permutation.csv`, `inner_selection.csv`, `chosen_hyperparameters.csv`, `runtimes.csv`. Changed source sidecars: `main_results.source.txt`, `main_results_full.source.txt`, `calendar_holdout.source.txt`, `sensitivity.source.txt`, `permutation.source.txt`, `all_events.source.txt`, `per_pond.source.txt`, `runtimes.source.txt`. No artifacts were added or removed.

All Python commands below ran from `/home/dev/development/publications`, with no new packages.

The original checker passed before changes. With the Revision 2 checks added but the old artifacts retained, it exited 1:

```text
FAIL: sensitivity: negative numbers require ASCII hyphen-minus
```

After regeneration, this command exited 0:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I reporting/tables/check_tables.py
PASS: independent Part B cell recomputation, complete rows, stops, seed precision and provenance for 16 panels
PASS: all data cells, structure, row order and provenance for all five revised Part A tables
```

The corruption and regeneration command exited 0:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I reporting/tables/verify_part_b.py --regenerate
PASS: 99 total mismatches rejected, including revision cells and footers; restored copies pass
PASS: byte-identical regeneration; all 42 CSVs and sidecars unchanged
```

The snapshot audit exited 0:

```text
env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I /tmp/audit_tables_revision2.py
PASS: every header, row count and row order retained; CSV changes are only stop-cell presentation and ASCII minus
PASS: sidecar changes are only represented stop notes, associated wording and the ASCII minus example
PASS: all accepted Part A CSVs and sidecars are byte-identical to the snapshot
```

All 16 Part B panel dimensions remain as listed under Revision 1. `git diff --check -- reporting/tables manuscript/src/tables` exited 0. All four items are complete; no requested item was blocked. The saved-data limitations recorded above still apply. No commit, no writes outside the two allowed table directories and `/tmp`, and no read of the prohibited verification scores file.
