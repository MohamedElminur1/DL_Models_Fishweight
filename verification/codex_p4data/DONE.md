# Completed independent verification

Implemented from SPEC.md and the eight named raw CSVs only. No other project source or results were read.

## Output files

- `codex_events.csv`: 70 rows.
- `codex_qc_counts.csv`: 48 rows.
- `codex_duplicate_counts.csv`: 8 rows.
- `codex_daily.csv`: 12144 rows.
- `codex_wq_by_date.csv`: 128 rows.
- `codex_t1.csv`: 63 rows.
- `codex_t2.csv`: 83 rows.
- `verify.py`: rerunnable implementation.
- `test_verify.py`: focused synthetic checks.
- `DONE.md`: this report.

## Counts

| Pond | Events | T1 eligible | T2 horizon 1 eligible | T2 horizon 2 eligible |
|---|---:|---:|---:|---:|
| 01 | 6 | 3 | 3 | 3 |
| 02 | 9 | 7 | 6 | 5 |
| 03 | 6 | 5 | 4 | 3 |
| 04 | 14 | 5 | 4 | 3 |
| 06 | 15 | 4 | 3 | 3 |
| 07 | 9 | 6 | 4 | 4 |
| 08 | 3 | 0 | 0 | 0 |
| 09 | 8 | 5 | 5 | 3 |

## Interpretations and assumptions

- Pond identifiers are zero-padded NN strings. Numeric outputs use 17 significant digits; missing values are empty CSV cells.
- Column labels are mapped by their meaning; values use their written numeric scales, with no unit conversion.
- Event modes use all rows dated that day, before sensor deduplication. The smallest numeric mode wins a tie. n_rows counts all rows dated that day, including rows lacking weight.
- Non-numeric fish values are missing. The specification gives no fish range or finiteness filter, so numeric infinities would remain fish values.
- The single pond 06 row with an empty timestamp is excluded from dated events and sensor processing; no date is inferred.
- Exact duplicate detection compares every parsed CSV cell as raw text, including auxiliary and unnamed columns. CSV quoting itself is not a column value.
- Flat-line equality compares numeric values before QC and before ammonia log10; textual representations of the same number compare equal. Range failures still participate in raw runs, but are counted under range first.
- Missing and interpolated days retain the actual number of distinct hours with valid measurements in n_hours. Anchors are populated only for interpolated days.
- Daily interpolation uses only original observed anchors. Reference-date features reject interpolation with right_anchor equal to or later than the reference date.
- T2 emits only existing exact +14/+28-day targets and requires an earlier origin event. Eligibility depends on origin coverage only.

## Verification

Synthetic checks cover zone stripping and day-first dates, flat-line duration and gaps, QC precedence, daily coverage and short-gap interpolation, reference-date anchor exclusion, the three-day feature minimum, event modes, and T2 previous-event gaps. Run with:

```sh
/home/dev/development/publications/.venv/bin/python -I test_verify.py
/home/dev/development/publications/.venv/bin/python -I verify.py
```

The script checks output dimensions, unique event and daily keys, and QC count reconciliation. Re-running overwrites only its named outputs and report; the raw data remain read-only. If interrupted, rerun to regenerate the complete set.

Completed checks: all 5 synthetic tests passed; the full eight-pond run completed; an independent read-back audit recomputed all 7,168 feature and coverage cells from codex_daily.csv and passed daily, QC, and eligibility invariants. Totals: 70 events, 35 eligible T1 events, 29 eligible T2 horizon-1 samples, and 24 eligible T2 horizon-2 samples.
