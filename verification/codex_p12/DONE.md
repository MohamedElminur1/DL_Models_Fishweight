# Completed independent verification

- verify.py
- codex_duplicates.csv
- codex_weight_changes.csv
- codex_daily_weights.csv
- codex_grid.csv
- codex_grid_notes.md
- codex_phase2_persistence.csv
- codex_step.md
- validate.py
- validation.txt
- DONE.md

Assumptions and conventions:

- Raw timestamp means the first column (Unnamed: 0 in pond 08); auxiliary Date/DATE columns are not used. Clock times retain their written values with timezone suffixes removed; slash dates are day-first.
- Shared timestamp counts are distinct nonmissing raw strings. Repeated timestamps use a many-to-many join for matched row-pair counts; a separate count reports distinct timestamps with an identical sensor pair. Missing sensor values never count as identical. Values are compared numerically without unit conversion.
- Weight-change counts compare adjacent file-order rows with both weights present; the first row is not a change. Missing weights are excluded from daily modes. Tied numeric modes select the smallest value. Jan-00 in pond 04 length is treated as missing, not interpreted as a measurement.
- Grid intervals are closed at both ends and use consecutive observed dates, including day gaps. The last grid point has no subsequent interval. Fewer than two dates give blank increment statistics. The notes use a 1e-6 g/day tolerance; CSV statistics retain the computed deviations. Daily modes and file-order change counts are supplied as additional CSVs.
- Task C uses pandas defaults exactly as requested. Indices refer to the cleaned frame, zero-based. S0 does not specify L and does not require it for persistence; its L field is blank. MAPE is mean absolute error divided by the target weight, times 100. k in the step report identifies the later target of the pair.
- CSV floating values use six significant digits. Integer counts and row indices remain exact. No hypothesis of real sampling can be established solely from interpolated labels.

Validation: executable assertions check 55 pond pairs, 187 grid rows, four metric rows, and subset sizes. validate.py independently checks every pair using counted timestamp/sensor tuples and checks persistence with scalar arithmetic; validation.txt records its result. Re-run with `/home/dev/development/publications/.venv/bin/python -I verify.py`. Raw data are read only; all writes are confined to this directory.

Final checks ran successfully: `python -I verify.py` and `python -I validate.py` using the prescribed interpreter. All listed artifacts exist and are nonempty. The independent checks passed; see validation.txt. DONE.md was written last.
