# Independent verification task (Codex), phases 1 and 2

You are verifying another team's analysis blind. Work only inside this directory
(`verification/codex_p12/`). Read the raw data from the absolute paths below. Do NOT open,
list or read anything under `/home/dev/development/publications/src`,
`/home/dev/development/publications/results`, `/home/dev/development/publications/tasks`,
`/home/dev/development/publications/configs` or any other project file: the point is an
independent implementation from this spec alone.

Data (read-only; never modify): `/home/dev/development/publications/data/raw/pondNN.csv` for
NN in 01, 02, 03, 04, 06, 07, 08, 09, 10, 11, 12 (no pond 05). Public IoT aquaponics dataset
(Udanor et al. 2022): sensor rows with a timestamp, six water-quality columns (temperature,
turbidity, dissolved oxygen, pH, ammonia, nitrate), and fish columns (length, weight, and in
some files population). Column names, units and timestamp formats differ between files;
inspect each header yourself. Timestamps appear as "2021-06-19 00:00:05 CET",
"... UTC", "2021-12-06T11:34:43+01:00", or date-only day-first strings such as "5/12/2021"
or "16/01/2022". Use the clock time as written; ignore zone labels.

Python: `/home/dev/development/publications/.venv/bin/python -I` (pandas, numpy available).
Write your scripts and outputs in this directory only.

## Task A: shared sensor records between ponds

For every pair of pond files, report: number of rows in each, number of identical raw
timestamp strings shared, and among rows matched on the raw timestamp string, the number
where all six sensor values are identical. Output `codex_duplicates.csv`.

## Task B: structure of the fish weight labels

For each pond: the number of rows where weight differs from the previous row; per calendar
date, the most frequent weight value. Hypothesis to test: fish were sampled on a fixed 14-day
grid of dates starting 2021-06-19 (2021-06-19 + 14k, k = 0..16), and the daily values between
grid dates are linear interpolations. For every pond and grid date report: whether any row
falls on that date, the most frequent weight and length on that date, and, for the stretch
of observed dates between this grid date and the next one, the median per-day weight
increment and the maximum absolute deviation of single per-day increments from that median
(use calendar-day differences between consecutive observed dates). Output `codex_grid.csv`
and a short `codex_grid_notes.md` stating whether the hypothesis holds for each pond and
listing exceptions.

## Task C: deterministic numbers for pond 02 (original analysis design)

Preparation: read `pond02.csv` with pandas defaults, drop `entry_id`, then drop every row with
any missing value in the remaining columns. Call the result N rows, file order kept.
Sample i means: target = weight of row i, inputs = rows i-L .. i-1. Persistence predicts row
i's weight as the weight of row i-1.

- Design S0: T = int(0.8 N); test targets i in [T, N - 134).
- Design S1: a = int(0.7 N), b = int(0.8 N); L = 36; test targets i in [b + 36, N).

For S0 and S1, report N, the boundaries, the number of test targets, and persistence MAE (g),
RMSE (g) and MAPE (%) on (1) all test targets and (2) only test targets whose weight differs
from the previous row. Also for S0: in the series of test target weights (indexed k = 0, 1,
..., so k = i - T), find the largest absolute change between consecutive k, and report k, the
two row indices, their timestamps, the two weights, and the time elapsed between those rows.
Output `codex_phase2_persistence.csv` and `codex_step.md`.

Report numbers exactly as computed (no rounding beyond 6 significant digits in the CSVs).
End by writing `DONE.md` listing every output file and any assumption you had to make.
