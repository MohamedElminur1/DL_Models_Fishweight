# Independent verification task (Codex): fish events and water-quality tables

You are verifying another team's data preparation blind. Work only inside this directory
(`verification/codex_p4data/`). Read the raw data from the absolute paths below. Do NOT open,
list or read anything under `/home/dev/development/publications/src`, `results`, `tasks`,
`configs`, `data/processed`, `data/interim`, `verification/codex_p12`, or any other project
file. The point is an independent implementation from this spec alone.

Data (read-only; never modify): `/home/dev/development/publications/data/raw/pondNN.csv` for
NN in 01, 02, 03, 04, 06, 07, 08, 09 (the "cohort ponds"; ignore 10, 11, 12). Each file has
sensor rows with a timestamp, six water-quality columns (temperature, turbidity, dissolved
oxygen, pH, ammonia, nitrate) and fish columns (length, weight; some files also population).
Column names, units and timestamp formats differ between files; inspect each header
yourself. Timestamps appear as "2021-06-19 00:00:05 CET", "... UTC",
"2021-12-06T11:34:43+01:00", or date-only day-first strings such as "5/12/2021" or
"16/01/2022". Use the clock time as written and ignore zone labels. Non-numeric cells are
missing; the text "inf" is +infinity.

Python: `/home/dev/development/publications/.venv/bin/python -I` (pandas, numpy available).
Write scripts and outputs in this directory only. Sensor names in outputs: temperature,
turbidity, do, ph, ammonia_log10, nitrate.

## 1. Fish events

Grid dates: 2021-06-19 + 14k days, k = 0..16. For each pond and grid date that has at least
one row (timestamped or date-only) dated that day with a weight value, there is an event:
weight = the most frequent weight among those rows, length = the most frequent length,
t = days since 2021-06-19, n_rows = number of rows dated that day.
Output `codex_events.csv`: pond, date, t, weight, length, n_rows.

## 2. Sensor quality control (rows with a clock time only)

a. Drop a timestamped row if it repeats an earlier row of the same file exactly in every
   column (raw text). Count drops per pond.
b. Sort the timestamped rows by clock time, ties by file order.
c. Per sensor, a value is invalid if missing or non-finite, or out of range: temperature
   outside [10, 40]; ph outside [0, 14]; do <= 0 or > 20; turbidity outside [0, 100];
   ammonia <= 0 or > 500; nitrate <= 0.
d. Flat-line rule per sensor: a run is a maximal sequence of consecutive rows (sorted order)
   with the identical finite raw value. A run ends at a row whose raw value is missing or
   non-finite, and where the time between two consecutive rows exceeds 1 hour. If the last
   clock time minus the first clock time of a run is at least 24 hours, every value of the
   run is invalid.
e. Valid ammonia values are replaced by their log10 (sensor name ammonia_log10).
Output `codex_qc_counts.csv`: pond, sensor, n_values (timestamped rows after step a),
n_invalid_missing_or_nonfinite, n_out_of_range, n_flatline (a value counts under the first
rule it fails in the order missing/non-finite, range, flat-line), n_valid.

## 3. Daily table

For each pond, sensor and calendar date from 2021-05-22 to 2022-01-29: if the valid values
of that date fall in at least 12 distinct clock hours (hour 0 to 23 of the clock time), the
day is observed, with mean, sd (population SD, ddof 0), min and max of the valid values;
otherwise it is missing. Then interpolation: a run of 1 or 2 consecutive missing days with an
observed day immediately before and immediately after the run gets mean, sd, min and max by
linear interpolation in calendar days between those two observed days (left_anchor,
right_anchor); status interpolated. Longer runs and runs at the edges stay missing.
Output `codex_daily.csv`: pond, date, sensor, mean, sd, min, max, n_hours, status,
left_anchor, right_anchor.

## 4. Window features for every grid date k = 1..16

For a reference date d, a sensor-day counts as having a value if it is observed, or if it is
interpolated and its right_anchor is before d. (An interpolated day whose right_anchor is d or
later counts as missing for reference date d.)
Windows: T1 window = the 14 dates d-14 .. d-1; T1 cumulative = 2021-06-19 .. d-1; T2 window =
the 28 dates d-28 .. d-1.
Per sensor and window, over the window days on which that sensor has a value: mean of daily
means (`_mean_w`), mean of daily sds (`_sd_w`), min of daily mins (`_min_w`), max of daily
maxs (`_max_w`). If the sensor has a value on fewer than 3 days of the window, all four are
missing. T1 cumulative per sensor: mean of daily means over the cumulative range
(`_mean_cum`), missing if fewer than 3 days have a value.
n_covered_window = number of window days on which at least 3 of the 6 sensors have a value.
Output `codex_wq_by_date.csv`: pond, date, then with prefix `t1_` the 24 T1 window features,
the 6 cumulative features (`t1_{sensor}_mean_cum`) and `t1_n_covered_window`, and with prefix
`t2_` the 24 T2 window features and `t2_n_covered_window`. One row per pond and grid date
k = 1..16, whether or not the pond has an event that day.

## 5. Eligibility

T1: one row per event with t > 0; eligible when its `t1_n_covered_window` >= 7.
Output `codex_t1.csv`: pond, date, t, t1_n_covered_window, eligible.
T2: events of a pond in date order; an origin is an event that has an earlier event (k-1 is
the latest earlier event, whatever the gap). Horizon 1 exists when the pond has an event
dated exactly origin + 14 days; horizon 2 when it has one dated exactly origin + 28 days.
Eligible when the origin's `t2_n_covered_window` >= 14.
Output `codex_t2.csv`: pond, origin_date, target_date, horizon, dt_prev (origin t minus
previous event t), t2_n_covered_window, eligible.

Report numbers as computed (at least 9 significant digits). End by writing `DONE.md` listing
every output file, the counts of events, T1 eligible events and T2 eligible samples per pond
and horizon, and any assumption you had to make.
