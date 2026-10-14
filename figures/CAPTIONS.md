# Caption drafts for the revision figures

Drafts for the lead to number and edit. Each section gives the caption, then the script and
the files it reads. Rebuild any figure from the repo root with
`.venv/bin/python -I reporting/figs/<name>.py`. Dates are ISO as in `ANALYSIS_PLAN.md`,
weights are in g, and every pond keeps one colour in every figure.

## fig_events

Fish weight in the eight cohort ponds (panels a to h: ponds 1, 2, 3, 4, 6, 7, 8 and 9) on a
logarithmic axis. Markers are the real weighings on the 14-day grid that starts on 2021-06-19
(70 weighings, 63 after day 0). Filled circles are weighings that enter task T1: at least 7 of
the 14 days before the weighing are covered, a covered day being one on which at least 3 of the
6 sensors have a value (35 weighings). Open circles are real weighings that do not meet this
rule, and diamonds mark the weighing at t = 0 (stocking), which pond 8 does not have. The thin
grey line is the daily weight stored in the source files, a straight-line interpolation between
weighings, and it is broken on days without a row. The dashed vertical line is the cut-off of
the calendar hold-out, 2021-08-28.

Script `reporting/figs/fig_events.py` (helper `reporting/figs/_events.py`). Reads
`data/processed/events.csv` (weighings), `data/processed/t1_table.csv` (column `eligible_eval`),
`results/audit/daily_labels.csv` (column `weight_mode`, the grey line) and `configs/phase4.yaml`
(`chrono.cutoff`). Writes `figures/main/fig_events.png` and `figures/main/fig_events.pdf`.

## fig_coverage

Sensor availability after quality control in the eight cohort ponds, 2021-06-01 to 2022-02-05.
For each day the stack shows how many of the six sensors have a daily value (0 to 6). Blue cells
are observed sensor-days (valid readings in at least 12 distinct clock hours). Grey cells are
interpolated sensor-days (runs of at most 2 missing days with observed days on both sides). The
order within a stack does not identify a sensor. Markers below each strip are the weighings of
the pond, as in fig_events: filled circles are T1 events with at least 7 covered days
in the 14 days before, open circles are weighings that do not meet this rule, and diamonds are
t = 0. The strips show the status saved in the daily table; the eligibility rule also counts an
interpolated day as missing for an event when its later observed anchor falls on or after the
event date. Ponds 1 and 2 share one sensor stream for temperature, turbidity, DO, pH and
nitrate, and differ in usable days and in ammonia.

Script `reporting/figs/fig_coverage.py`. Reads `data/processed/daily_wq.csv` (columns `pond`,
`date`, `sensor`, `status`), `data/processed/events.csv` and `data/processed/t1_table.csv`.
Writes `figures/main/fig_coverage.png` and `figures/main/fig_coverage.pdf`.

## fig_original_test

Pond 2, test period of the original design (34,298 test targets, 2021-10-07 21:09 to
2021-10-31 03:39, CET), plotted against the time stamp `created_at`. (a) The whole period.
(b) A zoom on 2021-10-12 to 2021-10-25. The thick grey line is the label `y_true`, the daily
weight of the source file, which is a straight-line interpolation between weighings and is
drawn as a step line. Blue is persistence, which predicts each target with the previous label
value. Purple lines are the CNN-LSTM retrained on the notebook design (seeds 0, 1 and 2). Green
dots are the CNN-LSTM trained on a random split (seed 0), restricted to its targets inside this
period. Shaded bands are stretches longer than 24 h without a logged row. The darker band is the
7.5-day gap from 2021-10-14 06:11 to 2021-10-21 18:11, across which the label rises by 29.05 g
(328.75 g to 357.80 g); a gap of 2 days follows with a further 8.30 g. Lines are not drawn
across gaps, and a row that stands alone between two gaps is a single dot. Filled circles with
a horizontal bar are the two real weighings of pond 2 in the period (2021-10-09 and
2021-10-23), which carry a date and no time; the bar spans that day.

Script `reporting/figs/fig_original_test.py`. Reads
`results/phase2/predictions/S0_notebook__none__persistence__sNA.csv.gz`,
`results/phase2/predictions/S0_notebook__notebook__cnn_lstm__s0.csv.gz` (also `s1`, `s2`) and
`results/phase2/predictions/S2_random__notebook__cnn_lstm__s0.csv.gz`, plus
`data/processed/events.csv` for the two weighings. The label jumps of 29.05 g and 8.30 g are
quoted from `results/audit/step_row12000.md`. Writes `figures/main/fig_original_test.png` and
`figures/main/fig_original_test.pdf`.

## fig_workflow

Design of the revised analysis, from the 11 raw pond files of Udanor et al. (2022) to the
statistics. The audit identified the real weighings on the 14-day grid, fixed the quality-control
rules before any model of the new design was run, and found that ponds 1 and 2 share a sensor
stream. The cohort has 8 ponds and 70 real weighings. Sensor readings that fail a rule become
missing, and daily aggregates use the 2-day gap rule. Task T1 estimates the weight at a weighing
from time since stocking and the water quality of the 14 days before it. Task T2 predicts the
weight 14 and 28 days after a weighing from past weights, time and the water quality of the 28
days before it. Models are compared under leave-one-pond-out validation with nested tuning (3
inner folds, 20 Optuna trials per model and fold, 5 seeds for the deep models) and a descriptive
calendar hold-out. Statistics are the primary statistic S, a fixed-sequence Wilcoxon test, a
permutation test and a pond-level bootstrap.

Replaces Figures 2 and 4 of the original submission. Script
`reporting/figs/fig_workflow.py`. Every number in the diagram is read from a file: the count of
`data/raw/pond*.csv`, `results/audit/summary.csv` (logging interval), `data/processed/events.csv`
and `data/processed/t1_table.csv` (weighings), `results/qc/eligibility.csv` (scored samples),
`configs/phase4.yaml` and `configs/phase4_data.yaml` (windows, gap rule, tuning budget, seeds,
bootstrap, cut-off, day 0). Writes `figures/main/fig_workflow.png`, `.pdf` and `.svg`.

## fig_tft

Architecture of the temporal fusion transformer as implemented in this study (own drawing,
adapted from Lim et al. (2021)). Static inputs (T1: t, the days since 2021-06-19; T2: log W_k,
log W_(k-1), t_k, t_k - t_(k-1) and the target gap) pass through static variable selection, and
four static covariate encoders turn its output into the contexts c_s, c_h, c_c and c_e. The six
sensors, each given as four daily aggregates and a validity mask, are embedded and weighted for
every day by temporal variable selection with context c_s. An LSTM encoder starts from c_h and
c_c, a gated skip connection adds the selected inputs, and static enrichment uses c_e.
Interpretable multi-head attention takes the last real day as its only query over all days,
with padded days masked. A gated skip, a position-wise GRN and a third gated skip (from the
output of the first, taken at the last day) lead to a linear layer that returns one value.
Unlike Lim et al. (2021), the model has no decoder and no known future inputs, uses one query,
and gives a point prediction trained on mean squared error. Shading groups the blocks: static
(blue), temporal inputs and selection (green), LSTM and GRN layers (orange), attention
(purple), gated skips (grey) and output (vermillion). In the small grid, rows are sensors,
columns are days, hatched columns are padding and the outlined column is the last real day.

Replaces Figure 8 of the original submission. Script `reporting/figs/fig_tft.py`. Drawn from
`src/models/tft.py` (read, not imported); the counts, day 0 and target gaps are read from
`configs/phase4.yaml`. Writes `figures/main/fig_tft.png`, `.pdf` and `.svg`.
