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


## fig_pond_scores

Per-pond mean absolute log error S for every prespecified model on the eligible
leave-one-pond-out samples. Panels show T1 and T2 at 14 and 28 days ahead. Models follow
the configured order within the labelled groups; sensitivity variants and post hoc results
are excluded. Coloured circles are ponds; black diamonds are the saved model S, the mean
over scored ponds. Neural S values are the saved means over five seeds. The primary pair
is labelled from inner-validation selection. T1 uses a symmetric-logarithmic horizontal
axis, linear from 0 to 0.5 and logarithmic above, to retain the large ridge errors. A light
vertical line at 0.5 and a note below the axis mark this change; T2 axes are linear. T2 Ridge
(past weights + WQ) stopped because a learned-model run produced a non-finite prediction
under plan 9.0; its row has no score or pond dots. An infinite S would be shown by an arrow
at the right edge, annotated with the saved count of predictions at or below 0 g; no such
prediction occurs among the models plotted here. Lower S means smaller error.

Script `reporting/figs/fig_pond_scores.py`. Saved data sources:
`results/metrics/completeness.csv`,
`results/metrics/per_pond.csv`,
`results/metrics/scores.csv`,
`results/selection/primary_T1.json`,
`results/selection/primary_T2.json`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/main/fig_pond_scores.png` and `figures/main/fig_pond_scores.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_pond_scores.py`.


## fig_confirmatory

Differences in S between the primary deep model and primary baseline,
with descriptive pond-bootstrap 95% percentile intervals. Panels show T1, LSTM minus Cohort
mean; T2 at 14 days ahead, CNN-LSTM minus Ridge (past weights); and T2 at 28 days ahead,
the same pair. Black diamonds and horizontal bars are the saved mean difference and interval
from pairwise.csv. Each coloured dot is a pond's saved seed-averaged S for the primary deep
model minus its saved S for the primary baseline, read from per_pond.csv for LOPO, eligible
samples and the panel's task-horizon. Small vertical offsets separate the seven pond dots;
pond colours are fixed across figures. Negative values favour the deep model. One-sided exact Wilcoxon p-values
come from confirmatory.csv for T1 and T2 at 14 days, and pairwise.csv for the descriptive
28-day comparison. The T2 14-day step was not tested in the fixed sequence because T1 did
not reject; its saved nominal p-value is displayed with that status. P-values are displayed
to three decimals. The intervals
are descriptive and are not the fixed-sequence rejection rule.

Script `reporting/figs/fig_confirmatory.py`. Saved data sources:
`results/metrics/per_pond.csv`,
`results/selection/primary_T1.json`,
`results/selection/primary_T2.json`,
`results/stats/confirmatory.csv`,
`results/stats/pairwise.csv`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/main/fig_confirmatory.png` and `figures/main/fig_confirmatory.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_confirmatory.py`.


## fig_permutation

Saved permutation distributions of S under leave-one-pond-out validation, with
the observed S as a black vertical line. Columns follow the configured water-quality tabular
models, then the selected primary deep model; rows show T1 and T2 at 14 days ahead.
Each histogram counts the saved permutation runs, using 25 equal-width bins. The p-values
are read from the saved permutation statistics and displayed to three decimals, with the
lower-tail rule of plan 8.2.
Tabular models use 999 permutations and deep models 199. The observed and permuted deep
runs use seed 0, rather than the five-seed mean used in the score figures. Water-quality
values and masks are reassigned jointly between stream units within a date, with model
settings fixed. T2 Ridge (past weights + WQ) has no saved permutation distribution because
its original LOPO run stopped under plan 9.0; panel (d) states that no permutation test was run.
These distributions concern pond-specific sensor information and do not establish a causal
effect of water quality on growth.

Script `reporting/figs/fig_permutation.py`. Saved data sources:
`results/metrics/completeness.csv`,
`results/permutation/T1/gbm_time_wq.csv`,
`results/permutation/T1/lstm.csv`,
`results/permutation/T1/ridge_time_wq.csv`,
`results/permutation/T2/cnn_lstm.csv`,
`results/permutation/T2/gbm_pw_wq.csv`,
`results/selection/primary_T1.json`,
`results/selection/primary_T2.json`,
`results/stats/permutation.csv`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/main/fig_permutation.png` and `figures/main/fig_permutation.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_permutation.py`.


## fig_trajectories

T1 weight estimation under leave-one-pond-out validation, one panel per pond
with eligible scored events. The eighth position of the 2 by 4 grid is unused because pond 8
has no eligible event. Filled circles are real weighings, including unscored weighings for
context. Hollow markers and line styles distinguish the fixed shown models: the primary
baseline and deep model, the configured permutation-test tabular models, and Static MLP.
Predictions occur only at eligible event dates; connecting segments are visual guides and
do not add observations or daily predictions. LSTM and Static MLP predictions are arithmetic
means of predicted grams across the five seeds. Light bands span the seed minimum and
maximum; vertical ranges make the spread visible at individual dates. Every panel uses a
logarithmic gram axis with one shared range that includes every real weighing shown. A
prediction outside that range is an open triangle at the panel edge pointing outward, with
its value labelled in grams. The calendar axis ends on 31 October 2021; later weighings,
which were not scored, are not shown. Pond colours are fixed across figures. LOPO transfers between ponds over an
overlapping calendar period and does not test forecasting into an unseen future.

Script `reporting/figs/fig_trajectories.py`. Saved data sources:
`data/processed/events.csv`,
`results/predictions/T1/lopo/cohort_mean.csv`,
`results/predictions/T1/lopo/gbm_time_wq.csv`,
`results/predictions/T1/lopo/lstm.csv`,
`results/predictions/T1/lopo/mlp_static.csv`,
`results/predictions/T1/lopo/ridge_time_wq.csv`,
`results/selection/primary_T1.json`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/main/fig_trajectories.png` and `figures/main/fig_trajectories.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_trajectories.py`.


## fig_t2_scatter

T2 target-weight predictions against observed weight under leave-one-pond-out
validation, at 14 and 28 days ahead. Both axes show grams on a logarithmic scale with equal
limits; the dashed diagonal is equality. Pond colours and model-specific marker shapes
identify the eligible origin-target samples of the fixed shown-model set. Separate legend
blocks show models by marker and ponds by colour. CNN-LSTM and
Static MLP points are arithmetic means of predicted grams across the five seeds; the other
models are deterministic. The selected baseline is Ridge (past weights), whose inputs also
include time. Ridge (past weights + WQ) has no points because its LOPO run stopped under
plan 9.0, and its legend entry records that absence. T2 evaluates transfer of growth over
14 or 28 days between ponds during an overlapping calendar period. The 28-day comparison
is descriptive; neither panel is a test of forecasting into an unseen future.

Script `reporting/figs/fig_t2_scatter.py`. Saved data sources:
`results/metrics/completeness.csv`,
`results/predictions/T2/lopo/cnn_lstm.csv`,
`results/predictions/T2/lopo/gbm_pw_wq.csv`,
`results/predictions/T2/lopo/mlp_static.csv`,
`results/predictions/T2/lopo/persistence.csv`,
`results/predictions/T2/lopo/ridge_pw.csv`,
`results/selection/primary_T2.json`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/main/fig_t2_scatter.png` and `figures/main/fig_t2_scatter.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_t2_scatter.py`.


## fig_calendar

Descriptive T1 calendar hold-out, one panel per pond with a test event.
The dashed line marks the cut-off, 28 August 2021. Filled circles are real weighings from
the displayed period, including unscored events for context. Hollow markers show saved
predictions only on eligible test dates on or after the cut-off; no later weighing enters
training. Line styles distinguish the primary pair, the configured permutation-test tabular
models and Static MLP. Neural predictions are arithmetic means in grams over the five seeds,
with light bands and vertical ranges spanning the minimum to maximum seed prediction.
Panels share one logarithmic gram range that includes every real weighing shown. A
prediction outside the range is an open triangle at the panel edge pointing outward, with
its value labelled in grams. The common calendar axis ends on 31 October 2021; later
weighings, which were not scored, are not shown. Connecting
segments only join saved test predictions; they are not daily estimates. A pond with one
test event has markers and ranges but no prediction line segment. The empty eighth grid
position corresponds to no additional test pond. This hold-out is descriptive, with no
p-value or confidence interval; seed ranges describe fitted-seed variation only.

Script `reporting/figs/fig_calendar.py`. Saved data sources:
`data/processed/events.csv`,
`results/predictions/T1/chrono/cohort_mean.csv`,
`results/predictions/T1/chrono/gbm_time_wq.csv`,
`results/predictions/T1/chrono/lstm.csv`,
`results/predictions/T1/chrono/mlp_static.csv`,
`results/predictions/T1/chrono/ridge_time_wq.csv`,
`results/selection/primary_T1.json`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/supp/fig_calendar.png` and `figures/supp/fig_calendar.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_calendar.py`.


## fig_learning_curves

Saved learning curves for all four deep models and Static MLP under
leave-one-pond-out validation. Columns show T1 and T2; rows follow the configured neural
model order. Dashed lines are validation MSE from the chosen trial only, one line for each
outer-fold and inner-fold combination, using tuning seed 0. Thin solid lines are training
MSE for each final outer-fold and seed fit, using seeds 0 to 4. Colour identifies the pond
held out in the outer fold. Open squares mark the final training epoch for every final-fit
seed; short marks along the top show the configured final epoch count read from chosen.csv
for each outer fold. MSE is on the fold-standardized target, log weight for T1 and log target
to origin weight ratio for T2. No final-fit validation or outer-test loss is drawn. Axes
retain every saved epoch of these selected curves; loss and epoch limits vary by panel.
These are training diagnostics, not an additional model comparison or hypothesis test.

Script `reporting/figs/fig_learning_curves.py`. Saved data sources:
`results/tuning/T1/lopo/cnn_lstm/chosen.csv`,
`results/tuning/T1/lopo/cnn_lstm/curves.csv.gz`,
`results/tuning/T1/lopo/gru/chosen.csv`,
`results/tuning/T1/lopo/gru/curves.csv.gz`,
`results/tuning/T1/lopo/lstm/chosen.csv`,
`results/tuning/T1/lopo/lstm/curves.csv.gz`,
`results/tuning/T1/lopo/mlp_static/chosen.csv`,
`results/tuning/T1/lopo/mlp_static/curves.csv.gz`,
`results/tuning/T1/lopo/tft/chosen.csv`,
`results/tuning/T1/lopo/tft/curves.csv.gz`,
`results/tuning/T2/lopo/cnn_lstm/chosen.csv`,
`results/tuning/T2/lopo/cnn_lstm/curves.csv.gz`,
`results/tuning/T2/lopo/gru/chosen.csv`,
`results/tuning/T2/lopo/gru/curves.csv.gz`,
`results/tuning/T2/lopo/lstm/chosen.csv`,
`results/tuning/T2/lopo/lstm/curves.csv.gz`,
`results/tuning/T2/lopo/mlp_static/chosen.csv`,
`results/tuning/T2/lopo/mlp_static/curves.csv.gz`,
`results/tuning/T2/lopo/tft/chosen.csv`,
`results/tuning/T2/lopo/tft/curves.csv.gz`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/supp/fig_learning_curves.png` and `figures/supp/fig_learning_curves.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_learning_curves.py`.


## fig_ablation

Descriptive ablation scores on the eligible leave-one-pond-out samples, shown
for T1 and T2 at 14 and 28 days ahead. T1 includes time, water quality, time plus water
quality, and time plus water quality plus length, for ridge and gradient boosting. T2
includes past weights plus time, with and without water quality. Names follow the shared
model labels; T2's past-weight models also include time. Coloured dots are the saved
per-pond S and black diamonds the saved mean S over ponds. Model order follows the feature
sets of plan 8.1 within each estimator. T1's symmetric-logarithmic axis is linear from 0 to
0.5 and logarithmic above, with the change marked by a light vertical line at 0.5 and a note
below the axis; T2 axes are linear. The T2 ridge model with water quality stopped under plan 9.0 and
has no dots. These comparisons are descriptive; length is available only in the T1 ablation
and reference analyses, not as an input to the primary T1 models.

Script `reporting/figs/fig_ablation.py`. Saved data sources:
`results/metrics/completeness.csv`,
`results/metrics/per_pond.csv`,
`results/metrics/scores.csv`,
`results/selection/primary_T1.json`,
`results/selection/primary_T2.json`.
Configuration `configs/phase4.yaml`; style `reporting/figstyle.py`; display labels `reporting/names.py`.
Writes `figures/supp/fig_ablation.png` and `figures/supp/fig_ablation.pdf`.
Rebuild from the repository root with `env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/fig_ablation.py`.
