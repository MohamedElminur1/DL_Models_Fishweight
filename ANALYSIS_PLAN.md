# ANALYSIS PLAN (draft v0.4, not frozen)

Status: draft by the lead, 2026-10-14. v0.2 filled in the Gate 1 items (`DECISIONS.md`).
v0.3 answers the pre-freeze statistical review (`results/plan_review.md`, items 1 to 17).
v0.4 fills in the counts and the cut-off date from the data pipeline (`src/events.py`,
`src/qc.py`, `src/features.py`; `results/qc/eligibility.csv`), computed before any model of
the new design was run. The annotated git tag
`analysis-plan-frozen` freezes this file together with `src/` and `configs/` (§12). Before
the tag, no model of the new design is evaluated on any test sample and no inner-validation
error is looked at.

## 1. Question

Do water-quality data logged by the IoT sensors improve the estimation of African catfish
(*Clarias gariepinus*) weight beyond time since stocking (T1), and the prediction of the
next measured weights beyond past measured weights (T2), when evaluation uses only real
measurements and ponds the models have not seen?

## 2. Data

2.1 Source: the 11 pond files of Udanor et al. (2022) (ponds 1-4, 6-12; no pond 5 in the
public archive), loaded with `src/load.py`. No row is dropped at load time.

2.2 Real measurement events. In the files, weight and length change once per calendar day;
between sampling dates the daily values follow straight lines (Phase 1 audit). Only the
values on sampling dates are real measurements. In the cohort ponds (§2.4) the sampling dates
are 2021-06-19 + 14k days, k = 0 to 16 (Gate 1).
- An event (p, d) exists when the file has at least one row dated d with a weight value. Its
  weight W and length L are the most frequent values among the rows dated d.
- Sampling dates without any row in the file are missing events. They are not reconstructed
  from neighbouring interpolation segments.
- Fish values on non-sampling dates never provide targets or inputs.
- Fish-only rows (no sensor values) count as events.

2.3 Time axis: t = days since 2021-06-19, the first sampling date of the cohort (day 0), for
every cohort pond. Dates come from each file's own clock as labelled (no time-zone
conversion), because the daily labels change at midnight in that clock.

2.4 Cohort. The main analysis uses the ponds whose first record with a weight is a
fingerling weight (<= 10 g) dated 18 or 19 June 2021, consistent with the 2 +/- 1.1 g
fingerlings described by Udanor et al. (2022): ponds 1, 2, 3, 4, 6, 7, 8, 9, with 6, 9, 6, 14,
15, 9, 3 and 8 events (70 in total). Ponds 10, 11 and 12 start on 2021-06-25 with juvenile
weights (27.6, 33.8 and 28.6 g), have few timestamped sensor rows and follow another sampling
grid; they have no comparable time since stocking, are described in the data section and are
excluded from model evaluation.

2.5 Sensor quality control, applied to timestamped raw rows (in time order) before any
aggregation or scaling. A value that fails a rule becomes missing (never clipped). Ranges
follow the sensors' measuring ranges where the readings are on that scale and physical limits
otherwise (Gate 1):
- temperature (DS18B20): 10 to 40 degC (removes the -127 disconnected-sensor code);
- pH (DFRobot analog probe): 0 to 14;
- dissolved oxygen (DFRobot galvanic probe, 0 to 20 mg/L): above 0 and at most 20;
- turbidity (TS-300B module, source values on a 0 to 100 scale): 0 to 100;
- ammonia (MQ137 gas sensor above the water): above 0 and at most 500; valid values are
  log10-transformed before daily aggregation;
- nitrate (MQ135 gas sensor above the water): above 0;
- every sensor: non-finite values are invalid; flat-line test: a run of identical consecutive
  values spanning at least 24 h of clock time is invalid (a missing value or a logging gap of
  more than 1 h ends a run). Exact duplicate lines are dropped before QC.
Units as labelled in the source; values are raw sensor readings. Rows without a clock time
(date-only fish rows) carry no sensor values and do not enter the sensor pipeline. The number
of values removed per pond, sensor and rule is reported.

2.6 Daily aggregation. For each pond, sensor and calendar date with valid readings in at
least 12 distinct clock hours: daily mean, SD (ddof 0), minimum and maximum. Otherwise the
sensor-day is missing.

2.7 Gap rule. Missing sensor-days are linearly interpolated only inside runs of at most two
consecutive missing days with observed days on both sides. Interpolated sensor-days are
flagged and counted per pond. Inputs for an event or origin on date d use only days before
d: an interpolated sensor-day whose later observed anchor is d or after counts as missing for
that event. A day is covered for an event when at least three of the six sensors have a value
on it under this rule.

2.8 Shared sensor streams. Ponds 1 and 2 share one sensor stream (81,215 common timestamps
with identical temperature, turbidity, DO, pH and nitrate; Gate 1). Each pond uses the sensor
values of its own file (the files differ in usable days and in ammonia). The two ponds form
one stream unit in the inner folds (§7.2) and the permutation test (§8.2), where pond 2's file
represents the unit as a donor; when one of them is held out, the other leaves the training
set (§7.1). This is reported as a property of the dataset.

## 3. Tasks

T1, estimation. Predict W at event e from t_e and the water-quality history before d_e.
Inputs exclude length and every weight value. Target modelled as log W.

T2, prediction from past weights. From origin event k with a previous available event k-1
(any gap), predict W at the event dated exactly d_k + 14 days (h = 1) and d_k + 28 days
(h = 2). An origin without such an event has no sample for that horizon. Inputs: log W_k,
log W_{k-1}, t_k, t_k - t_{k-1}, the target gap (14 or 28 days), and water quality in the 28
days before d_k. Target modelled as y = log(W_{k+h} / W_k); prediction W_k * exp(y_hat). One
sample per origin and horizon; one model serves both horizons (no recursive forecasting).
Scores are computed per horizon. Under leave-one-pond-out, the training ponds contain weights
measured after the test pond's origin dates, so T2 measures how well growth over the next 14
or 28 days transfers between ponds over the same period; it is not forecasting into an unseen
future. The calendar hold-out (§7.4) is the forecasting check.

Water quality on the event or origin day itself is not used (the sampling time of day is
unknown).

## 4. Evaluation and training sets

4.1 T1: events with t > 0 in cohort ponds whose 14 days before d_e contain at least 7 covered
days (§2.7). 35 of the 63 events with t > 0 are eligible: ponds 1, 2, 3, 4, 6, 7 and 9 have 3,
7, 5, 5, 4, 6 and 5; pond 8 has none. They span t = 14 to 112 days. Time-only models are also
reported on all 63 events with t > 0 as a secondary table.

4.2 T2: samples (origin, horizon) with at least 14 covered days among the 28 days before d_k.
h = 1: 29 of 43 samples (ponds 1, 2, 3, 4, 6, 7, 9: 3, 6, 4, 4, 3, 4, 5); h = 2: 24 of 40
(3, 5, 3, 3, 3, 4, 3); pond 8 has none. Target dates span t = 28 to 126 days.

4.3 Training sets. In every fold, every model of a task is trained on the eligible samples of
the training ponds, by the rules of §4.1-4.2, so that models with and without water quality
see the same training events. The secondary time-only table of §4.1 trains and scores on all
events with t > 0. Ponds without an eligible sample contribute neither training nor test
samples.

4.4 All models of a task are scored on exactly the same samples.

## 5. Features

5.1 Tabular models.
- Time: t and t^2 (T1); t_k, t_k - t_{k-1}, target gap (T2).
- Water quality, per sensor over the days of the window on which that sensor has a value
  under §2.7 (window: the 14 days before d_e in T1, the 28 days before d_k in T2): mean of
  daily means, mean of daily SDs, minimum of daily minima, maximum of daily maxima
  (24 features). T1 adds each sensor's mean of daily means from day 0 to d_e - 1
  (6 features). A sensor with a value on fewer than 3 days of the window (or of day 0 to
  d_e - 1) gives missing features. Coverage (§2.7) decides eligibility only.
- Past weights (T2): log W_k, log W_{k-1}.
- Length (T1 ablation only): log L_e.
- Missing features: ridge imputes the training-fold median and adds one indicator per sensor
  (1 when the sensor has no window features); gradient boosting uses its native
  missing-value handling. A feature missing in every training sample of a fold is dropped for
  that fold.

5.2 Sequence models. Daily steps; channels: the 24 daily aggregates, standardized with
training-fold statistics, missing values set to 0, plus six per-sensor validity masks, all
under the rule of §2.7. T1 sequence: day 0 to d_e - 1, left-padded with zeros (masks 0) to the
longest T1 sequence. T2 sequence: the 28 days before d_k. Static inputs (standardized with
training-fold statistics): t_e (T1); log W_k, log W_{k-1}, t_k, t_k - t_{k-1} and the target
gap (T2).

5.3 Every scaler, imputer, median and target standardization is fitted on the training
ponds of the outer fold, and again inside every inner fold on its inner-training ponds.

## 6. Models

6.1 Time-only baselines.
- T1: `gc`, growth curves on t fitted to log W of the training events: Gompertz, von
  Bertalanffy, logistic; the curve with the lowest training AIC is used (all have three
  parameters). `cohort_mean`: exp of the mean log W of the training ponds' events on the same
  sampling date, if at least 3 training ponds have that date, else `gc`. `ridge_time` and
  `gbm_time` on the time features.
- T2: `persistence` (W_k); `linear`, the straight line through (t_{k-1}, W_{k-1}) and
  (t_k, W_k), scored as it is, also when it predicts a weight <= 0; `sgr`, the straight line
  through log W_{k-1} and log W_k; `anchored_gc`, W_k * G(t_{k+h}) / G(t_k) with G fitted as
  in T1 on the origin and target events of the training samples; `cohort_growth`,
  W_k * exp(mean y of the training samples with the same origin date and target date) if at
  least 3 training ponds have them, else `anchored_gc`; `ridge_pw` and `gbm_pw` on past
  weights and time.
- A growth-curve fit is not retried: a curve whose fit fails is skipped in the AIC choice; if
  all three fail, `gc` uses `cohort_mean` and `anchored_gc` uses `sgr` for that fold, and the
  failure is counted and reported.

6.2 Tabular models with water quality: ridge and gradient boosting (scikit-learn
`HistGradientBoostingRegressor`) with the feature sets of §8.1.

6.3 Deep models (PyTorch, one shared training loop): LSTM, GRU, CNN-LSTM, and a temporal
fusion transformer (TFT) following Lim et al. (2021): gated residual networks, variable
selection for static and temporal inputs (one temporal variable per sensor: its four
aggregates and its mask), LSTM encoder initialized from static contexts, static enrichment,
interpretable multi-head attention with the last step as the query, gated skip connections,
point output (mean squared error instead of the quantile loss of Lim et al.). Loss: mean
squared error on the standardized target. AdamW. Early stopping on validation loss.
`mlp_static`: the static-inputs-only twin (a two-layer network on the static inputs alone),
tuned and seeded like the deep models, separates the contribution of the water-quality
sequence from that of the architecture.

6.4 Allometric reference (reported separately, uses length): W = a L^b fitted on log-log
training events, predicted from L_e (T1).

## 7. Validation

7.1 Primary: outer leave-one-pond-out (LOPO) over the cohort ponds with at least one eligible
sample. The held-out pond's samples are the test set of that fold. When pond 1 or pond 2 is
held out, the other is removed from the training ponds of that fold (§2.8).

7.2 Tuning inside the training ponds only: three inner folds grouped by stream unit over the
units with an eligible training sample (units sorted by their smallest pond id and assigned
round-robin; ponds 1 and 2 are one unit). Optuna TPE sampler (seed 0), 20 trials per model per
outer fold for every deep model, `mlp_static` and gradient boosting; ridge alpha from
10^-3 ... 10^3 (13 values) by the same inner folds. One Optuna study per task, scheme, model
and outer fold. Objective: mean over inner folds of S (§9.0) on the inner-validation ponds.
Early stopping inside each inner training on its inner-validation fold. Final model per outer
fold: trained on all training ponds for the median best epoch of the best trial's inner folds.
Every model, baselines included, is also scored on the inner folds (inner-validation S) for
the selection of §9.4.

7.3 Seeds: deep models and `mlp_static` are trained with seeds 0-4 in the final fit of every
fold (tuning uses seed 0). Gradient boosting and ridge are deterministic.

7.4 Calendar hold-out, descriptive only (no p-values, no intervals). Cut-off date
c = 2021-08-28 (t = 70), the latest grid date such that at least 10 eligible T1 events in at
least 5 ponds are dated on or after c: 11 T1 events in 7 ponds; T2 test samples: 10 (h = 1)
and 12 (h = 2). T1: train on eligible events dated before c, test on those dated c
or later. T2: train on eligible samples whose target is dated before c, test on those whose
target is dated c or later. Nothing dated c or later enters training, same-date baselines
included (they fall back as in §6.1). Tuning: one inner split whose validation set is the
training samples at the latest training date. Same budgets and seeds as §7.2-7.3.

7.5 Calendar overlap. In LOPO the training ponds cover the same calendar period as the test
pond. This is reported as a limitation; §7.4 tests prediction of later dates.

## 8. Controls

8.1 Ablations (descriptive) with ridge and gradient boosting.
- T1: time only; water quality only; time + water quality; time + water quality + length.
- T2: past weights + time; past weights + time + water quality.
- Deep models against `mlp_static`.

8.2 Permutation test. Null: within a sampling date, the water-quality inputs carry no
pond-specific information about weight. For each permutation r and each date d (T1 event
date; T2 origin date) with eligible samples in at least two stream units, the units'
water-quality inputs for date d (tabular features and sequences, under §2.7) are reassigned
among those units by a uniformly random derangement (rng seed 20261014 + r); a date with one
unit stays unchanged and is counted. Training and test data alike; targets, eligibility and
the fold-wise tuned hyperparameters stay fixed; LOPO final fits are re-run without
re-tuning. Statistic: S. p = (1 + number of permutations with S_perm <= S_obs) / (R + 1).
R = 999 for ridge and gradient boosting with time (or past weights) and water quality;
R = 199 for the primary deep model of each task (§9.4), with seed 0 in the observed and the
permuted runs.

8.3 Sensitivity analyses (descriptive, Supplementary Materials): water-quality windows of 7
and 28 days (T1) and of 14 and 42 days (T2) for ridge and gradient boosting with water quality
(evaluation sets of §4 unchanged); ridge without missing indicators; the confirmatory test
with ponds 1 and 2 averaged into one unit; a leave-one-pond-out influence table (S difference
with each pond removed).

## 9. Metrics and statistics

9.0 Primary statistic S: for each pond, the mean absolute error of log W over its scored
samples (T1: |log W_hat - log W|; T2: |y_hat - y|, which is the same quantity at the target);
then the mean over the ponds with a scored sample. For deep models and `mlp_static`, the
per-pond values are averaged over the five seeds. All other metrics are descriptive.

9.1 Descriptive metrics, pooled over all test samples of all folds: MAE (g), RMSE (g),
MAPE (%), R^2; per pond: MAE (g), MAPE and S.

9.2 Deep models and `mlp_static`: every metric per seed, reported as mean +/- SD over the
five seeds. The spread across ponds is reported as the SD of per-pond S and MAE.

9.3 Pond-level cluster bootstrap (B = 2000, seed 20261014), descriptive: 95 % percentile
intervals of S and of the difference in S to the primary baseline, resampling ponds jointly
for the model and its comparator.

9.4 Primary models. The primary baseline of a task is the time-only model of §6.1 with the
lowest mean inner-validation S over the outer folds of the LOPO scheme; the primary deep
model is chosen among LSTM, GRU, CNN-LSTM and TFT the same way (T2: S over both horizons).
`src/select_primary.py` makes the choice from inner-validation scores only, and its output is
committed before any outer prediction is scored.

9.5 Confirmatory test. Exact one-sided Wilcoxon signed-rank test on the per-pond differences
in S (primary deep model minus primary baseline; alternative: the deep model's S is lower);
units are the ponds with a scored sample; zero differences are dropped and counted; ties use
midranks with the exact distribution over all 2^n sign flips. Fixed sequence: T1 at
alpha = 0.05; T2 h = 1 at alpha = 0.05 only if T1 rejected. Everything else is descriptive:
T2 h = 2, the calendar hold-out, ablations, sensitivity analyses, all other model pairs, and
any task-horizon with fewer than 6 scored ponds. Two-sided p-values are reported alongside.
With 7 ponds the test rejects only if the ranks of the ponds where the deep model is worse sum
to at most 3 (5 of 128 sign patterns); with 6 ponds, at most 2.

## 10. Decision rule for the framing (Phase 5)

Positive framing for a task if its confirmatory test (§9.5) rejects. Saying that water quality
contributes to that result additionally requires a permutation p < 0.05 for the primary deep
model (§8.2) and a lower S than `mlp_static`. If neither task rejects, the paper takes the
leakage-aware benchmark framing. A non-significant result is not evidence of no effect, and
the paper says so.

Claims the paper may make: model X had a lower S than the baseline in n of N ponds;
pond-specific water-quality information was or was not used (permutation test). Descriptive
only: ablations, T2 h = 2, the calendar hold-out, the all-events table, the allometric
reference and the sensitivity analyses. Not claimed: causal effects of water quality on
growth; performance beyond the largest t among scored samples (112 days in T1, 126 days in
T2); temporal forecasting from LOPO results; that deep learning is unsuitable for this problem
in general (the dataset has 35 scored T1 events).

## 11. Reporting

Every model, ablation and seed is reported, including poor results. Tables: hyperparameter
search spaces, trial budget, selected values per fold; inner-validation scores and the
selection of §9.4; runtimes and hardware. Learning curves (training and validation loss per
epoch) go to the Supplementary Materials. Predictions are saved per sample with pond, date,
model, seed and fold.

## 12. Freeze

The annotated tag `analysis-plan-frozen` freezes this file, `src/` and `configs/`. Before the
tag, code runs end to end only in `--dev` mode (inner folds of the first outer fold), which
prints, logs and reports no error value. The runner refuses outer test predictions unless
(a) the tag exists and its commit SHA is recorded in `decisions.tsv`, and (b) `ANALYSIS_PLAN.md`,
`src/` and `configs/` in the working tree are identical to the tag (no differences, no
untracked files there), or identical to a tag `analysis-fix-N` whose `ANALYSIS_PLAN.md` equals
the frozen one. Every fix tag is explained in `LOG.md` and `DECISIONS.md` before the run and
reported with the results. (c) The first complete outer run is the reported run; a model is
re-run only after a crash, and the re-run is logged.

## 13. Search spaces and training settings

Deep models and `mlp_static` (20 Optuna TPE trials per outer fold, sampler seed 0):

| parameter | LSTM | GRU | CNN-LSTM | TFT | mlp_static |
|---|---|---|---|---|---|
| hidden size | {8, 16, 32, 64} | {8, 16, 32, 64} | LSTM part {8, 16, 32, 64} | {8, 16, 32} | {8, 16, 32, 64} |
| recurrent layers | {1, 2} | {1, 2} | {1, 2} | 1 (encoder) | |
| convolution | | | filters {8, 16, 32}; kernel {3, 5, 7}; pooling {1, 2} | | |
| attention heads | | | | {1, 2, 4} | |
| dropout | U(0, 0.5) | U(0, 0.5) | U(0, 0.5) | U(0, 0.5) | U(0, 0.5) |
| learning rate | logU(1e-4, 1e-2) | same | same | same | same |
| weight decay | logU(1e-6, 1e-2) | same | same | same | same |
| batch size | {8, 16, 32} | same | same | same | same |

Static inputs enter LSTM, GRU and CNN-LSTM concatenated to the last hidden state before a
two-layer output head (the same head forms `mlp_static`), and TFT through the static covariate
encoders. At most 300 epochs; early stopping with patience 30 on validation MSE.

Gradient boosting (`HistGradientBoostingRegressor`, early_stopping=False, random_state 0, 20
trials per outer fold): learning rate logU(0.01, 0.3); max_iter int [50, 500]; max_depth
{2, 3, 4}; min_samples_leaf {3, 5, 10, 20}; l2_regularization logU(1e-4, 10).

Ridge: alpha in {10^-3, 10^-2.5, ..., 10^3}; features standardized on the training fold.

Growth curves (fitted to log W by `scipy.optimize.curve_fit`, three parameters each):
Gompertz log W = log A - b exp(-c t); von Bertalanffy log W = log Winf + 3 log(1 - exp(-k (t - t0)))
with t0 < 0; logistic log W = log A - log(1 + exp(-k (t - tm))). Starting values and bounds
are in `configs/phase4.yaml` and frozen with it.

## 14. Compute

CPU only (4 cores). Trainings run in up to four single-threaded worker processes. Runtime
and hardware are logged per training.
