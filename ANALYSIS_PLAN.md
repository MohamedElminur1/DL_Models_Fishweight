# ANALYSIS PLAN (draft v0.2, not frozen)

Status: draft by the lead, 2026-10-14. v0.2 fills in the Gate 1 items from the Phase 1 audit
(`DECISIONS.md`, Gate 1). Items marked `[PRE]` are counts filled in from the data pipeline
(`src/events.py`, `src/qc.py`, `src/features.py`) before the freeze. The plan is frozen by the
git tag `analysis-plan-frozen`. Before the tag, no model of the new design is evaluated on any
test event (§12).

## 1. Question

Do water-quality data logged by the IoT sensors improve the prediction of African catfish
(*Clarias gariepinus*) weight beyond what time since stocking (T1) or past measured weights
(T2) already provide, when evaluation uses only real measurements and ponds the models have
not seen?

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
- Rows on non-sampling dates never provide targets or inputs.
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
least 12 distinct clock hours: daily mean, SD, minimum and maximum. Otherwise the sensor-day
is missing.

2.7 Gap rule. Missing sensor-days are linearly interpolated only inside runs of at most two
consecutive missing days with observed days on both sides. Longer runs stay missing.
Interpolated sensor-days are flagged and counted per pond. A day is covered when at least
three of the six sensors have a value (observed or interpolated).

2.8 Shared sensor streams. Ponds 1 and 2 share one sensor stream (81,215 common timestamps
with identical temperature, turbidity, DO, pH and nitrate; Gate 1). They count as one stream
in the inner folds (§7.2) and the permutation test (§8.2), and when one of them is held out
the other leaves the training set (§7.1). This is reported as a property of the dataset.

## 3. Tasks

T1, estimation. Predict W at event e from t_e and the water-quality history before d_e.
Inputs exclude length and every weight value. Target modelled as log W.

T2, forecasting. From origin event k (k >= 1, so W_{k-1} and W_k are known), predict W at the
next measured event (h = 1, nominal 14 days) and the one after (h = 2, nominal 28 days).
Inputs: log W_k, log W_{k-1}, t_k, t_k - t_{k-1}, t_{k+h} - t_k, and water quality in the 28
days before d_k. Target modelled as log(W_{k+h} / W_k); prediction W_k * exp(y_hat). Direct
strategy: one sample per origin and horizon; one model serves both horizons, which enter
through the target gap t_{k+h} - t_k (no recursive forecasting, no masked outputs). Scores are
computed per horizon.

Water quality on the event or origin day itself is not used (the sampling time of day is
unknown).

## 4. Evaluation sets

4.1 T1: events with t > 0 in cohort ponds whose 14 days before d_e contain at least 7
covered days. `[PRE: counts per pond]`. Time-only models are also reported on all events
with t > 0 as a secondary table.

4.2 T2: origins with at least 14 covered days among the 28 days before d_k and an observed
target event. h = 1 and h = 2 are scored separately.

All models of a task are scored on exactly the same events.

4.3 Training sets. In every fold, every model of a task is trained on the eligible events
(T1) or eligible origin-horizon samples (T2) of the training ponds, by the same rules as
§4.1-4.2, so that models with and without water quality see the same training events. The
secondary time-only table of §4.1 trains and scores on all events with t > 0.

## 5. Features

5.1 Tabular models.
- Time: t and t^2 (T1); t_k, t_k - t_{k-1}, t_{k+h} - t_k (T2).
- Water quality, per sensor over the days of the window on which that sensor has a value
  (observed or interpolated; window: the 14 days before d_e in T1, the 28 days before d_k in
  T2): mean of daily means, mean of daily SDs, minimum of daily minima, maximum of daily
  maxima (24 features). T1 adds each sensor's mean of daily means from day 0 to d_e - 1
  (6 features). Coverage (§2.7) decides eligibility only.
- Past weights (T2): log W_k, log W_{k-1}.
- Length (T1 ablation only): log L_e.
- A sensor without a covered day in the window gives missing features: ridge imputes the
  training-fold median and adds one indicator per sensor; gradient boosting uses its
  native missing-value handling.

5.2 Sequence models. Daily steps; channels: the 24 daily aggregates, standardized with
training-fold statistics, missing values set to 0, plus six per-sensor validity masks.
T1 sequence: day 0 to d_e - 1, left-padded with zeros (masks 0) to the longest sequence in
the dataset. T2 sequence: the 28 days before d_k. Static inputs (standardized with
training-fold statistics): t_e (T1); log W_k, log W_{k-1}, t_k, t_k - t_{k-1} and the target
gap t_{k+h} - t_k (T2).

5.3 Every scaler, imputer and median is fitted on the training ponds of the fold only.

## 6. Models

6.1 Time-only baselines.
- T1: growth curves on t fitted to log W of the training events: Gompertz, von Bertalanffy,
  logistic; the curve with the lowest training AIC is used (all have three parameters).
  Same-date cohort mean: exp of the mean log W of the training ponds' events on the same
  sampling date (fallback to the growth curve when no training pond has that date).
  Ridge and gradient boosting on time features only.
- T2: persistence (W_k); linear extrapolation from (t_{k-1}, W_{k-1}) and (t_k, W_k);
  specific-growth-rate extrapolation (straight line through log W_{k-1} and log W_k);
  anchored growth curve W_k * G(t_{k+h}) / G(t_k) with G fitted as in T1 on the origin and
  target events of the training samples; same-date cohort growth W_k * exp(mean y of the
  training samples with the same origin date and target date) (fallback to the anchored
  growth curve when there is none); ridge and gradient boosting on past weights and time.
  Linear extrapolation is scored as it is, also when it predicts a weight <= 0.

6.2 Tabular models with water quality: ridge and gradient boosting (scikit-learn
`HistGradientBoostingRegressor`) with the feature sets of §8.1.

6.3 Deep models (PyTorch, one shared training loop): LSTM, GRU, CNN-LSTM, and a temporal
fusion transformer (TFT) following Lim et al. (2021): gated residual networks, variable
selection for static and temporal inputs, LSTM encoder, static enrichment, interpretable
multi-head attention, gated skip connections, point output (mean squared error instead of
the quantile loss of Lim et al.). Loss: mean squared error on the standardized target.
AdamW. Early stopping on validation loss.

6.4 Allometric reference (reported separately, uses length): W = a L^b fitted on log-log
training events, predicted from L_e.

## 7. Validation

7.1 Primary: outer leave-one-pond-out (LOPO) over the cohort ponds. The held-out pond's
events are the test set of that fold. When pond 1 or pond 2 is held out, the other pond of
the shared stream is removed from the training ponds of that fold (§2.8).

7.2 Tuning inside the training ponds only: three inner folds grouped by sensor stream
(streams sorted by their smallest pond id and assigned round-robin; ponds 1 and 2 are one
stream). Optuna TPE sampler, 20 trials per model per outer fold for
every deep model and for gradient boosting (identical budget); ridge alpha from
10^-3 ... 10^3 (13 values) by the same inner folds. Objective: mean inner-validation MAE in
grams. Early stopping inside each inner training on that inner validation fold. Final
model per outer fold: trained on all training ponds for the median best epoch of the best
trial's inner folds.

7.3 Seeds: deep models are trained with seeds 0-4 in the final fit of every fold (tuning
uses seed 0). Gradient boosting and ridge are deterministic.

7.4 Secondary: chronological hold-out. Test = each cohort pond's last two events (T1) and
the origins whose targets are among them (T2). Training = all earlier events of all cohort
ponds. Inner validation for tuning: each pond's last training event. Same budgets.

7.5 Calendar overlap. In LOPO the training ponds cover the same calendar period as the
test pond. This is reported as a limitation; §7.4 tests extrapolation to later dates.

## 8. Controls

8.1 Ablations with ridge and gradient boosting.
- T1: time only; water quality only; time + water quality; time + water quality + length.
- T2: past weights + time; past weights + time + water quality.

8.2 Permutation test. Water-quality histories are reassigned between the seven cohort
sensor streams by a random derangement (aligned on t; ponds 1 and 2 receive the same
stream), in training and test data alike; the evaluation set stays as defined in §4; hyperparameters fixed at
the fold-wise tuned values; LOPO re-run; statistic: pooled MAE. p = (1 + number of
permutations with MAE <= observed) / (R + 1). R = 200 for ridge and gradient boosting with
water quality; R = 50 (seed 0) for the deep model with the lowest LOPO MAE in each task.

## 9. Metrics and statistics

9.1 Per event: error in grams and absolute percentage error. Pooled over all test events of
all folds: MAE (g), RMSE (g), MAPE (%), R^2. Per pond: MAE and MAPE.

9.2 Deep models: every metric per seed, reported as mean +/- SD over the five seeds. The
spread across ponds is reported as the SD of per-pond MAE.

9.3 Pond-level cluster bootstrap (B = 2000, seed 20261014) for 95 % CIs of pooled MAE and
of the MAE difference to the best time-only baseline.

9.4 Best time-only baseline: the time-only model of §6.1 with the lowest pooled LOPO MAE in
that task and horizon (chosen after the run, which makes the comparison harder, not easier,
for the deep models).

9.5 Wilcoxon signed-rank test (exact, one-sided, alternative: the deep model's per-pond MAE
is lower) on per-pond MAE (seed-averaged) of each deep model versus the best time-only
baseline. Units are the cohort ponds with at least one test event. Holm correction within
each task-horizon family (T1; T2 h = 1; T2 h = 2), four tests per family. One-sided because
the question is directional; with 8 ponds a two-sided exact test with Holm over four models
can reject only if a model wins in every pond, so the one-sided choice is fixed here, before
any result. Two-sided p-values are reported alongside.

## 10. Decision rule for the framing (Phase 5)

Positive framing for a task-horizon if at least one deep model using water quality has a
lower pooled LOPO MAE than the best time-only baseline and a Holm-adjusted Wilcoxon
p < 0.05. Saying that water quality contributes additionally requires a permutation p < 0.05
for that model. If no task-horizon meets the rule, the paper takes the leakage-aware
benchmark framing.

## 11. Reporting

Every model, ablation and seed is reported, including poor results. Tables: hyperparameter
search spaces, trial budget, selected values per fold; runtimes and hardware. Predictions are
saved per event with pond, date, model, seed and fold.

## 12. Development before the freeze

Before the tag, code may run end to end only on inner folds of the training ponds. The
runner refuses to produce outer test predictions unless (a) the git tag
`analysis-plan-frozen` exists and (b) the SHA-256 of `ANALYSIS_PLAN.md` in the working tree
equals the one recorded in the tagged commit. No outer test metric is computed or looked at
until the tag exists.

## 13. Search spaces and training settings

Deep models (every model gets 20 Optuna TPE trials per outer fold, sampler seed 0):

| parameter | LSTM | GRU | CNN-LSTM | TFT |
|---|---|---|---|---|
| hidden size | {8, 16, 32, 64} | {8, 16, 32, 64} | LSTM part {8, 16, 32, 64} | {8, 16, 32} |
| recurrent layers | {1, 2} | {1, 2} | {1, 2} | 1 (encoder) |
| convolution | | | filters {8, 16, 32}; kernel {3, 5, 7}; pooling {1, 2} | |
| attention heads | | | | {1, 2, 4} |
| dropout | U(0, 0.5) | U(0, 0.5) | U(0, 0.5) | U(0, 0.5) |
| learning rate | logU(1e-4, 1e-2) | same | same | same |
| weight decay | logU(1e-6, 1e-2) | same | same | same |
| batch size | {8, 16, 32} | same | same | same |

Static inputs enter every model through the same route: concatenated to the final hidden
state before a two-layer output head (LSTM, GRU, CNN-LSTM) or through the static covariate
encoders (TFT). At most 300 epochs; early stopping with patience 30 on validation MSE.

Gradient boosting (`HistGradientBoostingRegressor`, 20 trials per outer fold):
learning rate logU(0.01, 0.3); max_iter int [50, 500]; max_depth {2, 3, 4};
min_samples_leaf {3, 5, 10, 20}; l2_regularization logU(1e-4, 10).

Ridge: alpha in {10^-3, 10^-2.5, ..., 10^3}; features standardized on the training fold.

Growth curves (fitted to log W by `scipy.optimize.curve_fit`, three parameters each):
Gompertz log W = log A - b exp(-c t); von Bertalanffy log W = log Winf + 3 log(1 - exp(-k (t - t0)));
logistic log W = log A - log(1 + exp(-k (t - tm))). Starting values and bounds are fixed in
`configs/` before the freeze.

## 14. Compute

CPU only (4 cores). Trainings run in up to four single-threaded worker processes. Runtime
and hardware are logged per training.
