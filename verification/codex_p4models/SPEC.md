# Independent verification task (Codex): folds, baselines and ridge models

You are verifying another team's model evaluation blind. Work only inside this directory
(`verification/codex_p4models/`). Read the two input tables below. Do NOT open, list or read
anything else under `/home/dev/development/publications/` (in particular not `src`,
`results`, `configs`, `tasks`, `tests`, `data/interim`, other `verification/` folders or any
Markdown file outside this directory). The point is an independent implementation from this
spec alone.

Python: `/home/dev/development/publications/.venv/bin/python -I` (pandas, numpy, scipy,
scikit-learn available). Write scripts and outputs in this directory only. Put your code in
`verify.py`, add `test_verify.py` with small hand-checkable cases for the fold rules, the
statistic S and the T2 baselines, and run it with `-m pytest`.

## 0. Inputs (read-only)

`/home/dev/development/publications/data/processed/t1_table.csv`, one row per fish weighing
event with t > 0 (T1). Columns used here: `pond`, `date`, `t` (days since 2021-06-19), `t2`
(t squared), `log_w` (natural log of the weight in g), `w`, `log_l` (log length), the 24
window features `{sensor}_{stat}_w` and the 6 cumulative features `{sensor}_mean_cum`
(sensors temperature, turbidity, do, ph, ammonia_log10, nitrate; stats mean, sd, min, max),
and `eligible_eval`.

`/home/dev/development/publications/data/processed/t2_table.csv`, one row per (origin event,
horizon) (T2). Columns used here: `pond`, `origin_date`, `target_date`, `horizon` (1 or 2),
`t_k` (t of the origin), `dt_prev` (t of the origin minus t of the previous event), `dt_target`
(14 or 28), `log_w_k`, `log_w_prev`, `w_origin`, `w_target`, `y` = log(w_target / w_origin),
the 24 window features `{sensor}_{stat}_w`, and `eligible_eval`.

Feature values may be missing (empty cells). Use only rows with `eligible_eval` true, for
training and for testing, everywhere in this task.

## 1. Outer folds (leave one pond out)

Cohort ponds: 1, 2, 3, 4, 6, 7, 8, 9. For every cohort pond with at least one eligible row,
in increasing pond order, there is one outer fold per task: its test set is the eligible rows
of that pond. Its training set is the eligible rows of every other cohort pond, except that
ponds 1 and 2 share one sensor stream: when pond 1 is held out, pond 2 is removed from
training, and vice versa.

## 2. Inner folds (used to choose ridge alpha and to report inner scores)

Inside one outer fold: take the training ponds that have at least one training row. Group
them into stream units (ponds 1 and 2 form one unit when both are present; every other pond
is its own unit). Sort the units by their smallest pond id and number them 0, 1, 2, ...;
unit i goes to inner fold i mod 3. Inner fold j validates on the training rows of the units
in fold j and fits on the training rows of all other training ponds.

## 3. The statistic S

Absolute log error per sample: T1 |log W_hat - log_w|; T2 |y_hat - y|. A predicted weight of
0 g or less has log error +infinity. S of a set of samples: mean absolute log error per
pond, then the mean of these per-pond values over the ponds present.
T2 tuning statistic (inner folds): S over the horizon-1 samples and S over the horizon-2
samples, computed separately as above, then their mean.

## 4. Models

Every model is fitted on the training rows it is given and on nothing else (medians,
scalers, means and curves are computed inside each fit). T1 models predict log W
(W_hat = exp); T2 models predict y (W_hat = w_origin * exp(y_hat)). One T2 model serves both
horizons; `dt_target` tells it which.

Growth curve fit G (used by `gc` and `anchored_gc`): fit each of the three curves below to
the points (t, log W) with `scipy.optimize.curve_fit(f, t, logW, p0=start,
bounds=(lower, upper), max_nfev=20000)` from each of its three starts, in the order given;
a start that raises is skipped. Per curve keep the start with the lowest residual sum of
squares (RSS). Among the curves that fitted, choose the lowest AIC = n log(RSS / n) + 6
(n = number of points). If no curve fits, see the model's fallback.

| curve | formula for log W | lower | upper | starts |
|---|---|---|---|---|
| gompertz | log_a - b exp(-c t) | (2.3, 0.01, 1e-4) | (11.5, 30, 1) | (7, 6, 0.02), (6, 5, 0.04), (9, 8, 0.01) |
| von_bertalanffy | log_winf + 3 log(1 - exp(-k (t - t0))) | (2.3, 1e-4, -300) | (11.5, 1, -0.001) | (7, 0.02, -10), (6.5, 0.01, -30), (8, 0.005, -50) |
| logistic | log_a - log(1 + exp(-k (t - tm))) | (2.3, 1e-4, -100) | (11.5, 1, 600) | (7, 0.03, 100), (6, 0.05, 60), (8, 0.02, 150) |

Ridge fit with alpha (used by every `ridge_*` model): drop every feature column that is
missing in all training rows; replace remaining missing values by the training median of
their column; if the model has water-quality features, append one indicator per sensor,
equal to 1 when `{sensor}_mean_w` is missing in that row (6 indicators); standardize every
column (indicators included) with the training mean and population SD (ddof 0; an SD of 0
is replaced by 1); fit `sklearn.linear_model.Ridge(alpha=alpha)` (with intercept) on the
log-scale target. Apply the same medians, indicators and scaling to the rows predicted.
Choosing alpha: the grid is 10^-3, 10^-2.5, ..., 10^3 (13 values). For each alpha, fit on the
inner-training rows and score each inner fold's validation rows (T1: S; T2: the tuning
statistic); the alpha with the lowest mean over the three inner folds wins, ties go to the
larger alpha. Then fit with that alpha on all training rows of the outer fold and predict the
test rows.

T1 models:
- `gc`: G fitted on (t, log_w) of the training rows, evaluated at the test t. If no curve
  fits: `cohort_mean` without its fallback.
- `cohort_mean`: for a test event at day t, if at least 3 different training ponds have a
  training row with the same t, predict the mean log_w of those rows; otherwise `gc`.
- `ridge_time`: features t, t2.
- `ridge_time_wq`: features t, t2, the 24 window features (sensor-major order:
  temperature_mean_w, temperature_sd_w, temperature_min_w, temperature_max_w, turbidity_...,
  do_..., ph_..., ammonia_log10_..., nitrate_...), then the 6 `{sensor}_mean_cum`.
- `allometric`: ordinary least squares of log_w on log_l with intercept; predict from log_l.

T2 models:
- `persistence`: y_hat = 0.
- `linear`: W_prev = exp(log_w_prev); W_hat = w_origin + (w_origin - W_prev) / dt_prev *
  dt_target. If W_hat <= 0 the log error is +infinity; count such predictions.
- `sgr`: y_hat = (log_w_k - log_w_prev) / dt_prev * dt_target.
- `anchored_gc`: points = the origin events (pond, origin_date, t_k, log_w_k) and the target
  events (pond, target_date, t_k + dt_target, log(w_target)) of the training rows, one point
  per (pond, date). Fit G on them; y_hat = G(t_k + dt_target) - G(t_k). If no curve fits:
  `sgr`.
- `cohort_growth`: if at least 3 different training ponds have a training row with the same
  origin_date and target_date, y_hat = the mean y of those rows; otherwise `anchored_gc`
  fitted on the same training rows.
- `ridge_pw`: features log_w_k, log_w_prev, t_k, dt_prev, dt_target.
- `ridge_pw_wq`: those 5, then the 24 window features in the order above (no cumulative
  features in T2).

## 5. What to compute

For every task, model and outer fold: the inner scores and the test predictions. For the
models without alpha the inner score of an inner fold is the model fitted on that fold's
inner-training rows and scored on its validation rows.

## 6. Outputs

- `codex_folds.csv`: task, held_out_pond, train_ponds and dropped_ponds (space-separated pond
  ids), n_train, n_test, inner0_val_ponds, inner1_val_ponds, inner2_val_ponds.
- `codex_inner.csv`: task, model, held_out_pond, inner_fold, n_val, S (T2: the tuning
  statistic), S_h1, S_h2 (T2 only), alpha (ridge only, the chosen alpha).
- `codex_predictions.csv`: task, model, held_out_pond, pond, date (T1: date; T2:
  target_date), origin_date (T2), horizon (0 for T1), y_true_log, y_pred_log (empty when the
  predicted weight is 0 g or less), w_true, w_pred (grams).
- `codex_curves.csv`: task, model (gc, anchored_gc, and their use inside cohort_mean and
  cohort_growth is not repeated), held_out_pond, chosen curve, its three parameters, RSS, AIC,
  and the curves that failed.
- `codex_scores.csv`: task, model, horizon (0 for T1; 1 and 2 for T2), S over all outer
  folds' test predictions, n_ponds, and one column per pond with that pond's mean absolute log
  error.

Report numbers as computed (at least 10 significant digits). End by writing `DONE.md`
listing every output file, the number of outer folds and test rows per task, the number of
`linear` predictions at or below 0 g, every fallback used (model, fold, count), and any
assumption you had to make.
