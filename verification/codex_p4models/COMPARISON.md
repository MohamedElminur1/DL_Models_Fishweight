# Codex blind rebuild of folds and baseline models against the pipeline

Tolerance: relative 1e-06, absolute 1e-09; the strict column counts differences beyond 1e-9. Tasks compared: T1, T2. No outer-test error is computed here.

## T1 outer folds (ours: gc chosen.csv and runtime plan rows)

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

- `dropped_ponds`: 0 of 7 differ.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_train | 7 | 0 | 0 | 0 |
| n_test | 7 | 0 | 0 | 0 |

## T1 gc: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 18 | 8.74e-08 |

## T1 cohort_mean: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 13 | 1.06e-08 |

## T1 ridge_time: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 8.05e-16 |

## T1 ridge_time: chosen alpha

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| alpha | 7 | 0 | 0 | 0 |

## T1 ridge_time_wq: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 2.02e-14 |

## T1 ridge_time_wq: chosen alpha

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| alpha | 7 | 0 | 0 | 0 |

## T1 allometric: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 3.61e-15 |

## T1 gc: chosen growth curve per outer fold

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

- `chosen_curve`: 0 of 7 differ.
- `n_failed`: 0 of 7 differ.

## T1 gc: outer test predictions

Keys ['pond', 'date']: 35 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 35 | 0 | 0 | 0 |
| w_pred | 35 | 0 | 24 | 4.37e-05 |
| y_true_log | 35 | 0 | 0 | 8.88e-16 |
| y_pred_log | 35 | 0 | 19 | 2e-07 |

## T1 cohort_mean: outer test predictions

Keys ['pond', 'date']: 35 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 35 | 0 | 0 | 0 |
| w_pred | 35 | 0 | 1 | 2.83e-06 |
| y_true_log | 35 | 0 | 0 | 8.88e-16 |
| y_pred_log | 35 | 0 | 1 | 7.97e-09 |

## T1 ridge_time: outer test predictions

Keys ['pond', 'date']: 35 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 35 | 0 | 0 | 0 |
| w_pred | 35 | 0 | 0 | 2.27e-13 |
| y_true_log | 35 | 0 | 0 | 8.88e-16 |
| y_pred_log | 35 | 0 | 0 | 1.33e-15 |

## T1 ridge_time_wq: outer test predictions

Keys ['pond', 'date']: 35 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 35 | 0 | 0 | 0 |
| w_pred | 35 | 0 | 0 | 3.02e-08 |
| y_true_log | 35 | 0 | 0 | 8.88e-16 |
| y_pred_log | 35 | 0 | 0 | 7.28e-14 |

## T1 allometric: outer test predictions

Keys ['pond', 'date']: 35 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 35 | 0 | 0 | 0 |
| w_pred | 35 | 0 | 0 | 1.48e-12 |
| y_true_log | 35 | 0 | 0 | 8.88e-16 |
| y_pred_log | 35 | 0 | 0 | 4.88e-15 |

## T2 outer folds (ours: persistence chosen.csv and runtime plan rows)

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

- `dropped_ponds`: 0 of 7 differ.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_train | 7 | 0 | 0 | 0 |
| n_test | 7 | 0 | 0 | 0 |

## T2 persistence: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 2.22e-16 |
| S_h1 | 21 | 0 | 0 | 1.11e-16 |
| S_h2 | 21 | 0 | 0 | 1.11e-16 |

## T2 linear: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 1.11e-16 |
| S_h1 | 21 | 0 | 0 | 1.67e-16 |
| S_h2 | 21 | 0 | 0 | 1.11e-16 |

## T2 sgr: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 6.11e-16 |
| S_h1 | 21 | 0 | 0 | 3.33e-16 |
| S_h2 | 21 | 0 | 0 | 6.66e-16 |

## T2 anchored_gc: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 16 | 6.42e-09 |
| S_h1 | 21 | 0 | 12 | 4.68e-09 |
| S_h2 | 21 | 0 | 16 | 8.15e-09 |

## T2 cohort_growth: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 13 | 5.05e-09 |
| S_h1 | 21 | 0 | 7 | 3.35e-09 |
| S_h2 | 21 | 0 | 13 | 6.75e-09 |

## T2 ridge_pw: inner-validation scores

Keys ['held_out_pond', 'inner_fold']: 21 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| n_val | 21 | 0 | 0 | 0 |
| S | 21 | 0 | 0 | 2.22e-16 |
| S_h1 | 21 | 0 | 0 | 6.11e-16 |
| S_h2 | 21 | 0 | 0 | 5e-16 |

## T2 ridge_pw: chosen alpha

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| alpha | 7 | 0 | 0 | 0 |

## T2 ridge_pw_wq: not compared

The pipeline has no `results/tuning/T2/lopo/ridge_pw_wq/inner_scores.csv`: the run of this model stopped on a non-finite inner-validation prediction (plan 9.0). Codex's values are kept for reference.

## T2 anchored_gc: chosen growth curve per outer fold

Keys ['held_out_pond']: 7 matched, 0 only in ours, 0 only in Codex.

- `chosen_curve`: 0 of 7 differ.
- `n_failed`: 0 of 7 differ.

## T2 persistence: outer test predictions

Keys ['pond', 'origin_date', 'horizon']: 53 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 53 | 0 | 0 | 0 |
| w_pred | 53 | 0 | 0 | 0 |
| y_true_log | 53 | 0 | 0 | 1.11e-16 |
| y_pred_log | 53 | 0 | 0 | 0 |

## T2 linear: outer test predictions

Keys ['pond', 'origin_date', 'horizon']: 53 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 53 | 0 | 0 | 0 |
| w_pred | 53 | 0 | 0 | 7.11e-14 |
| y_true_log | 53 | 0 | 0 | 1.11e-16 |
| y_pred_log | 53 | 0 | 0 | 8.05e-16 |

## T2 sgr: outer test predictions

Keys ['pond', 'origin_date', 'horizon']: 53 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 53 | 0 | 0 | 0 |
| w_pred | 53 | 0 | 0 | 3.41e-13 |
| y_true_log | 53 | 0 | 0 | 1.11e-16 |
| y_pred_log | 53 | 0 | 0 | 1.33e-15 |

## T2 anchored_gc: outer test predictions

Keys ['pond', 'origin_date', 'horizon']: 53 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 53 | 0 | 0 | 0 |
| w_pred | 53 | 0 | 27 | 4.49e-06 |
| y_true_log | 53 | 0 | 0 | 1.11e-16 |
| y_pred_log | 53 | 0 | 36 | 3.53e-08 |

## T2 cohort_growth: outer test predictions

Keys ['pond', 'origin_date', 'horizon']: 53 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 53 | 0 | 0 | 0 |
| w_pred | 53 | 0 | 8 | 4.49e-06 |
| y_true_log | 53 | 0 | 0 | 1.11e-16 |
| y_pred_log | 53 | 0 | 8 | 1.17e-08 |

## T2 ridge_pw: outer test predictions

Keys ['pond', 'origin_date', 'horizon']: 53 matched, 0 only in ours, 0 only in Codex.

| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |
|---|---|---|---|---|
| w_true | 53 | 0 | 0 | 0 |
| w_pred | 53 | 0 | 0 | 2.56e-13 |
| y_true_log | 53 | 0 | 0 | 1.11e-16 |
| y_pred_log | 53 | 0 | 0 | 1.3e-15 |

## T2 ridge_pw_wq: not compared

The pipeline has no `results/predictions/T2/lopo/ridge_pw_wq.csv`: the run of this model stopped on a non-finite inner-validation prediction (plan 9.0). Codex's values are kept for reference.

Result: all compared values agree. Not compared (no pipeline output, plan 9.0 stop): T2 ridge_pw_wq.
