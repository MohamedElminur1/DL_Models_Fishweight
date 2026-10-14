"""The primary statistic S of ANALYSIS_PLAN.md section 9.0, shared by every script.

S is computed on the log scale. For each pond, the mean absolute error of log W over its
scored samples (T1: |log W_hat - log W|; T2: |y_hat - y|, which is the same quantity at the
target); S is the mean of those per-pond values over the ponds that have a scored sample.
For deep models and `mlp_static`, the per-pond values of the five seeds are averaged first.

T2 has one S per horizon. The statistic that tunes and selects a T2 model (inner objective,
ridge alpha, `select_primary.py`) is the mean of the S at horizon 1 and the S at horizon 2 on
the inner-validation samples (`tuning_s`); the permutation statistic is S at horizon 1.

`train.py` (inner objective and inner scores), `permute.py` and `evaluate.py` all use these
functions, so the objective that selects a model, the statistic that is permuted and the
statistic that is tested cannot drift apart.

A prediction that is NaN has no logarithm (the `linear` baseline when it extrapolates to a
weight that is not positive, plan 6.1). Its log error is +inf, and so is the S of every pond
and every set of ponds that contains it. It is not floored or replaced.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def abs_log_error(pred_log, true_log) -> np.ndarray:
    """Absolute log-scale error per sample; a NaN prediction gives +inf, a NaN target raises."""
    pred = np.asarray(pred_log, dtype=float)
    true = np.asarray(true_log, dtype=float)
    if pred.shape != true.shape:
        raise ValueError(f"prediction and target shapes differ: {pred.shape} and {true.shape}")
    if not np.isfinite(true).all():
        raise ValueError("a log-scale target is not finite")
    return np.where(np.isnan(pred), np.inf, np.abs(pred - true))


def pond_scores(ponds, errors) -> pd.Series:
    """Mean absolute log error per pond (index: pond id, sorted); needs at least one sample."""
    ponds = np.asarray(ponds)
    errors = np.asarray(errors, dtype=float)
    if ponds.shape != errors.shape or errors.size == 0:
        raise ValueError("pond_scores needs one error per sample and at least one sample")
    return pd.Series(errors).groupby(ponds).mean().sort_index().rename_axis("pond").rename("S")


def s_value(per_pond: pd.Series | np.ndarray) -> float:
    """S: the mean of the per-pond values (every scored pond counts once)."""
    values = np.asarray(per_pond, dtype=float)
    if values.size == 0:
        raise ValueError("S needs at least one scored pond")
    return float(values.mean())


def s_statistic(ponds, pred_log, true_log) -> float:
    """S of one set of predictions: per-pond means of the absolute log error, then their mean."""
    return s_value(pond_scores(ponds, abs_log_error(pred_log, true_log)))


def horizon_s(ponds, horizons, errors) -> dict[int, float]:
    """S of each horizon present: per-pond means over that horizon's samples, then their mean."""
    ponds, horizons = np.asarray(ponds), np.asarray(horizons)
    errors = np.asarray(errors, dtype=float)
    if not ponds.shape == horizons.shape == errors.shape:
        raise ValueError("ponds, horizons and errors need one entry per sample")
    return {int(h): s_value(pond_scores(ponds[horizons == h], errors[horizons == h]))
            for h in sorted(set(horizons.tolist()))}


def tuning_s(task: str, ponds, errors, horizons=None, expected_horizons=(1, 2)) -> float:
    """The statistic that tunes and selects models on an inner-validation set (plan 7.2, 9.4).

    T1: S. T2: the mean of the S of each horizon in `expected_horizons`, so that the horizon
    with more samples does not dominate; a horizon without a scored sample raises ValueError.
    `errors` are absolute log errors (`abs_log_error`), one per sample.
    """
    if task == "T1":
        return s_value(pond_scores(ponds, errors))
    if horizons is None:
        raise ValueError("T2 needs the horizon of every sample")
    per_horizon = horizon_s(ponds, horizons, errors)
    missing = [h for h in expected_horizons if h not in per_horizon]
    if missing:
        raise ValueError(f"no scored T2 sample at horizon(s) {missing}: the tuning S needs every horizon")
    return float(np.mean([per_horizon[h] for h in expected_horizons]))


def seed_averaged_pond_scores(frame: pd.DataFrame, seed_col: str = "seed",
                              pond_col: str = "pond", err_col: str = "abs_err") -> pd.Series:
    """Per-pond values averaged over seeds: mean over seeds of each seed's per-pond mean error.

    `frame` has one row per sample and seed. A pond missing from one seed raises, so the
    seeds always cover the same ponds.
    """
    per_seed = frame.groupby([pond_col, seed_col])[err_col].mean().unstack(seed_col)
    if per_seed.isna().any().any():
        raise ValueError("the seeds do not cover the same ponds")
    return per_seed.mean(axis=1).sort_index().rename_axis("pond").rename("S")


def permutation_p(s_obs: float, s_perm) -> float:
    """p = (1 + number of permutations with S_perm <= S_obs) / (R + 1) (plan 8.2).

    A smaller S is better, so a permuted run that does as well as the observed one counts
    against the observed result. NaN in either argument raises.
    """
    s_perm = np.asarray(s_perm, dtype=float)
    if np.isnan(s_perm).any() or np.isnan(s_obs) or len(s_perm) == 0:
        raise ValueError("the observed and permuted statistics must be numbers (and at least one permutation)")
    return (1 + int(np.sum(s_perm <= s_obs))) / (len(s_perm) + 1)
