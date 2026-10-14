"""Time-only baselines, tabular models and the allometric reference (plan 6.1, 6.2, 6.4).

Every model here, and the deep models in src/models, implements the `Learner` interface:

    info = learner.fit(train, val=None, epochs=None)   # returns FitInfo
    log_pred = learner.predict_log(df)                  # T1: log W; T2: y = log(W_target / W_origin)
    grams = learner.predict_grams(df)                   # T1: W; T2: W_target
    learner.notes()                                     # free text for the runtime log

`train`, `val` and `df` are row subsets of the T1 or T2 table (src/folds.py `select_samples`).
`val` and `epochs` matter only for the deep models. Every learner is fitted on the training
rows it is given and nothing else: imputers, scalers, medians and cohort means are computed
inside `fit`.

`make_baseline` builds a learner from its model name; `feature_columns` lists the columns of
a tabular model. Error is measured in grams: T1 exp(pred) against `w`, T2 `w_origin` times
exp(pred) against `w_target`. The primary statistic S (src/scoring.py) uses the log scale.

Missing-data rules (plan 5.1, 6.1): a feature that is missing in every training row is dropped
for that fit; `cohort_mean` and `cohort_growth` need at least `min_ponds_same_date` training
ponds with the date (pair), else they fall back; a growth curve that fails is not retried, and
if all three fail `gc` uses `cohort_mean` and `anchored_gc` uses `sgr`. Failures and fallbacks
are counted in `notes()`.
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

LOG = logging.getLogger("phase4")


# ----------------------------------------------------------------------------------------
# Targets and the common interface
# ----------------------------------------------------------------------------------------

def target_log(task: str, df: pd.DataFrame) -> np.ndarray:
    """The log-scale target: T1 log W, T2 y = log(W_target / W_origin)."""
    return df["log_w" if task == "T1" else "y"].to_numpy(dtype=float)


def target_grams(task: str, df: pd.DataFrame) -> np.ndarray:
    """The target in grams: T1 W, T2 W_target."""
    return df["w" if task == "T1" else "w_target"].to_numpy(dtype=float)


def grams_from_log(task: str, pred_log: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Convert a log-scale prediction to grams (T1 exp(pred), T2 W_origin * exp(pred))."""
    with np.errstate(over="ignore"):
        grams = np.exp(np.asarray(pred_log, dtype=float))
    if task == "T2":
        grams = df["w_origin"].to_numpy(dtype=float) * grams
    return grams


@dataclass
class FitInfo:
    """What one `fit` call reports: size, wall time and, for deep models, the epoch record.

    `curves` holds one tuple (epoch, train_mse, val_mse, seconds) per epoch; `val_mse` is NaN
    when no validation set was used. `best_epoch` is the 1-based epoch of the lowest
    validation MSE.
    """

    n_train: int = 0
    seconds: float = 0.0
    epochs_run: int = 0
    best_epoch: int | None = None
    best_val_mse: float | None = None
    curves: list[tuple[int, float, float, float]] = field(default_factory=list)


class Learner:
    """Base class: subclasses implement `_fit` and `predict_log` (and may override the rest)."""

    task: str = ""

    def fit(self, train: pd.DataFrame, val: pd.DataFrame | None = None,
            epochs: int | None = None) -> FitInfo:
        """Fit on `train`; time the call and record the training size."""
        start = time.perf_counter()
        info = self._fit(train, val, epochs)
        info.seconds = time.perf_counter() - start
        info.n_train = len(train)
        return info

    def _fit(self, train: pd.DataFrame, val: pd.DataFrame | None,
             epochs: int | None) -> FitInfo:
        raise NotImplementedError

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_grams(self, df: pd.DataFrame) -> np.ndarray:
        return grams_from_log(self.task, self.predict_log(df), df)

    def notes(self) -> str:
        return ""


# ----------------------------------------------------------------------------------------
# Growth curves (plan 6.1 and 13)
# ----------------------------------------------------------------------------------------

def _gompertz(t, log_a, b, c):
    return log_a - b * np.exp(-c * t)


def _von_bertalanffy(t, log_winf, k, t0):
    x = np.maximum(k * (t - t0), 1e-12)
    return log_winf + 3.0 * np.log(-np.expm1(-x))


def _logistic(t, log_a, k, tm):
    return log_a - np.logaddexp(0.0, -k * (t - tm))


CURVE_FUNCTIONS = {
    "gompertz": _gompertz,
    "von_bertalanffy": _von_bertalanffy,
    "logistic": _logistic,
}
N_CURVE_PARAMS = 3


class GrowthFitError(RuntimeError):
    """No growth curve could be fitted (all three failed, or there are too few events)."""


@dataclass(frozen=True)
class GrowthCurve:
    """A fitted curve on log W: `name`, parameters, residual sum of squares, AIC."""

    name: str
    params: tuple[float, ...]
    rss: float
    n: int
    aic: float

    def __call__(self, t) -> np.ndarray:
        return CURVE_FUNCTIONS[self.name](np.asarray(t, dtype=float), *self.params)


@dataclass
class GrowthFit:
    """Result of `fit_growth_curve`: the chosen curve, all fitted curves, failure messages."""

    curve: GrowthCurve
    candidates: list[GrowthCurve]
    failures: list[str]


def fit_growth_curve(t, log_w, curves_cfg: dict) -> GrowthFit:
    """Fit Gompertz, von Bertalanffy and logistic to log W and keep the lowest AIC.

    Each curve is fitted with scipy `curve_fit` (bounded least squares) from every start in
    `curves_cfg`, in the order given, and the lowest residual sum of squares is kept. A start
    that fails is recorded in `failures`; a curve with no successful start is logged and
    skipped; nothing is tried after that. All three failing, or no more than three events,
    raises GrowthFitError. AIC = n log(RSS / n) + 2 * 3 (all curves have three parameters).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(log_w, dtype=float)
    if len(t) <= N_CURVE_PARAMS:
        raise GrowthFitError(f"need more than {N_CURVE_PARAMS} events to fit a growth curve, got {len(t)}")
    candidates: list[GrowthCurve] = []
    failures: list[str] = []
    for name, spec in curves_cfg["curves"].items():
        func = CURVE_FUNCTIONS[name]
        best: GrowthCurve | None = None
        for start in spec["starts"]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", OptimizeWarning)
                    popt, _ = curve_fit(
                        func, t, y, p0=start, bounds=(spec["lower"], spec["upper"]),
                        max_nfev=curves_cfg["max_nfev"],
                    )
            except (RuntimeError, ValueError) as exc:
                failures.append(f"{name} start {start}: {exc}")
                continue
            resid = y - func(t, *popt)
            rss = float(resid @ resid)
            if not np.isfinite(rss):
                failures.append(f"{name} start {start}: non-finite residuals")
                continue
            aic = len(t) * np.log(max(rss, 1e-300) / len(t)) + 2.0 * N_CURVE_PARAMS
            curve = GrowthCurve(name, tuple(float(p) for p in popt), rss, len(t), float(aic))
            if best is None or curve.rss < best.rss:
                best = curve
        if best is None:
            LOG.warning("growth curve %s failed from every start and is skipped", name)
            continue
        candidates.append(best)
    if not candidates:
        raise GrowthFitError("all growth curves failed to fit: " + "; ".join(failures))
    return GrowthFit(min(candidates, key=lambda c: c.aic), candidates, failures)


def _curve_notes(fit: GrowthFit | None, curves_cfg: dict) -> str:
    if fit is None:
        return "curve=none; curve_failures=1"
    skipped = [n for n in curves_cfg["curves"] if n not in {c.name for c in fit.candidates}]
    text = f"curve={fit.curve.name}; curve_failures=0"
    return text + (f"; skipped={','.join(skipped)}" if skipped else "")


class GrowthCurveModel(Learner):
    """T1 `gc`: the growth curve with the lowest training AIC, evaluated at t.

    If all three curves fail, the fit is not retried: the model predicts the cohort mean of
    the training events on the same sampling date (`CohortMean` without a growth-curve
    fallback) and the failure is counted in `notes()`.
    """

    task = "T1"

    def __init__(self, curves_cfg: dict, min_ponds: int = 3):
        self.curves_cfg = curves_cfg
        self.min_ponds = min_ponds
        self.fit_result: GrowthFit | None = None
        self._cohort: CohortMean | None = None

    def _fit(self, train, val, epochs) -> FitInfo:
        self._cohort = None
        try:
            self.fit_result = fit_growth_curve(train["t"], train["log_w"], self.curves_cfg)
        except GrowthFitError as exc:
            LOG.warning("gc: %s; using cohort_mean for this fit", exc)
            self.fit_result = None
            self._cohort = CohortMean(self.curves_cfg, self.min_ponds, gc_fallback=False)
            self._cohort._fit(train, None, None)
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        if self.fit_result is None:
            return self._cohort.predict_log(df)
        return self.fit_result.curve(df["t"].to_numpy(dtype=float))

    def notes(self) -> str:
        return _curve_notes(self.fit_result, self.curves_cfg)


class CohortMean(Learner):
    """T1 `cohort_mean`: mean log W of the training events on the same sampling day.

    Used only when at least `min_ponds` training ponds have an event on that day; otherwise
    the prediction is the growth curve (`gc`) fitted on the same training events. With
    `gc_fallback=False` (the fallback of `gc` itself) a day without enough ponds has no
    prediction and `predict_log` raises RuntimeError.
    """

    task = "T1"

    def __init__(self, curves_cfg: dict, min_ponds: int = 3, gc_fallback: bool = True):
        self.min_ponds = min_ponds
        self._fallback = GrowthCurveModel(curves_cfg, min_ponds) if gc_fallback else None
        self._by_day: dict[int, float] = {}
        self._n_fallback = 0

    def _fit(self, train, val, epochs) -> FitInfo:
        by_day = train.groupby("t")
        means, ponds = by_day["log_w"].mean(), by_day["pond"].nunique()
        self._by_day = {int(day): float(m) for day, m in means.items() if ponds[day] >= self.min_ponds}
        if self._fallback is not None:
            self._fallback._fit(train, None, None)
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        days = df["t"].to_numpy()
        out = np.array([self._by_day.get(int(round(d)), np.nan) for d in days], dtype=float)
        missing = np.isnan(out)
        self._n_fallback = int(missing.sum())
        if missing.any():
            if self._fallback is None:
                raise RuntimeError(
                    f"no growth curve and fewer than {self.min_ponds} training ponds on day(s) "
                    f"{sorted({int(d) for d in days[missing]})}: cohort_mean has no prediction"
                )
            out[missing] = self._fallback.predict_log(df[missing])
        return out

    def notes(self) -> str:
        text = f"n_fallback_gc={self._n_fallback}"
        return text + (f"; {self._fallback.notes()}" if self._fallback is not None else "")


class Allometric(Learner):
    """T1 `allometric`: log W = log a + b log L by ordinary least squares (plan 6.4)."""

    task = "T1"

    def __init__(self):
        self.coef_: tuple[float, float] | None = None

    def _fit(self, train, val, epochs) -> FitInfo:
        x = train["log_l"].to_numpy(dtype=float)
        design = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(design, train["log_w"].to_numpy(dtype=float), rcond=None)
        self.coef_ = (float(beta[0]), float(beta[1]))
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        intercept, slope = self.coef_
        return intercept + slope * df["log_l"].to_numpy(dtype=float)

    def notes(self) -> str:
        return f"log_a={self.coef_[0]:.6g}; b={self.coef_[1]:.6g}"


# ----------------------------------------------------------------------------------------
# T2 time-only baselines (plan 6.1)
# ----------------------------------------------------------------------------------------

class Persistence(Learner):
    """T2 `persistence`: W_target = W_origin, that is y_hat = 0."""

    task = "T2"

    def _fit(self, train, val, epochs) -> FitInfo:
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(df))


class LinearExtrapolation(Learner):
    """T2 `linear`: straight line through (t_{k-1}, W_{k-1}) and (t_k, W_k) in grams.

    The raw extrapolated weight is the prediction in grams, even when it is not positive.
    `predict_log` returns NaN for those cases (the log is undefined); the count of such
    predictions of the last call is reported by `notes`.
    """

    task = "T2"

    def __init__(self):
        self._n_nonpositive = 0

    def _fit(self, train, val, epochs) -> FitInfo:
        return FitInfo()

    def _extrapolate(self, df: pd.DataFrame) -> np.ndarray:
        w0 = df["w_origin"].to_numpy(dtype=float)
        w_prev = np.exp(df["log_w_prev"].to_numpy(dtype=float))
        slope = (w0 - w_prev) / df["dt_prev"].to_numpy(dtype=float)
        grams = w0 + slope * df["dt_target"].to_numpy(dtype=float)
        self._n_nonpositive = int(np.sum(grams <= 0))
        return grams

    def predict_grams(self, df: pd.DataFrame) -> np.ndarray:
        return self._extrapolate(df)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        grams = self._extrapolate(df)
        w0 = df["w_origin"].to_numpy(dtype=float)
        positive = grams > 0
        ratio = np.where(positive, grams, 1.0) / w0
        return np.where(positive, np.log(ratio), np.nan)

    def notes(self) -> str:
        return f"n_nonpositive_pred={self._n_nonpositive}"


class SpecificGrowthRate(Learner):
    """T2 `sgr`: straight line through log W_{k-1} and log W_k, extended to the target day."""

    task = "T2"

    def _fit(self, train, val, epochs) -> FitInfo:
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        rate = (df["log_w_k"].to_numpy(dtype=float) - df["log_w_prev"].to_numpy(dtype=float)) \
            / df["dt_prev"].to_numpy(dtype=float)
        return rate * df["dt_target"].to_numpy(dtype=float)


def anchor_events(train: pd.DataFrame) -> pd.DataFrame:
    """Origin and target events of T2 training samples as (pond, date, t, log_w), deduplicated.

    Raises ValueError when two samples give different weights for the same pond and date.
    """
    origin = pd.DataFrame({
        "pond": train["pond"].to_numpy(), "date": pd.to_datetime(train["origin_date"]).to_numpy(),
        "t": train["t_k"].to_numpy(dtype=float), "log_w": train["log_w_k"].to_numpy(dtype=float),
    })
    target = pd.DataFrame({
        "pond": train["pond"].to_numpy(), "date": pd.to_datetime(train["target_date"]).to_numpy(),
        "t": (train["t_k"] + train["dt_target"]).to_numpy(dtype=float),
        "log_w": np.log(train["w_target"].to_numpy(dtype=float)),
    })
    both = pd.concat([origin, target], ignore_index=True)
    spread = both.groupby(["pond", "date"])["log_w"].agg(lambda s: s.max() - s.min())
    if (spread > 1e-9).any():
        raise ValueError("the same pond and date carries two different weights in the T2 table")
    return both.drop_duplicates(["pond", "date"]).reset_index(drop=True)


class AnchoredGrowthCurve(Learner):
    """T2 `anchored_gc`: W_k * G(t_{k+h}) / G(t_k), with G fitted as `gc` on the events.

    The fitting events are the origin and target events of the training samples, one row per
    pond and date. If all three curves fail, the fit is not retried and the model predicts
    like `sgr` instead; the failure is counted in `notes()`.
    """

    task = "T2"

    def __init__(self, curves_cfg: dict):
        self.curves_cfg = curves_cfg
        self.fit_result: GrowthFit | None = None
        self._sgr = SpecificGrowthRate()

    def _fit(self, train, val, epochs) -> FitInfo:
        events = anchor_events(train)
        try:
            self.fit_result = fit_growth_curve(events["t"], events["log_w"], self.curves_cfg)
        except GrowthFitError as exc:
            LOG.warning("anchored_gc: %s; using sgr for this fit", exc)
            self.fit_result = None
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        if self.fit_result is None:
            return self._sgr.predict_log(df)
        curve = self.fit_result.curve
        t_origin = df["t_k"].to_numpy(dtype=float)
        return curve(t_origin + df["dt_target"].to_numpy(dtype=float)) - curve(t_origin)

    def notes(self) -> str:
        return _curve_notes(self.fit_result, self.curves_cfg)


def _date_text(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.strftime("%Y-%m-%d")


class CohortGrowth(Learner):
    """T2 `cohort_growth`: mean y of the training samples with the same origin and target date.

    Used only when at least `min_ponds` training ponds have a sample with both dates;
    otherwise the prediction is `anchored_gc` fitted on the same training samples.
    """

    task = "T2"

    def __init__(self, curves_cfg: dict, min_ponds: int = 3):
        self.min_ponds = min_ponds
        self._fallback = AnchoredGrowthCurve(curves_cfg)
        self._mean: dict[tuple[str, str], float] = {}
        self._n_fallback = 0

    def _fit(self, train, val, epochs) -> FitInfo:
        keyed = pd.DataFrame({
            "o": _date_text(train["origin_date"]).to_numpy(),
            "g": _date_text(train["target_date"]).to_numpy(),
            "y": train["y"].to_numpy(dtype=float),
            "pond": train["pond"].to_numpy(),
        })
        by_pair = keyed.groupby(["o", "g"])
        means, ponds = by_pair["y"].mean(), by_pair["pond"].nunique()
        self._mean = {k: float(v) for k, v in means.items() if ponds[k] >= self.min_ponds}
        self._fallback._fit(train, None, None)
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        keys = zip(_date_text(df["origin_date"]), _date_text(df["target_date"]))
        out = np.array([self._mean.get(k, np.nan) for k in keys], dtype=float)
        missing = np.isnan(out)
        self._n_fallback = int(missing.sum())
        if missing.any():
            out[missing] = self._fallback.predict_log(df[missing])
        return out

    def notes(self) -> str:
        return f"n_fallback_anchored_gc={self._n_fallback}; {self._fallback.notes()}"


# ----------------------------------------------------------------------------------------
# Tabular models: ridge and gradient boosting (plan 6.1, 6.2, 8.1, 8.3)
# ----------------------------------------------------------------------------------------

def model_spec(task: str, model: str, cfg: dict) -> dict:
    """Feature spec of a tabular model with defaults: groups, wq_prefix (None) and indicators."""
    spec = cfg["model_features"][task][model]
    return {"groups": list(spec["groups"]), "wq_prefix": spec.get("wq_prefix"),
            "indicators": bool(spec.get("indicators", True))}


def wq_columns(task: str, cfg: dict, prefix: str | None = None) -> list[str]:
    """Water-quality feature columns: 24 window features, plus 6 cumulative ones in T1.

    With a `prefix` the window features are those of that supplementary window length
    (`{prefix}_{sensor}_{stat}_w`, merged from wq_by_date.csv by src/data.py); the cumulative
    features have no windowed version and keep their names.
    """
    sensors, stats = cfg["design"]["sensors"], cfg["design"]["stats"]
    head = f"{prefix}_" if prefix else ""
    columns = [f"{head}{s}_{k}_w" for s in sensors for k in stats]
    if task == "T1":
        columns += [f"{s}_mean_cum" for s in sensors]
    return columns


def uses_water_quality(task: str, model: str, cfg: dict) -> bool:
    return "wq" in model_spec(task, model, cfg)["groups"]


def feature_columns(task: str, model: str, cfg: dict) -> list[str]:
    """Columns of a tabular model, in the order of its feature groups (config)."""
    spec = model_spec(task, model, cfg)
    columns: list[str] = []
    for group in spec["groups"]:
        if group == "wq":
            columns += wq_columns(task, cfg, spec["wq_prefix"])
        else:
            columns += cfg["feature_groups"][task][group]
    return columns


def indicator_columns(task: str, model: str, cfg: dict) -> list[str]:
    """Columns whose missingness is a ridge indicator: one per sensor, none if switched off."""
    spec = model_spec(task, model, cfg)
    if "wq" not in spec["groups"] or not spec["indicators"]:
        return []
    head = f"{spec['wq_prefix']}_" if spec["wq_prefix"] else ""
    return [f"{head}{s}_mean_w" for s in cfg["design"]["sensors"]]


def window_prefixes(task: str, cfg: dict) -> list[str]:
    """Supplementary window prefixes that the tabular models of `task` use (sorted)."""
    return sorted({m["wq_prefix"] for m in cfg["model_features"][task].values() if m.get("wq_prefix")})


class RidgeModel(Learner):
    """Ridge regression on the log target.

    A column that is missing in every training row is dropped. The others are imputed with
    the training median, one missing-sensor indicator per entry of `indicator_columns` is
    added (1 when the sensor has no value in the window), everything is standardized on the
    training fold, then `sklearn.linear_model.Ridge(alpha)` is fitted. With no column left
    the model predicts the training mean.
    """

    def __init__(self, task: str, columns: list[str], indicator_columns: list[str], alpha: float):
        self.task = task
        self.columns = columns
        self.indicator_columns = indicator_columns
        self.alpha = float(alpha)

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.active_].to_numpy(dtype=float)
        x = np.where(np.isnan(x), self.median_, x)
        if self.indicator_columns:
            flags = np.column_stack([df[c].isna().to_numpy(dtype=float) for c in self.indicator_columns])
            x = np.hstack([x, flags])
        return x

    def _fit(self, train, val, epochs) -> FitInfo:
        raw = train[self.columns].to_numpy(dtype=float)
        keep = ~np.isnan(raw).all(axis=0)
        self.active_ = [c for c, k in zip(self.columns, keep) if k]
        self.n_dropped_ = int((~keep).sum())
        self.median_ = np.nanmedian(raw[:, keep], axis=0) if keep.any() else np.zeros(0)
        target = target_log(self.task, train)
        design = self._design(train)
        self.constant_ = float(target.mean()) if design.shape[1] == 0 else None
        if self.constant_ is None:
            self.mean_ = design.mean(axis=0)
            scale = design.std(axis=0)
            self.scale_ = np.where(scale > 0, scale, 1.0)
            self.model_ = Ridge(alpha=self.alpha).fit((design - self.mean_) / self.scale_, target)
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(df), self.constant_)
        return self.model_.predict((self._design(df) - self.mean_) / self.scale_)

    def notes(self) -> str:
        return f"alpha={self.alpha:.6g}; n_dropped_all_missing={self.n_dropped_}"


class GBMModel(Learner):
    """Gradient boosting on the log target: HistGradientBoostingRegressor, native NaN handling.

    Early stopping is off, so exactly `max_iter` boosting rounds run and the fit is
    deterministic for a fixed `random_state`. A column that is missing in every training row
    is dropped; with no column left the model predicts the training mean.
    """

    def __init__(self, task: str, columns: list[str], hp: dict, random_state: int):
        self.task = task
        self.columns = columns
        self.hp = dict(hp)
        self.model_ = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=float(hp["learning_rate"]),
            max_iter=int(hp["max_iter"]),
            max_depth=int(hp["max_depth"]),
            min_samples_leaf=int(hp["min_samples_leaf"]),
            l2_regularization=float(hp["l2_regularization"]),
            early_stopping=False,
            random_state=random_state,
        )

    def _fit(self, train, val, epochs) -> FitInfo:
        raw = train[self.columns].to_numpy(dtype=float)
        keep = ~np.isnan(raw).all(axis=0)
        self.active_ = [c for c, k in zip(self.columns, keep) if k]
        self.n_dropped_ = int((~keep).sum())
        target = target_log(self.task, train)
        self.constant_ = float(target.mean()) if not self.active_ else None
        if self.constant_ is None:
            self.model_.fit(raw[:, keep], target)
        return FitInfo()

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        if self.constant_ is not None:
            return np.full(len(df), self.constant_)
        return self.model_.predict(df[self.active_].to_numpy(dtype=float))

    def notes(self) -> str:
        return f"n_dropped_all_missing={self.n_dropped_}"


# ----------------------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------------------

def baseline_kind(task: str, model: str, cfg: dict) -> str:
    """'ridge', 'gbm' or 'fixed' (no hyperparameters) for a non-deep model; KeyError if unknown."""
    if model in cfg["model_features"][task]:
        return "ridge" if model.startswith("ridge_") else "gbm"
    fixed = {
        "T1": {"gc", "cohort_mean", "allometric"},
        "T2": {"persistence", "linear", "sgr", "anchored_gc", "cohort_growth"},
    }
    if model in fixed[task]:
        return "fixed"
    raise KeyError(f"unknown {task} model {model!r}")


def make_baseline(task: str, model: str, cfg: dict, hp: dict | None = None,
                  seed: int = 0) -> Learner:
    """Build the learner called `model` for `task`.

    `hp` carries `alpha` for ridge models and the five gradient-boosting parameters for gbm
    models; `seed` is the gradient-boosting random_state.
    """
    kind = baseline_kind(task, model, cfg)
    curves = cfg["growth_curves"]
    min_ponds = curves["min_ponds_same_date"]
    if kind == "fixed":
        builders = {
            ("T1", "gc"): lambda: GrowthCurveModel(curves, min_ponds),
            ("T1", "cohort_mean"): lambda: CohortMean(curves, min_ponds),
            ("T1", "allometric"): Allometric,
            ("T2", "persistence"): Persistence,
            ("T2", "linear"): LinearExtrapolation,
            ("T2", "sgr"): SpecificGrowthRate,
            ("T2", "anchored_gc"): lambda: AnchoredGrowthCurve(curves),
            ("T2", "cohort_growth"): lambda: CohortGrowth(curves, min_ponds),
        }
        return builders[(task, model)]()
    columns = feature_columns(task, model, cfg)
    if kind == "ridge":
        return RidgeModel(task, columns, indicator_columns(task, model, cfg), hp["alpha"])
    return GBMModel(task, columns, hp, seed)
