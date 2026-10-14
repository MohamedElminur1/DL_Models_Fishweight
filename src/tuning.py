"""Tuning and final fitting of one model on one outer fold (plan 7.2, 7.3, 13).

`make_context` bundles what a run needs (tables, config, the sequence sample set of the deep
models). `tune_fold` evaluates a model on the inner splits of one outer fold and returns its
chosen hyperparameters and every record the runner writes: the trials, the inner scores of the
best trial, the learning curves of the inner fits. Fixed baselines are evaluated once, ridge
over its alpha grid, gradient boosting and the neural models (the four sequence models and
`mlp_static`) with an Optuna TPE study of `n_trials` trials. The objective of every search
is the mean over the inner splits of the tuning statistic (src/scoring.py: S, and for T2 the
mean of the S at horizon 1 and at horizon 2). `final_fits` fits the chosen model on all
training rows of the outer fold and predicts its test rows; it refuses to run unless the
freeze guard passes.

A prediction that is not finite stops the run with NonFinitePredictionError (a predicted
weight that is not positive has no logarithm and is allowed: its log error is +inf, and it is
counted). Nothing is replaced silently.

In dev mode (`ctx.dev`) the study lives in memory, and no error value, objective value,
learning-curve loss or inner score leaves this module: the records carry NaN in their place.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import optuna
import pandas as pd

from src import guard
from src.artifacts import model_key, runtime_row
from src.baselines import FitInfo, Learner, baseline_kind, make_baseline, target_grams, target_log
from src.folds import FoldPlan, InnerPlan
from src.models import NETWORK_MODELS
from src.models.common import DeepRegressor, SampleSet
from src.scoring import abs_log_error, horizon_s, tuning_s

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna prints every objective value otherwise


class NonFinitePredictionError(RuntimeError):
    """A model produced a prediction that is not finite; the run of that model stops."""


@dataclass
class Context:
    """Everything a model needs on every fold of one (task, scheme, model, sample set) run."""

    task: str
    scheme: str
    model: str
    sample_set: str
    cfg: dict
    table: pd.DataFrame
    sampleset: SampleSet | None
    dev: bool = False

    @property
    def key(self) -> str:
        return model_key(self.model, self.sample_set)

    @property
    def kind(self) -> str:
        return model_kind(self.task, self.model, self.cfg)

    def learner(self, hp: dict, seed: int) -> Learner:
        """A fresh learner with hyperparameters `hp`; `seed` seeds the neural models."""
        if self.kind == "deep":
            return DeepRegressor(self.task, self.model, hp, self.sampleset, self.cfg["budget"], seed)
        return make_baseline(self.task, self.model, self.cfg, hp, seed=self.cfg["seeds"]["gbm"])


def model_kind(task: str, model: str, cfg: dict) -> str:
    """'deep' (the four sequence models and mlp_static), 'ridge', 'gbm' or 'fixed'."""
    return "deep" if model in NETWORK_MODELS else baseline_kind(task, model, cfg)


def search_space(task: str, model: str, cfg: dict) -> dict[str, dict]:
    """Parameter specs in the order Optuna is asked for them (the order changes the TPE draws)."""
    kind = model_kind(task, model, cfg)
    if kind == "gbm":
        return dict(cfg["gbm"]["space"])
    if kind == "deep":
        return {**cfg["deep"]["arch_space"][model], **cfg["deep"]["common_space"]}
    raise KeyError(f"{model} has no Optuna search space")


def ridge_alphas(cfg: dict) -> list[float]:
    spec = cfg["ridge"]["alpha_log10"]
    return [float(10.0 ** e) for e in np.linspace(spec["start"], spec["stop"], int(spec["num"]))]


def suggest(trial: optuna.Trial, spec: dict[str, dict]) -> dict:
    """Ask Optuna for every parameter of `spec`, in order."""
    hp = {}
    for name, p in spec.items():
        if p["type"] == "categorical":
            hp[name] = trial.suggest_categorical(name, p["choices"])
        elif p["type"] == "float":
            hp[name] = trial.suggest_float(name, p["low"], p["high"], log=bool(p.get("log", False)))
        elif p["type"] == "int":
            hp[name] = trial.suggest_int(name, p["low"], p["high"], log=bool(p.get("log", False)))
        else:
            raise ValueError(f"unknown parameter type {p['type']!r} for {name}")
    return hp


# ----------------------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------------------

def predict_checked(ctx: Context, learner: Learner, df: pd.DataFrame, where: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Log-scale and gram predictions for `df`, and how many are not positive weights.

    A weight that is not positive (the straight-line baseline, plan 6.1) has no logarithm: its
    log prediction is NaN, its log error +inf, and it is counted. Any other prediction that is
    not finite raises NonFinitePredictionError, naming the run and the stage and counting
    the offending predictions, never their values.
    """
    pred = learner.predict_log(df)
    grams = learner.predict_grams(df)
    nonpositive = np.isfinite(grams) & (grams <= 0) & np.isnan(pred)
    bad = (~np.isfinite(pred) & ~nonpositive) | ~np.isfinite(grams)
    if bad.any():
        raise NonFinitePredictionError(
            f"{ctx.task} {ctx.scheme} {ctx.key}: {int(bad.sum())} of {len(df)} predictions are not "
            f"finite ({where}); the run of this model stops and nothing is replaced")
    return pred, grams, int(nonpositive.sum())


# ----------------------------------------------------------------------------------------
# One trial: the inner splits
# ----------------------------------------------------------------------------------------

@dataclass
class InnerResult:
    index: int
    n_samples: int
    s: float
    s_by_horizon: dict[int, float]
    info: FitInfo
    n_nonpositive: int
    notes: str = ""


@dataclass
class TrialRecord:
    """One evaluated hyperparameter setting: its inner results and objective value."""

    number: int
    hp: dict
    value: float
    inner: list[InnerResult]
    seconds: float

    def best_epochs(self) -> list[int]:
        return [r.info.best_epoch for r in self.inner]

    def attrs(self, dev: bool) -> dict:
        """Scalars stored with the trial: epochs and seconds always, inner scores never in dev."""
        out: dict = {}
        for r in self.inner:
            out[f"inner{r.index}_n"] = r.n_samples
            out[f"inner{r.index}_seconds"] = round(r.info.seconds, 4)
            if r.info.best_epoch is not None:
                out[f"inner{r.index}_best_epoch"] = r.info.best_epoch
                out[f"inner{r.index}_epochs_run"] = r.info.epochs_run
            if not dev:
                out[f"inner{r.index}_S"] = r.s
        return out


def evaluate_inner(ctx: Context, hp: dict, inner: InnerPlan, seed: int) -> InnerResult:
    """Fit on the inner training rows, early-stopping on the validation rows (neural models), score."""
    train, val = ctx.table.loc[list(inner.train_idx)], ctx.table.loc[list(inner.val_idx)]
    learner = ctx.learner(hp, seed)
    info = learner.fit(train, val) if ctx.kind == "deep" else learner.fit(train)
    pred, _grams, nonpositive = predict_checked(ctx, learner, val, f"inner split {inner.index}")
    errors = abs_log_error(pred, target_log(ctx.task, val))
    horizons = val["horizon"].to_numpy() if ctx.task == "T2" else None
    s = tuning_s(ctx.task, val["pond"].to_numpy(), errors, horizons, tuple(ctx.cfg["design"]["horizons"]))
    by_horizon = horizon_s(val["pond"].to_numpy(), horizons, errors) if horizons is not None else {}
    return InnerResult(inner.index, len(val), s, by_horizon, info, nonpositive, learner.notes())


def run_trial(ctx: Context, plan: FoldPlan, hp: dict, number: int) -> TrialRecord:
    """Evaluate `hp` on every inner split of `plan`; the objective is the mean tuning statistic."""
    started = time.perf_counter()
    seed = ctx.cfg["seeds"]["tuning"]
    results = [evaluate_inner(ctx, hp, inner, seed) for inner in plan.inner]
    value = float(np.mean([r.s for r in results]))
    return TrialRecord(number, dict(hp), value, results, time.perf_counter() - started)


# ----------------------------------------------------------------------------------------
# One outer fold: search, then the records
# ----------------------------------------------------------------------------------------

@dataclass
class TuneOutcome:
    """The result of tuning on one outer fold (values are NaN in dev mode)."""

    hp: dict
    epochs: int | None
    best_trial: int
    best_value: float
    trials: pd.DataFrame
    inner_scores: pd.DataFrame
    curves: list[tuple]
    runtime: list[dict]
    n_nonpositive: int
    timing: dict


def _curve_rows(ctx: Context, fold: int, record: TrialRecord) -> list[tuple]:
    rows = []
    for r in record.inner:
        for epoch, train_mse, val_mse, seconds in r.info.curves:
            if ctx.dev:
                train_mse = val_mse = float("nan")
            rows.append((ctx.task, ctx.scheme, ctx.key, fold, "inner", record.number, r.index,
                         ctx.cfg["seeds"]["tuning"], epoch, train_mse, val_mse, seconds))
    return rows


def _trial_frame(ctx: Context, records: list[TrialRecord], states: dict[int, str]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {"number": rec.number, "state": states[rec.number],
               "value": float("nan") if ctx.dev else rec.value,
               "duration": round(rec.seconds, 4)}
        row.update({f"params_{k}": v for k, v in rec.hp.items()})
        row.update({f"user_attrs_{k}": v for k, v in rec.attrs(ctx.dev).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def _inner_scores_frame(ctx: Context, fold: int, best: TrialRecord) -> pd.DataFrame:
    rows = []
    for r in best.inner:
        row = {"outer_fold": fold, "inner_fold": r.index, "n_samples": r.n_samples,
               "S": float("nan") if ctx.dev else r.s}
        if ctx.task == "T2":
            for h in ctx.cfg["design"]["horizons"]:
                row[f"S_h{h}"] = float("nan") if ctx.dev else r.s_by_horizon[h]
        rows.append(row)
    return pd.DataFrame(rows)


def path_counts(notes: list[str]) -> dict[str, int]:
    """Counts of the code paths the learners report in their notes (fallbacks, failed fits, dropped columns).

    Only integer counters (`n_...`, `curve_failures`) are summed, and `curve=<name>` is
    tallied per curve. Fitted values in the notes (alpha, coefficients) are ignored.
    """
    counts: dict[str, int] = {}
    for text in notes:
        for part in filter(None, (p.strip() for p in text.split(";"))):
            key, _, value = part.partition("=")
            if key == "curve" and value:
                counts[f"curve_{value}"] = counts.get(f"curve_{value}", 0) + 1
            elif (key.startswith("n_") or key == "curve_failures") and value.lstrip("-").isdigit():
                counts[key] = counts.get(key, 0) + int(value)
    return counts


def timing_summary(ctx: Context, plan: FoldPlan, records: list[TrialRecord]) -> dict:
    """Seconds per trial and per fit, and the final-fit time they imply (timings and counts only).

    A final fit trains on all `len(plan.train_idx)` training rows. For the neural models its
    time is estimated as the median best epoch of the inner fits times the measured seconds
    per epoch and training row; for the others as the mean inner fit time scaled by the
    training size. `path_counts` tallies the fallbacks and failed fits of the inner fits.
    """
    fits = [r for rec in records for r in rec.inner]
    trial_seconds = [rec.seconds for rec in records]
    n_train_inner = float(np.mean([f.info.n_train for f in fits]))
    fit_seconds = float(np.mean([f.info.seconds for f in fits]))
    out = {
        "n_trials": len(records), "n_inner_fits": len(fits),
        "trial_seconds_mean": float(np.mean(trial_seconds)), "trial_seconds_max": float(np.max(trial_seconds)),
        "inner_fit_seconds_mean": fit_seconds, "n_train_inner_mean": n_train_inner,
        "n_train_outer": len(plan.train_idx),
        "path_counts": path_counts([f.notes for f in fits]),
    }
    if ctx.kind == "deep":
        epochs = np.array([f.info.epochs_run for f in fits], dtype=float)
        seconds = np.array([f.info.seconds for f in fits], dtype=float)
        per_epoch_row = float(seconds.sum() / (epochs.sum() * n_train_inner))
        best_median = float(np.median([f.info.best_epoch for f in fits]))
        out.update({
            "epochs_run_mean": float(epochs.mean()), "best_epoch_median": best_median,
            "seconds_per_epoch_mean": float(seconds.sum() / epochs.sum()),
            "est_final_fit_seconds": best_median * per_epoch_row * len(plan.train_idx),
        })
    else:
        out["est_final_fit_seconds"] = fit_seconds * len(plan.train_idx) / n_train_inner
    return out


def _median_best_epoch(best: TrialRecord) -> int:
    """Median of the best epochs of the inner splits (always a whole number for 1 or 3 splits)."""
    return max(1, int(round(float(np.median(best.best_epochs())))))


def tune_fold(ctx: Context, plan: FoldPlan, n_trials: int, db_path=None) -> TuneOutcome:
    """Search the hyperparameters of `ctx.model` on the inner splits of the outer fold `plan`.

    `n_trials` is the Optuna budget (gradient boosting and the neural models). `db_path` is
    the SQLite file of the study (ignored in dev mode, where the study is kept in memory).
    """
    kind = ctx.kind
    states: dict[int, str] = {}
    if kind == "fixed":
        records = [run_trial(ctx, plan, {}, 0)]
    elif kind == "ridge":
        records = [run_trial(ctx, plan, {"alpha": a}, i) for i, a in enumerate(ridge_alphas(ctx.cfg))]
    else:
        records, states = _optuna_search(ctx, plan, n_trials, db_path)
    for rec in records:
        states.setdefault(rec.number, "COMPLETE")
    if kind == "ridge":  # ties go to the larger alpha (more shrinkage)
        best = min(records, key=lambda r: (r.value, -r.hp["alpha"]))
    else:  # lowest value, ties to the earliest trial
        best = min(records, key=lambda r: (r.value, r.number))
    runtime, curves = [], []
    for rec in records:
        runtime.append(runtime_row(
            ctx.task, ctx.scheme, ctx.key, plan.index, "trial", rec.number,
            epochs=sum(r.info.epochs_run for r in rec.inner) or "", seconds=rec.seconds))
        curves += _curve_rows(ctx, plan.index, rec)
    return TuneOutcome(
        hp=dict(best.hp),
        epochs=_median_best_epoch(best) if kind == "deep" else None,
        best_trial=best.number,
        best_value=float("nan") if ctx.dev else best.value,
        trials=_trial_frame(ctx, records, states) if kind != "fixed" else pd.DataFrame(),
        inner_scores=_inner_scores_frame(ctx, plan.index, best),
        curves=curves,
        runtime=runtime,
        n_nonpositive=sum(r.n_nonpositive for r in best.inner),
        timing=timing_summary(ctx, plan, records),
    )


def _optuna_search(ctx: Context, plan: FoldPlan, n_trials: int, db_path):
    storage = None if (ctx.dev or db_path is None) else f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=f"{ctx.task}-{ctx.scheme}-{ctx.key}-fold{plan.index:02d}", storage=storage,
        sampler=optuna.samplers.TPESampler(seed=ctx.cfg["seeds"]["sampler"]), direction="minimize")
    spec = search_space(ctx.task, ctx.model, ctx.cfg)
    records: dict[int, TrialRecord] = {}

    def objective(trial: optuna.Trial) -> float:
        record = run_trial(ctx, plan, suggest(trial, spec), trial.number)
        records[trial.number] = record
        for name, value in record.attrs(ctx.dev).items():
            trial.set_user_attr(name, value)
        return record.value

    study.optimize(objective, n_trials=n_trials, catch=())
    states = {t.number: t.state.name for t in study.trials}
    return [records[n] for n in sorted(records)], states


# ----------------------------------------------------------------------------------------
# Final fits (outer test prediction): guarded
# ----------------------------------------------------------------------------------------

@dataclass
class FinalFit:
    seed: int
    info: FitInfo
    predictions: pd.DataFrame
    n_nonpositive: int
    notes: str


def prediction_frame(ctx: Context, plan: FoldPlan, test: pd.DataFrame, pred_log, grams, seed: int) -> pd.DataFrame:
    """One row per test sample: ids, truth and prediction on both scales."""
    t2 = ctx.task == "T2"
    frame = pd.DataFrame({
        "task": ctx.task, "scheme": ctx.scheme, "model": ctx.model, "sample_set": ctx.sample_set,
        "fold": plan.index, "seed": seed, "row_id": test.index.to_numpy(),
        "pond": test["pond"].to_numpy(), "stream": test["stream"].to_numpy(),
        "horizon": test["horizon"].to_numpy() if t2 else 0,
        "date": test["target_date" if t2 else "date"].to_numpy(),
        "origin_date": test["origin_date"].to_numpy() if t2 else "",
        "t": (test["t_k"] + test["dt_target"]).to_numpy() if t2 else test["t"].to_numpy(),
        "w_true": target_grams(ctx.task, test), "w_pred": grams,
        "y_true_log": target_log(ctx.task, test), "y_pred_log": pred_log,
        "w_origin": test["w_origin"].to_numpy() if t2 else np.nan,
    })
    return frame


def final_fits(ctx: Context, plan: FoldPlan, hp: dict, epochs: int | None, seeds: list[int]) -> list[FinalFit]:
    """Fit on all training rows of `plan`, predict its test rows, once per seed.

    Neural models are trained for `epochs` epochs (the median best epoch of the best trial)
    with each seed in `seeds`; every other model is deterministic and is fitted once (seed 0).
    Raises GuardError unless the analysis plan is frozen and unchanged (plan 12).
    """
    guard.assert_test_allowed()
    return fit_and_predict(ctx, plan, hp, epochs, seeds)


def fit_and_predict(ctx: Context, plan: FoldPlan, hp: dict, epochs: int | None, seeds: list[int]) -> list[FinalFit]:
    """`final_fits` without the guard call, for a caller that has passed the guard itself.

    src/permute.py calls the guard once per permutation and then fits the seven or eight
    outer folds of that permutation through this function; no other caller should use it.
    """
    train, test = ctx.table.loc[list(plan.train_idx)], ctx.table.loc[list(plan.test_idx)]
    fits = []
    for seed in (seeds if ctx.kind == "deep" else [0]):
        learner = ctx.learner(hp, seed)
        info = learner.fit(train, epochs=epochs) if ctx.kind == "deep" else learner.fit(train)
        pred, grams, nonpositive = predict_checked(ctx, learner, test, f"final fit, seed {seed}")
        fits.append(FinalFit(seed, info, prediction_frame(ctx, plan, test, pred, grams, seed),
                             nonpositive, learner.notes()))
    return fits
