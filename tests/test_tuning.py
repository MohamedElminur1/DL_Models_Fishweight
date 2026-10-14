"""Unit tests for src/tuning.py: the tuning objective, search, tie rules, non-finite predictions,
dev masking and the guarded final fit.

Data are the synthetic tables of tests/synth_p4.py (test-only constructions). Most tests replace the
learner by a fake whose absolute log error is known exactly (truth plus an offset), so the
expected objective is computed by hand from the inner splits; a few run the real models with
tiny budgets.
"""

import copy
import json

import numpy as np
import optuna
import pandas as pd
import pytest

from src import guard, tuning
from src.baselines import FitInfo, Learner, target_log
from src.folds import load_config, plan_folds, select_samples
from src.models import NETWORK_MODELS, common
from src.models.common import DailyStore, build_sampleset
from src.tuning import Context, NonFinitePredictionError
from tests.synth_p4 import SENSORS, STATS, make_tables

common.configure_torch()


def small_config():
    cfg = copy.deepcopy(load_config())
    cfg["budget"] = {**cfg["budget"], "max_epochs": 6, "patience": 3}
    cfg["gbm"]["space"]["max_iter"] = {"type": "int", "low": 5, "high": 8}
    return cfg


@pytest.fixture(scope="module")
def world():
    cfg = small_config()
    t1, t2, daily = make_tables(n_events=7, seed=11, sparse_ponds=(8,))
    store = DailyStore(daily, SENSORS, STATS)
    tables = {"T1": t1, "T2": t2}
    plans = {task: [p for p in plan_folds(task, "lopo", tables[task], "eligible", cfg) if not p.skipped]
             for task in tables}
    sets = {task: build_sampleset(task, select_samples(tables[task], cfg["design"]["cohort"], "eligible"),
                                  store, cfg) for task in tables}
    return cfg, tables, plans, sets


def make_context(world, task, model, dev=False, scheme="lopo", cfg=None):
    base, tables, _, sets = world
    return Context(task, scheme, model, "eligible", cfg or base, tables[task],
                   sets[task] if model in NETWORK_MODELS else None, dev)


def first_plan(world, task):
    return world[2][task][0]


class OffsetLearner(Learner):
    """Predicts the truth plus `offset_of(df)`, so that every absolute log error is known."""

    def __init__(self, task, offset_of, best_epoch=None, curves=(), notes=""):
        self.task, self.offset_of, self.best_epoch = task, offset_of, best_epoch
        self.curves, self._notes = list(curves), notes

    def _fit(self, train, val, epochs):
        return FitInfo(epochs_run=len(self.curves), best_epoch=self.best_epoch, curves=list(self.curves))

    def predict_log(self, df):
        return target_log(self.task, df) + np.asarray(self.offset_of(df), dtype=float)

    def notes(self):
        return self._notes


def use_learner(monkeypatch, factory):
    """Make every `Context.learner(hp, seed)` call return `factory(ctx, hp, seed)`."""
    calls = []

    def learner(self, hp, seed):
        calls.append((dict(hp), seed))
        return factory(self, hp, seed)

    monkeypatch.setattr(Context, "learner", learner)
    return calls


def constant(task, value):
    return lambda ctx, hp, seed: OffsetLearner(task, lambda df: np.full(len(df), value))


# ----------------------------------------------------------------------------------------
# Search space and Optuna plumbing
# ----------------------------------------------------------------------------------------

def test_search_spaces_follow_the_config_order_and_ridge_has_none():
    cfg = load_config()
    assert list(tuning.search_space("T1", "gbm_time", cfg)) == list(cfg["gbm"]["space"])
    assert list(tuning.search_space("T1", "tft", cfg)) == ["hidden", "heads", "dropout", "lr", "weight_decay", "batch_size"]
    assert list(tuning.search_space("T2", "mlp_static", cfg))[0] == "hidden"
    with pytest.raises(KeyError, match="no Optuna search space"):
        tuning.search_space("T1", "ridge_time", cfg)


def test_ridge_alpha_grid_is_the_thirteen_log_spaced_values_of_the_config():
    alphas = tuning.ridge_alphas(load_config())
    assert len(alphas) == 13
    assert alphas[0] == pytest.approx(1e-3) and alphas[-1] == pytest.approx(1e3)
    np.testing.assert_allclose(np.diff(np.log10(alphas)), 0.5)


def test_suggest_draws_each_parameter_type_within_its_range():
    spec = {"c": {"type": "categorical", "choices": [3, 5]},
            "f": {"type": "float", "low": 1e-4, "high": 1e-2, "log": True},
            "u": {"type": "float", "low": 0.0, "high": 0.5},
            "i": {"type": "int", "low": 5, "high": 9}}
    study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    for _ in range(8):
        trial = study.ask()
        hp = tuning.suggest(trial, spec)
        assert list(hp) == ["c", "f", "u", "i"]
        assert hp["c"] in (3, 5) and 1e-4 <= hp["f"] <= 1e-2 and 0 <= hp["u"] <= 0.5 and 5 <= hp["i"] <= 9
        study.tell(trial, 0.0)
    with pytest.raises(ValueError, match="unknown parameter type"):
        tuning.suggest(study.ask(), {"x": {"type": "weird"}})


# ----------------------------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------------------------

class FixedPrediction(Learner):
    def __init__(self, log, grams):
        self.log, self.grams = np.asarray(log, dtype=float), np.asarray(grams, dtype=float)

    def predict_log(self, df):
        return self.log

    def predict_grams(self, df):
        return self.grams


def test_a_nonpositive_weight_is_counted_and_any_other_non_finite_prediction_stops_the_run(world):
    ctx = make_context(world, "T2", "linear")
    df = pd.DataFrame({"x": [0, 1, 2]})
    pred, grams, n = tuning.predict_checked(ctx, FixedPrediction([0.1, np.nan, np.nan], [5.0, -3.0, 0.0]), df, "t")
    assert n == 2 and np.isnan(pred[1:]).all() and pred[0] == 0.1
    for log, g in (([0.1, 0.2, np.nan], [5.0, 6.0, np.nan]),        # NaN weight
                   ([0.1, np.inf, 0.3], [5.0, np.inf, 6.0]),        # infinite prediction
                   ([0.1, np.nan, 0.3], [5.0, 4.0, 6.0]),           # NaN log of a positive weight
                   ([0.1, 0.2, 0.3], [5.0, 6.0, np.inf])):          # overflow in grams
        with pytest.raises(NonFinitePredictionError) as error:
            tuning.predict_checked(ctx, FixedPrediction(log, g), df, "inner split 2")
        message = str(error.value)
        assert "T2 lopo linear" in message and "inner split 2" in message and "of 3 predictions" in message
        assert "nan" not in message.lower() and "inf" not in message.lower()   # counts, never values


def test_a_non_finite_prediction_stops_tune_fold_with_the_run_named(world, monkeypatch):
    ctx = make_context(world, "T1", "ridge_time")
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner("T1", lambda df: np.full(len(df), np.nan)))
    with pytest.raises(NonFinitePredictionError, match="T1 lopo ridge_time"):
        tuning.tune_fold(ctx, first_plan(world, "T1"), 3)


# ----------------------------------------------------------------------------------------
# The objective
# ----------------------------------------------------------------------------------------

def test_t1_objective_is_the_mean_over_inner_splits_of_the_mean_over_ponds(world, monkeypatch):
    ctx, plan = make_context(world, "T1", "gc"), first_plan(world, "T1")
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner("T1", lambda df: 0.1 * df["pond"].to_numpy()))
    outcome = tuning.tune_fold(ctx, plan, 3)
    by_split = [np.mean([0.1 * p for p in set(ctx.table.loc[list(i.val_idx), "pond"])]) for i in plan.inner]
    assert outcome.best_value == pytest.approx(np.mean(by_split))
    pooled = np.mean([0.1 * p for i in plan.inner for p in ctx.table.loc[list(i.val_idx), "pond"]])
    assert pooled != pytest.approx(np.mean(by_split))          # the objective is not the pooled mean
    np.testing.assert_allclose(outcome.inner_scores["S"], by_split)
    assert list(outcome.inner_scores["inner_fold"]) == [0, 1, 2]
    assert list(outcome.inner_scores["n_samples"]) == [len(i.val_idx) for i in plan.inner]


def test_t2_objective_is_the_mean_of_the_horizon_1_and_horizon_2_scores(world, monkeypatch):
    ctx, plan = make_context(world, "T2", "persistence"), first_plan(world, "T2")
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner(
        "T2", lambda df: np.where(df["horizon"].to_numpy() == 1, 0.1, 0.5)))
    outcome = tuning.tune_fold(ctx, plan, 3)
    assert outcome.best_value == pytest.approx(0.3)
    val = ctx.table.loc[list(plan.inner[0].val_idx)]
    pooled = np.mean(np.where(val["horizon"] == 1, 0.1, 0.5))
    assert pooled != pytest.approx(0.3)                          # horizons have different sample counts
    np.testing.assert_allclose(outcome.inner_scores["S_h1"], 0.1)
    np.testing.assert_allclose(outcome.inner_scores["S_h2"], 0.5)
    np.testing.assert_allclose(outcome.inner_scores["S"], 0.3)


def test_fixed_models_run_one_trial_and_write_no_trials_file(world, monkeypatch):
    use_learner(monkeypatch, constant("T1", 0.2))
    outcome = tuning.tune_fold(make_context(world, "T1", "cohort_mean"), first_plan(world, "T1"), 20)
    assert outcome.hp == {} and outcome.epochs is None and outcome.best_trial == 0
    assert len(outcome.trials) == 0 and len(outcome.inner_scores) == 3
    assert [r["stage"] for r in outcome.runtime] == ["trial"]


# ----------------------------------------------------------------------------------------
# Ridge and the tie rules
# ----------------------------------------------------------------------------------------

def test_ridge_tries_every_alpha_and_ties_go_to_the_larger_alpha(world, monkeypatch):
    ctx, plan = make_context(world, "T1", "ridge_time"), first_plan(world, "T1")
    alphas = tuning.ridge_alphas(ctx.cfg)
    delta = {i: 0.5 for i in range(13)}
    delta[4] = delta[7] = 0.2                                    # two alphas tie at the minimum
    seen = []

    def factory(c, hp, seed):
        seen.append(hp["alpha"])
        return OffsetLearner("T1", lambda df, d=delta[alphas.index(hp["alpha"])]: np.full(len(df), d))

    use_learner(monkeypatch, factory)
    outcome = tuning.tune_fold(ctx, plan, 5)
    assert sorted(set(seen)) == sorted(alphas) and len(outcome.trials) == 13
    assert outcome.hp == {"alpha": alphas[7]} and outcome.best_trial == 7
    assert outcome.best_value == pytest.approx(0.2)


def test_a_strictly_lower_ridge_value_beats_a_larger_alpha(world, monkeypatch):
    ctx, plan = make_context(world, "T1", "ridge_time"), first_plan(world, "T1")
    alphas = tuning.ridge_alphas(ctx.cfg)
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner(
        "T1", lambda df, a=hp["alpha"]: np.full(len(df), 0.1 if a == alphas[2] else 0.3)))
    assert tuning.tune_fold(ctx, plan, 5).hp == {"alpha": alphas[2]}


# ----------------------------------------------------------------------------------------
# Optuna searches (gradient boosting and neural models)
# ----------------------------------------------------------------------------------------

def test_gbm_search_runs_n_trials_picks_the_lowest_value_and_stores_the_study(world, monkeypatch, tmp_path):
    ctx, plan = make_context(world, "T1", "gbm_time"), first_plan(world, "T1")
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner(
        "T1", lambda df, v=hp["learning_rate"]: np.full(len(df), v)))
    db = tmp_path / "fold" / "study.db"
    db.parent.mkdir()
    outcome = tuning.tune_fold(ctx, plan, 6, db)
    trials = outcome.trials
    assert len(trials) == 6 and set(trials["state"]) == {"COMPLETE"}
    best = trials.loc[trials["value"].idxmin()]
    assert outcome.best_trial == int(best["number"]) and outcome.best_value == pytest.approx(best["value"])
    assert outcome.hp == {k[len("params_"):]: best[k] for k in trials.columns if k.startswith("params_")}
    assert outcome.hp["learning_rate"] == pytest.approx(best["value"])       # the fake's error is the learning rate
    assert list(trials.filter(like="params_").columns) == [f"params_{k}" for k in ctx.cfg["gbm"]["space"]]
    stored = optuna.load_study(study_name=f"T1-lopo-gbm_time-fold{plan.index:02d}", storage=f"sqlite:///{db}")
    assert len(stored.trials) == 6


def test_the_sampler_seed_makes_the_search_repeatable(world, monkeypatch):
    ctx, plan = make_context(world, "T1", "gbm_time"), first_plan(world, "T1")
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner(
        "T1", lambda df, v=hp["learning_rate"]: np.full(len(df), v)))
    first = tuning.tune_fold(ctx, plan, 5).trials.filter(like="params_")
    again = tuning.tune_fold(ctx, plan, 5).trials.filter(like="params_")
    pd.testing.assert_frame_equal(first, again)
    other_cfg = copy.deepcopy(ctx.cfg)
    other_cfg["seeds"]["sampler"] = 7
    other = tuning.tune_fold(make_context(world, "T1", "gbm_time", cfg=other_cfg), plan, 5).trials.filter(like="params_")
    assert not first.equals(other)


def test_equal_values_go_to_the_earliest_trial(world, monkeypatch):
    use_learner(monkeypatch, constant("T1", 0.25))
    outcome = tuning.tune_fold(make_context(world, "T1", "gbm_time"), first_plan(world, "T1"), 4)
    assert outcome.best_trial == 0 and outcome.hp == {
        k[len("params_"):]: outcome.trials.loc[0, k] for k in outcome.trials.filter(like="params_").columns}


def test_the_final_epoch_count_is_the_median_best_epoch_of_the_best_trial(world, monkeypatch):
    ctx, plan = make_context(world, "T1", "lstm"), first_plan(world, "T1")
    pattern = iter([7, 3, 12] * 10)
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner(
        "T1", lambda df: np.full(len(df), 0.3), best_epoch=next(pattern), curves=[(1, 0.5, 0.4, 0.01)]))
    outcome = tuning.tune_fold(ctx, plan, 2)
    assert outcome.epochs == 7
    chrono = tuning.tune_fold(make_context(world, "T1", "lstm"),
                              type(plan)(**{**plan.__dict__, "inner": plan.inner[:1]}), 2)
    assert chrono.epochs == 7                                    # one split: its own best epoch (first of the pattern)


def test_inner_fits_record_their_learning_curves_and_dev_masks_the_losses(world, monkeypatch):
    plan = first_plan(world, "T1")
    curves = [(1, 0.5, 0.4, 0.01), (2, 0.3, 0.2, 0.03)]
    use_learner(monkeypatch, lambda c, hp, seed: OffsetLearner(
        "T1", lambda df: np.full(len(df), 0.3), best_epoch=2, curves=curves))
    real = tuning.tune_fold(make_context(world, "T1", "lstm"), plan, 2)
    assert len(real.curves) == 2 * 3 * 2                          # trials x inner splits x epochs
    task, scheme, model, fold, stage, trial, inner, seed, epoch, train_mse, val_mse, seconds = real.curves[0]
    assert (task, scheme, model, fold, stage, trial, inner, seed, epoch) == ("T1", "lopo", "lstm", plan.index, "inner", 0, 0, 0, 1)
    assert (train_mse, val_mse, seconds) == (0.5, 0.4, 0.01)
    masked = tuning.tune_fold(make_context(world, "T1", "lstm", dev=True), plan, 2)
    assert len(masked.curves) == len(real.curves)
    assert all(np.isnan(row[9]) and np.isnan(row[10]) and row[11] in (0.01, 0.03) for row in masked.curves)
    assert [row[8] for row in masked.curves] == [row[8] for row in real.curves]   # epochs and timings kept


# ----------------------------------------------------------------------------------------
# Dev mode: no error value, objective value or loss leaves the module
# ----------------------------------------------------------------------------------------

DEV_CASES = [("T1", "gc"), ("T1", "ridge_time_wq"), ("T1", "gbm_time_wq"), ("T2", "persistence"),
             ("T2", "ridge_pw"), ("T1", "lstm"), ("T2", "mlp_static"), ("T2", "tft")]


@pytest.mark.parametrize("task,model", DEV_CASES)
def test_dev_mode_carries_nan_in_every_value_column_and_prints_nothing(world, capsys, task, model):
    plan = first_plan(world, task)
    outcome = tuning.tune_fold(make_context(world, task, model, dev=True), plan, 2)
    assert np.isnan(outcome.best_value)
    assert outcome.inner_scores.filter(regex=r"^S").isna().all().all()
    if len(outcome.trials):
        assert outcome.trials["value"].isna().all()
        assert not outcome.trials.filter(regex=r"inner\d+_S$").columns.tolist()
    assert all(np.isnan(row[9]) and np.isnan(row[10]) for row in outcome.curves)
    assert outcome.hp is not None and outcome.timing["trial_seconds_mean"] > 0
    flat = json.dumps(outcome.timing)
    assert "nan" not in flat.lower()
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""


@pytest.mark.parametrize("task,model", [("T1", "ridge_time_wq"), ("T2", "gbm_pw"), ("T1", "gru"), ("T2", "persistence")])
def test_a_full_run_carries_finite_values_and_the_inner_scores_match_the_objective(world, task, model):
    plan = first_plan(world, task)
    outcome = tuning.tune_fold(make_context(world, task, model), plan, 2)
    assert np.isfinite(outcome.best_value) and np.isfinite(outcome.inner_scores["S"]).all()
    assert outcome.inner_scores["S"].mean() == pytest.approx(outcome.best_value)
    if len(outcome.trials):
        row = outcome.trials.loc[outcome.trials["number"] == outcome.best_trial].iloc[0]
        assert row["value"] == pytest.approx(outcome.best_value)
        assert row["user_attrs_inner0_S"] == pytest.approx(outcome.inner_scores["S"].iloc[0])
    if tuning.model_kind(task, model, world[0]) == "deep":
        assert 1 <= outcome.epochs <= world[0]["budget"]["max_epochs"]
        assert len(outcome.curves) == sum(int(r["epochs"]) for r in outcome.runtime)
        assert all(np.isfinite(row[9]) and np.isfinite(row[10]) for row in outcome.curves)
    else:
        assert outcome.epochs is None


def test_dev_runs_keep_the_study_in_memory(world, tmp_path):
    db = tmp_path / "study.db"
    tuning.tune_fold(make_context(world, "T1", "gbm_time", dev=True), first_plan(world, "T1"), 2, db)
    assert not db.exists()


# ----------------------------------------------------------------------------------------
# Non-positive weights of the straight-line baseline
# ----------------------------------------------------------------------------------------

def test_the_linear_baseline_counts_nonpositive_weights_and_scores_them_infinite(world):
    plan = first_plan(world, "T2")
    ctx = make_context(world, "T2", "linear")
    ctx.table = ctx.table.copy()
    row = plan.inner[0].val_idx[0]
    ctx.table.loc[row, "log_w_prev"] = ctx.table.loc[row, "log_w_k"] + np.log(50.0)   # a fall to 1/50: the line crosses zero
    ctx.table.loc[row, "dt_prev"] = ctx.table.loc[row, "dt_target"]
    result = tuning.evaluate_inner(ctx, {}, plan.inner[0], 0)
    assert result.n_nonpositive >= 1 and result.s == np.inf
    others = [tuning.evaluate_inner(ctx, {}, split, 0) for split in plan.inner[1:]]
    assert all(r.n_nonpositive == 0 and np.isfinite(r.s) for r in others)      # the other splits are untouched
    outcome = tuning.tune_fold(ctx, plan, 3)
    assert outcome.n_nonpositive >= 1 and np.isinf(outcome.best_value)
    assert np.isinf(outcome.inner_scores["S"].iloc[0])
    dev = tuning.tune_fold(make_context(world, "T2", "linear", dev=True), plan, 3)
    assert dev.inner_scores["S"].isna().all()


# ----------------------------------------------------------------------------------------
# Timing summary
# ----------------------------------------------------------------------------------------

def record(number, fits, seconds):
    inner = [tuning.InnerResult(i, 5, 0.0, {}, info, 0, note) for i, (info, note) in enumerate(fits)]
    return tuning.TrialRecord(number, {}, 0.0, inner, seconds)


def test_final_fit_time_of_a_neural_model_is_scaled_from_the_inner_fit_epochs(world):
    ctx, plan = make_context(world, "T1", "lstm"), first_plan(world, "T1")
    fits = [(FitInfo(n_train=10, seconds=1.0, epochs_run=10, best_epoch=4), ""),
            (FitInfo(n_train=12, seconds=2.0, epochs_run=20, best_epoch=6), ""),
            (FitInfo(n_train=14, seconds=3.0, epochs_run=30, best_epoch=8), "")]
    summary = tuning.timing_summary(ctx, plan, [record(0, fits, 7.0), record(1, fits, 9.0)])
    # seconds per epoch and row: 6 s / (60 epochs x 12 rows); median best epoch 6
    expected = 6 * (6.0 / (60 * 12.0)) * len(plan.train_idx)
    assert summary["est_final_fit_seconds"] == pytest.approx(expected)
    assert summary["trial_seconds_mean"] == pytest.approx(8.0) and summary["trial_seconds_max"] == 9.0
    assert summary["best_epoch_median"] == 6 and summary["n_trials"] == 2 and summary["n_inner_fits"] == 6
    assert summary["seconds_per_epoch_mean"] == pytest.approx(0.1)


def test_final_fit_time_of_a_tabular_model_is_scaled_by_the_training_size(world):
    ctx, plan = make_context(world, "T1", "gbm_time"), first_plan(world, "T1")
    fits = [(FitInfo(n_train=10, seconds=1.0), ""), (FitInfo(n_train=10, seconds=3.0), "")]
    summary = tuning.timing_summary(ctx, plan, [record(0, fits, 4.0)])
    assert summary["est_final_fit_seconds"] == pytest.approx(2.0 * len(plan.train_idx) / 10)


def test_path_counts_sum_counters_and_tally_curves_and_ignore_fitted_values():
    counts = tuning.path_counts([
        "curve=gompertz; curve_failures=0", "curve=none; curve_failures=1; skipped=logistic",
        "n_fallback_gc=2; curve=gompertz; curve_failures=0", "alpha=0.5; n_dropped_all_missing=3",
        "log_a=1.5; b=0.25", ""])
    assert counts == {"curve_gompertz": 2, "curve_none": 1, "curve_failures": 1, "n_fallback_gc": 2,
                      "n_dropped_all_missing": 3}


# ----------------------------------------------------------------------------------------
# Final fits: guarded
# ----------------------------------------------------------------------------------------

def test_final_fits_refuse_to_run_when_the_guard_fails(world, monkeypatch):
    calls = use_learner(monkeypatch, constant("T1", 0.1))

    def refuse():
        raise guard.GuardError("the plan is not frozen")

    monkeypatch.setattr(tuning.guard, "assert_test_allowed", refuse)
    with pytest.raises(guard.GuardError, match="not frozen"):
        tuning.final_fits(make_context(world, "T1", "ridge_time"), first_plan(world, "T1"), {"alpha": 1.0}, None, [0])
    assert calls == []                                           # nothing was fitted or predicted


@pytest.mark.parametrize("task,model,hp,epochs", [("T1", "ridge_time_wq", {"alpha": 1.0}, None),
                                                  ("T2", "persistence", {}, None)])
def test_a_tabular_final_fit_predicts_every_test_row_once(world, monkeypatch, task, model, hp, epochs):
    monkeypatch.setattr(tuning.guard, "assert_test_allowed", lambda: ("tag",))
    ctx, plan = make_context(world, task, model), first_plan(world, task)
    fits = tuning.final_fits(ctx, plan, hp, epochs, [0, 1, 2, 3, 4])
    assert [f.seed for f in fits] == [0]                         # deterministic models are fitted once
    frame = fits[0].predictions
    assert list(frame["row_id"]) == list(plan.test_idx) and len(frame) == len(plan.test_idx)
    test = ctx.table.loc[list(plan.test_idx)]
    np.testing.assert_allclose(frame["y_true_log"], target_log(task, test))
    assert set(frame["pond"]) == set(plan.test_ponds)
    assert (frame["fold"] == plan.index).all() and (frame["model"] == model).all()


@pytest.mark.parametrize("task,model", [("T1", "gru"), ("T2", "mlp_static")])
def test_a_neural_final_fit_runs_once_per_seed_for_the_given_number_of_epochs(world, monkeypatch, task, model):
    monkeypatch.setattr(tuning.guard, "assert_test_allowed", lambda: ("tag",))
    ctx, plan = make_context(world, task, model), first_plan(world, task)
    hp = {"hidden": 8, "layers": 1, "dropout": 0.1, "lr": 0.01, "weight_decay": 1e-4, "batch_size": 8}
    if model == "mlp_static":
        hp.pop("layers")
    fits = tuning.final_fits(ctx, plan, hp, 3, [0, 1, 2])
    assert [f.seed for f in fits] == [0, 1, 2]
    assert all(f.info.epochs_run == 3 and len(f.predictions) == len(plan.test_idx) for f in fits)
    assert all(np.isnan(c[2]) for f in fits for c in f.info.curves)           # no validation set in a final fit
    assert not np.allclose(fits[0].predictions["y_pred_log"], fits[1].predictions["y_pred_log"])


def test_prediction_frame_has_the_columns_of_the_plan_for_both_tasks(world):
    for task, date_column in (("T1", "date"), ("T2", "target_date")):
        ctx, plan = make_context(world, task, "persistence" if task == "T2" else "gc"), first_plan(world, task)
        test = ctx.table.loc[list(plan.test_idx)]
        pred = np.zeros(len(test))
        frame = tuning.prediction_frame(ctx, plan, test, pred, np.ones(len(test)), 3)
        assert list(frame.columns) == ["task", "scheme", "model", "sample_set", "fold", "seed", "row_id", "pond",
                                       "stream", "horizon", "date", "origin_date", "t", "w_true", "w_pred",
                                       "y_true_log", "y_pred_log", "w_origin"]
        assert list(frame["date"]) == list(test[date_column]) and (frame["seed"] == 3).all()
        if task == "T2":
            np.testing.assert_allclose(frame["t"], test["t_k"] + test["dt_target"])
            np.testing.assert_allclose(frame["w_true"], test["w_target"])
            assert list(frame["horizon"]) == list(test["horizon"]) and list(frame["origin_date"]) == list(test["origin_date"])
        else:
            np.testing.assert_allclose(frame["w_true"], test["w"])
            assert (frame["horizon"] == 0).all() and frame["w_origin"].isna().all()
