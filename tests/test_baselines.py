"""Unit tests for src/baselines.py on synthetic frames built inside the tests.

Curves are generated without noise so that the fitted parameters and predictions have exact
expected values; extrapolations are checked against hand-computed numbers. None of this is
fish data.
"""

import copy
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import baselines as bl
from src.folds import load_config, select_samples
from tests.synth_p4 import make_inputs

CFG = load_config()
CURVES = CFG["growth_curves"]
GBM_HP = {"learning_rate": 0.1, "max_iter": 50, "max_depth": 3, "min_samples_leaf": 3,
          "l2_regularization": 1e-3}
REPO = Path(__file__).resolve().parents[1]
T_GRID = np.arange(14, 226, 14).astype(float)


def curve_values(name, params, t=T_GRID):
    return bl.CURVE_FUNCTIONS[name](t, *params)


def t2_rows(func, ponds=(3, 4), n_events=7):
    """T2 rows from a noiseless curve: every pond follows the same curve, h = 1 and 2."""
    rows = []
    for pond in ponds:
        t = 14.0 * np.arange(n_events) + 14.0
        w = np.exp(func(t))
        for k in range(1, n_events):
            for h in (1, 2):
                if k + h >= n_events:
                    continue
                rows.append({
                    "pond": pond, "horizon": h,
                    "origin_date": pd.Timestamp("2021-06-19") + pd.Timedelta(days=t[k]),
                    "target_date": pd.Timestamp("2021-06-19") + pd.Timedelta(days=t[k + h]),
                    "t_k": t[k], "dt_prev": t[k] - t[k - 1], "dt_target": t[k + h] - t[k],
                    "log_w_k": np.log(w[k]), "log_w_prev": np.log(w[k - 1]),
                    "w_origin": w[k], "w_target": w[k + h], "y": np.log(w[k + h] / w[k]),
                })
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def inputs():
    return make_inputs(sparse_ponds=(8,), seed=1)


@pytest.fixture(scope="module")
def tables(inputs):
    """T1 and T2 with datetime dates and the supplementary window columns merged in."""
    t1, t2 = inputs.t1.copy(), inputs.t2.copy()
    t1["date"] = pd.to_datetime(t1["date"])
    for col in ("origin_date", "target_date"):
        t2[col] = pd.to_datetime(t2[col])
    wq = inputs.wq.copy()
    wq["date"] = pd.to_datetime(wq["date"])
    for task, table, key in (("T1", t1, "date"), ("T2", t2, "origin_date")):
        for prefix in bl.window_prefixes(task, CFG):
            cols = [c for c in wq.columns if c.startswith(prefix + "_") and not c.endswith("n_covered_window")]
            merged = table[["pond", key]].merge(wq[["pond", "date", *cols]], left_on=["pond", key],
                                                right_on=["pond", "date"], how="left")
            for col in cols:
                table[col] = merged[col].to_numpy()
    return t1, t2


# ----------------------------------------------------------------------------- targets

def test_targets_and_grams_for_both_tasks():
    t1 = pd.DataFrame({"log_w": [np.log(10.0)], "w": [10.0]})
    t2 = pd.DataFrame({"y": [np.log(2.0)], "w_target": [20.0], "w_origin": [10.0]})
    assert bl.target_log("T1", t1)[0] == pytest.approx(np.log(10.0))
    assert bl.target_grams("T1", t1)[0] == 10.0
    assert bl.target_grams("T2", t2)[0] == 20.0
    assert bl.grams_from_log("T1", np.array([np.log(10.0)]), t1)[0] == pytest.approx(10.0)
    assert bl.grams_from_log("T2", np.array([np.log(2.0)]), t2)[0] == pytest.approx(20.0)


# ----------------------------------------------------------------------- growth curves

def test_curve_functions_match_the_formulas_of_plan_section_13():
    """Written out from the plan with `math`, independent of the module's own functions."""
    import math

    t = 50.0
    gompertz = 7.0 - 6.0 * math.exp(-0.03 * t)                       # log A - b exp(-c t)
    vb = 7.2 + 3.0 * math.log(1.0 - math.exp(-0.025 * (t + 12.0)))    # log Winf + 3 log(1 - exp(-k (t - t0)))
    logistic = 6.9 - math.log(1.0 + math.exp(-0.045 * (t - 80.0)))     # log A - log(1 + exp(-k (t - tm)))
    assert bl.CURVE_FUNCTIONS["gompertz"](np.array([t]), 7.0, 6.0, 0.03)[0] == pytest.approx(gompertz)
    assert bl.CURVE_FUNCTIONS["von_bertalanffy"](np.array([t]), 7.2, 0.025, -12.0)[0] == pytest.approx(vb)
    assert bl.CURVE_FUNCTIONS["logistic"](np.array([t]), 6.9, 0.045, 80.0)[0] == pytest.approx(logistic)
    assert set(CURVES["curves"]) == set(bl.CURVE_FUNCTIONS)
    for spec in CURVES["curves"].values():
        assert len(spec["lower"]) == len(spec["upper"]) == bl.N_CURVE_PARAMS
        for start in spec["starts"]:
            assert all(lo < s < hi for lo, s, hi in zip(spec["lower"], start, spec["upper"]))


@pytest.mark.parametrize("name, params", [
    ("gompertz", (7.0, 6.0, 0.03)),
    ("von_bertalanffy", (7.2, 0.025, -12.0)),
    ("logistic", (6.9, 0.045, 80.0)),
])
def test_growth_curve_selection_recovers_the_generating_curve(name, params):
    fit = bl.fit_growth_curve(T_GRID, curve_values(name, params), CURVES)
    assert fit.curve.name == name
    assert fit.curve.rss < 1e-10
    np.testing.assert_allclose(fit.curve.params, params, rtol=1e-3)


def test_aic_is_n_log_rss_over_n_plus_two_per_parameter():
    rng = np.random.default_rng(0)
    y = curve_values("gompertz", (7.0, 6.0, 0.03)) + rng.normal(0, 0.05, len(T_GRID))
    fit = bl.fit_growth_curve(T_GRID, y, CURVES)
    for curve in fit.candidates:
        assert curve.aic == pytest.approx(curve.n * np.log(curve.rss / curve.n) + 6.0)
    assert fit.curve.aic == min(c.aic for c in fit.candidates)


def test_a_curve_that_fails_is_logged_and_skipped(caplog):
    cfg = copy.deepcopy(CURVES)
    cfg["curves"]["logistic"]["starts"] = [[7.0, 5.0, 100.0]]  # k = 5 is above its upper bound 1
    y = curve_values("gompertz", (7.0, 6.0, 0.03))
    with caplog.at_level(logging.WARNING, logger="phase4"):
        fit = bl.fit_growth_curve(T_GRID, y, cfg)
    assert {c.name for c in fit.candidates} == {"gompertz", "von_bertalanffy"}
    assert any("logistic" in message for message in caplog.messages)
    assert any("logistic" in failure for failure in fit.failures)


def test_all_curves_failing_is_an_error():
    cfg = copy.deepcopy(CURVES)
    for spec in cfg["curves"].values():
        spec["starts"] = [[1e9, 1e9, 1e9]]
    with pytest.raises(bl.GrowthFitError, match="all growth curves failed"):
        bl.fit_growth_curve(T_GRID, curve_values("gompertz", (7.0, 6.0, 0.03)), cfg)


def test_too_few_events_for_three_parameters_is_a_fit_error():
    with pytest.raises(bl.GrowthFitError, match="more than 3 events"):
        bl.fit_growth_curve([14.0, 28.0, 42.0], [1.0, 2.0, 3.0], CURVES)


def test_von_bertalanffy_t0_is_bounded_below_zero_in_the_config():
    vb = CURVES["curves"]["von_bertalanffy"]
    assert vb["params"][2] == "t0" and vb["upper"][2] < 0
    assert all(start[2] < 0 for start in vb["starts"])


def test_gc_predicts_the_fitted_curve_and_reports_it():
    y = curve_values("gompertz", (7.0, 6.0, 0.03))
    train = pd.DataFrame({"t": T_GRID, "log_w": y})
    model = bl.make_baseline("T1", "gc", CFG)
    info = model.fit(train)
    assert info.n_train == len(train) and info.seconds >= 0
    np.testing.assert_allclose(model.predict_log(train), y, atol=1e-5)
    assert "curve=gompertz" in model.notes() and "curve_failures=0" in model.notes()
    np.testing.assert_allclose(model.predict_grams(train), np.exp(y), rtol=1e-5)


def failing_curves():
    cfg = copy.deepcopy(CURVES)
    for spec in cfg["curves"].values():
        spec["starts"] = [[1e9, 1e9, 1e9]]  # outside the bounds: every fit fails
    return cfg


def test_gc_uses_cohort_mean_when_all_three_curves_fail_and_counts_the_failure():
    y = curve_values("gompertz", (7.0, 6.0, 0.03))
    train = pd.DataFrame({"pond": np.repeat([3, 4, 6], len(T_GRID)), "t": np.tile(T_GRID, 3),
                          "log_w": np.tile(y, 3) + np.repeat([0.0, 0.1, 0.2], len(T_GRID))})
    model = bl.GrowthCurveModel(failing_curves())
    model.fit(train)
    assert "curve_failures=1" in model.notes()
    probe = pd.DataFrame({"t": [28.0, 42.0]})
    np.testing.assert_allclose(model.predict_log(probe), [y[1] + 0.1, y[2] + 0.1])
    with pytest.raises(RuntimeError, match="no growth curve"):  # a day without 3 ponds: no prediction
        model.predict_log(pd.DataFrame({"t": [100.0]}))


# ------------------------------------------------------------------------- cohort_mean

def cohort_frame(n_ponds=3):
    y = curve_values("gompertz", (7.0, 6.0, 0.03))
    ponds = np.repeat(np.arange(n_ponds), len(T_GRID))
    shift = np.repeat(0.1 * np.arange(n_ponds), len(T_GRID))
    return y, pd.DataFrame({"pond": ponds, "t": np.tile(T_GRID, n_ponds), "log_w": np.tile(y, n_ponds) + shift})


def test_cohort_mean_uses_the_same_day_and_falls_back_to_the_growth_curve():
    y, train = cohort_frame(3)
    model = bl.make_baseline("T1", "cohort_mean", CFG)
    model.fit(train)
    test = pd.DataFrame({"t": [28.0, 100.0]})  # 28 is a training day, 100 is not
    pred = model.predict_log(test)
    assert pred[0] == pytest.approx(y[1] + 0.1)  # mean of the three ponds' log W on day 28
    assert "n_fallback_gc=1" in model.notes()
    gc = bl.make_baseline("T1", "gc", CFG)
    gc.fit(train)
    assert pred[1] == pytest.approx(gc.predict_log(test)[1])


def test_cohort_mean_needs_three_training_ponds_on_the_date():
    y, train = cohort_frame(3)
    train = train[~((train["pond"] == 2) & (train["t"] == 28.0))]  # day 28 now has two ponds
    model = bl.make_baseline("T1", "cohort_mean", CFG)
    model.fit(train)
    gc = bl.make_baseline("T1", "gc", CFG)
    gc.fit(train)
    probe = pd.DataFrame({"t": [28.0, 42.0]})
    pred = model.predict_log(probe)
    assert pred[0] == pytest.approx(gc.predict_log(probe)[0])  # two ponds: the growth curve
    assert pred[1] == pytest.approx(y[2] + 0.1)  # three ponds: the mean
    assert "n_fallback_gc=1" in model.notes()


# --------------------------------------------------------------------------- allometric

def test_allometric_recovers_the_log_log_line():
    log_l = np.log(np.array([8.0, 12.0, 16.0, 22.0, 30.0]))
    train = pd.DataFrame({"log_l": log_l, "log_w": 1.2 + 2.9 * log_l})
    model = bl.make_baseline("T1", "allometric", CFG)
    model.fit(train)
    test = pd.DataFrame({"log_l": [np.log(18.0)]})
    assert model.predict_log(test)[0] == pytest.approx(1.2 + 2.9 * np.log(18.0))
    assert "b=2.9" in model.notes()


# ------------------------------------------------------------------- T2 time-only models

def test_persistence_predicts_no_change():
    df = t2_rows(lambda t: curve_values("gompertz", (7.0, 6.0, 0.03), t))
    model = bl.make_baseline("T2", "persistence", CFG)
    model.fit(df)
    np.testing.assert_array_equal(model.predict_log(df), 0.0)
    np.testing.assert_allclose(model.predict_grams(df), df["w_origin"])


def test_sgr_extrapolates_the_log_line_exactly_for_exponential_growth():
    df = t2_rows(lambda t: 2.0 + 0.02 * t)  # exponential growth: log W is a straight line
    model = bl.make_baseline("T2", "sgr", CFG)
    model.fit(df)
    np.testing.assert_allclose(model.predict_log(df), df["y"], atol=1e-12)


def test_linear_extrapolates_in_grams_and_counts_non_positive_predictions():
    df = pd.DataFrame({
        "w_origin": [10.0, 5.0, 10.0], "log_w_prev": np.log([8.0, 50.0, 10.0]),
        "dt_prev": [14.0, 14.0, 14.0], "dt_target": [28.0, 28.0, 14.0],
        "log_w_k": np.log([10.0, 5.0, 10.0]),
    })
    model = bl.make_baseline("T2", "linear", CFG)
    model.fit(df)
    grams = model.predict_grams(df)
    # 10 + (10 - 8) / 14 * 28 = 14; 5 + (5 - 50) / 14 * 28 = -85; flat series stays at 10
    np.testing.assert_allclose(grams, [14.0, -85.0, 10.0])
    log_pred = model.predict_log(df)
    assert log_pred[0] == pytest.approx(np.log(14.0 / 10.0))
    assert np.isnan(log_pred[1])  # not positive: the log is undefined, no floor is applied
    assert log_pred[2] == pytest.approx(0.0)
    assert model.notes() == "n_nonpositive_pred=1"


def test_anchored_gc_events_are_deduplicated_and_prediction_is_exact_on_the_curve():
    func = lambda t: curve_values("gompertz", (7.0, 6.0, 0.03), t)
    df = t2_rows(func, ponds=(3, 4), n_events=7)
    events = bl.anchor_events(df)
    assert len(events) == 12  # events 1..6 of two ponds; each is an origin or a target once
    assert not events.duplicated(["pond", "date"]).any()
    model = bl.make_baseline("T2", "anchored_gc", CFG)
    model.fit(df)
    np.testing.assert_allclose(model.predict_log(df), df["y"], atol=1e-5)
    assert "curve=gompertz" in model.notes()


def test_anchor_events_reject_two_weights_for_one_pond_and_date():
    df = t2_rows(lambda t: 2.0 + 0.02 * t, ponds=(3,))
    broken = df.copy()
    broken.loc[broken.index[0], "w_target"] *= 1.5  # contradicts the same event as origin elsewhere
    with pytest.raises(ValueError, match="two different weights"):
        bl.anchor_events(broken)


def test_cohort_growth_matches_origin_and_target_date_and_falls_back():
    func = lambda t: curve_values("gompertz", (7.0, 6.0, 0.03), t)
    train = t2_rows(func, ponds=(3, 4, 6), n_events=7)
    train.loc[train["pond"] == 4, "y"] += 0.3  # pond 4 grows faster on every pair of dates
    model = bl.make_baseline("T2", "cohort_growth", CFG)
    model.fit(train)
    probe = train[train["pond"] == 3].iloc[[0, 1]].copy()
    expected = train[train["pond"] == 3]["y"].to_numpy()[:2] + 0.1  # mean of the three ponds
    np.testing.assert_allclose(model.predict_log(probe), expected)
    assert "n_fallback_anchored_gc=0" in model.notes()
    odd = probe.copy()
    odd["target_date"] = odd["target_date"] + pd.Timedelta(days=7)  # no training pair has it
    odd["dt_target"] = odd["dt_target"] + 7
    pred = model.predict_log(odd)
    assert "n_fallback_anchored_gc=2" in model.notes()
    fallback = bl.make_baseline("T2", "anchored_gc", CFG)
    fallback.fit(train)
    np.testing.assert_allclose(pred, fallback.predict_log(odd))


def test_anchored_gc_uses_sgr_when_all_three_curves_fail_and_counts_the_failure():
    df = t2_rows(lambda t: 2.0 + 0.02 * t, ponds=(3, 4))
    model = bl.AnchoredGrowthCurve(failing_curves())
    model.fit(df)
    assert "curve_failures=1" in model.notes()
    sgr = bl.make_baseline("T2", "sgr", CFG)
    np.testing.assert_allclose(model.predict_log(df), sgr.predict_log(df))


def test_cohort_growth_needs_three_training_ponds_with_the_date_pair():
    func = lambda t: curve_values("gompertz", (7.0, 6.0, 0.03), t)
    train = t2_rows(func, ponds=(3, 4, 6), n_events=7)
    train.loc[train["pond"] == 4, "y"] += 0.3
    pair = train[(train["pond"] == 3)].iloc[0]
    # remove pond 6 from the first pair of dates: two ponds are left on it
    drop = (train["pond"] == 6) & (train["origin_date"] == pair["origin_date"]) \
        & (train["target_date"] == pair["target_date"])
    train = train[~drop]
    model = bl.make_baseline("T2", "cohort_growth", CFG)
    model.fit(train)
    anchored = bl.make_baseline("T2", "anchored_gc", CFG)
    anchored.fit(train)
    probe = train[train["pond"] == 3].iloc[[0, 1]]
    pred = model.predict_log(probe)
    assert pred[0] == pytest.approx(anchored.predict_log(probe)[0])  # two ponds: fallback
    three = train[(train["origin_date"] == probe.iloc[1]["origin_date"])
                  & (train["target_date"] == probe.iloc[1]["target_date"])]
    assert three["pond"].nunique() == 3 and pred[1] == pytest.approx(three["y"].mean())
    assert "n_fallback_anchored_gc=1" in model.notes()


# ----------------------------------------------------------------------------- features

def test_feature_columns_per_model(tables):
    t1, t2 = tables
    assert bl.feature_columns("T1", "ridge_time", CFG) == ["t", "t2"]
    assert len(bl.feature_columns("T1", "ridge_wq", CFG)) == 30
    assert len(bl.feature_columns("T1", "gbm_time_wq", CFG)) == 32
    assert bl.feature_columns("T1", "gbm_time_wq_len", CFG)[-1] == "log_l"
    assert bl.feature_columns("T2", "ridge_pw", CFG) == ["log_w_k", "log_w_prev", "t_k", "dt_prev", "dt_target"]
    assert len(bl.feature_columns("T2", "gbm_pw_wq", CFG)) == 5 + 24
    for model in CFG["model_features"]["T1"]:
        assert set(bl.feature_columns("T1", model, CFG)) <= set(t1.columns)
    for model in CFG["model_features"]["T2"]:
        assert set(bl.feature_columns("T2", model, CFG)) <= set(t2.columns)


def test_window_variants_use_the_prefixed_window_and_the_unprefixed_cumulative_features():
    main = bl.feature_columns("T1", "ridge_time_wq", CFG)
    w07 = bl.feature_columns("T1", "ridge_time_wq_w07", CFG)
    assert len(w07) == len(main) == 32
    assert w07[:2] == ["t", "t2"]
    assert w07[2] == "t1w07_temperature_mean_w" and w07[25] == "t1w07_nitrate_max_w"
    assert w07[26:] == [f"{s}_mean_cum" for s in CFG["design"]["sensors"]]  # not windowed
    assert bl.feature_columns("T2", "gbm_pw_wq_w42", CFG)[5] == "t2w42_temperature_mean_w"
    assert bl.window_prefixes("T1", CFG) == ["t1w07", "t1w28"]
    assert bl.window_prefixes("T2", CFG) == ["t2w14", "t2w42"]


def test_indicator_columns_follow_the_water_quality_prefix_and_the_noind_switch():
    sensors = CFG["design"]["sensors"]
    assert bl.indicator_columns("T1", "ridge_time", CFG) == []
    assert bl.indicator_columns("T1", "ridge_time_wq", CFG) == [f"{s}_mean_w" for s in sensors]
    assert bl.indicator_columns("T1", "ridge_time_wq_noind", CFG) == []
    assert bl.indicator_columns("T2", "ridge_pw_wq_noind", CFG) == []
    assert bl.indicator_columns("T1", "ridge_time_wq_w28", CFG) == [f"t1w28_{s}_mean_w" for s in sensors]
    assert bl.feature_columns("T1", "ridge_time_wq_noind", CFG) == bl.feature_columns("T1", "ridge_time_wq", CFG)


@pytest.mark.skipif(not (REPO / "data/processed/wq_by_date.csv").exists(), reason="tables not built")
def test_feature_columns_exist_in_the_real_table_headers():
    """Header check only: reads the first row of each file, no values are used."""
    wq_header = set(pd.read_csv(REPO / "data/processed/wq_by_date.csv", nrows=0).columns)
    for task, name in (("T1", "t1_table"), ("T2", "t2_table")):
        header = set(pd.read_csv(REPO / "data/processed" / f"{name}.csv", nrows=0).columns)
        for model in CFG["model_features"][task]:
            columns = bl.feature_columns(task, model, CFG)
            prefix = bl.model_spec(task, model, CFG)["wq_prefix"]
            # window columns of a supplementary length live in wq_by_date.csv, merged by src/data.py
            windowed = {c for c in columns if prefix and c.startswith(prefix + "_")}
            assert windowed <= wq_header, model
            assert set(columns) - windowed <= header, model
            assert set(bl.indicator_columns(task, model, CFG)) <= set(columns), model


# ------------------------------------------------------------------------------- ridge

def ridge_frame(rows=40, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"a": rng.normal(5, 2, rows), "b": rng.normal(100, 30, rows)})
    df["log_w"] = 1.0 + 0.5 * df["a"] + 0.01 * df["b"]
    return df


def test_ridge_imputes_with_the_training_median_and_adds_an_indicator():
    train = ridge_frame()
    model = bl.RidgeModel("T1", ["a", "b"], ["a"], alpha=1e-9)
    model.fit(train)
    test = pd.DataFrame({"a": [np.nan, 4.0], "b": [120.0, 80.0]})
    design = model._design(test)
    median_a = float(np.median(train["a"]))
    np.testing.assert_allclose(design[:, 0], [median_a, 4.0])
    np.testing.assert_array_equal(design[:, 2], [1.0, 0.0])  # indicator: sensor a has no value
    expected = 1.0 + 0.5 * median_a + 0.01 * 120.0
    assert model.predict_log(test)[0] == pytest.approx(expected, abs=1e-6)
    assert model.predict_log(test)[1] == pytest.approx(1.0 + 0.5 * 4.0 + 0.01 * 80.0, abs=1e-6)


def test_ridge_statistics_come_from_the_training_fold_only():
    train = ridge_frame()
    model = bl.RidgeModel("T1", ["a", "b"], [], alpha=1.0)
    model.fit(train)
    reference = model.predict_log(train.iloc[:3])
    extreme = pd.concat([train.iloc[:3], pd.DataFrame({"a": [1e6], "b": [1e6]})], ignore_index=True)
    assert model.predict_log(extreme)[:3] == pytest.approx(reference)  # a test row cannot shift the scaler
    np.testing.assert_allclose(model.mean_, train[["a", "b"]].mean().to_numpy())
    np.testing.assert_allclose(model.scale_, train[["a", "b"]].std(ddof=0).to_numpy())


def test_ridge_large_alpha_shrinks_to_the_training_mean():
    train = ridge_frame()
    model = bl.RidgeModel("T1", ["a", "b"], [], alpha=1e12)
    model.fit(train)
    np.testing.assert_allclose(model.predict_log(train), train["log_w"].mean(), atol=1e-6)


def test_ridge_drops_a_column_that_is_missing_in_every_training_row():
    train = ridge_frame()
    train["c"] = np.nan
    model = bl.RidgeModel("T1", ["a", "b", "c"], ["c"], alpha=1e-9)
    model.fit(train)
    assert model.active_ == ["a", "b"] and "n_dropped_all_missing=1" in model.notes()
    # the dropped column cannot influence a prediction, even when a test row has a value
    test = pd.DataFrame({"a": [5.0, 5.0], "b": [100.0, 100.0], "c": [np.nan, 9e9]})
    pred = model.predict_log(test)
    assert np.isfinite(pred).all()
    assert pred[0] == pytest.approx(1.0 + 0.5 * 5.0 + 0.01 * 100.0, abs=1e-6)
    assert design_width(model, test) == 2 + 1  # a, b and the indicator of c


def design_width(model, frame):
    return model._design(frame).shape[1]


def test_ridge_with_no_usable_column_predicts_the_training_mean():
    train = ridge_frame()
    train["a"] = np.nan
    train["b"] = np.nan
    model = bl.RidgeModel("T1", ["a", "b"], [], alpha=1.0)
    model.fit(train)
    np.testing.assert_allclose(model.predict_log(train), train["log_w"].mean())


def test_ridge_without_indicators_has_one_column_per_feature():
    train = ridge_frame()
    train.loc[train.index[::4], "a"] = np.nan
    with_flags = bl.RidgeModel("T1", ["a", "b"], ["a"], alpha=1.0)
    without = bl.RidgeModel("T1", ["a", "b"], [], alpha=1.0)
    with_flags.fit(train)
    without.fit(train)
    assert design_width(with_flags, train) == 3 and design_width(without, train) == 2


# --------------------------------------------------------------------------------- gbm

def test_gbm_accepts_missing_values_and_is_deterministic():
    train = ridge_frame(60)
    train.loc[train.index[::5], "a"] = np.nan
    first = bl.GBMModel("T1", ["a", "b"], GBM_HP, random_state=0)
    second = bl.GBMModel("T1", ["a", "b"], GBM_HP, random_state=0)
    first.fit(train)
    second.fit(train)
    test = pd.DataFrame({"a": [np.nan, 5.0], "b": [100.0, 90.0]})
    np.testing.assert_array_equal(first.predict_log(test), second.predict_log(test))
    assert np.isfinite(first.predict_log(test)).all()
    assert first.model_.get_params()["early_stopping"] is False
    assert first.model_.get_params()["max_iter"] == 50


def test_gbm_drops_a_column_that_is_missing_in_every_training_row():
    train = ridge_frame(60)
    train["c"] = np.nan
    model = bl.GBMModel("T1", ["a", "b", "c"], GBM_HP, random_state=0)
    model.fit(train)
    assert model.active_ == ["a", "b"] and "n_dropped_all_missing=1" in model.notes()
    base = bl.GBMModel("T1", ["a", "b"], GBM_HP, random_state=0)
    base.fit(train)
    test = pd.DataFrame({"a": [5.0, 6.0], "b": [100.0, 90.0], "c": [3.0, np.nan]})
    np.testing.assert_array_equal(model.predict_log(test), base.predict_log(test))


def test_gbm_with_no_usable_column_predicts_the_training_mean():
    train = ridge_frame(30)
    train[["a", "b"]] = np.nan
    model = bl.GBMModel("T1", ["a", "b"], GBM_HP, random_state=0)
    model.fit(train)
    np.testing.assert_allclose(model.predict_log(train), train["log_w"].mean())


# ------------------------------------------------------------------------------ registry

def test_registry_builds_every_non_deep_model_of_the_config():
    for task in ("T1", "T2"):
        group = CFG["models"][task]
        for model in group["time_only"] + group["water_quality"] + group["reference"] + group["sensitivity"]:
            kind = bl.baseline_kind(task, model, CFG)
            hp = {"alpha": 1.0} if kind == "ridge" else GBM_HP if kind == "gbm" else None
            assert isinstance(bl.make_baseline(task, model, CFG, hp), bl.Learner), (task, model)
    with pytest.raises(KeyError):
        bl.baseline_kind("T1", "no_such_model", CFG)


def test_every_non_deep_model_fits_and_predicts_on_a_held_out_pond(tables):
    t1, t2 = tables
    for task, table in (("T1", t1), ("T2", t2)):
        train = select_samples(table, [1, 2, 4, 6, 7, 9], "eligible")
        test = select_samples(table, [3], "eligible")
        assert len(test) > 0
        group = CFG["models"][task]
        for model in group["time_only"] + group["water_quality"] + group["reference"] + group["sensitivity"]:
            kind = bl.baseline_kind(task, model, CFG)
            hp = {"alpha": 1.0} if kind == "ridge" else GBM_HP if kind == "gbm" else None
            learner = bl.make_baseline(task, model, CFG, hp)
            learner.fit(train)
            grams = learner.predict_grams(test)
            assert grams.shape == (len(test),) and np.isfinite(grams).all(), (task, model)
            assert isinstance(learner.notes(), str)


def test_models_with_and_without_water_quality_get_the_same_training_rows(tables):
    """Plan 4.3: the sample rule does not depend on the feature set."""
    t1, _ = tables
    train = select_samples(t1, [1, 2, 4], "eligible")
    sizes = set()
    for model in ("ridge_time", "ridge_wq", "ridge_time_wq_len"):
        learner = bl.make_baseline("T1", model, CFG, {"alpha": 1.0})
        sizes.add(learner.fit(train).n_train)
    assert sizes == {len(train)}
