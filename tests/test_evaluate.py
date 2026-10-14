"""Tests for src/evaluate.py: the statistics, hand-computed on small cases, and the whole scorer
on synthetic prediction files whose per-pond errors are set by the tests.

Nothing here is a measurement. The prediction files are written to a temporary results
directory; the errors of every model are chosen so that S, the differences and the exact
p-values are known in advance (or enumerated independently in the test).
"""

import copy
import itertools
import json
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy import stats as scipy_stats

from src import evaluate as ev
from src import guard, scoring
from src.artifacts import Layout, write_json_atomic
from src.folds import load_config

CFG = load_config()


# ----------------------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ----------------------------------------------------------------------------------------

def brute_force(d):
    """(p_lower, p_upper, p_two) from a list of every sign pattern; midranks of |d|, zeros dropped."""
    d = [round(float(x), 12) for x in d if round(float(x), 12) != 0]
    ranks = scipy_stats.rankdata(np.abs(d), method="average")
    observed = sum(r for r, x in zip(ranks, d) if x > 0)
    le = ge = 0
    for signs in itertools.product([0, 1], repeat=len(d)):
        t = sum(r for r, s in zip(ranks, signs) if s)
        le += t <= observed + 1e-9
        ge += t >= observed - 1e-9
    p_lower, p_upper = le / 2 ** len(d), ge / 2 ** len(d)
    return p_lower, p_upper, min(1.0, 2 * min(p_lower, p_upper))


@pytest.mark.parametrize("seed,n", [(0, 5), (1, 6), (2, 7), (3, 8)])
def test_exact_wilcoxon_matches_scipy_when_there_are_no_ties_or_zeros(seed, n):
    d = np.random.default_rng(seed).normal(size=n)
    result = ev.wilcoxon_exact(d)
    for alternative, got in (("less", result.p_lower), ("greater", result.p_upper), ("two-sided", result.p_two_sided)):
        assert got == pytest.approx(scipy_stats.wilcoxon(d, alternative=alternative, method="exact").pvalue)
    assert result.n == n and result.n_zero == 0 and result.t_plus + result.t_minus == n * (n + 1) / 2


def test_exact_wilcoxon_with_ties_and_zeros_uses_midranks_and_all_sign_flips():
    d = [0.5, -0.5, 1.0, -2.0, 0.0, 0.0, 3.0, 0.5, -1.0]
    result = ev.wilcoxon_exact(d)
    assert (result.n, result.n_zero) == (7, 2)
    ranks = scipy_stats.rankdata(np.abs([0.5, -0.5, 1.0, -2.0, 3.0, 0.5, -1.0]))      # 0.5 three times, 1.0 twice
    assert sorted(ranks) == [2.0, 2.0, 2.0, 4.5, 4.5, 6.0, 7.0]
    assert result.t_plus == pytest.approx(2 + 4.5 + 7 + 2) and result.t_minus == pytest.approx(2 + 6 + 4.5)
    assert (result.p_lower, result.p_upper, result.p_two_sided) == pytest.approx(brute_force(d))


def test_differences_that_agree_to_twelve_decimals_are_ties_and_zeros():
    d = np.array([0.30 - 0.28, 0.27 - 0.25, 0.25 - 0.20, -(0.45 - 0.40), 1e-14, -1e-13])
    result = ev.wilcoxon_exact(d)
    assert result.n_zero == 2 and result.n == 4
    assert (result.p_lower, result.p_upper, result.p_two_sided) == pytest.approx(brute_force(d))


def positives_to_d(positive, n):
    return [(r if r in positive else -r) for r in range(1, n + 1)]


def test_seven_ponds_reject_only_when_the_ranks_of_the_worse_ponds_sum_to_at_most_three():
    assert ev.wilcoxon_exact(positives_to_d({1, 2}, 7)).p_lower == pytest.approx(5 / 128)      # T+ = 3, 5 of 128 patterns
    assert ev.wilcoxon_exact(positives_to_d({3}, 7)).p_lower == pytest.approx(5 / 128)
    assert ev.wilcoxon_exact(positives_to_d({4}, 7)).p_lower == pytest.approx(7 / 128)         # T+ = 4: 0.0547 > 0.05
    assert 5 / 128 <= 0.05 < 7 / 128
    assert ev.wilcoxon_exact(positives_to_d(set(), 7)).p_lower == pytest.approx(1 / 128)


def test_six_ponds_reject_only_when_the_ranks_of_the_worse_ponds_sum_to_at_most_two():
    assert ev.wilcoxon_exact(positives_to_d({2}, 6)).p_lower == pytest.approx(3 / 64)          # T+ = 2
    assert ev.wilcoxon_exact(positives_to_d({3}, 6)).p_lower == pytest.approx(5 / 64)          # T+ = 3: 0.078
    assert 3 / 64 <= 0.05 < 5 / 64


def test_wilcoxon_edge_cases():
    none = ev.wilcoxon_exact([0.0, 1e-15])
    assert (none.n, none.n_zero, none.p_lower, none.p_two_sided) == (0, 2, 1.0, 1.0)
    one = ev.wilcoxon_exact([-1.0])
    assert one.p_lower == 0.5 and one.p_two_sided == 1.0
    with pytest.raises(ValueError, match="NaN"):
        ev.wilcoxon_exact([1.0, np.nan])
    infinite = ev.wilcoxon_exact([np.inf, -1.0, -2.0])                    # an infinite difference ranks last
    assert infinite.t_plus == 3.0 and infinite.t_minus == 3.0


def test_a_pond_where_both_models_are_infinite_is_a_zero_difference():
    d = ev.pond_differences([np.inf, np.inf, 0.4, 0.2], [np.inf, 0.3, 0.1, np.inf])
    assert d[0] == 0 and d[1] == np.inf and d[2] == pytest.approx(0.3) and d[3] == -np.inf


def test_merging_ponds_into_units_averages_the_scored_members():
    values = np.array([0.2, 0.4, 0.6, 0.8])
    np.testing.assert_allclose(ev.merge_units([1, 2, 3, 4], values, [{1, 2}]), [0.3, 0.6, 0.8])
    np.testing.assert_allclose(ev.merge_units([2, 3, 4], values[1:], [{1, 2}]), [0.4, 0.6, 0.8])   # pond 1 not scored
    np.testing.assert_allclose(ev.merge_units([3, 4], values[2:], [{1, 2}]), [0.6, 0.8])


def test_percentile_intervals_interpolate_linearly_and_keep_infinities():
    values = np.arange(101.0)
    assert ev.percentile_interval(values, 0.95) == pytest.approx((2.5, 97.5))
    low, high = ev.percentile_interval(np.array([1.0, 2.0, 3.0, np.inf]), 0.5)
    assert low == pytest.approx(1.75) and high == np.inf
    assert ev.percentile_interval([np.inf] * 10, 0.95) == (np.inf, np.inf)
    assert all(np.isnan(v) for v in ev.percentile_interval([1.0, np.nan], 0.95))
    assert ev.percentile_interval([4.0], 0.95) == (4.0, 4.0)


def test_pooled_metrics_by_hand():
    metrics = ev.pooled_metrics([100.0, 200.0, 300.0], [110.0, 180.0, 330.0])
    assert metrics["mae_g"] == pytest.approx(20.0)
    assert metrics["rmse_g"] == pytest.approx(np.sqrt(1400 / 3))
    assert metrics["mape_pct"] == pytest.approx(10.0)
    assert metrics["r2"] == pytest.approx(1 - 1400 / 20000)
    assert np.isnan(ev.pooled_metrics([5.0, 5.0], [5.0, 6.0])["r2"])        # constant truth: R2 undefined


def test_bootstrap_indices_replay_the_seeded_generator():
    first = ev.bootstrap_indices(np.random.default_rng(20261014), 5, 7)
    again = ev.bootstrap_indices(np.random.default_rng(20261014), 5, 7)
    assert first.shape == (5, 7) and (first == again).all() and first.min() >= 0 and first.max() <= 6
    assert not (first == ev.bootstrap_indices(np.random.default_rng(1), 5, 7)).all()


def test_permutation_p_counts_runs_at_least_as_good_as_the_observed_one():
    assert scoring.permutation_p(0.3, [0.4, 0.5, 0.3, 0.2]) == pytest.approx((1 + 2) / 5)     # 0.3 and 0.2 count
    assert scoring.permutation_p(0.1, [0.4] * 999) == pytest.approx(1 / 1000)
    assert scoring.permutation_p(0.9, [0.4] * 199) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        scoring.permutation_p(0.3, [0.4, np.nan])


# ----------------------------------------------------------------------------------------
# Synthetic prediction files
# ----------------------------------------------------------------------------------------

PONDS = [1, 2, 3, 4, 6, 7, 9]
OFFSETS = [-0.02, -0.01, 0.0, 0.01, 0.02]                    # added to a deep model's error, per seed; they average to 0
T1_BASE = {1: 0.30, 2: 0.20, 3: 0.40, 4: 0.25, 6: 0.35, 7: 0.50, 9: 0.45}
T1_GRU = {1: 0.25, 2: 0.22, 3: 0.30, 4: 0.18, 6: 0.32, 7: 0.30, 9: 0.37}      # worse than the baseline only in pond 2
T2_SGR = {(p, 1): s for p, s in zip(PONDS, [.30, .25, .35, .40, .20, .45, .30])}
T2_SGR.update({(p, 2): s for p, s in zip(PONDS, [.50, .40, .60, .50, .45, .55, .60])})
T2_LSTM = {(p, 1): s for p, s in zip(PONDS, [.28, .27, .30, .45, .22, .40, .34])}   # ties in |d|: 0.02 x3, 0.05 x3, 0.04
T2_LSTM.update({(p, 2): s for p, s in zip(PONDS, [.45, .42, .50, .40, .40, .50, .55])})


T1_COUNTS = [2, 3, 4, 2, 3, 4, 2]                            # samples per pond: unequal, so that a pooled mean
T2_COUNTS = {1: [1, 2, 1, 2, 1, 2, 1], 2: [2, 1, 2, 1, 2, 1, 2]}    # differs from the mean of the pond means


def sample_frame(task, ponds=PONDS, start=0):
    """Test samples shared by the models of a group, with a different number of samples per pond."""
    rows, rid = [], start
    for fold, pond in enumerate(ponds):
        i = fold % 7
        for h in ([0] * T1_COUNTS[i] if task == "T1" else [1] * T2_COUNTS[1][i] + [2] * T2_COUNTS[2][i]):
            rows.append({"row_id": rid, "fold": fold, "pond": pond, "horizon": h, "w_true": 60.0 + 7.0 * rid,
                         "y": 0.20 + 0.05 * (rid % 3)})
            rid += 1
    return pd.DataFrame(rows)


def write_model(layout, task, scheme, model, errors, *, frame=None, sample_set="eligible", seeds=(0,),
                nonpositive=(), drop_rows=()):
    """Write one prediction file. `errors` maps pond (or (pond, horizon)) to the absolute log error.

    The error of seed i is the error plus OFFSETS[i]; the sign alternates with the row id. Rows in
    `nonpositive` get a NaN log prediction and a weight of -1 g.
    """
    frame = sample_frame(task) if frame is None else frame
    rows = []
    for i, seed in enumerate(seeds):
        for r in frame.itertuples():
            if r.row_id in drop_rows:
                continue
            err = errors[(r.pond, r.horizon)] if (r.pond, r.horizon) in errors else errors[r.pond]
            err = err + (OFFSETS[i] if len(seeds) > 1 else 0.0)
            sign = 1.0 if r.row_id % 2 == 0 else -1.0
            if task == "T1":
                true_log, origin = float(np.log(r.w_true)), np.nan
            else:
                true_log, origin = r.y, r.w_true / np.exp(r.y)
            pred = true_log + sign * err
            grams = float(np.exp(pred)) if task == "T1" else float(origin * np.exp(pred))
            if r.row_id in nonpositive:
                pred, grams = np.nan, -1.0
            rows.append({"task": task, "scheme": scheme, "model": model, "sample_set": sample_set, "fold": r.fold,
                         "seed": seed, "row_id": r.row_id, "pond": r.pond, "stream": f"S{r.pond:02d}",
                         "horizon": r.horizon, "date": "2021-08-14", "origin_date": "" if task == "T1" else "2021-07-31",
                         "t": 70, "w_true": r.w_true, "w_pred": grams, "y_true_log": true_log, "y_pred_log": pred,
                         "w_origin": origin})
    path = layout.predictions(task, scheme, model if sample_set == "eligible" else f"{model}__{sample_set}")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def shifted(base, shift):
    return {k: v + shift for k, v in base.items()}


def write_selection(layout, task, baseline, deep):
    write_json_atomic(layout.selection(task), {"task": task, "primary_baseline": baseline, "primary_deep": deep})


def build_world(tmp_path, t1_deep=None):
    cfg = copy.deepcopy(CFG)
    cfg["paths"]["results"] = str(tmp_path / "results")
    layout = Layout(cfg)
    t1_deep = T1_GRU if t1_deep is None else t1_deep
    t1 = {"ridge_time": T1_BASE, "gc": shifted(T1_BASE, 0.10), "ridge_time_wq": shifted(T1_BASE, -0.01),
          "ridge_time_wq_w07": shifted(T1_BASE, -0.005), "ridge_time_wq_noind": shifted(T1_BASE, 0.01)}
    for model, errors in t1.items():
        write_model(layout, "T1", "lopo", model, errors)
    write_model(layout, "T1", "lopo", "gru", t1_deep, seeds=range(5))
    write_model(layout, "T1", "lopo", "mlp_static", shifted(T1_BASE, 0.02), seeds=range(5))
    chrono = sample_frame("T1", [3, 4, 6, 7, 9], start=500)                   # five ponds in the hold-out
    write_model(layout, "T1", "chrono", "ridge_time", T1_BASE, frame=chrono)
    write_model(layout, "T1", "chrono", "gru", T1_GRU, frame=chrono, seeds=range(5))
    everything = sample_frame("T1", start=1000)
    for model in ("gc", "ridge_time"):
        write_model(layout, "T1", "lopo", model, shifted(T1_BASE, 0.1 if model == "gc" else 0.0), frame=everything,
                    sample_set="all")
    t2_frame = sample_frame("T2")
    bad = set(t2_frame[(t2_frame["pond"] == 3) & (t2_frame["horizon"] == 1)]["row_id"])
    write_model(layout, "T2", "lopo", "sgr", T2_SGR)
    write_model(layout, "T2", "lopo", "persistence", {k: v + 0.2 for k, v in T2_SGR.items()})
    write_model(layout, "T2", "lopo", "linear", {k: v + 0.1 for k, v in T2_SGR.items()}, nonpositive=bad)
    write_model(layout, "T2", "lopo", "lstm", T2_LSTM, seeds=range(5))
    write_model(layout, "T2", "lopo", "mlp_static", {k: v + 0.05 for k, v in T2_SGR.items()}, seeds=range(5))
    write_selection(layout, "T1", "ridge_time", "gru")
    write_selection(layout, "T2", "sgr", "lstm")
    return cfg, layout


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    cfg, layout = build_world(tmp_path_factory.mktemp("eval"))
    return cfg, layout, ev.evaluate(cfg, layout)


def one(frame, **conditions):
    mask = np.ones(len(frame), dtype=bool)
    for column, value in conditions.items():
        mask &= (frame[column] == value).to_numpy()
    assert mask.sum() == 1, (conditions, int(mask.sum()))
    return frame[mask].iloc[0]


# ----------------------------------------------------------------------------------------
# Scores and metrics
# ----------------------------------------------------------------------------------------

def test_s_is_the_seed_averaged_per_pond_error_averaged_over_ponds(world):
    scores = world[2]["scores"]
    base = one(scores, task="T1", scheme="lopo", sample_set="eligible", horizon="0", model="ridge_time")
    assert base["S"] == pytest.approx(np.mean(list(T1_BASE.values()))) and base["S"] == pytest.approx(0.35)
    assert (base["n_ponds"], base["n_samples"], base["n_seeds"]) == (7, 20, 1) and np.isnan(base["S_seed_sd"])
    pooled_mean = sum(T1_BASE[p] * n for p, n in zip(PONDS, T1_COUNTS)) / sum(T1_COUNTS)
    assert pooled_mean != pytest.approx(base["S"], abs=1e-3)                      # S is not the mean over samples
    assert base["sd_pond_S"] == pytest.approx(np.std(list(T1_BASE.values()), ddof=1))
    assert base["group"] == "time_only"
    gru = one(scores, task="T1", scheme="lopo", sample_set="eligible", horizon="0", model="gru")
    assert gru["S"] == pytest.approx(np.mean(list(T1_GRU.values()))) and gru["n_seeds"] == 5 and gru["group"] == "deep"
    assert gru["S_seed_sd"] == pytest.approx(np.std(OFFSETS, ddof=1))              # each seed's S is the mean plus its offset
    assert one(scores, task="T1", scheme="lopo", sample_set="eligible", horizon="0", model="mlp_static")["group"] == "static_only"
    assert one(scores, task="T1", scheme="lopo", sample_set="eligible", horizon="0",
               model="ridge_time_wq_w07")["group"] == "sensitivity"


def test_t2_has_one_score_per_horizon_and_one_for_both_together(world):
    scores = world[2]["scores"]
    for h in (1, 2):
        sgr = one(scores, task="T2", scheme="lopo", sample_set="eligible", horizon=str(h), model="sgr")
        assert sgr["S"] == pytest.approx(np.mean([T2_SGR[(p, h)] for p in PONDS]))
        assert sgr["n_samples"] == sum(T2_COUNTS[h])
    both = one(scores, task="T2", scheme="lopo", sample_set="eligible", horizon="all", model="sgr")
    per_pond = [(T2_SGR[(p, 1)] * a + T2_SGR[(p, 2)] * b) / (a + b)
                for p, a, b in zip(PONDS, T2_COUNTS[1], T2_COUNTS[2])]               # all samples of a pond, both horizons
    assert both["S"] == pytest.approx(np.mean(per_pond))
    assert both["n_samples"] == sum(T2_COUNTS[1]) + sum(T2_COUNTS[2])


def test_a_nonpositive_weight_makes_s_infinite_and_is_counted_but_grams_are_scored_as_they_are(world):
    scores, pooled = world[2]["scores"], world[2]["pooled"]
    linear = one(scores, task="T2", scheme="lopo", sample_set="eligible", horizon="1", model="linear")
    assert linear["S"] == np.inf and linear["n_nonpositive"] == T2_COUNTS[1][2]
    assert one(scores, task="T2", scheme="lopo", sample_set="eligible", horizon="2", model="linear")["S"] < np.inf
    row = one(pooled, task="T2", scheme="lopo", sample_set="eligible", horizon="1", model="linear")
    assert np.isfinite(row["mae_g"]) and row["mae_g"] > 0
    per_pond = world[2]["per_pond"]
    pond3 = one(per_pond, task="T2", scheme="lopo", sample_set="eligible", horizon="1", model="linear", pond=3)
    assert pond3["S"] == np.inf and pond3["n_nonpositive"] == T2_COUNTS[1][2]


def test_pooled_metrics_are_computed_over_all_test_samples_and_averaged_over_seeds(world):
    cfg, layout, tables = world
    frame = pd.read_csv(layout.predictions("T1", "lopo", "gru"))
    by_seed = [ev.pooled_metrics(g["w_true"], g["w_pred"]) for _, g in frame.groupby("seed")]
    row = one(tables["pooled"], task="T1", scheme="lopo", sample_set="eligible", horizon="0", model="gru")
    assert row["mae_g"] == pytest.approx(np.mean([m["mae_g"] for m in by_seed]))
    assert row["rmse_g_sd"] == pytest.approx(np.std([m["rmse_g"] for m in by_seed], ddof=1))
    assert row["n_samples"] == 20 and row["n_seeds"] == 5
    seeds = tables["pooled_by_seed"]
    assert len(seeds[(seeds["model"] == "gru") & (seeds["task"] == "T1") & (seeds["scheme"] == "lopo")]) == 5
    single = one(tables["pooled"], task="T1", scheme="lopo", sample_set="eligible", horizon="0", model="ridge_time")
    assert np.isnan(single["mae_g_sd"]) and single["n_seeds"] == 1


def test_per_pond_tables_hold_the_seed_rows_and_their_average(world):
    tables = world[2]
    rows = tables["per_pond_seed"]
    gru = rows[(rows["task"] == "T1") & (rows["scheme"] == "lopo") & (rows["model"] == "gru") & (rows["pond"] == 3)]
    assert sorted(gru["seed"]) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(sorted(gru["S"]), sorted(T1_GRU[3] + np.array(OFFSETS)), atol=1e-9)
    average = one(tables["per_pond"], task="T1", scheme="lopo", sample_set="eligible", horizon="0", model="gru", pond=3)
    assert average["S"] == pytest.approx(T1_GRU[3]) and average["n_seeds"] == 5 and average["n"] == T1_COUNTS[2]


def test_the_calendar_hold_out_and_the_all_events_table_are_scored_but_not_tested(world):
    tables = world[2]
    scores = tables["scores"]
    chrono = one(scores, task="T1", scheme="chrono", sample_set="eligible", horizon="0", model="ridge_time")
    assert chrono["n_ponds"] == 5 and chrono["S"] == pytest.approx(np.mean([T1_BASE[p] for p in (3, 4, 6, 7, 9)]))
    everything = one(scores, task="T1", scheme="lopo", sample_set="all", horizon="0", model="gc")
    assert everything["S"] == pytest.approx(np.mean(list(shifted(T1_BASE, 0.1).values()))) and everything["n_samples"] == 20
    for name in ("pairwise", "bootstrap_S", "influence"):
        assert "chrono" not in set(tables[name].get("scheme", [])) and len(tables[name]) > 0
    assert set(tables["pairwise"]["task"]) == {"T1", "T2"}


def test_completeness_lists_models_without_predictions(world):
    table = world[2]["completeness"]
    assert one(table, task="T1", scheme="lopo", sample_set="eligible", model="gc")["has_predictions"]
    assert not one(table, task="T1", scheme="lopo", sample_set="eligible", model="ridge_wq")["has_predictions"]
    assert one(table, task="T1", scheme="lopo", sample_set="all", model="ridge_time")["has_predictions"]
    assert not one(table, task="T2", scheme="chrono", sample_set="eligible", model="sgr")["has_predictions"]


# ----------------------------------------------------------------------------------------
# Tests and intervals
# ----------------------------------------------------------------------------------------

def test_t1_confirmatory_step_by_hand(world):
    row = one(world[2]["confirmatory"], step=1)
    assert (row["task"], row["horizon"], row["model"], row["comparator"]) == ("T1", "0", "gru", "ridge_time")
    # differences gru - base: -0.05, +0.02, -0.10, -0.07, -0.03, -0.20, -0.08; the one worse pond has the smallest |d|
    assert (row["n_ponds"], row["n_nonzero"], row["n_zero_dropped"], row["n_lower"]) == (7, 7, 0, 6)
    assert row["t_plus"] == 1.0 and row["t_minus"] == 27.0
    assert row["p_one_sided_lower"] == pytest.approx(2 / 128) and row["p_two_sided"] == pytest.approx(4 / 128)
    assert row["status"] == "rejected" and row["reject"] and row["alpha"] == 0.05


def test_t2_horizon_1_is_tested_after_t1_rejects_with_midranks(world):
    row = one(world[2]["confirmatory"], step=2)
    d = [T2_LSTM[(p, 1)] - T2_SGR[(p, 1)] for p in PONDS]
    expected = brute_force(d)
    assert (row["task"], row["horizon"], row["model"], row["comparator"]) == ("T2", "1", "lstm", "sgr")
    assert row["t_plus"] == pytest.approx(14.0)                       # ranks 2, 6, 2, 4 for the four worse ponds
    assert row["p_one_sided_lower"] == pytest.approx(expected[0]) and row["p_two_sided"] == pytest.approx(expected[2])
    assert row["status"] == "not rejected" and not row["reject"]


def test_the_confirmatory_test_is_one_sided(tmp_path):
    # ponds 1 and 2 worse by 0.01 and 0.02 (ranks 1 and 2), the other five better by 0.10 to 0.14: T+ = 3
    shifts = {1: 0.01, 2: 0.02, 3: -0.10, 4: -0.11, 6: -0.12, 7: -0.13, 9: -0.14}
    cfg, layout = build_world(tmp_path, t1_deep={p: T1_BASE[p] + shifts[p] for p in PONDS})
    first = ev.evaluate(cfg, layout)["confirmatory"].iloc[0]
    assert first["t_plus"] == pytest.approx(3.0)
    assert first["p_one_sided_lower"] == pytest.approx(5 / 128) and first["p_two_sided"] == pytest.approx(10 / 128)
    assert first["status"] == "rejected"                              # 0.039 <= 0.05 although the two-sided p is 0.078


def test_the_sequence_stops_at_the_first_step_that_does_not_reject(tmp_path):
    mixed = {1: 0.29, 2: 0.22, 3: 0.35, 4: 0.30, 6: 0.30, 7: 0.60, 9: 0.40}       # worse than the baseline in 3 ponds
    cfg, layout = build_world(tmp_path, t1_deep=mixed)
    tables = ev.evaluate(cfg, layout)
    first, second = tables["confirmatory"].iloc[0], tables["confirmatory"].iloc[1]
    assert first["status"] == "not rejected" and not first["reject"]
    assert second["status"] == "not tested (an earlier step did not reject)" and not second["reject"]
    assert second["p_one_sided_lower"] == pytest.approx(brute_force([T2_LSTM[(p, 1)] - T2_SGR[(p, 1)] for p in PONDS])[0])


def six_pond_world(tmp_path, deep_errors):
    cfg, layout = build_world(tmp_path)
    six = sample_frame("T1", [1, 2, 3, 4, 6, 7])
    for model, errors, seeds in (("ridge_time", T1_BASE, (0,)), ("gru", deep_errors, range(5))):
        write_model(layout, "T1", "lopo", model, errors, frame=six, seeds=seeds)
    for model in ("gc", "ridge_time_wq", "ridge_time_wq_w07", "ridge_time_wq_noind", "mlp_static"):
        (layout.predictions("T1", "lopo", model)).unlink()
    return ev.evaluate(cfg, layout)


def test_exactly_six_scored_ponds_are_enough_for_the_test(tmp_path):
    better = {p: s - 0.05 - 0.01 * p for p, s in T1_BASE.items()}                # lower in every pond: T+ = 0, p = 1/64
    first = six_pond_world(tmp_path / "a", better).get("confirmatory").iloc[0]
    assert first["n_ponds"] == 6 and first["status"] == "rejected" and first["p_one_sided_lower"] == pytest.approx(1 / 64)
    worse_in_one = {**better, 1: T1_BASE[1] + 0.2}      # worse in pond 1 whose |d| is the largest: T+ = 6, not enough
    second = six_pond_world(tmp_path / "b", worse_in_one).get("confirmatory").iloc[0]
    assert second["n_ponds"] == 6 and second["status"] == "not rejected" and second["t_plus"] == 6.0


def test_a_task_with_fewer_than_six_scored_ponds_is_descriptive_and_stops_the_sequence(tmp_path):
    cfg, layout = build_world(tmp_path)
    five = sample_frame("T1", [1, 2, 3, 4, 6])
    for model, errors, seeds in (("ridge_time", T1_BASE, (0,)), ("gru", T1_GRU, range(5))):
        write_model(layout, "T1", "lopo", model, errors, frame=five, seeds=seeds)
    for model in ("gc", "ridge_time_wq", "ridge_time_wq_w07", "ridge_time_wq_noind", "mlp_static"):
        (layout.predictions("T1", "lopo", model)).unlink()
    tables = ev.evaluate(cfg, layout)
    first, second = tables["confirmatory"].iloc[0], tables["confirmatory"].iloc[1]
    assert first["status"] == "descriptive (fewer than 6 scored ponds)" and not first["reject"] and first["n_ponds"] == 5
    assert second["status"] == "not tested (an earlier step did not reject)"
    pair = one(tables["pairwise"], task="T1", model="gru", comparator="ridge_time")
    assert pair["descriptive_only"] and np.isfinite(pair["p_one_sided_lower"])
    pair2 = one(tables["pairwise"], task="T2", horizon="1", model="lstm", comparator="sgr")
    assert not pair2["descriptive_only"]


def test_pairs_cover_the_baseline_the_static_twin_and_the_sensitivity_variants(world):
    pairs = world[2]["pairwise"]
    t1 = pairs[pairs["task"] == "T1"]
    kinds = {(r.model, r.comparator): r.kind for r in t1.itertuples()}
    assert kinds[("gru", "ridge_time")] == "primary_deep_vs_primary_baseline"
    assert kinds[("gc", "ridge_time")] == "vs_primary_baseline"
    assert kinds[("gru", "gc")] == "primary_deep_vs_baseline"
    assert kinds[("gru", "mlp_static")] == "deep_vs_static"
    assert kinds[("ridge_time_wq_w07", "ridge_time_wq")] == "sensitivity_vs_base"
    assert kinds[("ridge_time_wq_noind", "ridge_time_wq")] == "sensitivity_vs_base"
    assert ("ridge_time", "ridge_time") not in kinds
    assert pairs["confirmatory_pair"].sum() == 1 + 3 and set(pairs["horizon"]) == {"0", "1", "2", "all"}
    gc = one(t1, model="gc", comparator="ridge_time")
    assert (gc["n_higher"], gc["n_lower"], gc["n_equal"]) == (7, 0, 0)
    assert gc["p_one_sided_lower"] == 1.0 and gc["p_two_sided"] == pytest.approx(2 / 128)      # worse in every pond
    assert gc["diff"] == pytest.approx(0.10)


def test_the_bootstrap_resamples_ponds_jointly_with_the_seeded_generator(world):
    tables = world[2]
    rng = np.random.default_rng(20261014)
    index = rng.integers(0, 7, size=(2000, 7))                         # the first stratum drawn is T1
    base = np.array([T1_BASE[p] for p in PONDS])
    gru = np.array([T1_GRU[p] for p in PONDS])
    low, high = np.percentile(base[index].mean(axis=1), [2.5, 97.5])
    row = one(tables["bootstrap_S"], task="T1", horizon="0", model="ridge_time")
    assert (row["ci_low"], row["ci_high"], row["B"]) == (pytest.approx(low), pytest.approx(high), 2000)
    assert row["S"] == pytest.approx(base.mean())
    diff = gru[index].mean(axis=1) - base[index].mean(axis=1)
    low, high = np.percentile(diff, [2.5, 97.5])
    pair = one(tables["pairwise"], task="T1", horizon="0", model="gru", comparator="ridge_time")
    assert (pair["diff_ci_low"], pair["diff_ci_high"]) == (pytest.approx(low), pytest.approx(high))
    second = rng.integers(0, 7, size=(2000, 7))                        # then T2 horizon 1
    sgr1 = np.array([T2_SGR[(p, 1)] for p in PONDS])
    lstm1 = np.array([T2_LSTM[(p, 1)] for p in PONDS])
    low, high = np.percentile(lstm1[second].mean(axis=1) - sgr1[second].mean(axis=1), [2.5, 97.5])
    pair = one(tables["pairwise"], task="T2", horizon="1", model="lstm", comparator="sgr")
    assert (pair["diff_ci_low"], pair["diff_ci_high"]) == (pytest.approx(low), pytest.approx(high))


def test_a_model_compared_with_a_copy_of_itself_has_a_zero_interval_and_no_test(tmp_path):
    cfg, layout = build_world(tmp_path)
    write_model(layout, "T1", "lopo", "ridge_time_wq", T1_BASE)                  # the same errors as the baseline
    tables = ev.evaluate(cfg, layout)
    pair = one(tables["pairwise"], task="T1", horizon="0", model="ridge_time_wq", comparator="ridge_time")
    assert (pair["diff_ci_low"], pair["diff_ci_high"], pair["diff"]) == (0.0, 0.0, 0.0)
    assert (pair["n_nonzero"], pair["n_zero_dropped"], pair["p_one_sided_lower"], pair["p_two_sided"]) == (0, 7, 1.0, 1.0)


def test_influence_table_removes_one_pond_at_a_time(world):
    table = world[2]["influence"]
    pair = table[(table["task"] == "T1") & (table["horizon"] == "0") & (table["model"] == "gru")
                 & (table["comparator"] == "ridge_time")]
    assert sorted(pair["pond_removed"]) == PONDS and (pair["n_ponds_left"] == 6).all()
    row = one(pair, pond_removed=2)
    others = [p for p in PONDS if p != 2]
    assert row["S_model"] == pytest.approx(np.mean([T1_GRU[p] for p in others]))
    assert row["S_comparator"] == pytest.approx(np.mean([T1_BASE[p] for p in others]))
    assert row["diff"] == pytest.approx(row["S_model"] - row["S_comparator"])
    full = np.mean(list(T1_GRU.values())) - np.mean(list(T1_BASE.values()))
    assert row["diff"] < full                                           # pond 2 is where the deep model is worse


def test_the_merged_units_sensitivity_averages_ponds_1_and_2(world):
    table = world[2]["merged_units"]
    row = one(table, task="T1", horizon="0")
    assert (row["model"], row["comparator"], row["n_units"]) == ("gru", "ridge_time", 6)
    # unit values 0.235 vs 0.25 and the other five ponds: the deep model is lower in every unit
    assert row["t_plus"] == 0.0 and row["p_one_sided_lower"] == pytest.approx(1 / 64) and row["descriptive_only"]
    assert row["S_model"] == pytest.approx(np.mean([(T1_GRU[1] + T1_GRU[2]) / 2] + [T1_GRU[p] for p in (3, 4, 6, 7, 9)]))
    assert set(table["task"]) == {"T1", "T2"}


# ----------------------------------------------------------------------------------------
# Refusals
# ----------------------------------------------------------------------------------------

def test_models_must_be_scored_on_the_same_samples(tmp_path):
    cfg, layout = build_world(tmp_path)
    write_model(layout, "T1", "lopo", "gc", shifted(T1_BASE, 0.1), drop_rows={0})
    with pytest.raises(ev.EvaluationError, match="different samples"):
        ev.evaluate(cfg, layout)


def test_seeds_must_be_the_expected_ones(tmp_path):
    cfg, layout = build_world(tmp_path)
    write_model(layout, "T1", "lopo", "gru", T1_GRU, seeds=range(4))
    with pytest.raises(ev.EvaluationError, match="seeds \\[0, 1, 2, 3\\]"):
        ev.evaluate(cfg, layout)
    write_model(layout, "T1", "lopo", "gru", T1_GRU, seeds=range(5))
    write_model(layout, "T1", "lopo", "ridge_time", T1_BASE, seeds=(0, 1))
    with pytest.raises(ev.EvaluationError, match="seeds \\[0, 1\\], expected \\[0\\]"):
        ev.evaluate(cfg, layout)


def test_a_sample_predicted_twice_a_wrong_file_and_a_missing_primary_model_are_refused(tmp_path):
    cfg, layout = build_world(tmp_path)
    path = layout.predictions("T1", "lopo", "gc")
    frame = pd.read_csv(path)
    pd.concat([frame, frame.iloc[:1]]).to_csv(path, index=False)
    with pytest.raises(ev.EvaluationError, match="predicted twice"):
        ev.evaluate(cfg, layout)
    frame.assign(model="ridge_time").to_csv(path, index=False)
    with pytest.raises(ev.EvaluationError, match="another task, scheme or model"):
        ev.evaluate(cfg, layout)
    frame.to_csv(path, index=False)
    layout.predictions("T1", "lopo", "gru").unlink()
    with pytest.raises(ev.EvaluationError, match="primary models ridge_time and gru have no predictions"):
        ev.evaluate(cfg, layout)


def test_models_scored_on_different_ponds_in_one_stratum_are_refused(tmp_path):
    cfg, layout = build_world(tmp_path)
    frame = sample_frame("T1")
    short = frame[frame["pond"] != 9]
    write_model(layout, "T1", "lopo", "gc", shifted(T1_BASE, 0.1), frame=short)
    with pytest.raises(ev.EvaluationError, match="different samples|different ponds"):
        ev.evaluate(cfg, layout)


def test_a_missing_selection_file_is_refused(tmp_path):
    cfg, layout = build_world(tmp_path)
    layout.selection("T2").unlink()
    with pytest.raises(ev.EvaluationError, match="missing selection file"):
        ev.evaluate(cfg, layout)


def test_a_nonfinite_weight_in_a_prediction_file_is_refused(tmp_path):
    cfg, layout = build_world(tmp_path)
    path = layout.predictions("T1", "lopo", "gc")
    frame = pd.read_csv(path)
    frame.loc[0, "w_pred"] = np.nan
    frame.to_csv(path, index=False)
    with pytest.raises(ev.EvaluationError, match="not finite"):
        ev.evaluate(cfg, layout)


def test_the_outputs_are_repeatable_byte_for_byte(tmp_path):
    cfg, layout = build_world(tmp_path)
    ev.evaluate(cfg, layout)
    first = {p: p.read_bytes() for p in sorted((layout.root).glob("metrics/*.csv")) + sorted(layout.root.glob("stats/*.csv"))}
    assert len(first) >= 11
    ev.evaluate(cfg, layout)
    assert {p: p.read_bytes() for p in first} == first


# ----------------------------------------------------------------------------------------
# Permutation p-values
# ----------------------------------------------------------------------------------------

def write_permutation(layout, task, model, s_perm):
    path = layout.permutation(task, model)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"r": range(len(s_perm)), "S_perm": s_perm}).to_csv(path, index=False)


def test_permutation_p_values_use_the_observed_statistic_of_the_saved_predictions(tmp_path):
    cfg, layout = build_world(tmp_path)
    # observed S: ridge_time_wq = 0.34 (T1); the primary deep model at seed 0 only: mean(T1_GRU) + OFFSETS[0]
    write_model(layout, "T1", "lopo", "ridge_time_wq", shifted(T1_BASE, -0.01))
    s_perm = np.linspace(0.30, 0.50, 999)
    write_permutation(layout, "T1", "ridge_time_wq", s_perm)
    deep_obs = np.mean(list(T1_GRU.values())) + OFFSETS[0]
    deep_perm = np.linspace(deep_obs - 0.05, deep_obs + 0.05, 199)
    write_permutation(layout, "T1", "gru", deep_perm)
    write_permutation(layout, "T2", "lstm", np.full(199, 0.9))
    tables = ev.evaluate(cfg, layout)
    table = tables["permutation"]
    tab = one(table, task="T1", model="ridge_time_wq")
    s_obs = float(np.mean([v - 0.01 for v in T1_BASE.values()]))
    assert tab["S_obs"] == pytest.approx(s_obs) and tab["statistic"] == "S" and tab["R"] == 999
    assert tab["n_perm_le_obs"] == int(np.sum(s_perm <= s_obs)) and tab["p"] == pytest.approx((1 + tab["n_perm_le_obs"]) / 1000)
    deep = one(table, task="T1", model="gru")
    assert deep["seed"] == 0 and deep["S_obs"] == pytest.approx(deep_obs) and deep["R"] == 199
    assert deep["p"] == pytest.approx((1 + np.sum(deep_perm <= deep["S_obs"])) / 200)
    t2 = one(table, task="T2", model="lstm")
    h1 = np.mean([T2_LSTM[(p, 1)] for p in PONDS]) + OFFSETS[0]
    assert t2["statistic"] == "S_h1" and t2["S_obs"] == pytest.approx(h1) and t2["p"] == pytest.approx(1 / 200)
    assert set(table["model"]) == {"ridge_time_wq", "gru", "lstm"}                      # the others have no file


def test_a_permutation_file_with_the_wrong_number_of_runs_is_refused(tmp_path):
    cfg, layout = build_world(tmp_path)
    write_model(layout, "T1", "lopo", "ridge_time_wq", shifted(T1_BASE, -0.01))
    write_permutation(layout, "T1", "ridge_time_wq", np.linspace(0.3, 0.5, 998))
    with pytest.raises(ev.EvaluationError, match="r = 0..998 expected|permutations 998 listed"):
        ev.evaluate(cfg, layout)


# ----------------------------------------------------------------------------------------
# The committed-selection rule and the command line
# ----------------------------------------------------------------------------------------

def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    done = subprocess.run(["git", "-c", "user.name=test", "-c", "user.email=test@example.org",
                           "-c", "commit.gpgsign=false", *args], cwd=repo, env=env, check=True,
                          capture_output=True, text=True)
    return done.stdout.strip()


@pytest.fixture
def repo_world(tmp_path, monkeypatch):
    cfg, layout = build_world(tmp_path)
    git(tmp_path, "init", "-q")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    config = tmp_path / "phase4_test.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return cfg, layout, tmp_path, str(config)


def test_the_scorer_refuses_selection_files_that_are_not_committed_or_were_modified(repo_world, capsys):
    cfg, layout, repo, config = repo_world
    assert ev.main(["--config", config]) == 2
    assert "not tracked by git" in capsys.readouterr().err
    assert not (layout.root / "metrics").exists()                      # nothing was scored
    git(repo, "add", "results/selection/primary_T1.json", "results/selection/primary_T2.json")
    assert ev.main(["--config", config]) == 2
    assert "not in the last commit" in capsys.readouterr().err
    git(repo, "commit", "-q", "-m", "selection")
    assert ev.main(["--config", config]) == 0
    out = capsys.readouterr().out
    assert "wrote" in out and "rejected" in out and (layout.root / "stats" / "confirmatory.csv").is_file()
    path = layout.selection("T1")
    path.write_text(path.read_text(encoding="utf-8").replace("gru", "lstm"), encoding="utf-8")
    assert ev.main(["--config", config]) == 2
    assert "differs from the last commit" in capsys.readouterr().err


def test_the_scorer_reports_invalid_predictions_as_a_refusal(repo_world, capsys):
    cfg, layout, repo, config = repo_world
    git(repo, "add", "results/selection")
    git(repo, "commit", "-q", "-m", "selection")
    layout.predictions("T1", "lopo", "gc").write_text("not,a,prediction,file\n1,2,3,4\n", encoding="utf-8")
    assert ev.main(["--config", config]) == 2
    assert "lacks columns" in capsys.readouterr().err


def test_the_scorer_reads_prediction_and_selection_files_only(world):
    cfg, layout, tables = world
    scratch = layout.root / "tuning" / "T1" / "lopo" / "gru"
    scratch.mkdir(parents=True)
    (scratch / "inner_scores.csv").write_bytes(b"\xff\xfe garbage")
    (scratch / "chosen.csv").write_bytes(b"\xff\xfe garbage")
    again = ev.evaluate(cfg, layout)
    pd.testing.assert_frame_equal(again["scores"], tables["scores"])
    assert json.loads(layout.selection("T1").read_text(encoding="utf-8"))["primary_deep"] == "gru"
