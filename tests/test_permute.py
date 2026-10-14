"""Tests for src/permute.py: the within-date derangement, the donor inputs, and the whole procedure on
synthetic tables (tests/synth_p4.py), including a case where the target depends on the water
quality so that the permutation has to destroy something.
"""

import concurrent.futures
import copy
import json
import multiprocessing
import os
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from src import guard, permute, train
from src.artifacts import write_json_atomic
from src.baselines import make_baseline, wq_columns
from src.folds import load_config, plan_folds, select_samples
from tests.p4env import make_env, quiet

CFG = load_config()


def allow():
    """A stand-in for the freeze guard that can be pickled for the worker processes."""
    return {"tag": "analysis-plan-frozen", "commit": "0" * 40}


def no_selection_check(layout, task):
    return None


def refuse_after_two():
    """A guard that passes twice and then fails; the counter lives in a file named by the environment."""
    path = os.environ["P4_GUARD_COUNTER"]
    with open(path, "a+") as handle:
        handle.seek(0)
        calls = len(handle.read())
        handle.write("x")
    if calls >= 2:
        raise guard.GuardError("the tree changed")
    return allow()


# ----------------------------------------------------------------------------------------
# Derangements and donors
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("k", [2, 3, 4, 5, 6])
def test_derangements_have_no_fixed_point(k):
    rng = np.random.default_rng(0)
    for _ in range(300):
        order = permute.derange(rng, k)
        assert sorted(order) == list(range(k)) and not (order == np.arange(k)).any()


def test_derangements_are_uniform_over_all_derangements():
    rng = np.random.default_rng(1)
    three = Counter(tuple(permute.derange(rng, 3)) for _ in range(4000))
    assert set(three) == {(1, 2, 0), (2, 0, 1)} and all(1800 < v < 2200 for v in three.values())
    four = Counter(tuple(permute.derange(rng, 4)) for _ in range(9000))
    assert len(four) == 9 and all(800 < v < 1200 for v in four.values())            # 9 derangements of 4 items
    with pytest.raises(ValueError, match="at least two"):
        permute.derange(rng, 1)


def test_the_draws_depend_only_on_the_seed():
    first = [permute.derange(np.random.default_rng(20261014 + r), 5).tolist() for r in range(5)]
    again = [permute.derange(np.random.default_rng(20261014 + r), 5).tolist() for r in range(5)]
    assert first == again and len({tuple(x) for x in first}) > 1


def test_units_and_donor_ponds_follow_the_config():
    unit_of = permute.unit_lookup(CFG)
    assert unit_of[1] == unit_of[2] == (1, 2) and unit_of[3] == (3,) and set(unit_of) == set(CFG["design"]["cohort"])
    assert permute.donor_of(CFG, (1, 2)) == 2 and permute.donor_of(CFG, (4,)) == 4
    bad = copy.deepcopy(CFG)
    bad["permutation"]["donor_pond"] = {1: 1, 2: 2}
    with pytest.raises(permute.PermutationError, match="different donor ponds"):
        permute.donor_of(bad, (1, 2))


ROWS = pd.DataFrame({
    "pond": [1, 3, 4, 2, 1, 2, 3],
    "date": ["2021-07-17"] * 3 + ["2021-07-31"] + ["2021-08-14"] * 3,
    "stream": ["S12", "S03", "S04", "S12", "S12", "S12", "S03"],
}, index=[10, 11, 12, 13, 14, 15, 16])


def test_each_date_with_two_or_more_units_is_deranged_and_a_single_unit_date_is_left_alone():
    seed = 20261014 + 3
    donors, counts = permute.permuted_donors(ROWS, "T1", CFG, np.random.default_rng(seed))
    # the same draws by hand: 07-17 has the units (1,2), (3,), (4,) with donor ponds 2, 3, 4
    rng = np.random.default_rng(seed)
    order = permute.derange(rng, 3)
    donor_pond = [2, 3, 4]
    expected = {1: donor_pond[order[0]], 3: donor_pond[order[1]], 4: donor_pond[order[2]]}
    assert [donors[0], donors[1], donors[2]] == [expected[1], expected[3], expected[4]]
    assert donors[3] == 2                                                   # 07-31: one unit, the pond keeps itself
    permute.derange(rng, 2)                                                 # 08-14: units (1,2) and (3,) swap
    assert list(donors[4:]) == [3, 3, 2]                                    # ponds 1 and 2 share a unit: same donor
    assert counts == {"n_dates": 3, "n_dates_unchanged": 1, "n_rows_changed": 6}
    assert not (donors[:3] == [2, 3, 4]).any()                              # nobody keeps its unit's inputs


def test_a_unit_never_receives_its_own_inputs_and_the_donors_are_a_permutation_of_the_units():
    for r in range(40):
        donors, _ = permute.permuted_donors(ROWS, "T1", CFG, np.random.default_rng(20261014 + r))
        first = donors[:3]                                                  # one row per unit on 07-17
        assert sorted(first) == [2, 3, 4] and first[0] != 2 and first[1] != 3 and first[2] != 4
        assert donors[3] == 2 and donors[4] == donors[5] == 3 and donors[6] == 2


def test_the_order_of_the_rows_does_not_change_who_receives_what():
    for seed in range(25):
        forward, _ = permute.permuted_donors(ROWS, "T1", CFG, np.random.default_rng(seed))
        backward, _ = permute.permuted_donors(ROWS.iloc[::-1], "T1", CFG, np.random.default_rng(seed))
        assert dict(zip(ROWS.index, forward)) == dict(zip(ROWS.iloc[::-1].index, backward))


def test_samples_that_are_not_passed_in_do_not_affect_the_draws():
    only = ROWS[ROWS["date"] == "2021-08-14"]
    donors, counts = permute.permuted_donors(only, "T1", CFG, np.random.default_rng(0))
    assert list(donors) == [3, 3, 2] and counts["n_dates"] == 1 and counts["n_rows_changed"] == 3


def test_t2_dates_are_origin_dates():
    rows = ROWS.rename(columns={"date": "origin_date"}).copy()
    donors, counts = permute.permuted_donors(rows, "T2", CFG, np.random.default_rng(0))
    assert counts["n_dates"] == 3 and counts["n_dates_unchanged"] == 1 and sorted(donors[:3]) == [2, 3, 4]


# ----------------------------------------------------------------------------------------
# The permuted table
# ----------------------------------------------------------------------------------------

def test_a_permuted_table_takes_the_donors_features_and_changes_nothing_else(tmp_path):
    env = make_env(tmp_path)
    for task, key in (("T1", "date"), ("T2", "origin_date")):
        table = env.tables.table(task)
        rows = select_samples(table, env.cfg["design"]["cohort"], "eligible")
        donors, counts = permute.permuted_donors(rows, task, env.cfg, np.random.default_rng(7))
        before = table.copy()
        out = permute.permuted_table(table, rows, donors, env.tables.wq, task, env.cfg)
        pd.testing.assert_frame_equal(table, before)                       # the input is not modified
        columns = wq_columns(task, env.cfg)
        prefix = "t1" if task == "T1" else "t2"
        wq = env.tables.wq.set_index(["pond", "date"])
        changed = donors != rows["pond"].to_numpy()
        assert changed.sum() == counts["n_rows_changed"] > 0
        for label, pond, donor, date in zip(rows.index, rows["pond"], donors, rows[key]):
            source = wq.loc[(donor, date), [f"{prefix}_{c}" for c in columns]].to_numpy(dtype=float)
            np.testing.assert_allclose(out.loc[label, columns].to_numpy(dtype=float), source, equal_nan=True)
        other = [c for c in table.columns if c not in columns]
        pd.testing.assert_frame_equal(out[other], table[other])           # targets, time, past weights, eligibility
        outside = table.index.difference(rows.index)
        pd.testing.assert_frame_equal(out.loc[outside], table.loc[outside])


def test_missing_features_move_with_the_donor(tmp_path):
    env = make_env(tmp_path)
    table = env.tables.t1
    rows = select_samples(table, env.cfg["design"]["cohort"], "eligible")
    wq = env.tables.wq.copy()
    wq.loc[wq["pond"] == 3, [c for c in wq.columns if c.startswith("t1_nitrate")]] = np.nan      # pond 3 has no nitrate
    donors = rows["pond"].to_numpy().copy()
    donors[(rows["pond"] == 4).to_numpy()] = 3                                   # pond 4 receives pond 3's inputs
    out = permute.permuted_table(table, rows, donors, wq, "T1", env.cfg)
    four = rows.index[(rows["pond"] == 4).to_numpy()]
    assert out.loc[four, "nitrate_mean_w"].isna().all() and table.loc[four, "nitrate_mean_w"].notna().any()
    three = rows.index[(rows["pond"] == 3).to_numpy()]
    pd.testing.assert_frame_equal(out.loc[three], table.loc[three])


def test_a_donor_without_a_row_at_the_date_is_refused(tmp_path):
    env = make_env(tmp_path)
    rows = select_samples(env.tables.t1, env.cfg["design"]["cohort"], "eligible")
    donors = np.full(len(rows), 3)
    wq = env.tables.wq[env.tables.wq["pond"] != 3]
    with pytest.raises(permute.PermutationError, match="no wq_by_date row"):
        permute.permuted_table(env.tables.t1, rows, donors, wq, "T1", env.cfg)


@pytest.mark.parametrize("task", ["T1", "T2"])
def test_a_neural_model_gets_the_donors_sequences_and_keeps_its_own_statics_and_targets(tmp_path, task):
    env = make_env(tmp_path)
    cfg = env.cfg
    tables, rows, _plans, store = permute.prepare(cfg, task, "gru")
    own, _ = permute.permuted_context(cfg, tables, rows, store, task, "gru", np.random.default_rng(99999))
    donors, _ = permute.permuted_donors(rows, task, cfg, np.random.default_rng(5))
    ctx, counts = permute.permuted_context(cfg, tables, rows, store, task, "gru", np.random.default_rng(5))
    ss = ctx.sampleset
    assert counts["n_rows_changed"] > 0 and list(ss.ids) == list(rows.index)
    assert list(ss.data_pond_idx) == [store.pond_index[int(d)] for d in donors]            # sequences cut from the donor
    assert list(ss.pond_idx) == [store.pond_index[int(p)] for p in rows["pond"]]           # the sample stays its own
    base = permute.build_sampleset(task, rows, store, cfg)
    np.testing.assert_array_equal(ss.statics, base.statics)
    np.testing.assert_array_equal(ss.y, base.y)
    changed = np.array(ss.data_pond_idx) != np.array(ss.pond_idx)
    assert changed.any() and not np.array_equal(ss.vals[changed], base.vals[changed])    # the water quality differs
    assert own.table is not None and ctx.table is not tables.table(task)
    tabular, _ = permute.permuted_context(cfg, tables, rows, None, task, "ridge_time_wq" if task == "T1" else "ridge_pw_wq",
                                          np.random.default_rng(5))
    assert tabular.sampleset is None


def test_the_statistic_is_s_and_for_t2_s_at_horizon_one():
    frame = pd.DataFrame({"pond": [1, 1, 2, 2, 2], "horizon": [1, 2, 1, 1, 2],
                          "y_true_log": [0.0] * 5, "y_pred_log": [0.1, 0.9, 0.2, 0.4, 0.8]})
    assert permute.permutation_statistic("T2", frame) == pytest.approx((0.1 + 0.3) / 2)
    assert permute.permutation_statistic("T1", frame) == pytest.approx((0.5 + (0.2 + 0.4 + 0.8) / 3) / 2)


def test_chosen_hyperparameters_are_cast_to_the_types_of_the_search_space():
    spec = {"max_iter": {"type": "int", "low": 50, "high": 500}, "max_depth": {"type": "categorical", "choices": [2, 3, 4]},
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3}, "act": {"type": "categorical", "choices": ["a", "b"]}}
    assert permute.cast(120.0, spec["max_iter"]) == 120 and isinstance(permute.cast(120.0, spec["max_iter"]), int)
    assert permute.cast(3.0, spec["max_depth"]) == 3 and isinstance(permute.cast(3.0, spec["max_depth"]), int)
    assert permute.cast("0.25", spec["learning_rate"]) == 0.25
    assert permute.cast("b", spec["act"]) == "b"


# ----------------------------------------------------------------------------------------
# The whole procedure
# ----------------------------------------------------------------------------------------

@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(guard, "assert_test_allowed", allow)


def trained(env, task, model):
    train.run_model(env.cfg, task, "lopo", model, log=quiet)
    return env


def permute_args(env):
    return {"guard_check": allow, "selection_check": no_selection_check, "log": quiet}


def test_a_tabular_permutation_run_writes_one_statistic_per_permutation(tmp_path, frozen):
    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    result = permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=5, **permute_args(env))
    assert result["status"] == "complete"
    table = pd.read_csv(env.live.permutation("T1", "ridge_time_wq"))
    assert list(table["r"]) == [0, 1, 2, 3, 4] and list(table.columns) == permute.PART_COLUMNS
    assert np.isfinite(table["S_perm"]).all() and table["S_perm"].nunique() == 5
    expected = permute.describe(env.cfg, "T1")
    assert (table["n_dates"] == expected["n_dates"]).all() and (table["n_dates_unchanged"] == expected["n_dates_unchanged"]).all()
    assert (table["n_rows_changed"] == expected["n_rows_changed_in_permutation_0"]).all()
    meta = json.loads(env.live.permutation("T1", "ridge_time_wq").with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["R"] == 5 and meta["statistic"] == "S" and meta["complete"] and meta["fit_seed"] == 0
    saved = pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time_wq"))
    errors = np.abs(saved["y_pred_log"] - saved["y_true_log"])
    s_obs = errors.groupby(saved["pond"]).mean().mean()
    assert meta["observed_check"]["observed_S"] == pytest.approx(s_obs)
    assert not env.live.permutation("T1", "ridge_time_wq").with_suffix(".parts").exists()
    log = pd.read_csv(env.live.runtime_log(), dtype=str, keep_default_na=False)
    assert set(log.loc[log["scheme"] == "permutation", "stage"]) == {"check", "permutation"}


def test_the_permuted_statistic_is_the_one_a_hand_replay_of_that_permutation_gives(tmp_path, frozen):
    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, **permute_args(env))
    table = pd.read_csv(env.live.permutation("T1", "ridge_time_wq"))
    # replay permutation 2 with the library pieces, in a different order of calls
    cfg, t1, wq = env.cfg, env.tables.t1, env.tables.wq
    rows = select_samples(t1, cfg["design"]["cohort"], "eligible")
    donors, _ = permute.permuted_donors(rows, "T1", cfg, np.random.default_rng(cfg["permutation"]["seed"] + 2))
    table2 = permute.permuted_table(t1, rows, donors, wq, "T1", cfg)
    plans = [p for p in plan_folds("T1", "lopo", t1, "eligible", cfg) if not p.skipped]
    chosen = pd.read_csv(env.live.tuning("T1", "lopo", "ridge_time_wq") / "chosen.csv").set_index("outer_fold")
    frames = []
    for plan in plans:
        hp = {"alpha": float(chosen.loc[plan.index, "hp_alpha"])}
        model = make_baseline("T1", "ridge_time_wq", cfg, hp, seed=cfg["seeds"]["gbm"])
        model.fit(table2.loc[list(plan.train_idx)])
        test = table2.loc[list(plan.test_idx)]
        frames.append(pd.DataFrame({"pond": test["pond"].to_numpy(), "e": np.abs(model.predict_log(test) - test["log_w"].to_numpy())}))
    s = pd.concat(frames).groupby("pond")["e"].mean().mean()
    assert table.loc[2, "S_perm"] == pytest.approx(s)


def test_a_finished_test_is_not_repeated_and_other_inputs_are_refused(tmp_path, frozen):
    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, **permute_args(env))
    path = env.live.permutation("T1", "ridge_time_wq")
    before = path.read_bytes()
    assert permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, **permute_args(env))["status"] == "no-op"
    assert path.read_bytes() == before
    with pytest.raises(permute.PermutationError, match="another config or other inputs"):
        permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=4, **permute_args(env))


def test_a_run_that_stopped_resumes_from_its_finished_permutations(tmp_path, frozen, monkeypatch):
    env = trained(make_env(tmp_path / "a"), "T1", "ridge_time_wq")
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=6, **permute_args(env))
    clean = pd.read_csv(env.live.permutation("T1", "ridge_time_wq"))
    env2 = trained(make_env(tmp_path / "b"), "T1", "ridge_time_wq")
    real = permute.permutation_statistic
    calls = []

    def dying(task, predictions):
        calls.append(1)
        if len(calls) == 5:                      # call 1 is the check of the observed fits
            raise RuntimeError("the machine went away")
        return real(task, predictions)

    monkeypatch.setattr(permute, "permutation_statistic", dying)
    with pytest.raises(RuntimeError, match="went away"):
        permute.run_permutations(env2.cfg, "T1", "ridge_time_wq", n_perm=6, **permute_args(env2))
    parts = env2.live.permutation("T1", "ridge_time_wq").with_suffix(".parts")
    assert sorted(p.name for p in parts.glob("r*.json")) == ["r0000.json", "r0001.json", "r0002.json"]
    assert not env2.live.permutation("T1", "ridge_time_wq").exists()
    monkeypatch.setattr(permute, "permutation_statistic", real)
    permute.run_permutations(env2.cfg, "T1", "ridge_time_wq", n_perm=6, **permute_args(env2))
    pd.testing.assert_frame_equal(pd.read_csv(env2.live.permutation("T1", "ridge_time_wq")), clean)


def test_parts_of_another_run_are_refused(tmp_path, frozen):
    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    parts = env.live.permutation("T1", "ridge_time_wq").with_suffix(".parts")
    write_json_atomic(parts / "r0000.json", {"r": 0, "S_perm": 0.1, "hash": "stale", "seconds": 1.0})
    with pytest.raises(permute.PermutationError, match="belongs to another run"):
        permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, **permute_args(env))


def test_the_guard_is_checked_before_every_permutation(tmp_path, frozen, monkeypatch):
    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    monkeypatch.setenv("P4_GUARD_COUNTER", str(tmp_path / "counter"))
    args = {**permute_args(env), "guard_check": refuse_after_two}
    # one call in run_permutations, then one per permutation: the second permutation is refused
    with pytest.raises(guard.GuardError, match="tree changed"):
        permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=5, **args)
    parts = env.live.permutation("T1", "ridge_time_wq").with_suffix(".parts")
    assert [p.name for p in parts.glob("r*.json")] == ["r0000.json"]


def test_the_real_guard_and_the_selection_rule_stop_the_run_before_anything_is_written(tmp_path, tmp_path_factory, monkeypatch):
    env = trained_unguarded(make_env(tmp_path), monkeypatch)
    monkeypatch.undo()
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path_factory.mktemp("not_a_repo"))
    with pytest.raises(guard.GuardError):
        permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, log=quiet)
    assert not env.live.permutation("T1", "ridge_time_wq").parent.exists()


def trained_unguarded(env, monkeypatch):
    monkeypatch.setattr(guard, "assert_test_allowed", allow)
    return trained(env, "T1", "ridge_time_wq")


def test_only_the_models_of_the_plan_are_permuted(tmp_path, frozen):
    env = make_env(tmp_path)
    layout = env.live
    for model in ("gc", "lstm", "ridge_time"):
        with pytest.raises(permute.PermutationError, match="neither a permutation model"):
            permute.model_kind_for_permutation(env.cfg, layout, "T1", model)
    write_json_atomic(layout.selection("T1"), {"primary_baseline": "ridge_time", "primary_deep": "gru"})
    assert permute.model_kind_for_permutation(env.cfg, layout, "T1", "gru") == ("deep", 199, 0)
    assert permute.model_kind_for_permutation(env.cfg, layout, "T1", "gbm_time_wq") == ("tabular", 999, 0)
    assert permute.model_kind_for_permutation(env.cfg, layout, "T2", "ridge_pw_wq") == ("tabular", 999, 0)
    with pytest.raises(permute.PermutationError, match="neither a permutation model"):
        permute.model_kind_for_permutation(env.cfg, layout, "T1", "lstm")


def test_a_repeat_of_the_observed_fits_that_differs_from_the_saved_predictions_stops_the_run(tmp_path, frozen):
    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    path = env.live.predictions("T1", "lopo", "ridge_time_wq")
    frame = pd.read_csv(path)
    frame.loc[0, "y_pred_log"] += 1e-3
    frame.to_csv(path, index=False)
    with pytest.raises(permute.PermutationError, match="does not reproduce the saved predictions"):
        permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, **permute_args(env))
    assert not env.live.permutation("T1", "ridge_time_wq").exists()


def test_two_runs_cannot_overlap(tmp_path, frozen):
    import fcntl

    env = trained(make_env(tmp_path), "T1", "ridge_time_wq")
    lock = env.live.permutation("T1", "ridge_time_wq").with_suffix(".lock")
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(permute.PermutationError, match="another process"):
            permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=3, **permute_args(env))


def test_workers_give_the_same_statistics_as_the_main_process(tmp_path, frozen):
    env = trained(make_env(tmp_path / "a"), "T1", "ridge_time_wq")
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=6, **permute_args(env))
    single = pd.read_csv(env.live.permutation("T1", "ridge_time_wq"))
    shutil_target = env.live.permutation("T1", "ridge_time_wq")
    for suffix in (".csv", ".json"):
        shutil_target.with_suffix(suffix).unlink()
    messages = []
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=6, workers=2, guard_check=allow,
                             selection_check=no_selection_check, log=messages.append)
    pd.testing.assert_frame_equal(pd.read_csv(env.live.permutation("T1", "ridge_time_wq")), single)
    assert sum("finished" in m for m in messages) >= 2
    meta = json.loads(env.live.permutation("T1", "ridge_time_wq").with_suffix(".json").read_text(encoding="utf-8"))
    assert meta["processes"] and os.getpid() not in meta["processes"]                  # the work ran in worker processes


# ----------------------------------------------------------------------------------------
# A target that depends on the water quality
# ----------------------------------------------------------------------------------------

def make_dependent(inputs):
    """T1 weights driven by the temperature window feature (and time): the permutation must lose that."""
    t1 = inputs.t1.copy()
    temperature = t1["temperature_mean_w"]
    signal = (temperature.fillna(temperature.mean()) - temperature.mean()) / temperature.std()
    t1["log_w"] = 3.0 + 0.01 * t1["t"] + 0.8 * signal
    t1["w"] = np.exp(t1["log_w"])
    return inputs._replace(t1=t1)


def test_when_the_target_depends_on_the_water_quality_the_permuted_runs_are_worse_than_the_observed_one(tmp_path, frozen):
    env = trained(make_env(tmp_path, n_events=9, edit=make_dependent), "T1", "ridge_time_wq")
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=19, **permute_args(env))
    permuted = pd.read_csv(env.live.permutation("T1", "ridge_time_wq"))["S_perm"]
    saved = pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time_wq"))
    s_obs = (saved["y_pred_log"] - saved["y_true_log"]).abs().groupby(saved["pond"]).mean().mean()
    assert s_obs < permuted.min()                          # no permuted run does as well: p = 1/20
    assert permuted.median() > 2 * s_obs


def test_when_the_target_ignores_the_water_quality_the_permuted_runs_straddle_the_observed_one(tmp_path, frozen):
    env = trained(make_env(tmp_path, n_events=9), "T1", "ridge_time_wq")
    permute.run_permutations(env.cfg, "T1", "ridge_time_wq", n_perm=19, **permute_args(env))
    permuted = pd.read_csv(env.live.permutation("T1", "ridge_time_wq"))["S_perm"]
    saved = pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time_wq"))
    s_obs = (saved["y_pred_log"] - saved["y_true_log"]).abs().groupby(saved["pond"]).mean().mean()
    assert permuted.min() < s_obs * 1.2 and permuted.max() > s_obs * 0.8     # nothing separates them


# ----------------------------------------------------------------------------------------
# Neural models, T2 and the structure report
# ----------------------------------------------------------------------------------------

def test_the_primary_deep_model_is_permuted_with_the_deep_seed_and_reproduces_across_processes(tmp_path, frozen):
    env = make_env(tmp_path)
    write_json_atomic(env.live.selection("T2"), {"primary_baseline": "sgr", "primary_deep": "gru"})
    trained(env, "T2", "gru")
    plans = [p for p in plan_folds("T2", "lopo", env.tables.t2, "eligible", env.cfg) if not p.skipped]
    chosen = permute.load_chosen(env.live, env.cfg, "T2", "gru", plans)
    assert all(isinstance(epochs, int) and 1 <= epochs <= 6 for _hp, epochs in chosen.values())
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        elsewhere = pool.submit(permute.check_observed, env.cfg, env.live, "T2", "gru", chosen, 0).result()
    here = permute.check_observed(env.cfg, env.live, "T2", "gru", chosen, 0)           # both match the saved predictions
    assert elsewhere["observed_S"] == pytest.approx(here["observed_S"])
    permute.run_permutations(env.cfg, "T2", "gru", n_perm=2, **permute_args(env))
    table = pd.read_csv(env.live.permutation("T2", "gru"))
    meta = json.loads(env.live.permutation("T2", "gru").with_suffix(".json").read_text(encoding="utf-8"))
    assert list(table["r"]) == [0, 1] and np.isfinite(table["S_perm"]).all()
    assert meta["kind"] == "deep" and meta["fit_seed"] == 0 and meta["statistic"] == "S_h1"


def test_the_structure_report_counts_dates_units_and_changed_samples(tmp_path):
    env = make_env(tmp_path)
    for task in ("T1", "T2"):
        report = permute.describe(env.cfg, task)
        rows = select_samples(env.tables.table(task), env.cfg["design"]["cohort"], "eligible")
        assert report["n_eligible_samples"] == len(rows)
        assert sum(report["units_per_date"].values()) == report["n_dates"]
        assert report["n_dates_unchanged"] == report["units_per_date"].get(1, 0)
        assert report["n_rows_changed_in_permutation_0"] <= len(rows)
    assert permute.main(["--task", "T1", "--describe", "--config", str(write_cfg(env))]) == 0


def write_cfg(env):
    import yaml

    path = env.root / "cfg.yaml"
    path.write_text(yaml.safe_dump(env.cfg), encoding="utf-8")
    return path


def test_the_command_line_refuses_a_missing_model(tmp_path, capsys):
    env = make_env(tmp_path)
    assert permute.main(["--task", "T1", "--config", str(write_cfg(env))]) == 2
    assert "refused" in capsys.readouterr().err
