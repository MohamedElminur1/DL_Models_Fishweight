"""Unit tests for src/models: sequences and scalers, architectures, shared trainer, wrapper.

All data are the synthetic tables of tests/synth_p4.py (test-only constructions). Budgets are
tiny so the file runs in seconds on one core.
"""

import copy
import math

import numpy as np
import pandas as pd
import pytest
import torch

from src.folds import load_config, select_samples
from src.models import DEEP_MODELS, NETWORK_MODELS, build_network
from src.models import common
from src.models.common import (
    DailyStore, DeepRegressor, build_sampleset, fit_scalers, gather_last, last_real_index,
    make_batch, train_network,
)
from tests.synth_p4 import DAY0, SENSORS, STATS, make_tables

CFG = load_config()
BUDGET = {"max_epochs": 6, "patience": 3}
HP = {"dropout": 0.1, "lr": 0.01, "weight_decay": 1e-4, "batch_size": 8}
ARCH_HP = {
    "lstm": {"hidden": 8, "layers": 1},
    "gru": {"hidden": 8, "layers": 2},
    "cnn_lstm": {"hidden": 8, "layers": 1, "filters": 8, "kernel": 3, "pool": 2},
    "tft": {"hidden": 8, "heads": 2},
    "mlp_static": {"hidden": 8},
}
common.configure_torch()


@pytest.fixture(scope="module")
def world():
    t1, t2, daily = make_tables(n_events=6, seed=3, sparse_ponds=(8,))
    store = DailyStore(daily, SENSORS, STATS)
    t1e, t2e = select_samples(t1, [1, 2, 3, 4, 6, 7, 8, 9], "eligible"), select_samples(t2, [1, 2, 3, 4, 6, 7, 8, 9], "eligible")
    return {"t1": t1e, "t2": t2e, "daily": daily, "store": store,
            "ss1": build_sampleset("T1", t1e, store, CFG), "ss2": build_sampleset("T2", t2e, store, CFG)}


def store_value(world, pond, date, sensor, stat, reference):
    """Daily statistic and visibility straight from the daily table, bypassing DailyStore.

    Visible means: observed, or interpolated with a right anchor before the `reference` date
    (the causal gap rule of plan 2.7, written out independently of the module).
    """
    d = world["daily"]
    row = d[(d["pond"] == pond) & (d["date"] == pd.Timestamp(date).strftime("%Y-%m-%d")) & (d["sensor"] == sensor)]
    assert len(row) == 1
    row = row.iloc[0]
    visible = row["status"] == "observed" or (
        row["status"] == "interpolated" and pd.Timestamp(row["right_anchor_date"]) < pd.Timestamp(reference))
    return row[stat], bool(visible)


# ------------------------------------------------------------------------ daily store

def test_daily_store_marks_valid_sensor_days_from_the_status(world):
    store, daily = world["store"], world["daily"]
    assert store.vals.shape[2:] == (6, 4)
    assert store.valid.sum() == (daily["status"] != "missing").sum()  # the store keeps every non-missing cell
    row = daily[(daily["status"] != "missing")].iloc[5]
    day = (pd.Timestamp(row["date"]) - store.start).days
    s = SENSORS.index(row["sensor"])
    assert store.valid[store.pond_index[row["pond"]], day, s]
    assert store.vals[store.pond_index[row["pond"]], day, s, 0] == row["mean"]


def test_daily_store_records_the_right_anchor_day_of_interpolated_sensor_days(world):
    store, daily = world["store"], world["daily"]
    interpolated = daily[daily["status"] == "interpolated"]
    assert len(interpolated) > 20  # the synthetic tables contain many gap days
    assert (store.right_anchor >= 0).sum() == len(interpolated)
    row = interpolated.iloc[3]
    day = (pd.Timestamp(row["date"]) - store.start).days
    anchor_day = (pd.Timestamp(row["right_anchor_date"]) - store.start).days
    assert store.right_anchor[store.pond_index[row["pond"]], day, SENSORS.index(row["sensor"])] == anchor_day
    assert anchor_day > day
    observed = daily[daily["status"] == "observed"].iloc[0]
    assert store.right_anchor[store.pond_index[observed["pond"]],
                              (pd.Timestamp(observed["date"]) - store.start).days,
                              SENSORS.index(observed["sensor"])] == -1


def test_daily_store_rejects_inconsistent_anchors(world):
    daily = world["daily"]
    no_anchor = daily.copy()
    no_anchor.loc[no_anchor.index[no_anchor["status"] == "interpolated"][0], "right_anchor_date"] = None
    with pytest.raises(ValueError, match="no right anchor"):
        DailyStore(no_anchor, SENSORS, STATS)
    stray = daily.copy()
    stray.loc[stray.index[stray["status"] == "observed"][0], "right_anchor_date"] = "2021-07-01"
    with pytest.raises(ValueError, match="not interpolated"):
        DailyStore(stray, SENSORS, STATS)
    backwards = daily.copy()
    first = backwards.index[backwards["status"] == "interpolated"][0]
    backwards.loc[first, "right_anchor_date"] = backwards.loc[first, "date"]
    with pytest.raises(ValueError, match="not after"):
        DailyStore(backwards, SENSORS, STATS)
    with pytest.raises(ValueError, match="lacks columns"):
        DailyStore(daily.drop(columns="right_anchor_date"), SENSORS, STATS)


def test_daily_store_rejects_malformed_tables(world):
    daily = world["daily"]
    with pytest.raises(ValueError, match="more than one row"):
        DailyStore(pd.concat([daily, daily.iloc[:1]]), SENSORS, STATS)
    bad = daily.copy()
    bad.loc[bad.index[bad["status"] == "missing"][0], "mean"] = 1.0
    with pytest.raises(ValueError, match="missing"):
        DailyStore(bad, SENSORS, STATS)
    bad = daily.copy()
    bad.loc[bad.index[bad["status"] == "observed"][0], "sd"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        DailyStore(bad, SENSORS, STATS)
    with pytest.raises(ValueError, match="unknown sensors"):
        DailyStore(daily.assign(sensor=daily["sensor"].replace("ph", "acidity")), SENSORS, STATS)
    with pytest.raises(ValueError, match="status"):
        DailyStore(daily.assign(status=daily["status"].replace("observed", "ok")), SENSORS, STATS)


# --------------------------------------------------------------------------- sequences

def test_t1_sequences_run_from_day_zero_to_the_day_before_the_event_and_are_left_padded(world):
    ss, table, store = world["ss1"], world["t1"], world["store"]
    assert ss.length == table["t"].max()
    for i in (0, len(table) // 2, len(table) - 1):
        row = table.iloc[i]
        t = int(row["t"])
        real = ~ss.pad[i]
        assert real.sum() == t and real[-t:].all() and not real[:-t].any() if t < ss.length else real.all()
        assert not ss.valid[i][ss.pad[i]].any()  # masks are 0 on padding
        first_date = DAY0
        last_date = pd.Timestamp(row["date"]) - pd.Timedelta(days=1)
        for date, step in ((first_date, ss.length - t), (last_date, ss.length - 1)):
            for s, sensor in enumerate(SENSORS):
                expected, ok = store_value(world, row["pond"], date, sensor, "mean", row["date"])
                assert ss.valid[i, step, s] == ok
                if ok:
                    assert ss.vals[i, step, s, 0] == pytest.approx(expected)
    assert (ss.statics[:, 0] == table["t"].to_numpy()).all()
    np.testing.assert_allclose(ss.y, table["log_w"])


def test_t2_sequences_are_the_28_days_before_the_origin_and_unpadded(world):
    ss, table = world["ss2"], world["t2"]
    assert ss.length == 28 and not ss.pad.any()
    for i in (0, len(table) // 2, len(table) - 1):
        row = table.iloc[i]
        origin = pd.Timestamp(row["origin_date"])
        for step in (0, 13, 27):
            date = origin - pd.Timedelta(days=28 - step)
            for s, sensor in enumerate(SENSORS):
                expected, ok = store_value(world, row["pond"], date, sensor, "max", row["origin_date"])
                assert ss.valid[i, step, s] == ok
                if ok:
                    assert ss.vals[i, step, s, 3] == pytest.approx(expected)
    np.testing.assert_allclose(ss.statics[:, 0], table["log_w_k"])
    np.testing.assert_allclose(ss.statics[:, 4], table["dt_target"])
    np.testing.assert_allclose(ss.y, table["y"])


def test_sample_set_rejects_inconsistent_inputs(world):
    bad = world["t1"].copy()
    bad.loc[bad.index[0], "t"] += 1
    with pytest.raises(ValueError, match="days since day 0"):
        build_sampleset("T1", bad, world["store"], CFG)
    early = world["t2"].copy()
    early.loc[early.index[0], "origin_date"] = "2021-05-30"  # window starts before the table
    with pytest.raises(ValueError, match="outside the daily table"):
        build_sampleset("T2", early, world["store"], CFG)
    with pytest.raises(KeyError):
        world["ss1"].rows([10 ** 6])


# ---------------------------------------------------------------- causal gap rule (plan 2.7)

def tiny_store(right_anchors):
    """One pond, one sensor, ten days from 2021-06-01. Days listed in `right_anchors`
    ({day: right anchor day}) are interpolated, the others observed."""
    rows = []
    for day in range(10):
        status = "interpolated" if day in right_anchors else "observed"
        date = pd.Timestamp("2021-06-01") + pd.Timedelta(days=day)
        rows.append({
            "pond": 1, "date": date.strftime("%Y-%m-%d"), "sensor": "temperature", "status": status,
            "mean": 10.0 + day, "sd": 1.0, "min": 9.0, "max": 11.0,
            "right_anchor_date": (pd.Timestamp("2021-06-01") + pd.Timedelta(days=right_anchors[day])
                                  ).strftime("%Y-%m-%d") if day in right_anchors else None,
        })
    return DailyStore(pd.DataFrame(rows), ["temperature"], STATS)


def test_interpolated_day_is_missing_when_its_right_anchor_is_the_reference_day_or_later():
    store = tiny_store({4: 6, 5: 6})  # gap days 4 and 5, observed again on day 6
    for reference, hidden in ((5, [4]), (6, [4, 5]), (7, [])):
        vals, valid = store.visible_window(0, 2, reference)  # days 2 .. reference - 1
        shown = [2 + i for i in range(len(valid)) if not valid[i, 0]]
        assert shown == hidden, (reference, shown)
        for day in hidden:
            assert np.isnan(vals[day - 2]).all()  # value NaN, so it becomes 0 with mask 0 in the batch
        for day in set(range(2, reference)) - set(hidden):
            assert valid[day - 2, 0] and vals[day - 2, 0, 0] == 10.0 + day
    # the store itself still holds the interpolated cells
    assert store.valid[0, 4:6, 0].all()


def test_observed_days_are_never_masked_and_the_window_is_a_copy():
    store = tiny_store({4: 6, 5: 6})
    vals, valid = store.visible_window(0, 0, 6)
    assert valid[[0, 1, 2, 3], 0].all()
    vals[...] = -1.0
    assert store.vals[0, 0, 0, 0] == 10.0  # the stored arrays were not changed


def sample_cells(world, task):
    """(row position, step, sensor) of sensor-days that the store has but the sample cannot see."""
    ss = world["ss1"] if task == "T1" else world["ss2"]
    cells = []
    for i in range(len(ss.ids)):
        store_valid = ss.store.valid[ss.data_pond_idx[i]][np.maximum(ss.day[i], 0)]
        hidden = store_valid & ~ss.valid[i] & ~ss.pad[i][:, None]
        cells += [(i, step, sensor) for step, sensor in np.argwhere(hidden)]
    return cells


@pytest.mark.parametrize("task", ["T1", "T2"])
def test_sequences_hide_interpolated_days_that_close_on_or_after_the_sample_date(world, task):
    ss = world["ss1"] if task == "T1" else world["ss2"]
    table = world["t1"] if task == "T1" else world["t2"]
    daily = world["daily"]
    cells = sample_cells(world, task)
    assert len(cells) > 5  # the synthetic data contain the situation
    date_col = "date" if task == "T1" else "origin_date"
    for i, step, sensor in cells[:12]:
        row = table.iloc[i]
        day = ss.store.start + pd.Timedelta(days=int(ss.day[i, step]))
        source = daily[(daily["pond"] == row["pond"]) & (daily["date"] == day.strftime("%Y-%m-%d"))
                       & (daily["sensor"] == SENSORS[sensor])].iloc[0]
        assert source["status"] == "interpolated"
        assert pd.Timestamp(source["right_anchor_date"]) >= pd.Timestamp(row[date_col])
        assert not ss.valid[i, step, sensor] and np.isnan(ss.vals[i, step, sensor]).all()
    # nothing is hidden that the rule allows: every other non-missing cell is in the sequence
    visible_rule = ss.store.valid[ss.data_pond_idx[:, None], np.maximum(ss.day, 0)] & ~ss.pad[..., None]
    hidden_total = (visible_rule & ~ss.valid).sum()
    assert hidden_total == len(cells)


def test_a_masked_cell_is_zero_with_mask_zero_in_the_batch(world):
    ss, table = world["ss1"], world["t1"]
    i, step, sensor = sample_cells(world, "T1")[0]
    batch = make_batch(ss, [table.index[i]], fit_scalers(ss, table.index.to_numpy()))
    x = batch.x.numpy()[0]
    assert (x[step, sensor * 4:(sensor + 1) * 4] == 0).all() and x[step, 24 + sensor] == 0


def window_features_from_sequence(ss, i, length):
    """The 24 window features of the last `length` steps of sample i, from the sequence alone."""
    out = {}
    for s, sensor in enumerate(SENSORS):
        ok = ss.valid[i, -length:, s]
        block = ss.vals[i, -length:, s][ok]
        found = ({"mean": block[:, 0].mean(), "sd": block[:, 1].mean(), "min": block[:, 2].min(),
                  "max": block[:, 3].max()} if ok.sum() >= 3 else dict.fromkeys(STATS, np.nan))
        out.update({f"{sensor}_{stat}_w": found[stat] for stat in STATS})
    return out


@pytest.mark.parametrize("task, length", [("T1", 14), ("T2", 28)])
def test_sequence_and_table_agree_on_the_window_features(world, task, length):
    """Two independent implementations of plan 2.7 (the table's and the store's) must agree."""
    ss = world["ss1"] if task == "T1" else world["ss2"]
    table = world["t1"] if task == "T1" else world["t2"]
    for i in range(len(table)):
        expected = window_features_from_sequence(ss, i, length)
        for name, value in expected.items():
            assert table.iloc[i][name] == pytest.approx(value, nan_ok=True), (i, name)


def test_donor_sequences_come_from_the_donor_file_at_the_sample_date(world):
    store, table = world["store"], world["t2"]
    origin = table["origin_date"].iloc[0]
    same_date = table[(table["origin_date"] == origin) & (table["horizon"] == 1)]
    assert len(same_date) >= 3
    ponds = same_date["pond"].to_numpy()
    donors = np.roll(ponds, 1)  # every sample takes the previous sample's pond
    sub = same_date.copy()
    ss_own = build_sampleset("T2", sub, store, CFG)
    ss_donor = build_sampleset("T2", sub, store, CFG, donors=donors)
    for k in range(len(sub)):
        j = int(np.flatnonzero(ponds == donors[k])[0])
        np.testing.assert_array_equal(ss_donor.vals[k], ss_own.vals[j])  # the donor's own sequence
        np.testing.assert_array_equal(ss_donor.valid[k], ss_own.valid[j])
        assert ss_donor.data_pond_idx[k] == store.pond_index[donors[k]]
        assert ss_donor.pond_idx[k] == store.pond_index[ponds[k]]
    np.testing.assert_array_equal(ss_donor.statics, ss_own.statics)  # statics and targets stay own
    np.testing.assert_array_equal(ss_donor.y, ss_own.y)
    same = build_sampleset("T2", sub, store, CFG, donors=ponds)
    np.testing.assert_array_equal(same.vals, ss_own.vals)
    with pytest.raises(ValueError, match="one pond id per table row"):
        build_sampleset("T2", sub, store, CFG, donors=ponds[:-1])


def test_donor_sequences_apply_the_gap_rule_at_the_sample_date_not_the_donors_own(world):
    """A donor day that closes after the sample date is hidden, whichever pond it belongs to."""
    store, daily, t2 = world["store"], world["daily"], world["t2"]
    for _, row in t2[t2["horizon"] == 1].iterrows():
        donor = 3 if row["pond"] != 3 else 4
        ss = build_sampleset("T2", t2.loc[[row.name]], store, CFG, donors=[donor])
        origin = pd.Timestamp(row["origin_date"])
        window = daily[(daily["pond"] == donor) & (daily["date"] >= (origin - pd.Timedelta(days=28)).strftime("%Y-%m-%d"))
                       & (daily["date"] < origin.strftime("%Y-%m-%d"))]
        late = window[(window["status"] == "interpolated")
                      & (pd.to_datetime(window["right_anchor_date"]) >= origin)]
        for _, cell in late.iterrows():
            step = (pd.Timestamp(cell["date"]) - (origin - pd.Timedelta(days=28))).days
            assert not ss.valid[0, step, SENSORS.index(cell["sensor"])]
        if len(late):
            return
    pytest.fail("no donor window with a closing gap in the synthetic data")


# ------------------------------------------------------------------------------ scalers

def expected_channel_stats(world, ss_table, task):
    """Mean and SD of each channel over the sensor-days that some training sample can see.

    A cell is counted once per pond file, date and sensor, and only if at least one training
    sample sees it under the causal gap rule (written out here from the daily table).
    """
    daily = world["daily"]
    visible = set()
    for _, row in ss_table.iterrows():
        if task == "T2":
            reference = pd.Timestamp(row["origin_date"])
            first = reference - pd.Timedelta(days=28)
        else:
            reference, first = pd.Timestamp(row["date"]), DAY0
        days = {d.strftime("%Y-%m-%d") for d in pd.date_range(first, reference - pd.Timedelta(days=1))}
        sub = daily[(daily["pond"] == row["pond"]) & daily["date"].isin(days) & (daily["status"] != "missing")]
        for index, cell in sub.iterrows():
            if cell["status"] == "interpolated" and pd.Timestamp(cell["right_anchor_date"]) >= reference:
                continue
            visible.add(index)
    frame = daily.loc[sorted(visible)]
    out = np.zeros((6, 4)), np.zeros((6, 4))
    for s, sensor in enumerate(SENSORS):
        sub = frame[frame["sensor"] == sensor]
        for k, stat in enumerate(STATS):
            out[0][s, k], out[1][s, k] = sub[stat].mean(), sub[stat].std(ddof=0)
    return out


def test_scalers_use_each_pond_day_once_and_only_training_samples(world):
    for task, ss, table in (("T2", world["ss2"], world["t2"]), ("T1", world["ss1"], world["t1"])):
        train = table[table["pond"].isin([3, 4, 6])]
        sc = fit_scalers(ss, train.index.to_numpy())
        mean, std = expected_channel_stats(world, train, task)
        np.testing.assert_allclose(sc.ch_mean, mean, rtol=1e-9)
        np.testing.assert_allclose(sc.ch_std, std, rtol=1e-9)
        key = "log_w" if task == "T1" else "y"
        assert sc.y_mean == pytest.approx(train[key].mean())
        assert sc.y_std == pytest.approx(train[key].std(ddof=0))


def test_scalers_ignore_held_out_ponds(world):
    ss, table = world["ss2"], world["t2"]
    train_ids = table.index[table["pond"].isin([3, 4, 6])].to_numpy()
    before = fit_scalers(ss, train_ids)
    changed = copy.deepcopy(ss)
    held_out = np.isin(changed.pond_idx, [world["store"].pond_index[7]])
    changed.vals[held_out] = 1e6
    changed.statics[held_out] = 1e6
    changed.y[held_out] = 1e6
    store = copy.deepcopy(ss.store)
    store.vals[store.pond_index[7]] = 1e6
    changed.store = store
    after = fit_scalers(changed, train_ids)
    for field in ("ch_mean", "ch_std", "st_mean", "st_std"):
        np.testing.assert_array_equal(getattr(before, field), getattr(after, field))
    assert before.y_mean == after.y_mean and before.y_std == after.y_std


def test_scalers_ignore_cells_that_no_training_sample_can_see(world):
    ss, table = world["ss2"], world["t2"]
    i, step, sensor = next(c for c in sample_cells(world, "T2") if c[1] == ss.length - 1)
    only = [table.index[i]]  # a single training sample: the hidden cell is invisible to it
    before = fit_scalers(ss, only)
    store = copy.deepcopy(ss.store)
    day, pond = int(ss.day[i, step]), int(ss.data_pond_idx[i])
    store.vals[pond, day, sensor] = 1e6  # the hidden cell
    hidden_changed = copy.deepcopy(ss)
    hidden_changed.store = store
    after = fit_scalers(hidden_changed, only)
    np.testing.assert_array_equal(before.ch_mean, after.ch_mean)
    np.testing.assert_array_equal(before.ch_std, after.ch_std)
    # control: a cell the sample does see does move the statistics
    seen = np.argwhere(ss.valid[i] & ~ss.pad[i][:, None])[0]
    store2 = copy.deepcopy(ss.store)
    store2.vals[pond, int(ss.day[i, seen[0]]), seen[1]] = 1e6
    visible_changed = copy.deepcopy(ss)
    visible_changed.store = store2
    assert not np.array_equal(fit_scalers(visible_changed, only).ch_mean, before.ch_mean)


def test_batch_layout_missing_values_padding_and_standardization(world):
    ss, table = world["ss1"], world["t1"]
    ids = table.index.to_numpy()
    sc = fit_scalers(ss, ids)
    batch = make_batch(ss, ids, sc)
    n, length = len(table), ss.length
    assert batch.x.shape == (n, length, 30) and batch.static.shape == (n, 1) and batch.pad.shape == (n, length)
    x = batch.x.numpy()
    masks = x[..., 24:]
    assert set(np.unique(masks)) <= {0.0, 1.0}
    np.testing.assert_array_equal(masks, ss.valid.astype(float))
    aggregates = x[..., :24].reshape(n, length, 6, 4)
    assert (aggregates[~ss.valid] == 0).all()                   # missing and padded steps are 0
    assert (x[ss.pad] == 0).all()                               # padding is all zeros, masks included
    i, step, s = np.argwhere(ss.valid)[7]
    np.testing.assert_allclose(aggregates[i, step, s], (ss.vals[i, step, s] - sc.ch_mean[s]) / sc.ch_std[s])
    np.testing.assert_allclose(batch.static.numpy().mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(batch.static.numpy().std(), 1.0, atol=1e-5)
    np.testing.assert_allclose(batch.y.numpy().mean(), 0.0, atol=1e-6)


# -------------------------------------------------------------------------- architectures

def build(arch, n_static, **override):
    hp = {**HP, **ARCH_HP[arch], **override}
    return build_network(arch, hp, n_static=n_static, n_sensors=6, n_stats=4)


def toy_batch(batch=5, length=12, n_static=2, pad_left=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(batch, length, 30, generator=g)
    pad = torch.zeros(batch, length, dtype=torch.bool)
    pad[:, :pad_left] = True
    x[pad] = 0.0
    x[..., 24:] = (x[..., 24:] > 0).float() * (~pad).unsqueeze(-1)
    return x, torch.randn(batch, n_static, generator=g), pad


@pytest.mark.parametrize("arch", DEEP_MODELS)
def test_every_architecture_returns_one_finite_number_per_sample_and_trains(arch):
    net = build(arch, n_static=2)
    x, static, pad = toy_batch()
    out = net(x, static, pad)
    assert out.shape == (5,) and torch.isfinite(out).all()
    out.pow(2).mean().backward()
    for name, p in net.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


@pytest.mark.parametrize("arch", DEEP_MODELS)
def test_statics_reach_the_output(arch):
    net = build(arch, n_static=2).eval()
    x, static, pad = toy_batch()
    with torch.no_grad():
        assert not torch.allclose(net(x, static, pad), net(x, static + 1.0, pad))


@pytest.mark.parametrize("arch", DEEP_MODELS)
def test_eval_mode_is_deterministic_and_train_mode_uses_dropout(arch):
    net = build(arch, n_static=2, dropout=0.5)
    x, static, pad = toy_batch()
    net.eval()
    with torch.no_grad():
        assert torch.equal(net(x, static, pad), net(x, static, pad))
    net.train()
    with torch.no_grad():
        assert not torch.equal(net(x, static, pad), net(x, static, pad))


def test_architecture_options_of_the_search_space_all_build_and_run():
    x, static, pad = toy_batch(length=9, pad_left=2)  # odd length exercises ceil-mode pooling
    for hidden in (8, 16, 32, 64):
        for layers in (1, 2):
            for arch in ("lstm", "gru"):
                assert build(arch, 2, hidden=hidden, layers=layers)(x, static, pad).shape == (5,)
    for filters in (8, 16, 32):
        for kernel in (3, 5, 7):
            for pool in (1, 2):
                net = build("cnn_lstm", 2, filters=filters, kernel=kernel, pool=pool, layers=2)
                assert net(x, static, pad).shape == (5,)
    for hidden in (8, 16, 32):
        for heads in (1, 2, 4):
            assert build("tft", 2, hidden=hidden, heads=heads)(x, static, pad).shape == (5,)
    with pytest.raises(KeyError):
        build_network("transformer", HP, 2, 6, 4)


def test_mlp_static_uses_the_statics_only_and_is_a_two_layer_head():
    net = build("mlp_static", n_static=5, hidden=16).eval()
    x, static, pad = toy_batch(n_static=5)
    with torch.no_grad():
        out = net(x, static, pad)
        assert out.shape == (5,) and torch.isfinite(out).all()
        assert torch.equal(out, net(torch.randn_like(x), static, ~pad))  # sequence and padding are ignored
        assert not torch.allclose(out, net(x, static + 1.0, pad))
    assert sum(p.numel() for p in net.parameters()) == 5 * 16 + 16 + 16 + 1
    linear = [m for m in net.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linear] == [(5, 16), (16, 1)]
    assert "mlp_static" in NETWORK_MODELS and "mlp_static" not in DEEP_MODELS


def test_cnn_lstm_pooling_keeps_the_most_recent_day_when_the_length_is_odd():
    net = build("cnn_lstm", n_static=2, pool=2, kernel=3).eval()
    x, static, pad = toy_batch(length=9, pad_left=0)
    changed = x.clone()
    changed[:, -1, :24] += 5.0  # only the very last day differs
    with torch.no_grad():
        assert not torch.allclose(net(x, static, pad), net(changed, static, pad))


def test_last_real_index_handles_left_and_right_padding():
    pad = torch.tensor([[True, True, False, False], [False, False, False, False], [False, False, True, True]])
    assert last_real_index(pad).tolist() == [3, 3, 1]
    seq = torch.arange(12.0).reshape(3, 4, 1)
    assert gather_last(seq, pad).flatten().tolist() == [3.0, 7.0, 9.0]


def test_tft_attention_ignores_padded_days_and_selection_weights_sum_to_one():
    net = build("tft", n_static=3, hidden=16, heads=4).eval()
    x, static, pad = toy_batch(n_static=3, pad_left=4)
    with torch.no_grad():
        out, details = net(x, static, pad, return_details=True)
    attention = details["attention"]
    assert attention.shape == (5, 12)
    assert (attention[:, :4] == 0).all()                               # padded days get no attention
    np.testing.assert_allclose(attention.sum(dim=1).numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(details["static_weights"].sum(dim=-1).numpy(), 1.0, atol=1e-6)
    np.testing.assert_allclose(details["temporal_weights"].sum(dim=-1).numpy(), 1.0, atol=1e-6)
    assert details["temporal_weights"].shape == (5, 12, 6)
    with torch.no_grad():
        assert torch.equal(out, net(x, static, pad))  # same path, same numbers


def test_tft_query_is_the_last_real_day():
    """Right-padded input: the prediction must not depend on the padded tail beyond the LSTM."""
    net = build("tft", n_static=2).eval()
    x, static, _ = toy_batch(pad_left=0)
    pad = torch.zeros(5, 12, dtype=torch.bool)
    pad[:, -3:] = True
    with torch.no_grad():
        _, details = net(x, static, pad, return_details=True)
    assert (details["attention"][:, -3:] == 0).all()


def test_tft_with_a_single_static_variable_is_valid():
    net = build("tft", n_static=1).eval()
    x, static, pad = toy_batch(n_static=1)
    with torch.no_grad():
        out, details = net(x, static, pad, return_details=True)
    assert out.shape == (5,)
    np.testing.assert_allclose(details["static_weights"].numpy(), 1.0)


def test_same_seed_gives_identical_weights_and_different_seed_does_not():
    def init(seed):
        common.set_seed(seed)
        return build("lstm", 2)

    a, b, c = init(1), init(1), init(2)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert not all(torch.equal(p, q) for p, q in zip(a.parameters(), c.parameters()))


# ----------------------------------------------------------------------------- trainer

def make_regression(n_train=20, n_val=20, noise=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(30, generator=g)

    def batch(n, shift):
        x = torch.randn(n, 6, 30, generator=g)
        static = torch.zeros(n, 1)
        y = (x.mean(dim=1) @ w) * 0.3 + noise * torch.randn(n, generator=g) + shift
        return common.Batch(x, static, torch.zeros(n, 6, dtype=torch.bool), y)

    return batch(n_train, 0.0), batch(n_val, 0.0)


def fresh_net(seed=0):
    common.set_seed(seed)
    return build("lstm", 1, hidden=16)


def evaluate_mse(net, batch):
    net.eval()
    with torch.no_grad():
        return float(torch.nn.functional.mse_loss(net(batch.x, batch.static, batch.pad), batch.y))


KW = dict(lr=0.02, weight_decay=0.0, batch_size=8)


def test_early_stopping_restores_the_best_epoch_and_stops_after_patience():
    train, val = make_regression(n_train=12, n_val=30, noise=1.0)
    net = fresh_net()
    result = train_network(net, train, val, max_epochs=300, patience=5, fixed_epochs=None, seed=0, **KW)
    val_curve = np.array([row[2] for row in result.curves])
    assert result.best_epoch == int(np.argmin(val_curve)) + 1
    assert result.best_val_mse == pytest.approx(val_curve.min())
    assert evaluate_mse(net, val) == pytest.approx(result.best_val_mse, rel=1e-5)  # weights restored
    assert result.epochs_run == len(result.curves) <= 300
    assert result.epochs_run == result.best_epoch + 5 or result.epochs_run == 300


def test_max_epochs_is_a_hard_limit():
    train, val = make_regression()
    result = train_network(fresh_net(), train, val, max_epochs=4, patience=100, fixed_epochs=None, seed=0, **KW)
    assert result.epochs_run == 4


def test_fixed_epochs_train_exactly_n_epochs_without_validation():
    train, _ = make_regression()
    net = fresh_net()
    result = train_network(net, train, None, max_epochs=300, patience=30, fixed_epochs=7, seed=0, **KW)
    assert result.epochs_run == 7 and result.best_epoch == 7
    assert all(math.isnan(row[2]) for row in result.curves)


def test_trainer_needs_exactly_one_of_validation_or_fixed_epochs():
    train, val = make_regression()
    with pytest.raises(ValueError):
        train_network(fresh_net(), train, val, max_epochs=5, patience=3, fixed_epochs=5, seed=0, **KW)
    with pytest.raises(ValueError):
        train_network(fresh_net(), train, None, max_epochs=5, patience=3, fixed_epochs=None, seed=0, **KW)


def test_curve_rows_have_epoch_losses_and_seconds():
    train, val = make_regression()
    result = train_network(fresh_net(), train, val, max_epochs=6, patience=100, fixed_epochs=None, seed=0, **KW)
    assert [row[0] for row in result.curves] == [1, 2, 3, 4, 5, 6]
    for _, train_mse, val_mse, seconds in result.curves:
        assert train_mse > 0 and val_mse > 0 and seconds >= 0
    assert result.curves[-1][1] < result.curves[0][1]  # the training loss falls on this toy problem


def test_training_is_reproducible_for_a_seed_and_differs_across_seeds():
    train, val = make_regression()
    runs = {}
    for seed in (0, 0, 1):
        net = fresh_net(seed)
        runs.setdefault(seed, []).append(
            [row[1] for row in train_network(net, train, val, max_epochs=5, patience=100,
                                              fixed_epochs=None, seed=seed, **KW).curves]
        )
    assert runs[0][0] == runs[0][1]
    assert runs[0][0] != runs[1][0]


def test_batch_size_controls_the_number_of_updates_per_epoch():
    train, _ = make_regression(n_train=20)
    counts = {}
    for batch_size in (4, 20):
        net = fresh_net()
        steps = []
        original = torch.optim.AdamW.step

        def counting_step(self, *a, **k):
            steps.append(1)
            return original(self, *a, **k)

        torch.optim.AdamW.step = counting_step
        try:
            train_network(net, train, None, max_epochs=3, patience=1, fixed_epochs=2, seed=0,
                          lr=0.01, weight_decay=0.0, batch_size=batch_size)
        finally:
            torch.optim.AdamW.step = original
        counts[batch_size] = len(steps)
    assert counts == {4: 10, 20: 2}


# ----------------------------------------------------------------------- DeepRegressor

@pytest.mark.parametrize("task", ["T1", "T2"])
@pytest.mark.parametrize("arch", NETWORK_MODELS)
def test_deep_regressor_fits_with_validation_then_with_fixed_epochs(world, task, arch):
    ss = world["ss1"] if task == "T1" else world["ss2"]
    table = world["t1"] if task == "T1" else world["t2"]
    train, val = table[table["pond"].isin([1, 2, 3, 4])], table[table["pond"].isin([6, 7])]
    test = table[table["pond"] == 9]
    hp = {**HP, **ARCH_HP[arch]}
    model = DeepRegressor(task, arch, hp, ss, BUDGET, seed=0)
    info = model.fit(train, val)
    assert info.n_train == len(train)
    assert 1 <= info.best_epoch <= info.epochs_run <= BUDGET["max_epochs"]
    assert len(info.curves) == info.epochs_run and all(not math.isnan(row[2]) for row in info.curves)
    log_pred, grams = model.predict_log(test), model.predict_grams(test)
    assert log_pred.shape == (len(test),) and np.isfinite(log_pred).all()
    expected = np.exp(log_pred) if task == "T1" else test["w_origin"].to_numpy() * np.exp(log_pred)
    np.testing.assert_allclose(grams, expected)
    final = DeepRegressor(task, arch, hp, ss, BUDGET, seed=1)
    info = final.fit(pd.concat([train, val]), epochs=3)
    assert info.epochs_run == 3 and len(info.curves) == 3
    assert all(math.isnan(row[2]) for row in info.curves)
    assert np.isfinite(final.predict_grams(test)).all()


def test_deep_regressor_is_reproducible_per_seed(world):
    table = world["t2"]
    train, test = table[table["pond"].isin([1, 2, 3])], table[table["pond"] == 4]
    hp = {**HP, **ARCH_HP["gru"]}

    def run(seed):
        return DeepRegressor("T2", "gru", hp, world["ss2"], BUDGET, seed=seed).fit(train, epochs=4), seed

    first = DeepRegressor("T2", "gru", hp, world["ss2"], BUDGET, seed=0)
    second = DeepRegressor("T2", "gru", hp, world["ss2"], BUDGET, seed=0)
    other = DeepRegressor("T2", "gru", hp, world["ss2"], BUDGET, seed=1)
    for model in (first, second, other):
        model.fit(train, epochs=4)
    np.testing.assert_array_equal(first.predict_log(test), second.predict_log(test))
    assert not np.allclose(first.predict_log(test), other.predict_log(test))


def test_deep_regressor_scalers_come_from_its_training_rows_only(world):
    table = world["t2"]
    train, val = table[table["pond"].isin([1, 2, 3])], table[table["pond"].isin([4, 6])]
    model = DeepRegressor("T2", "lstm", {**HP, **ARCH_HP["lstm"]}, world["ss2"], BUDGET, seed=0)
    model.fit(train, val)
    expected = fit_scalers(world["ss2"], train.index.to_numpy())
    np.testing.assert_array_equal(model.scalers.ch_mean, expected.ch_mean)
    assert model.scalers.y_mean == expected.y_mean
