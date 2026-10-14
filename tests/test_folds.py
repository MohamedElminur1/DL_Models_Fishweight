"""Unit tests for src/folds.py: outer LOPO folds, stream-grouped inner folds, the calendar
hold-out, the cut-off rule and the sample rule. Tables are the synthetic ones of
tests/synth_p4.py or small frames built here; one test reads only the pond, date and
eligibility columns of the real tables."""

from pathlib import Path

import pandas as pd
import pytest

from src import folds
from src.folds import (
    inner_folds, lopo_folds, plan_chrono, plan_folds, plan_lopo, select_samples, sensor_streams,
    suggest_cutoff,
)
from tests.synth_p4 import make_inputs

CFG = folds.load_config()
COHORT = CFG["design"]["cohort"]
SHARED = CFG["design"]["shared_streams"]
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tables():
    """Synthetic T1 and T2 in which pond 8 has no eligible sample."""
    inputs = make_inputs(sparse_ponds=(8,), seed=2)
    return inputs.t1, inputs.t2


def test_config_matches_the_plan():
    assert COHORT == [1, 2, 3, 4, 6, 7, 8, 9]
    assert SHARED == [[1, 2]]
    assert CFG["design"]["n_inner_folds"] == 3
    assert CFG["chrono"]["cutoff"] == "2021-08-28"
    assert CFG["chrono"]["test_origin_margin_days"] == 14


def test_outer_folds_hold_out_each_cohort_pond_once():
    outer = lopo_folds(COHORT, SHARED)
    assert [f.test_ponds for f in outer] == [(p,) for p in COHORT]
    assert [f.index for f in outer] == list(range(8))


def test_stream_partner_leaves_training_when_pond_1_or_2_is_held_out():
    outer = {f.test_ponds[0]: f for f in lopo_folds(COHORT, SHARED)}
    assert outer[1].dropped_ponds == (2,)
    assert outer[1].train_ponds == (3, 4, 6, 7, 8, 9)
    assert outer[2].dropped_ponds == (1,)
    assert outer[2].train_ponds == (3, 4, 6, 7, 8, 9)


def test_other_folds_drop_nothing_and_keep_the_shared_stream_together():
    outer = {f.test_ponds[0]: f for f in lopo_folds(COHORT, SHARED)}
    assert outer[3].dropped_ponds == ()
    assert outer[3].train_ponds == (1, 2, 4, 6, 7, 8, 9)
    assert outer[9].train_ponds == (1, 2, 3, 4, 6, 7, 8)


def test_no_training_pond_shares_a_stream_with_the_test_pond():
    for fold in lopo_folds(COHORT, SHARED):
        test_streams = {p for s in sensor_streams(COHORT, SHARED) if set(s) & set(fold.test_ponds) for p in s}
        assert not test_streams & set(fold.train_ponds)
        assert set(fold.train_ponds) | set(fold.test_ponds) | set(fold.dropped_ponds) == set(COHORT)


def test_sensor_streams_are_sorted_by_smallest_pond():
    assert sensor_streams([9, 3, 2, 1, 4], SHARED) == [(1, 2), (3,), (4,), (9,)]
    # the shared pair is restricted to the ponds given
    assert sensor_streams([2, 5, 3], SHARED) == [(2,), (3,), (5,)]


def test_inner_folds_round_robin_over_the_given_ponds():
    inner = inner_folds([3, 4, 6, 7, 9], SHARED, 3)
    assert [f.val_ponds for f in inner] == [(3, 7), (4, 9), (6,)]
    assert [f.train_ponds for f in inner] == [(4, 6, 9), (3, 6, 7), (3, 4, 7, 9)]


def test_inner_folds_keep_ponds_1_and_2_together():
    inner = inner_folds([1, 2, 4, 6, 7, 8, 9], SHARED, 3)  # pond 3 held out
    # streams: (1,2) (4) (6) (7) (8) (9) -> folds 0 1 2 0 1 2
    assert [f.val_ponds for f in inner] == [(1, 2, 7), (4, 8), (6, 9)]
    for f in inner:
        assert (1 in f.val_ponds) == (2 in f.val_ponds)
        assert (1 in f.train_ponds) == (2 in f.train_ponds)


def test_inner_folds_need_at_least_as_many_streams_as_folds():
    with pytest.raises(ValueError):
        inner_folds([3, 4], SHARED, 3)
    with pytest.raises(ValueError):
        inner_folds([1, 2, 3], SHARED, 3)  # ponds 1 and 2 are one stream: only two streams


def test_select_samples_applies_the_eligibility_rule():
    table = pd.DataFrame(
        {"pond": [1, 1, 2, 3, 3], "t": [14, 28, 14, 14, 28],
         "eligible_eval": [True, False, True, False, True]}
    )
    eligible = select_samples(table, [1, 3], "eligible")
    assert eligible.index.tolist() == [0, 4]  # row labels are kept
    everything = select_samples(table, [1, 3], "all")
    assert everything.index.tolist() == [0, 1, 3, 4]
    with pytest.raises(ValueError):
        select_samples(table, [1], "secondary")


# ------------------------------------------------------------------------- LOPO plans

@pytest.mark.parametrize("task", ["T1", "T2"])
def test_a_pond_without_eligible_samples_has_no_fold_and_does_not_train(tables, task):
    table = tables[0] if task == "T1" else tables[1]
    plans = plan_lopo(task, table, "eligible", CFG)
    assert [p.index for p in plans] == list(range(8))
    by_pond = {p.test_ponds[0]: p for p in plans}
    assert by_pond[8].skipped == "no test samples" and by_pond[8].test_idx == () and by_pond[8].inner == ()
    for plan in plans:
        if plan.skipped:
            continue
        pond_of = table["pond"]
        assert 8 not in set(pond_of.loc[list(plan.train_idx)])
        assert set(pond_of.loc[list(plan.test_idx)]) == set(plan.test_ponds)
        assert table["eligible_eval"].loc[list(plan.train_idx + plan.test_idx)].all()


def test_inner_folds_are_built_over_the_streams_with_an_eligible_training_sample(tables):
    t1, _ = tables
    first = plan_lopo("T1", t1, "eligible", CFG)[0]  # pond 1 held out, pond 2 dropped
    assert first.dropped_ponds == (2,) and first.train_ponds == (3, 4, 6, 7, 8, 9)
    assert [i.val_ponds for i in first.inner] == [(3, 7), (4, 9), (6,)]  # pond 8 has no sample
    assert [i.train_ponds for i in first.inner] == [(4, 6, 9), (3, 6, 7), (3, 4, 7, 9)]


@pytest.mark.parametrize("task", ["T1", "T2"])
def test_inner_splits_partition_the_training_rows_by_stream(tables, task):
    table = tables[0] if task == "T1" else tables[1]
    for plan in plan_lopo(task, table, "eligible", CFG):
        if plan.skipped:
            continue
        validated = []
        for inner in plan.inner:
            assert not set(inner.train_idx) & set(inner.val_idx)
            assert set(inner.train_idx) | set(inner.val_idx) == set(plan.train_idx)
            assert set(table["pond"].loc[list(inner.val_idx)]) == set(inner.val_ponds)
            assert (1 in inner.val_ponds) == (2 in inner.val_ponds)  # the shared stream moves together
            assert (1 in inner.train_ponds) == (2 in inner.train_ponds)
            validated += list(inner.val_idx)
        assert sorted(validated) == sorted(plan.train_idx)  # every training row is validated once


def test_the_all_sample_set_keeps_ineligible_rows_and_gives_pond_8_a_fold(tables):
    t1, _ = tables
    plans = {p.test_ponds[0]: p for p in plan_lopo("T1", t1, "all", CFG)}
    assert not plans[8].skipped and len(plans[8].test_idx) == (t1["pond"] == 8).sum()
    assert len(plans[3].train_idx) > len(
        plan_lopo("T1", t1, "eligible", CFG)[2].train_idx)  # more training rows than the eligible set
    # pond 8 now has samples, so it forms an inner stream of its own
    assert any(8 in inner.val_ponds for inner in plans[1].inner)


def test_a_fold_whose_training_ponds_make_fewer_than_three_streams_is_an_error():
    table = pd.DataFrame({"pond": [1, 2, 3, 4], "eligible_eval": [True] * 4, "t": [14] * 4})
    with pytest.raises(ValueError, match="inner folds"):
        plan_lopo("T1", table, "eligible", {**CFG, "design": {**CFG["design"], "cohort": [1, 2, 3, 4]}})


# ----------------------------------------------------------------------- the cut-off rule

def event_frame(rows):
    return pd.DataFrame(rows, columns=["pond", "date", "eligible_eval"])


def test_cutoff_is_the_latest_date_with_enough_events_in_enough_ponds():
    rows = []
    for pond in (1, 2, 3, 4, 5):
        rows += [(pond, "2021-08-14", True), (pond, "2021-08-28", True), (pond, "2021-09-11", pond <= 2)]
    t1 = event_frame(rows)
    # on or after 09-11: 2 events in 2 ponds; after 08-28: 7 in 5; after 08-14: 12 in 5
    assert suggest_cutoff(t1, 10, 5) == "2021-08-14"
    assert suggest_cutoff(t1, 7, 5) == "2021-08-28"
    assert suggest_cutoff(t1, 2, 2) == "2021-09-11"


def test_cutoff_needs_both_the_event_count_and_the_pond_count():
    t1 = event_frame([(1, "2021-08-28", True), (1, "2021-09-11", True), (1, "2021-09-25", True),
                      (2, "2021-08-28", True), (2, "2021-09-11", True)])
    assert suggest_cutoff(t1, 5, 2) == "2021-08-28"
    with pytest.raises(ValueError, match="no date"):
        suggest_cutoff(t1, 5, 3)  # three ponds are never reached
    with pytest.raises(ValueError, match="no date"):
        suggest_cutoff(t1, 6, 2)


def test_cutoff_ignores_ineligible_events():
    t1 = event_frame([(1, "2021-08-28", True), (2, "2021-08-28", True), (3, "2021-09-11", False),
                      (4, "2021-09-11", False)])
    assert suggest_cutoff(t1, 2, 2) == "2021-08-28"


# ---------------------------------------------------------------------- calendar hold-out

def date_of(frame, idx, column):
    return pd.to_datetime(frame[column].loc[list(idx)])


def test_t1_chrono_trains_before_and_tests_from_the_cutoff(tables):
    t1, _ = tables
    plan = plan_chrono("T1", t1, "eligible", CFG, "2021-08-28")[0]
    cut = pd.Timestamp("2021-08-28")
    assert (date_of(t1, plan.train_idx, "date") < cut).all()
    assert (date_of(t1, plan.test_idx, "date") >= cut).all()
    assert set(plan.train_idx) | set(plan.test_idx) == set(t1.index[t1["eligible_eval"]])
    assert plan.excluded_idx == () and plan.dropped_ponds == () and 8 not in plan.train_ponds
    inner = plan.inner[0]
    last = date_of(t1, plan.train_idx, "date").max()
    assert (date_of(t1, inner.val_idx, "date") == last).all()
    assert (date_of(t1, inner.train_idx, "date") < last).all()
    assert set(inner.train_idx) | set(inner.val_idx) == set(plan.train_idx)


def test_t2_chrono_test_needs_a_late_target_and_a_recent_origin(tables):
    _, t2 = tables
    plan = plan_chrono("T2", t2, "eligible", CFG, "2021-08-28")[0]
    cut, margin = pd.Timestamp("2021-08-28"), pd.Timedelta(days=14)
    target = pd.to_datetime(t2["target_date"])
    origin = pd.to_datetime(t2["origin_date"])
    assert (target.loc[list(plan.train_idx)] < cut).all()
    test = t2.loc[list(plan.test_idx)]
    assert (pd.to_datetime(test["target_date"]) >= cut).all()
    assert (pd.to_datetime(test["origin_date"]) >= cut - margin).all()
    excluded = t2.loc[list(plan.excluded_idx)]
    assert len(excluded) > 0
    assert (pd.to_datetime(excluded["target_date"]) >= cut).all()
    assert (pd.to_datetime(excluded["origin_date"]) < cut - margin).all()
    eligible = set(t2.index[t2["eligible_eval"]])
    assert set(plan.train_idx) | set(plan.test_idx) | set(plan.excluded_idx) == eligible
    assert not (set(plan.train_idx) & set(plan.test_idx)) and not (set(plan.test_idx) & set(plan.excluded_idx))
    # horizon 1 samples are never excluded: their origin is 14 days before the target
    assert set(excluded["horizon"]) == {2}


def test_t2_chrono_on_hand_made_cases():
    c = "2021-08-28"
    rows = [  # pond, origin, target, horizon
        (1, "2021-07-17", "2021-07-31", 1), (1, "2021-07-31", "2021-08-14", 1),   # target before c: train
        (2, "2021-07-31", "2021-08-28", 2),   # target = c, origin 28 days before c: neither
        (3, "2021-08-14", "2021-08-28", 1),   # target = c, origin c - 14: test
        (4, "2021-08-14", "2021-09-11", 2),   # target c + 14, origin c - 14: test
        (5, "2021-08-28", "2021-09-11", 1),   # origin = c: test
        (6, "2021-07-17", "2021-08-14", 2),   # target c - 14 (before c): train, origin earlier
    ]
    table = pd.DataFrame(rows, columns=["pond", "origin_date", "target_date", "horizon"])
    table["eligible_eval"] = True
    cfg = {**CFG, "design": {**CFG["design"], "cohort": [1, 2, 3, 4, 5, 6]}}
    plan = plan_chrono("T2", table, "eligible", cfg, c)[0]
    assert plan.train_idx == (0, 1, 6)
    assert plan.test_idx == (3, 4, 5)
    assert plan.excluded_idx == (2,)
    assert [i.val_idx for i in plan.inner] == [(1, 6)]  # latest training target, 2021-08-14
    assert [i.train_idx for i in plan.inner] == [(0,)]


def test_chrono_needs_training_and_test_samples_and_two_training_dates(tables):
    t1, _ = tables
    with pytest.raises(ValueError, match="0 training"):
        plan_chrono("T1", t1, "eligible", CFG, "2021-01-01")
    with pytest.raises(ValueError, match="0 test"):
        plan_chrono("T1", t1, "eligible", CFG, "2030-01-01")
    one_date = t1[t1["t"] == 14]
    with pytest.raises(ValueError, match="single date"):
        plan_chrono("T1", pd.concat([one_date, t1[t1["t"] == 56]]), "eligible", CFG, "2021-08-14")


def test_plan_folds_dispatches_on_the_scheme(tables):
    t1, _ = tables
    assert len(plan_folds("T1", "lopo", t1, "eligible", CFG)) == 8
    assert plan_folds("T1", "chrono", t1, "eligible", CFG, cutoff="2021-08-28")[0].scheme == "chrono"
    with pytest.raises(ValueError, match="cut-off"):
        plan_folds("T1", "chrono", t1, "eligible", CFG)
    with pytest.raises(ValueError, match="scheme"):
        plan_folds("T1", "blocked", t1, "eligible", CFG)


# ------------------------------------------------------------- the real tables, counts only

@pytest.mark.skipif(not (REPO / "data/processed/t2_table.csv").exists(), reason="tables not built")
def test_real_plans_have_every_pond_but_8_and_both_horizons_in_every_t2_validation_fold():
    """Reads pond, date, horizon and eligibility columns only; no value of a target."""
    columns = ["pond", "date", "eligible_eval"]
    t1 = pd.read_csv(REPO / "data/processed/t1_table.csv", usecols=columns)
    t2 = pd.read_csv(REPO / "data/processed/t2_table.csv",
                     usecols=["pond", "origin_date", "target_date", "horizon", "eligible_eval"])
    assert suggest_cutoff(t1, CFG["chrono"]["min_events"], CFG["chrono"]["min_ponds"]) == CFG["chrono"]["cutoff"]
    for task, table in (("T1", t1), ("T2", t2)):
        plans = plan_lopo(task, table, "eligible", CFG)
        assert [p.test_ponds[0] for p in plans if p.skipped] == [8]
        for plan in plans:
            if plan.skipped:
                continue
            assert len(plan.inner) == 3
            if task == "T2":
                for inner in plan.inner:
                    assert set(t2["horizon"].loc[list(inner.val_idx)]) == {1, 2}, (plan.label, inner.index)
                    assert set(t2["horizon"].loc[list(inner.train_idx)]) == {1, 2}
        chrono = plan_chrono(task, table, "eligible", CFG, CFG["chrono"]["cutoff"])[0]
        if task == "T2":
            assert set(t2["horizon"].loc[list(chrono.inner[0].val_idx)]) == {1, 2}
            assert set(t2["horizon"].loc[list(chrono.test_idx)]) == {1, 2}
