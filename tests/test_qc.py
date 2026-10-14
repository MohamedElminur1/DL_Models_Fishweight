"""Unit tests for src/qc.py on small synthetic series built inside each test.

All series here are test-only constructions, not measurements. Flat-line spans and gaps are
derived from the thresholds in configs/phase4_data.yaml, so the tests follow the configured
limit; the range tests state the bounds of the plan (section 2.5) explicitly and check that
the config reproduces them.
"""

import numpy as np
import pandas as pd
import pytest

from src import qc
from src.pipeline_io import load_config

CFG = load_config()
LIMIT_H = CFG["qc"]["flatline_hours"]
GAP_H = CFG["qc"]["flatline_max_gap_hours"]
LIMIT = pd.Timedelta(hours=LIMIT_H).to_timedelta64()
MAX_GAP = pd.Timedelta(hours=GAP_H).to_timedelta64()
STEP_H = GAP_H / 2  # reading interval inside a toy run, well inside the gap rule
T0 = np.datetime64("2021-07-01T00:00:00", "us")  # test-only clock origin


def clock(hours) -> np.ndarray:
    """Timestamps at the given hours after T0."""
    micros = np.round(np.asarray(hours, dtype="float64") * 3_600_000_000).astype("int64")
    return T0 + micros.astype("timedelta64[us]")


def run_hours(span_h: float, start_h: float = 0.0) -> np.ndarray:
    """Reading times every STEP_H from start_h, the last reading exactly span_h later."""
    return start_h + np.append(np.arange(0.0, span_h, STEP_H), span_h)


def flagged(values, hours) -> np.ndarray:
    return qc.flatline_mask(np.asarray(values, dtype="float64"), clock(hours), LIMIT, MAX_GAP)


def spec(name: str) -> qc.SensorSpec:
    return next(s for s in qc.sensor_specs(CFG) if s.name == name)


def classify(name: str, values, hours) -> np.ndarray:
    return qc.classify(np.asarray(values, dtype="float64"), clock(hours), spec(name), LIMIT, MAX_GAP)


# ------------------------------------------------------------------ flat-line rule


def test_run_spanning_exactly_the_limit_is_flat_line():
    hours = run_hours(LIMIT_H)
    assert flagged(np.full(hours.size, 5.0), hours).all()


def test_run_just_under_the_limit_is_valid():
    hours = run_hours(LIMIT_H - 0.1)
    assert not flagged(np.full(hours.size, 5.0), hours).any()


def test_missing_value_ends_a_run():
    hours = run_hours(1.5 * LIMIT_H)
    values = np.full(hours.size, 5.0)
    assert flagged(values, hours).all()  # unbroken, the same series is one long run
    values[hours.size // 2] = np.nan
    assert not flagged(values, hours).any()  # each half is shorter than the limit


def test_non_finite_value_ends_a_run_and_is_never_part_of_one():
    hours = run_hours(1.5 * LIMIT_H)
    values = np.full(hours.size, 5.0)
    values[hours.size // 2] = np.inf
    assert not flagged(values, hours).any()
    assert not flagged(np.full(hours.size, np.inf), hours).any()
    assert not flagged(np.full(hours.size, np.nan), hours).any()


def test_a_different_value_ends_a_run():
    hours = run_hours(1.5 * LIMIT_H)
    values = np.full(hours.size, 5.0)
    values[hours.size // 2] = 5.1
    assert not flagged(values, hours).any()


def test_only_the_rows_of_the_long_run_are_flagged():
    hours = run_hours(3 * LIMIT_H)
    values = np.where(np.arange(hours.size) % 2 == 0, 1.0, 2.0)  # alternating: no run
    values[20:100] = 9.0  # about 39 h of identical readings
    expected = np.zeros(hours.size, dtype=bool)
    expected[20:100] = True
    assert (flagged(values, hours) == expected).all()


def test_two_identical_readings_across_a_long_logging_gap_are_not_a_run():
    hours = np.array([0.0, LIMIT_H + 1.0])
    assert not flagged([5.0, 5.0], hours).any()


def test_logging_gap_longer_than_the_limit_splits_a_run():
    """Two stretches of identical readings, 2 h apart, together longer than the flat-line limit."""
    first = run_hours(0.55 * LIMIT_H)
    second = run_hours(0.55 * LIMIT_H, start_h=first[-1] + 2 * GAP_H)
    hours = np.concatenate([first, second])
    assert hours[-1] - hours[0] > LIMIT_H
    assert not flagged(np.full(hours.size, 5.0), hours).any()


def test_gap_equal_to_the_limit_does_not_end_a_run():
    first = run_hours(0.55 * LIMIT_H)
    second = run_hours(0.55 * LIMIT_H, start_h=first[-1] + GAP_H)
    hours = np.concatenate([first, second])
    assert flagged(np.full(hours.size, 5.0), hours).all()


def test_equal_timestamps_stay_in_one_run():
    hours = np.append(np.repeat(run_hours(LIMIT_H)[:-1], 2), LIMIT_H)  # every reading twice
    assert flagged(np.full(hours.size, 5.0), hours).all()


def test_unsorted_timestamps_are_rejected():
    with pytest.raises(ValueError):
        flagged([1.0, 2.0], [1.0, 0.0])


# ------------------------------------------------------------------ rule order and ranges


def test_each_value_is_counted_under_the_first_rule_it_fails():
    hours = run_hours(1.5 * LIMIT_H)
    n = hours.size
    out_of_range_plateau = classify("ph", np.full(n, 15.45), hours)  # also flat for longer than the limit
    in_range_plateau = classify("ph", np.full(n, 7.2), hours)
    infinite = classify("ph", np.full(n, np.inf), hours)
    empty = classify("ph", np.full(n, np.nan), hours)
    varying = classify("ph", 7.0 + 0.01 * (np.arange(n) % 5), hours)
    assert (out_of_range_plateau == qc.OUT_OF_RANGE).all()
    assert (in_range_plateau == qc.FLATLINE).all()
    assert (infinite == qc.NONFINITE).all()
    assert (empty == qc.MISSING).all()
    assert (varying == qc.VALID).all()


RANGE_CASES = [
    ("temperature", 10.0, True), ("temperature", 9.99, False), ("temperature", 40.0, True),
    ("temperature", 40.01, False), ("temperature", -127.0, False),
    ("turbidity", 0.0, True), ("turbidity", 100.0, True), ("turbidity", -0.01, False),
    ("turbidity", 100.01, False),
    ("do", 0.0, False), ("do", 0.001, True), ("do", 20.0, True), ("do", 20.01, False),
    ("ph", 0.0, True), ("ph", 14.0, True), ("ph", -0.01, False), ("ph", 14.01, False),
    ("ammonia_log10", 0.0, False), ("ammonia_log10", 1e-9, True), ("ammonia_log10", 500.0, True),
    ("ammonia_log10", 500.01, False),
    ("nitrate", 0.0, False), ("nitrate", -1.0, False), ("nitrate", 1e12, True),
]


@pytest.mark.parametrize("sensor,value,is_valid", RANGE_CASES)
def test_range_bounds_of_the_plan(sensor, value, is_valid):
    code = classify(sensor, [value], [0.0])[0]
    assert (code == qc.VALID) == is_valid
    if not is_valid:
        assert code == qc.OUT_OF_RANGE


def test_sensor_names_and_order_follow_the_plan():
    names = [s.name for s in qc.sensor_specs(CFG)]
    assert names == ["temperature", "turbidity", "do", "ph", "ammonia_log10", "nitrate"]


def test_ammonia_is_log10_transformed_after_qc_and_other_sensors_are_not():
    ammonia = spec("ammonia_log10")
    raw = np.array([100.0, 0.1, 500.0, 0.0, 600.0, np.inf, np.nan])
    codes = qc.classify(raw, clock(np.arange(raw.size)), ammonia, LIMIT, MAX_GAP)
    out = qc.qc_values(raw, codes, ammonia)
    assert out[:3] == pytest.approx([2.0, -1.0, np.log10(500.0)])
    assert np.isnan(out[3:]).all()
    temperature = spec("temperature")
    values = np.array([22.5, -127.0])
    kept = qc.qc_values(values, qc.classify(values, clock([0.0, 1.0]), temperature, LIMIT, MAX_GAP), temperature)
    assert kept[0] == 22.5 and np.isnan(kept[1])


# ------------------------------------------------------------------ duplicates, order, counts


def toy_frame(records: list[dict]) -> pd.DataFrame:
    """A load_pond-like frame from dicts; unspecified measurement columns are missing."""
    base = {name: np.nan for name in CFG["qc"]["duplicate_key_columns"] if name != "ts_raw"}
    rows = []
    for number, record in enumerate(records):
        row = {**base, "row": number, "entry_id": number, "ts_kind": "datetime_label", **record}
        row.setdefault("ts_raw", str(row["ts"]))
        row["date"] = pd.Timestamp(row["ts"]).normalize()
        row["ts"] = pd.Timestamp(row["ts"])
        rows.append(row)
    return pd.DataFrame(rows)


def test_exact_duplicate_lines_are_dropped_keeping_the_earliest_file_row():
    same = {"ts": "2021-07-01 10:00:00", "entry_id": 7, "temperature": 24.0, "weight": 5.0}
    frame = toy_frame(
        [
            same,  # row 0, kept
            {"ts": "2021-07-01 10:01:00", "temperature": 24.5},  # row 1, kept
            dict(same),  # row 2: repeats row 0 exactly, dropped
            {**same, "entry_id": 8},  # row 3: another entry_id, kept
            {**same, "temperature": 24.1},  # row 4: another value, kept
            {"ts": "2021-07-01 10:05:00", "entry_id": 9},  # row 5: missing values, kept
            {"ts": "2021-07-01 10:05:00", "entry_id": 9},  # row 6: repeats row 5, NaN equals NaN, dropped
        ]
    )
    kept, n_dropped = qc.drop_exact_duplicates(frame.iloc[::-1], CFG["qc"]["duplicate_key_columns"])
    assert n_dropped == 2
    assert kept["row"].tolist() == [0, 1, 3, 4, 5]


def test_timed_rows_keep_clock_rows_only_and_sort_by_ts_then_file_row():
    frame = toy_frame(
        [
            {"ts": "2021-07-01 12:00:00"},  # row 0
            {"ts": "2021-07-01 09:00:00"},  # row 1
            {"ts": "2021-07-01 10:00:00", "ts_kind": "date_only"},  # row 2, no clock time
            {"ts": "2021-07-01 10:00:00", "temperature": 1.0},  # row 3
            {"ts": "2021-07-01 10:00:00", "temperature": 2.0},  # row 4, same ts as row 3
        ]
    )
    timed, n_dropped = qc.timed_rows(frame, CFG)
    assert timed["row"].tolist() == [1, 3, 4, 0]
    assert n_dropped == 0


def test_qc_pond_counts_every_value_once_and_reports_dropped_duplicates():
    duplicate = {"ts": "2021-07-01 00:30:00", "ph": 7.1, "entry_id": 500}
    frame = toy_frame(
        [
            {"ts": "2021-07-01 00:00:00", "ph": 7.0},  # valid
            dict(duplicate),  # valid
            dict(duplicate),  # exact duplicate of the row above, dropped before QC
            {"ts": "2021-07-01 01:00:00", "ph": -0.5},  # out of range
            {"ts": "2021-07-01 01:30:00", "ph": np.inf},  # non-finite
            {"ts": "2021-07-01 02:00:00"},  # missing
            {"ts": "2021-07-01 02:30:00", "ph": 7.2, "ts_kind": "date_only"},  # not a timed row
        ]
    )
    rows, counts = qc.qc_pond(frame, 99, CFG)
    ph = counts.loc[counts["sensor"] == "ph"].iloc[0]
    assert (ph["n_values"], ph["n_valid"], ph["n_out_of_range"], ph["n_nonfinite"], ph["n_missing_raw"]) == (5, 2, 1, 1, 1)
    assert ph["n_flatline"] == 0 and ph["n_duplicate_rows_dropped"] == 1
    assert list(counts.columns) == qc.COUNT_COLUMNS
    assert len(counts) == len(CFG["sensors"])
    assert rows["ph"].tolist()[:2] == [7.0, 7.1] and rows["ph"].isna().tolist()[2:] == [True, True, True]
    assert rows["hour"].tolist() == [0, 0, 1, 1, 2]
