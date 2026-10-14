"""Unit tests for src/features.py on small synthetic series built inside each test.

All series here are test-only constructions, not measurements. The thresholds come from
configs/phase4_data.yaml; `test_config_matches_the_plan` pins them to the values of
ANALYSIS_PLAN.md and the lead's amendments so a silent edit of the config fails a test.
"""

import numpy as np
import pandas as pd
import pytest

from src import features as f
from src.pipeline_io import grid_dates, load_config

CFG = load_config()
SENSORS = f.sensor_names(CFG)
MIN_HOURS = CFG["daily"]["min_hours"]
MAX_GAP = CFG["daily"]["max_gap_days"]
DDOF = CFG["daily"]["sd_ddof"]
MIN_SENSOR_DAYS = CFG["windows"]["min_sensor_days"]
DAY0 = pd.Timestamp(CFG["cohort"]["day0"])
DAY = pd.Timedelta(days=1)
T1_DAYS, T2_DAYS = CFG["windows"]["t1_days"], CFG["windows"]["t2_days"]


def test_config_matches_the_plan():
    assert (MIN_HOURS, MAX_GAP, DDOF, CFG["daily"]["min_sensors_covered"]) == (12, 2, 0, 3)
    assert (T1_DAYS, T2_DAYS, MIN_SENSOR_DAYS) == (14, 28, 3)
    assert CFG["windows"]["horizons"] == [1, 2] and CFG["windows"]["horizon_days"] == 14
    assert (CFG["eligibility"]["t1_min_covered_days"], CFG["eligibility"]["t2_min_covered_days"]) == (7, 14)
    assert CFG["cohort"]["day0"] == "2021-06-19"
    assert CFG["daily"]["first_date"] == str((DAY0 - 28 * DAY).date())
    assert SENSORS == ["temperature", "turbidity", "do", "ph", "ammonia_log10", "nitrate"]
    assert CFG["wq_by_date"]["k_first"] == 1
    assert CFG["wq_by_date"]["sensitivity_windows"] == {"t1w07": 7, "t1w28": 28, "t2w14": 14, "t2w42": 42}
    assert CFG["streams"] == {1: "S12", 2: "S12", 3: "S03", 4: "S04", 6: "S06", 7: "S07", 8: "S08", 9: "S09"}
    assert list(grid_dates(CFG)) == [DAY0 + 14 * k * DAY for k in range(17)]


# ------------------------------------------------------------------ toy builders


def toy_rows(days: dict, sensor: str = "temperature") -> pd.DataFrame:
    """QC-style rows. days maps a date to (hours, center, spread): in each listed hour two
    values, center - spread and center + spread, so the day has mean center, population SD
    spread, min center - spread and max center + spread."""
    records = []
    for date, (hours, center, spread) in days.items():
        for hour in hours:
            for value in (center - spread, center + spread):
                records.append({"date": pd.Timestamp(date), "hour": hour, sensor: value})
    return pd.DataFrame(records)


def hours(n: int) -> list[int]:
    return list(range(n))


def iso(date: pd.Timestamp) -> str:
    return str(date.date())


def sensor_days(rows: pd.DataFrame, start: str, n_days: int, sensor: str = "temperature") -> pd.DataFrame:
    days = pd.date_range(start, periods=n_days, freq="D")
    return f.sensor_day_table(rows, sensor, days, MIN_HOURS, DDOF)


def make_series(first: str, last: str, per_day: dict, right_anchors: dict | None = None) -> f.DailySeries:
    """DailySeries over first..last. per_day maps a Timestamp to {sensor: (mean, sd, min, max)};
    sensors not listed for a day have no value that day. right_anchors maps (Timestamp, sensor)
    to the right anchor of that interpolated sensor-day."""
    days = pd.date_range(first, last, freq="D")
    values = np.full((len(days), len(SENSORS), len(f.STATS)), np.nan)
    anchors = np.full((len(days), len(SENSORS)), np.datetime64("NaT", "ns"))
    for date, entry in per_day.items():
        for sensor, stats in entry.items():
            values[days.get_loc(date), SENSORS.index(sensor), :] = stats
    for (date, sensor), anchor in (right_anchors or {}).items():
        anchors[days.get_loc(date), SENSORS.index(sensor)] = anchor.to_datetime64()
    return f.DailySeries(days.as_unit("ns"), tuple(SENSORS), values, anchors)


def present(date_range, sensors=("temperature", "turbidity", "do"), stats=(1.0, 0.1, 0.5, 1.5)) -> dict:
    return {date: {s: stats for s in sensors} for date in date_range}


def series_through_pipeline(per_sensor: dict, first: str, last: str) -> tuple[pd.DataFrame, f.DailySeries]:
    """Long daily table and DailySeries built from toy QC rows by the production functions.
    per_sensor maps a sensor to the `days` argument of toy_rows; other sensors have no data."""
    rows = pd.concat([toy_rows(days, sensor) for sensor, days in per_sensor.items()], ignore_index=True)
    for sensor in SENSORS:
        if sensor not in rows:
            rows[sensor] = np.nan
    long = f.pond_daily(rows, 1, pd.date_range(first, last, freq="D"), CFG)
    return long, f.series_from_long(long, SENSORS)


# ------------------------------------------------------------------ 12 clock hours


def test_sensor_day_needs_the_configured_number_of_distinct_clock_hours():
    rows = toy_rows(
        {
            "2021-07-01": (hours(MIN_HOURS - 1), 20.0, 1.0),
            "2021-07-02": (hours(MIN_HOURS), 21.0, 1.0),
        }
    )
    table = sensor_days(rows, "2021-07-01", 2)
    assert table["status"].tolist() == ["missing", "observed"]
    assert table["n_hours"].tolist() == [MIN_HOURS - 1, MIN_HOURS]
    assert table.loc[table["status"] == "missing", list(f.STATS)].isna().all().all()  # partial day gives no statistics


def test_many_values_in_few_hours_do_not_make_an_observed_day():
    rows = toy_rows({"2021-07-01": (hours(MIN_HOURS - 2), 20.0, 1.0)})
    crowded = pd.DataFrame({"date": pd.Timestamp("2021-07-01"), "hour": 3, "temperature": np.linspace(19, 21, 500)})
    table = sensor_days(pd.concat([rows, crowded], ignore_index=True), "2021-07-01", 1)
    assert table.at[table.index[0], "n_hours"] == MIN_HOURS - 2
    assert table["status"].tolist() == ["missing"]


def test_hours_with_only_invalid_values_are_not_counted():
    rows = toy_rows({"2021-07-01": (hours(MIN_HOURS - 1), 20.0, 1.0)})
    invalid = pd.DataFrame({"date": pd.Timestamp("2021-07-01"), "hour": range(MIN_HOURS - 1, 24), "temperature": np.nan})
    table = sensor_days(pd.concat([rows, invalid], ignore_index=True), "2021-07-01", 1)
    assert table["n_hours"].tolist() == [MIN_HOURS - 1]
    assert table["status"].tolist() == ["missing"]


def test_daily_statistics_are_mean_population_sd_min_max_of_the_valid_values():
    rows = toy_rows({"2021-07-01": (hours(MIN_HOURS), 21.0, 1.0)})
    table = sensor_days(rows, "2021-07-01", 1)
    row = table.iloc[0]
    assert [row["mean"], row["sd"], row["min"], row["max"]] == pytest.approx([21.0, 1.0, 20.0, 22.0])  # ddof=1 gives 1.0215
    assert row["n_values"] == 2 * MIN_HOURS


# ------------------------------------------------------------------ 2-day interpolation and anchors


def gap_table() -> pd.DataFrame:
    """Observed days at offsets 2, 5, 6, 10, 12 of a 15-day table starting 2021-07-01."""
    observed = {2: (10.0, 1.0), 5: (16.0, 4.0), 6: (17.0, 4.0), 10: (20.0, 2.0), 12: (30.0, 6.0)}
    rows = toy_rows(
        {
            iso(pd.Timestamp("2021-07-01") + offset * DAY): (hours(MIN_HOURS), center, spread)
            for offset, (center, spread) in observed.items()
        }
    )
    return f.fill_short_gaps(sensor_days(rows, "2021-07-01", 15), MAX_GAP)


def test_two_day_gap_is_filled_three_day_and_edge_gaps_are_not():
    status = gap_table()["status"].tolist()
    assert status == [
        "missing", "missing",  # offsets 0, 1: edge gap, no observed day on the left
        "observed",
        "interpolated", "interpolated",  # offsets 3, 4: a 2-day gap
        "observed", "observed",
        "missing", "missing", "missing",  # offsets 7 to 9: a 3-day gap
        "observed",
        "interpolated",  # offset 11: a 1-day gap
        "observed",
        "missing", "missing",  # offsets 13, 14: edge gap, no observed day on the right
    ]


def test_interpolation_is_linear_in_each_statistic():
    table = gap_table()
    filled = table.iloc[[3, 4, 11]][list(f.STATS)].to_numpy()
    expected = np.array(
        [
            [12.0, 2.0, 10.0, 14.0],  # between (mean 10, sd 1, min 9, max 11) and (16, 4, 12, 20), one third
            [14.0, 3.0, 11.0, 17.0],  # two thirds
            [25.0, 4.0, 21.0, 29.0],  # between (20, 2, 18, 22) and (30, 6, 24, 36), halfway
        ]
    )
    assert filled == pytest.approx(expected)


def test_observed_days_are_unchanged_and_missing_days_stay_nan():
    table = gap_table()
    assert table.iloc[2][list(f.STATS)].tolist() == pytest.approx([10.0, 1.0, 9.0, 11.0])
    assert table.iloc[[0, 1, 7, 8, 9, 13, 14]][list(f.STATS)].isna().all().all()


def test_interpolated_days_record_the_observed_days_on_both_sides():
    table = gap_table()
    start = pd.Timestamp("2021-07-01")
    for offset, (left, right) in {3: (2, 5), 4: (2, 5), 11: (10, 12)}.items():
        assert table.iloc[offset]["left_anchor_date"] == start + left * DAY
        assert table.iloc[offset]["right_anchor_date"] == start + right * DAY
    others = table.drop(table.index[[3, 4, 11]])
    assert others["left_anchor_date"].isna().all() and others["right_anchor_date"].isna().all()


def test_gap_limit_is_the_configured_number_of_days():
    for gap, filled in ((MAX_GAP, True), (MAX_GAP + 1, False)):
        rows = toy_rows(
            {
                "2021-07-01": (hours(MIN_HOURS), 10.0, 1.0),
                iso(pd.Timestamp("2021-07-01") + (gap + 1) * DAY): (hours(MIN_HOURS), 20.0, 1.0),
            }
        )
        status = f.fill_short_gaps(sensor_days(rows, "2021-07-01", gap + 2), MAX_GAP)["status"].tolist()
        assert (status[1:-1] == ["interpolated"] * gap) == filled


def test_partial_day_inside_a_short_gap_is_replaced_and_keeps_its_own_counts():
    rows = toy_rows(
        {
            "2021-07-01": (hours(MIN_HOURS), 10.0, 1.0),
            "2021-07-02": (hours(MIN_HOURS - 1), 99.0, 1.0),  # too few hours: missing, then interpolated
            "2021-07-03": (hours(MIN_HOURS), 20.0, 1.0),
        }
    )
    table = f.fill_short_gaps(sensor_days(rows, "2021-07-01", 3), MAX_GAP)
    middle = table.iloc[1]
    assert middle["status"] == "interpolated" and middle["mean"] == pytest.approx(15.0)
    assert middle["n_hours"] == MIN_HOURS - 1 and middle["n_values"] == 2 * (MIN_HOURS - 1)


def test_pond_daily_layout_and_interpolation_counts():
    rows = toy_rows({"2021-07-01": (hours(MIN_HOURS), 10.0, 1.0), "2021-07-04": (hours(MIN_HOURS), 16.0, 1.0)})
    for sensor in SENSORS[1:]:
        rows[sensor] = np.nan
    days = pd.date_range("2021-07-01", "2021-07-04", freq="D")
    long = f.pond_daily(rows, 5, days, CFG)
    assert list(long.columns) == f.DAILY_COLUMNS
    assert len(long) == len(days) * len(SENSORS)
    assert long["sensor"].tolist()[: len(SENSORS)] == SENSORS  # sensors in config order within a date
    assert long["t"].tolist()[:1] == [(pd.Timestamp("2021-07-01") - DAY0).days]
    temperature = long.loc[long["sensor"] == "temperature"]
    assert temperature["status"].tolist() == ["observed", "interpolated", "interpolated", "observed"]
    assert temperature["right_anchor_date"].tolist()[1:3] == [pd.Timestamp("2021-07-04")] * 2
    assert temperature["left_anchor_date"].tolist()[1:3] == [pd.Timestamp("2021-07-01")] * 2
    counts = f.interpolation_counts(long).set_index("sensor")
    assert counts.at["temperature", "n_interpolated"] == 2
    assert counts.at["ph", "n_missing"] == 4 and counts.at["ph", "n_days"] == 4


# ------------------------------------------------------------------ coverage


def test_day_is_covered_with_the_configured_number_of_sensors():
    day = pd.Timestamp("2021-07-01")
    series = make_series(
        "2021-07-01", "2021-07-02",
        {
            day: {s: (1.0, 0.1, 0.5, 1.5) for s in SENSORS[: CFG["daily"]["min_sensors_covered"] - 1]},
            day + DAY: {s: (1.0, 0.1, 0.5, 1.5) for s in SENSORS[: CFG["daily"]["min_sensors_covered"]]},
        },
    )
    table = f.coverage_table(series, 1, CFG)
    assert table["covered"].tolist() == [False, True]
    assert table["n_sensors_with_value"].tolist() == [2, 3]


# ------------------------------------------------------------------ windows


EVENT = pd.Timestamp("2021-07-17")


def offset_series() -> f.DailySeries:
    """Temperature, turbidity and DO on every day from E - 40 to E + 5; the daily mean of
    temperature is the day's offset from the event date E, SD 1, min offset - 1, max offset + 1."""
    per_day = {}
    for date in pd.date_range(EVENT - 40 * DAY, EVENT + 5 * DAY, freq="D"):
        offset = float((date - EVENT).days)
        per_day[date] = {
            "temperature": (offset, 1.0, offset - 1.0, offset + 1.0),
            "turbidity": (50.0, 2.0, 40.0, 60.0),
            "do": (7.0, 0.5, 6.0, 8.0),
        }
    return make_series(iso(EVENT - 40 * DAY), iso(EVENT + 5 * DAY), per_day)


def test_window_is_the_days_before_the_event_and_excludes_the_event_date():
    days = f.window_days(EVENT, T1_DAYS)
    assert len(days) == T1_DAYS
    assert days[0] == EVENT - T1_DAYS * DAY and days[-1] == EVENT - DAY
    assert EVENT not in days
    days28 = f.window_days(EVENT, T2_DAYS)
    assert (days28[0], days28[-1], len(days28)) == (EVENT - 28 * DAY, EVENT - DAY, 28)


def test_t1_window_features_use_exactly_the_14_days_before_the_event():
    features, n_covered = f.window_row(offset_series(), EVENT, T1_DAYS, CFG)
    # offsets -14 .. -1: day E (offset 0), day E - 15 (offset -15) and day E + 1 would each change one number
    assert features["temperature_mean_w"] == pytest.approx(-7.5)
    assert features["temperature_sd_w"] == pytest.approx(1.0)
    assert features["temperature_min_w"] == -15.0  # E - 15 would give -16
    assert features["temperature_max_w"] == 0.0  # E would give 1
    assert features["turbidity_mean_w"] == 50.0 and features["do_max_w"] == 8.0
    assert n_covered == T1_DAYS


def test_t2_window_features_use_exactly_the_28_days_before_the_origin():
    features, n_covered = f.window_row(offset_series(), EVENT, T2_DAYS, CFG)
    assert features["temperature_mean_w"] == pytest.approx(-14.5)  # offsets -28 .. -1
    assert features["temperature_min_w"] == -29.0 and features["temperature_max_w"] == 0.0
    assert n_covered == T2_DAYS


def test_sensor_without_a_value_in_the_window_gives_nan_not_an_imputed_number():
    features, _ = f.window_row(offset_series(), EVENT, T1_DAYS, CFG)
    for stat in f.STATS:
        assert np.isnan(features[f"ph_{stat}_w"]) and np.isnan(features[f"nitrate_{stat}_w"])
    assert list(features) == [f"{s}_{stat}_w" for s in SENSORS for stat in f.STATS]  # 24 names, sensor-major


def test_a_sensor_needs_the_configured_number_of_window_days_for_its_features():
    for n_days, has_features in ((MIN_SENSOR_DAYS, True), (MIN_SENSOR_DAYS - 1, False)):
        per_day = present(pd.date_range(EVENT - 14 * DAY, EVENT - DAY, freq="D"), sensors=("turbidity", "do", "ph"))
        for k in range(1, n_days + 1):  # temperature on the last n_days window days only
            per_day[EVENT - k * DAY]["temperature"] = (30.0, 3.0, 20.0, 40.0)
        series = make_series(iso(EVENT - 20 * DAY), iso(EVENT + DAY), per_day)
        features, n_covered = f.window_row(series, EVENT, T1_DAYS, CFG)
        got = [features[f"temperature_{stat}_w"] for stat in f.STATS]
        if has_features:
            assert got == [30.0, 3.0, 20.0, 40.0]
        else:
            assert np.isnan(got).all()
        assert n_covered == T1_DAYS  # three other sensors cover every day, so the count does not depend on this sensor


def test_only_days_with_a_value_enter_a_sensors_window_statistics():
    per_day = present(pd.date_range(EVENT - 14 * DAY, EVENT - DAY, freq="D"), sensors=("turbidity", "do", "ph"))
    values = {1: (10.0, 1.0, 8.0, 12.0), 2: (20.0, 2.0, 17.0, 23.0), 5: (60.0, 3.0, 50.0, 71.0)}
    for k, stats in values.items():
        per_day[EVENT - k * DAY]["temperature"] = stats
    series = make_series(iso(EVENT - 20 * DAY), iso(EVENT + DAY), per_day)
    features, _ = f.window_row(series, EVENT, T1_DAYS, CFG)
    assert features["temperature_mean_w"] == pytest.approx(30.0)  # three days, not 14
    assert features["temperature_sd_w"] == pytest.approx(2.0)
    assert (features["temperature_min_w"], features["temperature_max_w"]) == (8.0, 71.0)


def test_covered_days_are_counted_in_the_window_only():
    series = offset_series()
    values = series.values.copy()
    do = SENSORS.index("do")
    values[series.days.get_loc(EVENT - DAY), do, :] = np.nan  # 2 sensors left on the last window day
    values[series.days.get_loc(EVENT - 14 * DAY), do, :] = np.nan  # and on the first window day
    values[series.days.get_loc(EVENT), do, :] = np.nan  # event date, outside the window
    values[series.days.get_loc(EVENT - 15 * DAY), do, :] = np.nan  # day before the window
    thinner = f.DailySeries(series.days, series.sensors, values, series.right_anchor)
    _, n_covered = f.window_row(thinner, EVENT, T1_DAYS, CFG)
    assert n_covered == T1_DAYS - 2


def test_cumulative_mean_runs_from_day_0_to_the_day_before_the_event():
    per_day = {}
    for date in pd.date_range(DAY0 - 3 * DAY, EVENT + DAY, freq="D"):
        mean = 5.0 if date == DAY0 else 2.0
        if date < DAY0 or date >= EVENT:
            mean = 1000.0  # outside [day 0, E - 1]: must not count
        per_day[date] = {"temperature": (mean, 1.0, mean - 1.0, mean + 1.0)}
    series = make_series(iso(DAY0 - 3 * DAY), iso(EVENT + DAY), per_day)
    n_days = (EVENT - DAY0).days
    cumulative = f.cumulative_means(series, EVENT, CFG)
    assert cumulative["temperature_mean_cum"] == pytest.approx((5.0 + 2.0 * (n_days - 1)) / n_days)
    assert np.isnan(cumulative["ph_mean_cum"])
    assert list(cumulative) == [f"{s}_mean_cum" for s in SENSORS]


def test_cumulative_mean_needs_the_configured_number_of_days():
    for n_days, has_value in ((MIN_SENSOR_DAYS, True), (MIN_SENSOR_DAYS - 1, False)):
        per_day = {EVENT - k * DAY: {"temperature": (4.0, 1.0, 3.0, 5.0)} for k in range(1, n_days + 1)}
        series = make_series(iso(DAY0 - DAY), iso(EVENT + DAY), per_day)
        value = f.cumulative_means(series, EVENT, CFG)["temperature_mean_cum"]
        assert (value == 4.0) if has_value else np.isnan(value)


def test_a_window_that_leaves_the_daily_table_is_an_error():
    series = make_series("2021-07-01", "2021-07-31", present(pd.date_range("2021-07-01", "2021-07-31", freq="D")))
    with pytest.raises(ValueError):
        f.window_row(series, pd.Timestamp("2021-07-10"), T1_DAYS, CFG)


# ------------------------------------------------------------------ gap filled from the event day


def gap_before_event() -> tuple[pd.DataFrame, f.DailySeries]:
    """Three sensors observed on every day of E - 20 .. E + 5 except E - 2 and E - 1; the
    temperature is 10 before the gap, 100 on E itself, so a gap day interpolated towards E
    carries a large value (40 on E - 2, 70 on E - 1)."""
    per_sensor = {}
    for sensor in ("temperature", "turbidity", "do"):
        days = {}
        for date in pd.date_range(EVENT - 20 * DAY, EVENT + 5 * DAY, freq="D"):
            if date in (EVENT - 2 * DAY, EVENT - DAY):
                continue
            center = 100.0 if (sensor == "temperature" and date == EVENT) else 10.0
            days[iso(date)] = (hours(MIN_HOURS), center, 1.0)
        per_sensor[sensor] = days
    return series_through_pipeline(per_sensor, iso(EVENT - 40 * DAY), iso(EVENT + 5 * DAY))


def test_gap_filled_from_the_event_day_counts_as_missing_for_that_event():
    long, series = gap_before_event()
    filled = long.loc[(long["sensor"] == "temperature") & (long["status"] == "interpolated")]
    assert filled["date"].tolist() == [EVENT - 2 * DAY, EVENT - DAY]
    assert filled["right_anchor_date"].tolist() == [EVENT, EVENT]  # both gap days lean on the event day
    assert filled["mean"].tolist() == pytest.approx([40.0, 70.0])
    features, n_covered = f.window_row(series, EVENT, T1_DAYS, CFG)
    assert features["temperature_max_w"] == 11.0  # the 70 of E - 1 is not used
    assert features["temperature_mean_w"] == pytest.approx(10.0)
    assert n_covered == T1_DAYS - 2  # the two gap days are not covered for this event
    cumulative = f.cumulative_means(series, EVENT, CFG)
    assert cumulative["temperature_mean_cum"] == pytest.approx(10.0)


def test_the_same_gap_is_usable_for_a_later_reference_date():
    _, series = gap_before_event()
    later = EVENT + DAY  # the right anchor E is before this date, so the gap days may be used
    features, n_covered = f.window_row(series, later, T1_DAYS, CFG)
    assert n_covered == T1_DAYS
    expected_mean = (11 * 10.0 + 40.0 + 70.0 + 100.0) / T1_DAYS  # E - 13 .. E - 3, the gap days, E
    assert features["temperature_mean_w"] == pytest.approx(expected_mean)
    assert f.cumulative_means(series, later, CFG)["temperature_mean_cum"] > 10.0


def test_a_gap_anchored_on_the_reference_date_or_later_is_masked_but_an_earlier_anchor_is_not():
    day = pd.Timestamp("2021-07-10")
    anchors = {(day, "temperature"): day + 3 * DAY}
    series = make_series("2021-07-01", "2021-07-31", {day: {"temperature": (5.0, 1.0, 4.0, 6.0)}}, anchors)
    assert np.isnan(series.block(pd.DatetimeIndex([day]), day + 3 * DAY)[0, 0, 0])  # anchor on the reference date
    assert np.isnan(series.block(pd.DatetimeIndex([day]), day + 2 * DAY)[0, 0, 0])
    assert series.block(pd.DatetimeIndex([day]), day + 4 * DAY)[0, 0, 0] == 5.0  # reference date after the anchor


# ------------------------------------------------------------------ features by date, T1 and T2 tables


def toy_events() -> pd.DataFrame:
    """Pond 1 has no event on its t = 42 grid date; pond 2 starts at t = 14 (like pond 8)."""
    records = []
    for pond, ts, weights in ((1, (0, 14, 28, 56, 70), (3.0, 10.0, 30.0, 100.0, 200.0)),
                              (2, (14, 28, 42), (8.0, 20.0, 45.0))):
        for t, w in zip(ts, weights):
            records.append(
                {"pond": pond, "stream": CFG["streams"][pond], "date": DAY0 + t * DAY, "t": t, "weight": w, "length": w / 2}
            )
    return pd.DataFrame(records)


def series_on_days(dates, stats=(1.0, 0.1, 0.5, 1.5)) -> dict[int, f.DailySeries]:
    """The daily table of ponds 1 and 2 (first date to last grid date), three sensors present on `dates`."""
    first, last = CFG["daily"]["first_date"], iso(grid_dates(CFG)[-1])
    series = make_series(first, last, present(dates, stats=stats))
    return {1: series, 2: series}


def all_days() -> pd.DatetimeIndex:
    return pd.date_range(CFG["daily"]["first_date"], grid_dates(CFG)[-1], freq="D")


def tables(events: pd.DataFrame, series_by_pond: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    wq = f.wq_by_date(series_by_pond, CFG)
    return wq, f.build_t1(events, wq, CFG), f.build_t2(events, wq, CFG)


def test_wq_by_date_has_every_grid_date_for_every_pond_event_or_not():
    wq = f.wq_by_date(series_on_days(all_days()), CFG)
    grid = list(grid_dates(CFG))[1:]
    assert len(wq) == 2 * len(grid) == 32
    assert wq[["pond", "date"]].values.tolist()[: len(grid)] == [[1, d] for d in grid]
    assert wq["grid_k"].tolist()[: len(grid)] == list(range(1, 17))
    assert (wq["t"] == (wq["date"] - DAY0).dt.days).all()
    assert wq.loc[wq["pond"] == 2, "stream"].unique().tolist() == ["S12"]
    assert wq.columns.tolist()[:5] == ["pond", "stream", "date", "t", "grid_k"]


def test_stream_follows_the_config_in_wq_by_date_and_both_tables():
    series = series_on_days(all_days())[1]
    ponds = {1: "S12", 3: "S03", 9: "S09"}
    wq = f.wq_by_date({pond: series for pond in ponds}, CFG)
    assert wq.groupby("pond")["stream"].unique().map(list).to_dict() == {pond: [s] for pond, s in ponds.items()}
    events = pd.DataFrame(
        [
            {"pond": pond, "stream": stream, "date": DAY0 + t * DAY, "t": t, "weight": w, "length": w / 2}
            for pond, stream in ponds.items()
            for t, w in ((14, 5.0), (28, 9.0), (42, 20.0))
        ]
    )
    t1, t2 = f.build_t1(events, wq, CFG), f.build_t2(events, wq, CFG)
    assert t1.groupby("pond")["stream"].unique().map(list).to_dict() == {pond: [s] for pond, s in ponds.items()}
    assert t2.groupby("pond")["stream"].unique().map(list).to_dict() == {pond: [s] for pond, s in ponds.items()}


def test_wq_by_date_columns_per_prefix():
    wq = f.wq_by_date(series_on_days(all_days()), CFG)
    window = [f"{s}_{stat}_w" for s in SENSORS for stat in f.STATS]
    expected = (
        [f"t1_{c}" for c in window] + [f"t1_{s}_mean_cum" for s in SENSORS] + ["t1_n_covered_window"]
        + [f"t2_{c}" for c in window] + ["t2_n_covered_window"]
    )
    for prefix in CFG["wq_by_date"]["sensitivity_windows"]:
        expected += [f"{prefix}_{c}" for c in window] + [f"{prefix}_n_covered_window"]
    assert wq.columns.tolist()[5:] == expected
    assert len(expected) == 24 + 6 + 1 + 24 + 1 + 4 * (24 + 1)


def test_wq_by_date_windows_follow_the_configured_lengths():
    per_day = {d: {s: ((d - DAY0).days, 1.0, (d - DAY0).days - 1.0, (d - DAY0).days + 1.0) for s in SENSORS[:3]} for d in all_days()}
    series = make_series(CFG["daily"]["first_date"], iso(grid_dates(CFG)[-1]), per_day)  # mean = t on every day
    wq = f.wq_by_date({1: series}, CFG).set_index("grid_k")
    for k in (1, 5, 16):
        t = 14 * k
        row = wq.loc[k]
        assert row["t1_temperature_mean_w"] == pytest.approx(t - 7.5)  # t - 14 .. t - 1
        assert row["t2_temperature_mean_w"] == pytest.approx(t - 14.5)  # t - 28 .. t - 1
        assert row["t1w07_temperature_mean_w"] == pytest.approx(t - 4.0)
        assert row["t1w28_temperature_mean_w"] == pytest.approx(t - 14.5)
        assert row["t2w14_temperature_mean_w"] == pytest.approx(t - 7.5)
        assert row["t2w42_temperature_mean_w"] == pytest.approx(t - 21.5)
        assert row["t1_temperature_mean_cum"] == pytest.approx((t - 1) / 2)  # mean of 0 .. t - 1
        assert row["t1_temperature_max_w"] == t  # (t - 1) + 1: the grid date itself is never used
        for prefix, length in (("t1", 14), ("t2", 28), ("t1w07", 7), ("t1w28", 28), ("t2w14", 14), ("t2w42", 42)):
            assert row[f"{prefix}_n_covered_window"] == length


def test_t1_and_t2_features_are_the_wq_by_date_columns_of_the_same_pond_and_date():
    events, series = toy_events(), series_on_days(all_days())
    wq, t1, t2 = tables(events, series)
    wq = wq.set_index(["pond", "date"])
    window = [f"{s}_{stat}_w" for s in SENSORS for stat in f.STATS]
    t1_features = window + [f"{s}_mean_cum" for s in SENSORS] + ["n_covered_window"]
    t2_features = window + ["n_covered_window"]
    assert len(t1_features) == 24 + 6 + 1 and len(t2_features) == 24 + 1
    assert set(t1_features) <= set(t1.columns) and set(t2_features) <= set(t2.columns)
    assert len(t1) > 0 and len(t2) > 0
    for row in t1.itertuples():
        source = wq.loc[(row.pond, row.date)]
        for column in t1_features:
            assert getattr(row, column) == pytest.approx(source[f"t1_{column}"], nan_ok=True)
    for row in t2.itertuples():
        source = wq.loc[(row.pond, row.origin_date)]
        for column in t2_features:
            assert getattr(row, column) == pytest.approx(source[f"t2_{column}"], nan_ok=True)


def test_event_dates_that_are_not_in_wq_by_date_are_an_error():
    events = toy_events().assign(date=lambda d: d["date"] + DAY, t=lambda d: d["t"] + 1)
    wq = f.wq_by_date(series_on_days(all_days()), CFG)
    with pytest.raises(ValueError):
        f.build_t1(events, wq, CFG)


def test_t1_has_one_row_per_event_after_day_0_with_log_targets_and_squared_time():
    _, t1, _ = tables(toy_events(), series_on_days(all_days()))
    assert t1[["pond", "t"]].values.tolist() == [[1, 14], [1, 28], [1, 56], [1, 70], [2, 14], [2, 28], [2, 42]]
    assert t1.columns.tolist()[:9] == ["pond", "stream", "date", "t", "t2", "log_w", "w", "log_l", "l"]
    assert t1.columns.tolist()[-2:] == ["n_covered_window", "eligible_eval"]
    assert t1["stream"].tolist() == ["S12"] * 7
    row = t1.iloc[2]
    assert (row["t2"], row["w"], row["l"]) == (56**2, 100.0, 50.0)
    assert row["log_w"] == pytest.approx(np.log(100.0)) and row["log_l"] == pytest.approx(np.log(50.0))
    assert row["date"] == DAY0 + 56 * DAY
    assert "temperature_mean_cum" in t1.columns and "temperature_mean_w" in t1.columns
    assert t1["eligible_eval"].all()


def test_t2_horizon_needs_an_event_exactly_14_h_days_after_the_origin():
    _, _, t2 = tables(toy_events(), series_on_days(all_days()))
    keys = t2[["pond", "t_k", "horizon"]].values.tolist()
    # pond 1 events at t = 0, 14, 28, 56, 70: origin 14 has +14 (t = 28) but no +28 (t = 42);
    # origin 28 has no +14 (t = 42) but has +28 (t = 56); origin 56 has +14 (t = 70) only.
    assert keys == [[1, 14, 1], [1, 28, 2], [1, 56, 1], [2, 28, 1]]
    assert (t2["dt_target"] == 14 * t2["horizon"]).all()
    assert t2["stream"].tolist() == ["S12"] * 4


def test_t2_previous_event_is_the_previous_available_one_whatever_the_gap():
    _, _, t2 = tables(toy_events(), series_on_days(all_days()))
    far = t2.query("pond == 1 and t_k == 56").iloc[0]  # the t = 42 event is missing: previous is t = 28
    assert far["dt_prev"] == 28 and far["log_w_prev"] == pytest.approx(np.log(30.0))
    assert (far["w_origin"], far["w_target"]) == (100.0, 200.0)
    assert far["y"] == pytest.approx(np.log(200.0 / 100.0))
    assert far["origin_date"] == DAY0 + 56 * DAY and far["target_date"] == DAY0 + 70 * DAY
    skip = t2.query("pond == 1 and t_k == 28").iloc[0]  # horizon 2 target is t = 56
    assert (skip["horizon"], skip["dt_prev"], skip["dt_target"]) == (2, 14, 28)
    assert skip["y"] == pytest.approx(np.log(100.0 / 30.0))
    assert skip["log_w_k"] == pytest.approx(np.log(30.0))


def test_t2_origin_needs_an_earlier_event_and_a_target_must_exist():
    events = toy_events()
    first_only = events.query("pond == 2 and t == 14")  # a lone first event: no origin, no sample
    _, _, none = tables(first_only, series_on_days(all_days()))
    assert len(none) == 0
    pair = events.query("pond == 2 and t in (14, 28)")  # origin t = 28 has an earlier event but no target
    _, _, still_none = tables(pair, series_on_days(all_days()))
    assert len(still_none) == 0
    triple = events.query("pond == 2")  # origin t = 28 now has the t = 42 target
    _, _, one = tables(triple, series_on_days(all_days()))
    assert one[["t_k", "horizon", "dt_prev", "dt_target"]].values.tolist() == [[28, 1, 14, 14]]


def test_t1_eligibility_needs_the_configured_number_of_covered_days():
    minimum = CFG["eligibility"]["t1_min_covered_days"]
    event = toy_events().query("pond == 1 and t == 28")
    last_days = [event["date"].iloc[0] - k * DAY for k in range(1, minimum + 1)]
    _, exactly, _ = tables(event, series_on_days(last_days))
    _, fewer, _ = tables(event, series_on_days(last_days[:-1]))
    assert (exactly.iloc[0]["n_covered_window"], exactly.iloc[0]["eligible_eval"]) == (minimum, True)
    assert (fewer.iloc[0]["n_covered_window"], fewer.iloc[0]["eligible_eval"]) == (minimum - 1, False)
    assert exactly.iloc[0]["temperature_mean_cum"] == pytest.approx(1.0)  # present days only
    _, none, _ = tables(event, series_on_days([]))
    assert none.iloc[0]["n_covered_window"] == 0 and not none.iloc[0]["eligible_eval"]
    assert np.isnan(none.iloc[0]["temperature_mean_w"]) and np.isnan(none.iloc[0]["temperature_mean_cum"])


def test_t2_eligibility_needs_the_configured_number_of_covered_days():
    minimum = CFG["eligibility"]["t2_min_covered_days"]
    events = toy_events().query("pond == 2")  # one sample: origin t = 28, h = 1
    origin = DAY0 + 28 * DAY
    before = [origin - k * DAY for k in range(1, T2_DAYS + 1)]
    _, _, exactly = tables(events, series_on_days(before[:minimum]))
    _, _, fewer = tables(events, series_on_days(before[: minimum - 1]))
    _, _, outside = tables(events, series_on_days([origin, origin - 29 * DAY]))
    assert (exactly.iloc[0]["n_covered_window"], exactly.iloc[0]["eligible_eval"]) == (minimum, True)
    assert (fewer.iloc[0]["n_covered_window"], fewer.iloc[0]["eligible_eval"]) == (minimum - 1, False)
    assert (outside.iloc[0]["n_covered_window"], outside.iloc[0]["eligible_eval"]) == (0, False)  # origin day and day 29 back never count


def test_eligibility_table_counts_origins_samples_and_eligible_per_pond_and_horizon():
    events = toy_events()
    _, t1, t2 = tables(events, series_on_days(all_days()))
    table = f.eligibility_table(events, t1, t2, {**CFG, "cohort": {**CFG["cohort"], "ponds": [1, 2]}})
    assert table.columns.tolist() == [
        "pond", "t1_events", "t1_eligible", "t2_origins",
        "t2_h1_samples", "t2_h1_eligible", "t2_h2_samples", "t2_h2_eligible",
    ]
    assert table.values.tolist() == [[1, 4, 4, 4, 2, 2, 1, 1], [2, 3, 3, 2, 1, 1, 0, 0]]
    # samples are counted before the eligibility rule: with no covered days none is eligible
    _, t1_bare, t2_bare = tables(events, series_on_days([]))
    bare = f.eligibility_table(events, t1_bare, t2_bare, {**CFG, "cohort": {**CFG["cohort"], "ponds": [1, 2]}})
    assert bare.values.tolist() == [[1, 4, 0, 4, 2, 0, 1, 0], [2, 3, 0, 2, 1, 0, 0, 0]]
