"""Step 3 of the Phase 4 data pipeline: daily water quality, features by date, T1 and T2 tables.

Run from the repo root:  .venv/bin/python -I src/features.py

Reads data/processed/events.csv (src/events.py) and data/interim/qc_rows_pondNN.csv.gz
(src/qc.py). For every cohort pond it builds the daily statistics of each sensor from
2021-05-22 to the last grid date, fills short gaps by linear interpolation and records the
observed days on both sides of every filled gap (the anchors). It then cuts the window
features of every pond at every grid date, event or not (data/processed/wq_by_date.csv), and
joins the T1 and T2 tables from that file, so both always agree with it.

Rules for the features of a reference date d (an event or origin date):
- the window is the days before d; d itself is never used;
- an interpolated sensor-day whose right anchor is d or later counts as missing for d;
- a sensor's features are NaN unless it has a value on enough days of the window;
- a day is covered when enough sensors have a value; coverage decides eligibility only;
- nothing is imputed here: a sensor without enough days gives NaN.

Writes data/processed/daily_wq.csv, wq_by_date.csv, t1_table.csv, t2_table.csv and
results/qc/coverage.csv, interpolation_counts.csv, eligibility.csv. Specification:
ANALYSIS_PLAN.md sections 2.6, 2.7, 3, 4, 5.1 and the lead's amendments in
tasks/phase4_experiments.md step 3. Thresholds: configs/phase4_data.yaml.
"""

from __future__ import annotations

import datetime
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.pipeline_io import grid_dates, load_config, log_run, repo_path, write_csv  # noqa: E402

# Daily statistics in column order of `DailySeries.values`.
STATS = ("mean", "sd", "min", "max")
STATUSES = ("observed", "interpolated", "missing")
DAILY_COLUMNS = [
    "pond", "date", "t", "sensor", "mean", "sd", "min", "max", "n_hours", "n_values", "status",
    "left_anchor_date", "right_anchor_date",
]
T2_SAMPLE_COLUMNS = [
    "pond", "stream", "origin_date", "target_date", "horizon", "t_k", "dt_prev", "dt_target",
    "log_w_k", "log_w_prev", "w_origin", "w_target", "y",
]
DAY = pd.Timedelta(days=1)


def sensor_names(cfg: dict) -> list[str]:
    """Sensor names of the config in output order."""
    return [entry["name"] for entry in cfg["sensors"]]


def read_qc_rows(pond: int, cfg: dict) -> pd.DataFrame:
    """The QC'd rows of one pond written by src/qc.py (ts and date parsed, floats exact)."""
    return pd.read_csv(
        repo_path(cfg["paths"]["qc_rows"].format(pond=pond)),
        parse_dates=["ts", "date"],
        float_precision="round_trip",
    )


# ------------------------------------------------------------------ daily statistics


def sensor_day_table(
    rows: pd.DataFrame, sensor: str, days: pd.DatetimeIndex, min_hours: int, ddof: int
) -> pd.DataFrame:
    """Daily statistics of one sensor for every date in `days`.

    `rows` has the columns date, hour and the QC'd `sensor` (NaN when invalid). Columns:
    n_values and n_hours (valid values, distinct clock hours holding one), mean, sd, min,
    max and status. A sensor-day is observed when n_hours >= min_hours; otherwise the four
    statistics are NaN and the status is missing, whatever partial data the day has.
    """
    valid = rows.loc[rows[sensor].notna(), ["date", "hour", sensor]]
    by_day = valid.groupby("date")
    table = pd.DataFrame(
        {
            "n_values": by_day[sensor].size(),
            "n_hours": by_day["hour"].nunique(),
            "mean": by_day[sensor].mean(),
            "sd": by_day[sensor].std(ddof=ddof),
            "min": by_day[sensor].min(),
            "max": by_day[sensor].max(),
        }
    )
    table.index = table.index.as_unit("ns")
    table = table.reindex(days.as_unit("ns"))
    counts = ["n_values", "n_hours"]
    table[counts] = table[counts].fillna(0).astype("int64")
    observed = table["n_hours"] >= min_hours
    table.loc[~observed, list(STATS)] = np.nan
    table["status"] = np.where(observed, "observed", "missing")
    return table


def fill_short_gaps(table: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Interpolate mean, sd, min and max linearly over short gaps of missing sensor-days.

    `table` is the output of `sensor_day_table` over consecutive calendar days. A run of 1 to
    `max_gap` missing days with an observed day on both sides is filled, each statistic
    separately, and its status becomes interpolated. Longer runs, and runs at the start or
    end of the table, stay missing. n_values and n_hours keep the day's own counts. Adds
    left_anchor_date and right_anchor_date: the observed days on both sides of a filled gap
    (NaT for every other day).
    """
    out = table.copy()
    stats = table[list(STATS)].to_numpy(dtype="float64")
    filled = stats.copy()
    interpolated = np.zeros(len(table), dtype=bool)
    left_anchor = np.full(len(table), np.datetime64("NaT", "ns"))
    right_anchor = left_anchor.copy()
    dates = table.index.to_numpy(dtype="datetime64[ns]")
    observed = np.flatnonzero((table["status"] == "observed").to_numpy())
    for left, right in zip(observed[:-1], observed[1:]):
        gap = right - left - 1
        if 1 <= gap <= max_gap:
            weight = (np.arange(left + 1, right) - left) / (right - left)
            filled[left + 1 : right] = stats[left] + np.outer(weight, stats[right] - stats[left])
            interpolated[left + 1 : right] = True
            left_anchor[left + 1 : right] = dates[left]
            right_anchor[left + 1 : right] = dates[right]
    out[list(STATS)] = filled
    out.loc[interpolated, "status"] = "interpolated"
    out["left_anchor_date"] = left_anchor
    out["right_anchor_date"] = right_anchor
    return out


def pond_daily(rows: pd.DataFrame, pond: int, days: pd.DatetimeIndex, cfg: dict) -> pd.DataFrame:
    """Long daily table of one pond: DAILY_COLUMNS, one row per date and sensor.

    Dates are `days` (consecutive calendar days); t is days since day 0. Sensors follow the
    config order within a date.
    """
    day0 = pd.Timestamp(cfg["cohort"]["day0"]).as_unit("ns")
    parts = []
    for rank, sensor in enumerate(sensor_names(cfg)):
        table = sensor_day_table(
            rows, sensor, days, cfg["daily"]["min_hours"], cfg["daily"]["sd_ddof"]
        )
        table = fill_short_gaps(table, cfg["daily"]["max_gap_days"])
        part = table.rename_axis("date").reset_index()
        part["pond"] = pond
        part["t"] = (part["date"] - day0).dt.days
        part["sensor"] = sensor
        part["rank"] = rank
        parts.append(part)
    long = pd.concat(parts, ignore_index=True).sort_values(["date", "rank"], kind="stable")
    return long[DAILY_COLUMNS].reset_index(drop=True)


# ------------------------------------------------------------------ windows and features


@dataclass(frozen=True)
class DailySeries:
    """One pond's daily statistics as an array `values[day, sensor, stat]` (stats in STATS).

    `days` are consecutive calendar dates. An entry is NaN where the sensor has no value
    (neither observed nor interpolated) on that day; the four statistics of a sensor-day are
    all present or all NaN. `right_anchor[day, sensor]` is the observed day that closes the
    gap an interpolated sensor-day was filled from, NaT for every other sensor-day.
    """

    days: pd.DatetimeIndex
    sensors: tuple[str, ...]
    values: np.ndarray
    right_anchor: np.ndarray

    def block(self, days: pd.DatetimeIndex, reference_date: pd.Timestamp) -> np.ndarray:
        """Values for `days` as seen from `reference_date`, shape (len(days), n_sensors, len(STATS)).

        An interpolated sensor-day whose right anchor is `reference_date` or later is
        returned as missing, so nothing observed on or after the reference date leaks into
        a window. Raises ValueError when a date is outside the series, so a window never
        silently shrinks.
        """
        position = self.days.as_unit("ns").get_indexer(days.as_unit("ns"))
        if (position < 0).any():
            raise ValueError(f"window {days[0]:%Y-%m-%d} to {days[-1]:%Y-%m-%d} leaves the daily table")
        block = self.values[position].copy()
        block[self.right_anchor[position] >= reference_date.to_datetime64()] = np.nan
        return block


def series_from_long(long: pd.DataFrame, sensors: list[str]) -> DailySeries:
    """DailySeries of one pond from its long daily table."""
    days = pd.DatetimeIndex(sorted(long["date"].unique())).as_unit("ns")
    values = np.full((len(days), len(sensors), len(STATS)), np.nan)
    right_anchor = np.full((len(days), len(sensors)), np.datetime64("NaT", "ns"))
    for index, sensor in enumerate(sensors):
        part = long.loc[long["sensor"] == sensor].set_index("date")
        part.index = part.index.as_unit("ns")
        part = part.reindex(days)
        values[:, index, :] = part[list(STATS)].to_numpy(dtype="float64")
        right_anchor[:, index] = part["right_anchor_date"].to_numpy(dtype="datetime64[ns]")
    return DailySeries(days, tuple(sensors), values, right_anchor)


def sensors_with_value(values: np.ndarray) -> np.ndarray:
    """Per day, the number of sensors that have a daily value; `values` is [day, sensor, stat]."""
    return np.isfinite(values[:, :, 0]).sum(axis=1)


def window_days(reference_date: pd.Timestamp, length: int) -> pd.DatetimeIndex:
    """The `length` calendar days that end the day before `reference_date`.

    For an event or origin on date d these are d - length .. d - 1; d itself is never in it.
    """
    return pd.date_range(end=reference_date - DAY, periods=length, freq="D")


def window_row(
    series: DailySeries, reference_date: pd.Timestamp, length: int, cfg: dict
) -> tuple[dict[str, float], int]:
    """Window features and the number of covered days for the window before `reference_date`.

    Per sensor, over the window days on which it has a value (as seen from the reference
    date): `{sensor}_mean_w` mean of daily means, `{sensor}_sd_w` mean of daily SDs,
    `{sensor}_min_w` minimum of daily minima, `{sensor}_max_w` maximum of daily maxima. All
    four are NaN when the sensor has a value on fewer than `min_sensor_days` window days.
    The count of covered days is independent of that: a day is covered when at least
    `min_sensors_covered` sensors have a value.
    """
    block = series.block(window_days(reference_date, length), reference_date)
    features: dict[str, float] = {}
    for index, sensor in enumerate(series.sensors):
        has = np.isfinite(block[:, index, 0])
        if has.sum() >= cfg["windows"]["min_sensor_days"]:
            mean_w = block[has, index, 0].mean()
            sd_w = block[has, index, 1].mean()
            min_w = block[has, index, 2].min()
            max_w = block[has, index, 3].max()
        else:
            mean_w = sd_w = min_w = max_w = np.nan
        features.update(
            {
                f"{sensor}_mean_w": mean_w,
                f"{sensor}_sd_w": sd_w,
                f"{sensor}_min_w": min_w,
                f"{sensor}_max_w": max_w,
            }
        )
    covered = sensors_with_value(block) >= cfg["daily"]["min_sensors_covered"]
    return features, int(covered.sum())


def cumulative_means(
    series: DailySeries, reference_date: pd.Timestamp, cfg: dict
) -> dict[str, float]:
    """`{sensor}_mean_cum`: mean of daily means from day 0 to the day before `reference_date`.

    Over the days on which the sensor has a value as seen from the reference date; NaN when
    that is fewer than `min_sensor_days` days.
    """
    day0 = pd.Timestamp(cfg["cohort"]["day0"])
    block = series.block(pd.date_range(day0, reference_date - DAY, freq="D"), reference_date)
    out: dict[str, float] = {}
    for index, sensor in enumerate(series.sensors):
        means = block[:, index, 0]
        has = np.isfinite(means)
        enough = has.sum() >= cfg["windows"]["min_sensor_days"]
        out[f"{sensor}_mean_cum"] = means[has].mean() if enough else np.nan
    return out


# ------------------------------------------------------------------ features by date


def _prefixed(prefix: str, features: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{name}": value for name, value in features.items()}


def wq_features(series: DailySeries, date: pd.Timestamp, cfg: dict) -> dict[str, float]:
    """Every wq_by_date feature column of one pond at one grid date.

    `t1_`: the 14-day window, the cumulative means and `t1_n_covered_window`. `t2_`: the
    28-day window and `t2_n_covered_window`. Then, per prefix of the config's
    sensitivity windows, that window's features and its n_covered_window.
    """
    out: dict[str, float] = {}
    window, n_covered = window_row(series, date, cfg["windows"]["t1_days"], cfg)
    out.update(_prefixed("t1", window))
    out.update(_prefixed("t1", cumulative_means(series, date, cfg)))
    out["t1_n_covered_window"] = n_covered
    window, n_covered = window_row(series, date, cfg["windows"]["t2_days"], cfg)
    out.update(_prefixed("t2", window))
    out["t2_n_covered_window"] = n_covered
    for prefix, length in cfg["wq_by_date"]["sensitivity_windows"].items():
        window, n_covered = window_row(series, date, length, cfg)
        out.update(_prefixed(prefix, window))
        out[f"{prefix}_n_covered_window"] = n_covered
    return out


def wq_by_date(series_by_pond: dict[int, DailySeries], cfg: dict) -> pd.DataFrame:
    """One row per pond and grid date k_first..k_last, whether or not the pond has an event.

    Columns: pond, stream, date, t, grid_k, then the columns of `wq_features`.
    """
    day0 = pd.Timestamp(cfg["cohort"]["day0"])
    k_first = cfg["cohort"]["grid"]["k_first"]
    rows = []
    for pond, series in series_by_pond.items():
        for k, date in enumerate(grid_dates(cfg), start=k_first):
            if k < cfg["wq_by_date"]["k_first"]:
                continue
            rows.append(
                {
                    "pond": pond,
                    "stream": cfg["streams"][pond],
                    "date": date,
                    "t": (date - day0).days,
                    "grid_k": k,
                    **wq_features(series, date, cfg),
                }
            )
    return pd.DataFrame(rows)


def window_set(wq: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """pond, date and the columns of one wq_by_date prefix (`t1`, `t2`, ...) with it removed."""
    names = [c for c in wq.columns if c.startswith(prefix + "_")]
    return wq[["pond", "date", *names]].rename(columns={c: c[len(prefix) + 1 :] for c in names})


def _ns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Dates as datetime64[ns], so frames from CSV and from the pipeline join on equal keys."""
    return frame.assign(**{c: frame[c].astype("datetime64[ns]") for c in columns})


# ------------------------------------------------------------------ T1 and T2 tables


def build_t1(events: pd.DataFrame, wq: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T1 table: one row per event with t > 0, features joined from wq_by_date (`t1_` columns).

    Columns: pond, stream, date, t, t2, log_w, w, log_l, l, the 24 window features of the 14
    days before the event, the 6 cumulative means, n_covered_window and eligible_eval
    (t > 0 and at least t1_min_covered_days covered days). Raises ValueError when an event
    date is not in wq_by_date.
    """
    selected = _ns(events.loc[events["t"] > 0], ["date"])
    features = window_set(_ns(wq, ["date"]), "t1")
    joined = selected.merge(features, on=["pond", "date"], how="left", validate="one_to_one")
    feature_columns = [c for c in features.columns if c not in ("pond", "date")]
    if joined["n_covered_window"].isna().any():
        raise ValueError("an event date is missing from wq_by_date")
    table = pd.DataFrame(
        {
            "pond": joined["pond"],
            "stream": joined["stream"],
            "date": joined["date"],
            "t": joined["t"],
            "t2": joined["t"] ** 2,
            "log_w": np.log(joined["weight"]),
            "w": joined["weight"],
            "log_l": np.log(joined["length"]),
            "l": joined["length"],
        }
    )
    table = pd.concat([table, joined[feature_columns]], axis=1)
    table["n_covered_window"] = table["n_covered_window"].astype("int64")
    table["eligible_eval"] = (table["t"] > 0) & (
        table["n_covered_window"] >= cfg["eligibility"]["t1_min_covered_days"]
    )
    return table.reset_index(drop=True)


def t2_samples(events: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T2 samples without features: one row per origin and horizon with a calendar target.

    Events are indexed per pond in date order. Origin k needs an earlier event k - 1, the
    pond's previous available event whatever the gap. Horizon h exists only when the pond
    has an event dated exactly `horizon_days * h` days after the origin; dt_target is
    therefore that many days. Columns: pond, stream, origin_date, target_date, horizon, t_k,
    dt_prev, dt_target, log_w_k, log_w_prev, w_origin, w_target, y = log(w_target / w_origin).
    """
    rows = []
    for pond, group in events.groupby("pond", sort=True):
        ev = group.sort_values("date").reset_index(drop=True)
        index_of_t = {t: i for i, t in enumerate(ev["t"])}
        for k in range(1, len(ev)):
            for horizon in cfg["windows"]["horizons"]:
                target = index_of_t.get(ev.at[k, "t"] + horizon * cfg["windows"]["horizon_days"])
                if target is None:
                    continue
                w_origin, w_target = ev.at[k, "weight"], ev.at[target, "weight"]
                rows.append(
                    {
                        "pond": pond,
                        "stream": ev.at[k, "stream"],
                        "origin_date": ev.at[k, "date"],
                        "target_date": ev.at[target, "date"],
                        "horizon": horizon,
                        "t_k": ev.at[k, "t"],
                        "dt_prev": ev.at[k, "t"] - ev.at[k - 1, "t"],
                        "dt_target": ev.at[target, "t"] - ev.at[k, "t"],
                        "log_w_k": np.log(w_origin),
                        "log_w_prev": np.log(ev.at[k - 1, "weight"]),
                        "w_origin": w_origin,
                        "w_target": w_target,
                        "y": np.log(w_target / w_origin),
                    }
                )
    return pd.DataFrame(rows, columns=T2_SAMPLE_COLUMNS)


def build_t2(events: pd.DataFrame, wq: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """T2 table: the samples of `t2_samples` with the `t2_` features of wq_by_date at the origin.

    Adds the 24 window features of the 28 days before the origin, n_covered_window and
    eligible_eval (at least t2_min_covered_days covered days). Raises ValueError when an
    origin date is not in wq_by_date.
    """
    samples = _ns(t2_samples(events, cfg), ["origin_date", "target_date"])
    features = window_set(_ns(wq, ["date"]), "t2").rename(columns={"date": "origin_date"})
    joined = samples.merge(features, on=["pond", "origin_date"], how="left", validate="many_to_one")
    if joined["n_covered_window"].isna().any():
        raise ValueError("an origin date is missing from wq_by_date")
    joined["n_covered_window"] = joined["n_covered_window"].astype("int64")
    joined["eligible_eval"] = joined["n_covered_window"] >= cfg["eligibility"]["t2_min_covered_days"]
    return joined


# ------------------------------------------------------------------ summaries


def coverage_table(series: DailySeries, pond: int, cfg: dict) -> pd.DataFrame:
    """Per date of one pond: n_sensors_with_value and covered (pond, date, t, ...).

    Describes the daily table itself, observed and interpolated sensor-days alike. The
    coverage an event sees (its n_covered_window) can be lower, see `DailySeries.block`.
    """
    day0 = pd.Timestamp(cfg["cohort"]["day0"]).as_unit("ns")
    n_sensors = sensors_with_value(series.values)
    return pd.DataFrame(
        {
            "pond": pond,
            "date": series.days,
            "t": (series.days - day0).days,
            "n_sensors_with_value": n_sensors,
            "covered": n_sensors >= cfg["daily"]["min_sensors_covered"],
        }
    )


def interpolation_counts(long: pd.DataFrame) -> pd.DataFrame:
    """Sensor-days per pond and sensor by status (observed, interpolated, missing)."""
    counts = (
        long.groupby(["pond", "sensor"], sort=False)["status"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=list(STATUSES), fill_value=0)
    )
    counts.insert(0, "n_days", counts.sum(axis=1))
    counts.columns = ["n_days", *("n_" + status for status in STATUSES)]
    return counts.reset_index()


def eligibility_table(
    events: pd.DataFrame, t1: pd.DataFrame, t2: pd.DataFrame, cfg: dict
) -> pd.DataFrame:
    """Per pond: T1 events (t > 0) and eligible; T2 origins, then samples and eligible per horizon.

    `t2_origins` counts events that have an earlier event. `t2_hN_samples` counts the samples
    of horizon N that exist (an event dated origin + 14 N days), before the eligibility rule;
    `t2_hN_eligible` counts those that pass it.
    """
    rows = []
    for pond in cfg["cohort"]["ponds"]:
        one = t1.loc[t1["pond"] == pond]
        n_events = int((events["pond"] == pond).sum())
        row = {
            "pond": pond,
            "t1_events": len(one),
            "t1_eligible": int(one["eligible_eval"].sum()),
            "t2_origins": max(n_events - 1, 0),
        }
        for horizon in cfg["windows"]["horizons"]:
            two = t2.loc[(t2["pond"] == pond) & (t2["horizon"] == horizon)]
            row[f"t2_h{horizon}_samples"] = len(two)
            row[f"t2_h{horizon}_eligible"] = int(two["eligible_eval"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _dates_as_text(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame.assign(**{c: frame[c].dt.strftime("%Y-%m-%d") for c in columns})


def main() -> None:
    started = datetime.datetime.now(datetime.timezone.utc)
    clock = time.perf_counter()
    cfg = load_config()
    sensors = sensor_names(cfg)
    events = pd.read_csv(
        repo_path(cfg["paths"]["events"]), parse_dates=["date"], float_precision="round_trip"
    )
    days = pd.date_range(cfg["daily"]["first_date"], grid_dates(cfg)[-1], freq="D")

    long_parts, coverage_parts, series_by_pond = [], [], {}
    for pond in cfg["cohort"]["ponds"]:
        long = pond_daily(read_qc_rows(pond, cfg), pond, days, cfg)
        series = series_from_long(long, sensors)
        series_by_pond[pond] = series
        long_parts.append(long)
        coverage_parts.append(coverage_table(series, pond, cfg))
        n_filled = (long["status"] == "interpolated").sum()
        print(f"pond {pond}: {len(days)} days, {n_filled} interpolated sensor-days")

    daily = pd.concat(long_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    wq = wq_by_date(series_by_pond, cfg)
    t1 = build_t1(events, wq, cfg)
    t2 = build_t2(events, wq, cfg)
    eligibility = eligibility_table(events, t1, t2, cfg)

    anchors = ["date", "left_anchor_date", "right_anchor_date"]
    write_csv(_dates_as_text(daily, anchors), repo_path(cfg["paths"]["daily_wq"]))
    write_csv(_dates_as_text(wq, ["date"]), repo_path(cfg["paths"]["wq_by_date"]))
    write_csv(_dates_as_text(coverage, ["date"]), repo_path(cfg["paths"]["coverage"]))
    write_csv(interpolation_counts(daily), repo_path(cfg["paths"]["interpolation_counts"]))
    write_csv(_dates_as_text(t1, ["date"]), repo_path(cfg["paths"]["t1_table"]))
    write_csv(_dates_as_text(t2, ["origin_date", "target_date"]), repo_path(cfg["paths"]["t2_table"]))
    write_csv(eligibility, repo_path(cfg["paths"]["eligibility"]))

    print(f"wq_by_date rows {len(wq)}, T1 rows {len(t1)}, T2 rows {len(t2)}")
    print(eligibility.to_string(index=False))
    seconds = time.perf_counter() - clock
    log_run(cfg, "features.py", started, seconds)
    print(f"done in {seconds:.1f} s")


if __name__ == "__main__":
    main()
