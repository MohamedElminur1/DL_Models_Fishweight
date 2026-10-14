"""Synthetic T1, T2, daily and wq_by_date tables for the Phase 4 unit tests.

Everything here is a test-only construction with the column layout of data/processed/
(t1_table.csv, t2_table.csv, daily_wq.csv, wq_by_date.csv). None of it is a measurement and
none of it is ever written to data/ or results/. Growth is a logistic curve with noise, sensor
values are random, short gaps are interpolated with anchors as in src/features.py, ponds 1
and 2 share one sensor stream (the same daily values except for ammonia, as in the dataset),
`sparse_ponds` get almost no sensor days so that they have no eligible rows, and `blackouts`
remove all sensors on chosen days. The window features follow the plan's rules: the window is
the days before the reference date, an interpolated sensor-day whose right anchor is the
reference date or later counts as missing, a sensor needs three window days.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

SENSORS = ["temperature", "turbidity", "do", "ph", "ammonia_log10", "nitrate"]
STATS = ["mean", "sd", "min", "max"]
DAY0 = pd.Timestamp("2021-06-19")
FIRST_DAY = pd.Timestamp("2021-05-22")
PONDS = (1, 2, 3, 4, 6, 7, 8, 9)
STREAMS = {1: "S12", 2: "S12", 3: "S03", 4: "S04", 6: "S06", 7: "S07", 8: "S08", 9: "S09"}
LEVELS = np.array([25.0, 40.0, 6.0, 7.2, 1.0, 300.0])
DAY0_OFFSET = (DAY0 - FIRST_DAY).days
SENSITIVITY_WINDOWS = {"t1w07": 7, "t1w28": 28, "t2w14": 14, "t2w42": 42}


class Inputs(NamedTuple):
    t1: pd.DataFrame
    t2: pd.DataFrame
    daily: pd.DataFrame
    wq: pd.DataFrame


def _fill_gaps(vals: np.ndarray, observed: np.ndarray, max_gap: int = 2):
    """Linear interpolation over runs of at most `max_gap` missing days with observed ends."""
    filled = vals.copy()
    status = np.where(observed, "observed", "missing").astype(object)
    right = np.full(observed.shape, -1, dtype=int)
    left = np.full(observed.shape, -1, dtype=int)
    for s in range(observed.shape[1]):
        seen = np.flatnonzero(observed[:, s])
        for a, b in zip(seen[:-1], seen[1:]):
            if 1 <= b - a - 1 <= max_gap:
                for j in range(a + 1, b):
                    filled[j, s] = vals[a, s] + (j - a) / (b - a) * (vals[b, s] - vals[a, s])
                    status[j, s], right[j, s], left[j, s] = "interpolated", b, a
    return filled, status != "missing", status, left, right


def _raw_pond(n_days: int, rng: np.random.Generator, rate: float) -> tuple[np.ndarray, np.ndarray]:
    mean = LEVELS + rng.normal(0, 0.1, size=(n_days, 6)) * LEVELS
    sd = np.abs(rng.normal(0.3, 0.1, size=(n_days, 6))) * LEVELS * 0.05 + 1e-3
    lo = mean - sd * rng.uniform(1, 3, size=(n_days, 6))
    hi = mean + sd * rng.uniform(1, 3, size=(n_days, 6))
    vals = np.stack([mean, sd, lo, hi], axis=-1)
    observed = rng.uniform(size=(n_days, 6)) > rate
    return np.where(observed[..., None], vals, np.nan), observed


def _daily_arrays(ponds, n_days: int, rng, missing_rate: float, sparse_ponds, blackouts) -> dict:
    """Per pond: filled stats [days, sensors, stats], valid, status, left and right anchors."""
    raw = {}
    for pond in ponds:
        rate = 0.97 if pond in sparse_ponds else missing_rate
        if pond == 2 and 1 in raw:
            vals, observed = (a.copy() for a in raw[1])  # shared stream ...
            own_vals, own_obs = _raw_pond(n_days, rng, rate)
            vals[:, 4], observed[:, 4] = own_vals[:, 4], own_obs[:, 4]  # ... except ammonia
            raw[2] = (vals, observed)
        else:
            raw[pond] = _raw_pond(n_days, rng, rate)
    arrays = {}
    for pond, (vals, observed) in raw.items():
        observed = observed.copy()
        for p, first, last in blackouts:
            if p == pond:
                observed[first:last + 1] = False
        vals = np.where(observed[..., None], vals, np.nan)
        arrays[pond] = _fill_gaps(vals, observed)
    return arrays


def make_daily(arrays: dict) -> pd.DataFrame:
    """The daily_wq.csv layout: one row per pond, date and sensor."""
    frames = []
    for pond, (vals, valid, status, left, right) in arrays.items():
        n_days = vals.shape[0]
        dates = FIRST_DAY + pd.to_timedelta(np.arange(n_days), unit="D")
        text = np.array(dates.strftime("%Y-%m-%d"))
        for s, name in enumerate(SENSORS):
            frames.append(pd.DataFrame({
                "pond": pond, "date": text, "t": (dates - DAY0).days, "sensor": name,
                "mean": vals[:, s, 0], "sd": vals[:, s, 1], "min": vals[:, s, 2], "max": vals[:, s, 3],
                "n_hours": np.where(valid[:, s], 14, 0), "n_values": np.where(valid[:, s], 500, 0),
                "status": status[:, s],
                "left_anchor_date": np.where(left[:, s] >= 0, text[np.maximum(left[:, s], 0)], None),
                "right_anchor_date": np.where(right[:, s] >= 0, text[np.maximum(right[:, s], 0)], None),
            }))
    return pd.concat(frames, ignore_index=True)


def _events(pond: int, n_events: int, rng: np.random.Generator) -> pd.DataFrame:
    t = 14 * np.arange(n_events)
    w = 600.0 * (pond % 3 + 1) / 3 / (1 + np.exp(-0.035 * (t - 110))) + 2.0
    w = np.round(w * np.exp(rng.normal(0, 0.05, size=n_events)), 2)
    length = np.round(7.0 + 2.2 * w ** 0.33, 2)
    return pd.DataFrame({"pond": pond, "t": t, "date": DAY0 + pd.to_timedelta(t, unit="D"),
                         "w": w, "l": length})


def _block(array, end: int, length: int):
    """Stats, validity of the `length` days before index `end`, as seen from `end`."""
    vals, valid, _status, _left, right = array
    window = slice(end - length, end)
    ok = valid[window] & ~(right[window] >= end)
    return vals[window], ok


def window_features(array, end: int, length: int) -> tuple[dict[str, float], int]:
    """The 24 window features and the number of covered days before index `end`."""
    vals, ok = _block(array, end, length)
    out = {}
    for s, name in enumerate(SENSORS):
        keep = ok[:, s]
        if keep.sum() >= 3:
            block = vals[keep, s]
            found = {"mean": block[:, 0].mean(), "sd": block[:, 1].mean(),
                     "min": block[:, 2].min(), "max": block[:, 3].max()}
        else:
            found = dict.fromkeys(STATS, np.nan)
        for stat in STATS:
            out[f"{name}_{stat}_w"] = found[stat]
    return out, int((ok.sum(axis=1) >= 3).sum())


def cumulative_features(array, end: int) -> dict[str, float]:
    """Each sensor's mean of daily means from day 0 to the day before index `end`."""
    vals, ok = _block(array, end, end - DAY0_OFFSET)
    out = {}
    for s, name in enumerate(SENSORS):
        keep = ok[:, s]
        out[f"{name}_mean_cum"] = float(vals[keep, s, 0].mean()) if keep.sum() >= 3 else np.nan
    return out


def _prefixed(prefix: str, features: dict) -> dict:
    return {f"{prefix}_{k}": v for k, v in features.items()}


def make_wq_by_date(arrays: dict, last_k: int) -> pd.DataFrame:
    """The wq_by_date.csv layout: every pond at every grid date k = 1 .. last_k."""
    rows = []
    for pond, array in arrays.items():
        for k in range(1, last_k + 1):
            end = DAY0_OFFSET + 14 * k
            row = {"pond": pond, "stream": STREAMS[pond],
                   "date": (DAY0 + pd.Timedelta(days=14 * k)).strftime("%Y-%m-%d"),
                   "t": 14 * k, "grid_k": k}
            window, covered = window_features(array, end, 14)
            row.update(_prefixed("t1", window))
            row.update(_prefixed("t1", cumulative_features(array, end)))
            row["t1_n_covered_window"] = covered
            window, covered = window_features(array, end, 28)
            row.update(_prefixed("t2", window))
            row["t2_n_covered_window"] = covered
            for prefix, length in SENSITIVITY_WINDOWS.items():
                window, covered = window_features(array, end, length)
                row.update(_prefixed(prefix, window))
                row[f"{prefix}_n_covered_window"] = covered
            rows.append(row)
    return pd.DataFrame(rows)


def make_inputs(ponds=PONDS, n_events: int | dict = 8, seed: int = 0, missing_rate: float = 0.15,
                sparse_ponds=(), blackouts=()) -> Inputs:
    """The four tables with string dates, as read from the CSV files.

    `blackouts` is a list of (pond, first_day_index, last_day_index): all sensors of the pond
    are missing on those days (index 0 is 2021-05-22).
    """
    rng = np.random.default_rng(seed)
    counts = {p: (n_events[p] if isinstance(n_events, dict) else n_events) for p in ponds}
    last_k = max(counts.values()) - 1
    n_days = DAY0_OFFSET + 14 * last_k + 1
    arrays = _daily_arrays(ponds, n_days, rng, missing_rate, set(sparse_ponds), list(blackouts))
    daily = make_daily(arrays)
    wq = make_wq_by_date(arrays, last_k)
    wq_key = wq.set_index(["pond", "date"])
    t1_rows, t2_rows = [], []
    for pond in ponds:
        events = _events(pond, counts[pond], rng)
        for k in range(1, len(events)):
            ev = events.iloc[k]
            date = ev["date"].strftime("%Y-%m-%d")
            feats = wq_key.loc[(pond, date)]
            row = {"pond": pond, "stream": STREAMS[pond], "date": date, "t": int(ev["t"]),
                   "t2": int(ev["t"]) ** 2, "log_w": float(np.log(ev["w"])), "w": float(ev["w"]),
                   "log_l": float(np.log(ev["l"])), "l": float(ev["l"])}
            for name in [f"{s}_{k2}_w" for s in SENSORS for k2 in STATS] + [f"{s}_mean_cum" for s in SENSORS]:
                row[name] = float(feats[f"t1_{name}"])
            row["n_covered_window"] = int(feats["t1_n_covered_window"])
            row["eligible_eval"] = row["n_covered_window"] >= 7
            t1_rows.append(row)
            prev = events.iloc[k - 1]
            for h in (1, 2):
                if k + h >= len(events):
                    continue
                tgt = events.iloc[k + h]
                row2 = {"pond": pond, "stream": STREAMS[pond], "origin_date": date,
                        "target_date": tgt["date"].strftime("%Y-%m-%d"), "horizon": h,
                        "t_k": int(ev["t"]), "dt_prev": int(ev["t"] - prev["t"]),
                        "dt_target": int(tgt["t"] - ev["t"]), "log_w_k": float(np.log(ev["w"])),
                        "log_w_prev": float(np.log(prev["w"])), "w_origin": float(ev["w"]),
                        "w_target": float(tgt["w"]), "y": float(np.log(tgt["w"] / ev["w"]))}
                for s in SENSORS:
                    for k2 in STATS:
                        row2[f"{s}_{k2}_w"] = float(feats[f"t2_{s}_{k2}_w"])
                row2["n_covered_window"] = int(feats["t2_n_covered_window"])
                row2["eligible_eval"] = row2["n_covered_window"] >= 14
                t2_rows.append(row2)
    return Inputs(pd.DataFrame(t1_rows), pd.DataFrame(t2_rows), daily, wq)


def make_tables(ponds=PONDS, n_events: int | dict = 8, seed: int = 0, missing_rate: float = 0.15,
                sparse_ponds=(), blackouts=()) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (t1, t2, daily) with string dates, as read from the CSV files."""
    inputs = make_inputs(ponds, n_events, seed, missing_rate, sparse_ponds, blackouts)
    return inputs.t1, inputs.t2, inputs.daily


def write_inputs(directory: Path, inputs: Inputs, cfg: dict) -> dict:
    """Write the four tables under `directory` and return `cfg` with its paths pointing there."""
    directory.mkdir(parents=True, exist_ok=True)
    names = {"t1_table": inputs.t1, "t2_table": inputs.t2, "daily_wq": inputs.daily,
             "wq_by_date": inputs.wq}
    paths = dict(cfg["paths"])
    for key, frame in names.items():
        target = directory / f"{key}.csv"
        frame.to_csv(target, index=False)
        paths[key] = str(target)
    paths["results"] = str(directory / "results")
    paths["dev_results"] = str(directory / "results" / "dev")
    return {**cfg, "paths": paths}
