"""Phase 1 data audit for the aquacj-4561230 revision (data-auditor role).

Run from the repo root:  .venv/bin/python -I src/audit.py

Regenerates every file in results/audit/ and figures/audit/ from data/raw/ in one run.
It reports what the files contain: row counts, timestamps, intervals, gaps, labels,
ranges and cross-pond overlaps. Nothing is cleaned, dropped, imputed or interpreted.
Scope and file list: tasks/phase1_audit.md. Parsing lives in src/load.py.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import math
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402

from src.load import (  # noqa: E402
    HEADER_MAP,
    NUMERIC_VARS,
    POND_IDS,
    RAW_DIR,
    SENSOR_VARS,
    TRAILING_UNNAMED,
    UNITS_AS_LABELLED,
    load_pond,
    raw_path,
    read_raw,
    unit_as_labelled,
)

RESULTS = REPO_ROOT / "results" / "audit"
FIGURES = REPO_ROOT / "figures" / "audit"
VARIABLES = SENSOR_VARS + ("population", "length", "weight")

GAP_SECONDS = 600.0
INTERVAL_EDGES = [-np.inf, 0, 5, 10, 15, 20, 30, 60, 600, np.inf]
INTERVAL_LABELS = [
    "<=0", "(0,5]", "(5,10]", "(10,15]", "(15,20]", "(20,30]", "(30,60]", "(60,600]", ">600",
]
AUDIT_BOUNDS = {
    "temperature": (0.0, 45.0),
    "ph": (0.0, 14.0),
    "do": (0.0, 20.0),
    "turbidity": (0.0, 1000.0),
    "ammonia": (0.0, 1000.0),
    "nitrate": (0.0, 5000.0),
}
PERCENTILES = (0.1, 1.0, 5.0, 25.0, 50.0, 75.0, 95.0, 99.0, 99.9)
PERCENTILE_NAMES = ("p0.1", "p1", "p5", "p25", "p50", "p75", "p95", "p99", "p99.9")
KNOT_MIN_ABS = 0.03
KNOT_MIN_REL = 0.02
GRID_START = pd.Timestamp("2021-06-19")
GRID_DATES = [GRID_START + pd.Timedelta(days=14 * k) for k in range(17)]
N_STEPS = 134
TRAIN_FRACTION = 0.8
DAY_FIRST_RX = re.compile(r"^(?P<d>\d{1,2})[-/](?P<mo>\d{1,2})[-/](?P<y>\d{4}|\d{2})$")


# ---------------------------------------------------------------- helpers


def _seconds(ts: pd.Series) -> np.ndarray:
    """Seconds since 1970-01-01 as float64, NaN where the timestamp is NaT."""
    return ((ts - pd.Timestamp(0)) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def _mode(values: np.ndarray) -> float:
    """Most frequent value. Ties resolve to the smallest value. NaN when empty."""
    if values.size == 0:
        return float("nan")
    uniq, counts = np.unique(values, return_counts=True)
    return float(uniq[counts == counts.max()].min())


def _longest_identical_run(values: np.ndarray) -> int:
    """Longest run of consecutive file rows holding the same non-missing value."""
    best = 0
    run = 0
    prev = float("nan")
    for value in values:
        if math.isnan(value):
            run = 0
            continue
        run = run + 1 if run and value == prev else 1
        prev = value
        best = max(best, run)
    return best


def _differs(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    """True where two consecutive values differ. Missing against a value counts as a change."""
    both_missing = np.isnan(prev) & np.isnan(cur)
    return ~((prev == cur) | both_missing)


def _raw_header(pond: int) -> list[str]:
    """Header cells as written in the file. Empty headers stay ""."""
    with raw_path(pond).open(newline="", encoding="ascii") as handle:
        return next(csv.reader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _published_sums() -> dict[Path, str]:
    """Entries of data/raw/SHA256SUMS keyed by the resolved path they describe."""
    sums: dict[Path, str] = {}
    for line in (RAW_DIR / "SHA256SUMS").read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            sums[(REPO_ROOT / rel).resolve()] = digest
    return sums


def _parse_day_first(text: str) -> pd.Timestamp:
    """Date from a day-first string (dd-mm-yy, dd-mm-yyyy, dd/mm/yyyy); NaT if invalid."""
    match = DAY_FIRST_RX.match(text)
    if match is None:
        return pd.NaT
    year = int(match["y"])
    year = year + 2000 if year < 100 else year
    try:
        return pd.Timestamp(year=year, month=int(match["mo"]), day=int(match["d"]))
    except ValueError:
        return pd.NaT


def _cell(value: object) -> str:
    if value is None or value is pd.NaT or value is pd.NA:
        return ""
    if isinstance(value, float):
        return "" if math.isnan(value) else f"{value:.6g}"
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value)


def _md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table for AUDIT_REPORT.md and step_row12000.md."""
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    rule = "|" + "|".join("---" for _ in frame.columns) + "|"
    body = ["| " + " | ".join(_cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, rule, *body])


# ---------------------------------------------------------------- per-pond audit


@dataclass
class PondAudit:
    """Per-pond tables. Each DataFrame is concatenated over ponds in `main`."""

    pond: int
    summary: dict
    schema: pd.DataFrame
    ts_runs: pd.DataFrame
    interval_hist: pd.DataFrame
    gaps: pd.DataFrame
    coverage: pd.DataFrame
    ranges: pd.DataFrame
    label_changes: pd.DataFrame
    daily_labels: pd.DataFrame
    label_segments: pd.DataFrame
    grid: pd.DataFrame
    offgrid: pd.DataFrame
    trailing: pd.DataFrame
    nonnumeric: pd.DataFrame
    faults: pd.DataFrame
    slim: pd.DataFrame


def _schema(pond: int, raw: pd.DataFrame, frame: pd.DataFrame, header: list[str]) -> pd.DataFrame:
    """One row per file column: names as written, unit text, and cell counts."""
    mapping = HEADER_MAP[pond]
    trailing = TRAILING_UNNAMED.get(pond, ())
    rows = []
    for index, name in enumerate(raw.columns):
        if name in mapping:
            canonical = mapping[name]
        elif name in trailing:
            canonical = "dropped_trailing"
        else:
            raise ValueError(f"pond {pond:02d}: column {name!r} has no mapping")
        nonempty = raw[name].ne("").to_numpy()
        numeric = canonical == "entry_id" or canonical in NUMERIC_VARS
        n_nonnumeric = (
            int((nonempty & frame[canonical].isna().to_numpy()).sum()) if numeric else None
        )
        rows.append(
            {
                "pond": pond,
                "file": f"data/raw/pond{pond:02d}.csv",
                "column_index": index,
                "original_name": header[index],
                "canonical_name": canonical,
                "unit_as_labelled": unit_as_labelled(header[index]),
                "n_nonempty": int(nonempty.sum()),
                "n_missing": int((~nonempty).sum()),
                "n_nonnumeric": n_nonnumeric,
            }
        )
    return pd.DataFrame(rows)


def _ts_runs(pond: int, frame: pd.DataFrame) -> pd.DataFrame:
    """Contiguous file-order runs of one ts_kind, with first and last data row."""
    kind = frame["ts_kind"].to_numpy()
    starts = np.r_[0, np.flatnonzero(kind[1:] != kind[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(kind) - 1]
    return pd.DataFrame(
        {
            "pond": pond,
            "ts_kind": kind[starts],
            "first_row": starts,
            "last_row": ends,
            "n_rows": ends - starts + 1,
        }
    )


def _interval_hist(pond: int, intervals: np.ndarray) -> pd.DataFrame:
    binned = pd.cut(
        pd.Series(intervals, dtype="float64"),
        bins=INTERVAL_EDGES,
        labels=INTERVAL_LABELS,
        right=True,
    )
    counts = binned.value_counts().reindex(INTERVAL_LABELS, fill_value=0)
    return pd.DataFrame({"pond": pond, "bin_s": INTERVAL_LABELS, "count": counts.to_numpy()})


def _quantiles(values: np.ndarray, probs: tuple[float, ...]) -> list[float]:
    if values.size == 0:
        return [float("nan")] * len(probs)
    return [float(q) for q in np.percentile(values, probs)]


def _summary(
    pond: int,
    frame: pd.DataFrame,
    raw: pd.DataFrame,
    intervals: np.ndarray,
    gap_idx: np.ndarray,
    checksums: dict[Path, str],
) -> dict:
    stamped = frame.loc[frame["ts"].notna()]
    dated = frame.loc[frame["date"].notna()]
    date_only_dates = np.sort(
        pd.to_datetime(frame.loc[frame["ts_kind"] == "date_only", "date"].dropna().unique()).to_numpy()
    )
    positive = intervals[intervals > 0]
    kinds = frame["ts_kind"].value_counts()
    tz_counts = frame.loc[frame["tz_label"] != "", "tz_label"].value_counts()
    eid = frame["entry_id"].astype("float64").to_numpy()
    eid_steps = np.diff(eid)
    eid_valid = ~np.isnan(eid_steps)
    pop = np.sort(frame["population"].dropna().unique())
    path = raw_path(pond)
    sha = _sha256(path.resolve())
    published = checksums.get(path.resolve())

    date_col = next((name for name in raw.columns if name in ("Date", "DATE")), None)
    if date_col is None:
        date_fields = {"n_date_col_disagree_with_ts": None, "n_date_col_unparsed": None}
    else:
        parsed = pd.Series(
            [_parse_day_first(text) for text in raw[date_col].to_list()],
            index=raw.index,
            dtype="datetime64[us]",
        )
        ts_norm = frame["ts"].dt.normalize()
        disagree = frame["ts"].notna() & parsed.notna() & (parsed != ts_norm)
        unparsed = raw[date_col].ne("") & parsed.isna()
        date_fields = {
            "n_date_col_disagree_with_ts": int(disagree.sum()),
            "n_date_col_unparsed": int(unparsed.sum()),
        }

    def _stamp(value: pd.Timestamp) -> str:
        return "" if pd.isna(value) else value.strftime("%Y-%m-%d %H:%M:%S")

    def _day(value: pd.Timestamp) -> str:
        return "" if pd.isna(value) else value.strftime("%Y-%m-%d")

    return {
        "pond": pond,
        "file": f"data/raw/pond{pond:02d}.csv",
        "n_rows": len(frame),
        "n_datetime_label": int(kinds.get("datetime_label", 0)),
        "n_iso_offset": int(kinds.get("iso_offset", 0)),
        "n_date_only": int(kinds.get("date_only", 0)),
        "n_ts_missing": int(kinds.get("missing", 0)),
        "n_ts_unparsed": int(kinds.get("unparsed", 0)),
        "n_rows_all_sensors_missing": int(frame[list(SENSOR_VARS)].isna().all(axis=1).sum()),
        "n_rows_without_date": len(frame) - len(dated),
        "first_ts_file_order": _stamp(stamped["ts"].iloc[0]) if len(stamped) else "",
        "last_ts_file_order": _stamp(stamped["ts"].iloc[-1]) if len(stamped) else "",
        "min_ts": _stamp(stamped["ts"].min()) if len(stamped) else "",
        "max_ts": _stamp(stamped["ts"].max()) if len(stamped) else "",
        "first_date_file_order": _day(dated["date"].iloc[0]) if len(dated) else "",
        "last_date_file_order": _day(dated["date"].iloc[-1]) if len(dated) else "",
        "min_date": _day(dated["date"].min()) if len(dated) else "",
        "max_date": _day(dated["date"].max()) if len(dated) else "",
        "span_days": (dated["date"].max() - dated["date"].min()).days if len(dated) else None,
        "n_dates_with_rows": int(dated["date"].nunique()),
        "tz_label_counts": ";".join(f"{k}={v}" for k, v in tz_counts.items()),
        "n_intervals": int(intervals.size),
        "interval_median_s": _quantiles(positive, (50.0,))[0],
        "interval_p05_s": _quantiles(positive, (5.0,))[0],
        "interval_p25_s": _quantiles(positive, (25.0,))[0],
        "interval_p75_s": _quantiles(positive, (75.0,))[0],
        "interval_p95_s": _quantiles(positive, (95.0,))[0],
        "interval_mean_s": float(positive.mean()) if positive.size else float("nan"),
        "n_intervals_nonpositive": int((intervals <= 0).sum()),
        "n_rows_with_duplicate_ts_raw": int(stamped["ts_raw"].duplicated(keep=False).sum()),
        "n_exact_duplicate_lines": int(raw.duplicated(keep="first").sum()),
        "n_entry_id_repeated_rows": int(frame["entry_id"].dropna().duplicated().sum()),
        "n_date_only_dates": int(date_only_dates.size),
        "date_only_first_date": _day(pd.Timestamp(date_only_dates[0])) if date_only_dates.size else "",
        "date_only_last_date": _day(pd.Timestamp(date_only_dates[-1])) if date_only_dates.size else "",
        "date_only_spacing_days": ";".join(str(v) for v in sorted(set(np.diff(date_only_dates).astype("timedelta64[D]").astype(int)))) if date_only_dates.size > 1 else "",
        "n_gaps_gt_10min": int(gap_idx.size),
        "total_gap_h": float(intervals[gap_idx].sum() / 3600.0) if gap_idx.size else 0.0,
        "longest_gap_h": float(intervals[gap_idx].max() / 3600.0) if gap_idx.size else float("nan"),
        "population_unique": ";".join(f"{v:g}" for v in pop),
        "entry_id_min": int(np.nanmin(eid)) if np.isfinite(eid).any() else None,
        "entry_id_max": int(np.nanmax(eid)) if np.isfinite(eid).any() else None,
        "n_entry_id_missing": int(frame["entry_id"].isna().sum()),
        "n_entry_id_not_plus1": int(np.sum(eid_steps[eid_valid] != 1)),
        **date_fields,
        "sha256": sha,
        "sha256_matches_SHA256SUMS": (published == sha) if published else None,
    }


def _coverage(pond: int, frame: pd.DataFrame) -> pd.DataFrame:
    dated = frame.loc[frame["date"].notna()].assign(hour=lambda d: d["ts"].dt.hour)
    cov = (
        dated.groupby("date")
        .agg(
            n_rows=("row", "size"),
            first_ts=("ts", "min"),
            last_ts=("ts", "max"),
            n_clock_hours_with_rows=("hour", "nunique"),
        )
        .reset_index()
    )
    cov.insert(0, "pond", pond)
    return cov


def _ranges(pond: int, frame: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    by_canonical = {canon: header for header, canon in HEADER_MAP[pond].items()}
    rows = []
    for var in VARIABLES:
        present = var in by_canonical
        if present:
            values = frame[var].to_numpy(dtype="float64")
            text = raw[by_canonical[var]]
            n_missing = int(text.eq("").sum())
            n_nonnumeric = int((text.ne("").to_numpy() & np.isnan(values)).sum())
        else:
            values = np.full(len(frame), np.nan)
            n_missing = n_nonnumeric = None
        present_values = ~np.isnan(values)
        finite = values[np.isfinite(values)]
        quant = _quantiles(finite, PERCENTILES)
        lo_hi = AUDIT_BOUNDS.get(var)
        row = {
            "pond": pond,
            "variable": var,
            "column_present": present,
            "n": int(present_values.sum()),
            "n_missing": n_missing,
            "n_nonnumeric": n_nonnumeric,
            "min": float(finite.min()) if finite.size else float("nan"),
            **dict(zip(PERCENTILE_NAMES, quant)),
            "max": float(finite.max()) if finite.size else float("nan"),
            "n_infinite": int(np.isinf(values).sum()),
            "n_negative": int((values < 0).sum()),
            "n_zero": int((values == 0).sum()),
            "n_equal_minus127": int((values == -127).sum()) if var == "temperature" else None,
            "longest_identical_run_rows": _longest_identical_run(values),
            "audit_lo": lo_hi[0] if lo_hi else None,
            "audit_hi": lo_hi[1] if lo_hi else None,
            "n_out_of_audit_bounds": (
                int(((values < lo_hi[0]) | (values > lo_hi[1])).sum()) if lo_hi else None
            ),
            "n_gt_1e3": int((values > 1e3).sum()) if var == "ammonia" else None,
            "n_gt_1e6": int((values > 1e6).sum()) if var == "ammonia" else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _label_changes(pond: int, frame: pd.DataFrame) -> pd.DataFrame:
    weight = frame["weight"].to_numpy(dtype="float64")
    length = frame["length"].to_numpy(dtype="float64")
    changed = _differs(weight[:-1], weight[1:]) | _differs(length[:-1], length[1:])
    cur = np.flatnonzero(changed) + 1
    prev = cur - 1
    secs = _seconds(frame["ts"])
    return pd.DataFrame(
        {
            "pond": pond,
            "row": frame["row"].to_numpy()[cur],
            "ts_raw": frame["ts_raw"].to_numpy()[cur],
            "ts": frame["ts"].to_numpy()[cur],
            "date": frame["date"].to_numpy()[cur],
            "weight_old": weight[prev],
            "weight_new": weight[cur],
            "length_old": length[prev],
            "length_new": length[cur],
            "hours_since_previous_row": (secs[cur] - secs[prev]) / 3600.0,
        }
    )


def _clock_or_kind(row: pd.Series) -> str:
    return row["ts"].strftime("%H:%M:%S") if pd.notna(row["ts"]) else str(row["ts_kind"])


def _daily_labels(pond: int, frame: pd.DataFrame) -> pd.DataFrame:
    dated = frame.loc[frame["date"].notna()]
    rows = []
    for date, group in dated.groupby("date", sort=True):
        weight = group["weight"].to_numpy(dtype="float64")
        length = group["length"].to_numpy(dtype="float64")
        wv = weight[~np.isnan(weight)]
        lv = length[~np.isnan(length)]
        wmode = _mode(wv)
        differs = ~np.isnan(weight) & (weight != wmode)
        first_three = [
            _clock_or_kind(group.iloc[i]) for i in np.flatnonzero(differs)[:3]
        ]
        rows.append(
            {
                "pond": pond,
                "date": date,
                "n_rows": len(group),
                "n_distinct_weight": int(np.unique(wv).size),
                "weight_mode": wmode,
                "weight_min": float(wv.min()) if wv.size else float("nan"),
                "weight_max": float(wv.max()) if wv.size else float("nan"),
                "n_distinct_length": int(np.unique(lv).size),
                "length_mode": _mode(lv),
                "length_min": float(lv.min()) if lv.size else float("nan"),
                "length_max": float(lv.max()) if lv.size else float("nan"),
                "n_weight_ne_mode": int(differs.sum()),
                "first3_clock_times_weight_ne_mode": ";".join(first_three),
            }
        )
    return pd.DataFrame(rows)


SEGMENT_COLUMNS = [
    "pond",
    "date_prev",
    "date",
    "days_between",
    "weight_mode_prev",
    "weight_mode",
    "slope_w",
    "slope_L",
    "delta_slope_w",
    "candidate_knot",
]


def _label_segments(pond: int, daily: pd.DataFrame) -> pd.DataFrame:
    """Consecutive days with a weight value. A candidate knot sits on date_prev."""
    seg = daily.loc[daily["weight_mode"].notna()].sort_values("date")
    dates = seg["date"].to_numpy()
    if dates.size < 2:
        return pd.DataFrame(columns=SEGMENT_COLUMNS)
    w = seg["weight_mode"].to_numpy(dtype="float64")
    length = seg["length_mode"].to_numpy(dtype="float64")
    days = ((dates[1:] - dates[:-1]) / np.timedelta64(1, "D")).astype("int64")
    slope_w = (w[1:] - w[:-1]) / days
    slope_l = (length[1:] - length[:-1]) / days
    delta = np.full(slope_w.shape, np.nan)
    delta[1:] = slope_w[1:] - slope_w[:-1]
    prev_slope = np.full(slope_w.shape, np.nan)
    prev_slope[1:] = slope_w[:-1]
    candidate = (np.abs(delta) > KNOT_MIN_ABS) & (
        np.abs(delta) > KNOT_MIN_REL * np.abs(prev_slope)
    )
    return pd.DataFrame(
        {
            "pond": pond,
            "date_prev": dates[:-1],
            "date": dates[1:],
            "days_between": days,
            "weight_mode_prev": w[:-1],
            "weight_mode": w[1:],
            "slope_w": slope_w,
            "slope_L": slope_l,
            "delta_slope_w": delta,
            "candidate_knot": candidate,
        }
    )


def _grid(
    pond: int, frame: pd.DataFrame, daily: pd.DataFrame, segments: pd.DataFrame
) -> pd.DataFrame:
    """For each grid date: rows, row source, modal labels, slopes either side, knot flag."""
    dated = frame.loc[frame["date"].notna()]
    daily_by_date = daily.set_index("date")
    rows = []
    for grid_date in GRID_DATES:
        on = dated.loc[dated["date"] == grid_date]
        if len(on) == 0:
            source = "none"
        else:
            has_sensor = on[list(SENSOR_VARS)].notna().any(axis=1).to_numpy()
            if has_sensor.all():
                source = "sensor"
            elif (~has_sensor).all():
                source = "fish_only"
            else:
                source = "mixed"
        if grid_date in daily_by_date.index:
            weight_mode = float(daily_by_date.at[grid_date, "weight_mode"])
            length_mode = float(daily_by_date.at[grid_date, "length_mode"])
        else:
            weight_mode = length_mode = float("nan")
        before = segments.loc[segments["date"] <= grid_date]
        after = segments.loc[segments["date_prev"] >= grid_date]
        rows.append(
            {
                "pond": pond,
                "grid_date": grid_date,
                "has_rows": len(on) > 0,
                "n_rows": len(on),
                "row_source": source,
                "weight_mode": weight_mode,
                "length_mode": length_mode,
                "pair_before_end": before["date"].iloc[-1] if len(before) else pd.NaT,
                "slope_w_before": before["slope_w"].iloc[-1] if len(before) else float("nan"),
                "pair_after_start": after["date_prev"].iloc[0] if len(after) else pd.NaT,
                "slope_w_after": after["slope_w"].iloc[0] if len(after) else float("nan"),
                "candidate_knot_here": bool(
                    (segments["candidate_knot"] & (segments["date_prev"] == grid_date)).any()
                )
                if len(segments)
                else False,
            }
        )
    return pd.DataFrame(rows)


OFFGRID_COLUMNS = [
    "pond",
    "knot_date",
    "next_date",
    "days_between",
    "slope_w_prev",
    "slope_w",
    "delta_slope_w",
    "nearest_grid_date",
    "days_from_nearest_grid",
]


def _offgrid(pond: int, segments: pd.DataFrame) -> pd.DataFrame:
    """Candidate knots whose date_prev is not a grid date, with the nearest grid date."""
    rows = []
    if len(segments):
        seg = segments.assign(slope_w_prev=segments["slope_w"].shift(1))
        off = seg.loc[seg["candidate_knot"] & ~seg["date_prev"].isin(GRID_DATES)]
        grid = np.array(GRID_DATES, dtype="datetime64[ns]")
        for _, knot in off.iterrows():
            d = np.datetime64(pd.Timestamp(knot["date_prev"]).to_datetime64(), "ns")
            nearest = grid[np.argmin(np.abs(grid - d))]
            rows.append(
                {
                    "pond": pond,
                    "knot_date": knot["date_prev"],
                    "next_date": knot["date"],
                    "days_between": knot["days_between"],
                    "slope_w_prev": knot["slope_w_prev"],
                    "slope_w": knot["slope_w"],
                    "delta_slope_w": knot["delta_slope_w"],
                    "nearest_grid_date": pd.Timestamp(nearest),
                    "days_from_nearest_grid": int((d - nearest) / np.timedelta64(1, "D")),
                }
            )
    return pd.DataFrame(rows, columns=OFFGRID_COLUMNS)


TRAILING_COLUMNS = ["pond", "row", "file_line", "column_name", "value"]
NONNUMERIC_COLUMNS = ["pond", "row", "file_line", "column", "canonical", "value"]


def _trailing(pond: int, raw: pd.DataFrame) -> pd.DataFrame:
    """Every non-empty cell in the trailing unnamed columns (dropped by load_pond)."""
    rows = []
    for name in TRAILING_UNNAMED.get(pond, ()):
        text = raw[name]
        for row in np.flatnonzero(text.ne("").to_numpy()):
            rows.append(
                {
                    "pond": pond,
                    "row": int(row),
                    "file_line": int(row) + 2,
                    "column_name": name,
                    "value": text.iloc[row],
                }
            )
    return pd.DataFrame(rows, columns=TRAILING_COLUMNS)


FAULT_COLUMNS = ["pond", "fault", "row", "file_line", "other_row", "ts_raw", "entry_id"]


def _faults(pond: int, raw: pd.DataFrame, frame: pd.DataFrame, intervals: np.ndarray, rows_t: np.ndarray) -> pd.DataFrame:
    """Rows that repeat an earlier raw line, and timestamped rows that go backwards in time.

    `other_row` is the earlier identical line for a duplicate, and the previous timestamped
    row for a non-positive interval.
    """
    hashed = pd.util.hash_pandas_object(raw, index=False).to_numpy()
    duplicate = raw.duplicated(keep="first").to_numpy()
    first_index: dict[int, int] = {}
    for i, value in enumerate(hashed):
        first_index.setdefault(int(value), i)
    rows = []

    def _entry(row: int, fault: str, other: int) -> dict:
        return {
            "pond": pond,
            "fault": fault,
            "row": int(row),
            "file_line": int(row) + 2,
            "other_row": int(other),
            "ts_raw": frame["ts_raw"].iloc[row],
            "entry_id": frame["entry_id"].iloc[row],
        }

    for i in np.flatnonzero(duplicate):
        rows.append(_entry(int(i), "exact_duplicate_line", first_index[int(hashed[i])]))
    for k in np.flatnonzero(intervals <= 0):
        rows.append(_entry(int(rows_t[k + 1]), "non_positive_interval", int(rows_t[k])))
    return pd.DataFrame(rows, columns=FAULT_COLUMNS)


def _nonnumeric(pond: int, raw: pd.DataFrame, frame: pd.DataFrame, header: list[str]) -> pd.DataFrame:
    """Non-empty numeric-column cells that did not parse (NaN in the loaded frame)."""
    rows = []
    names = list(raw.columns)
    for name, canonical in HEADER_MAP[pond].items():
        if not (canonical == "entry_id" or canonical in NUMERIC_VARS):
            continue
        text = raw[name]
        bad = text.ne("").to_numpy() & frame[canonical].isna().to_numpy()
        for row in np.flatnonzero(bad):
            rows.append(
                {
                    "pond": pond,
                    "row": int(row),
                    "file_line": int(row) + 2,
                    "column": header[names.index(name)],
                    "canonical": canonical,
                    "value": text.iloc[row],
                }
            )
    return pd.DataFrame(rows, columns=NONNUMERIC_COLUMNS)


def analyse_pond(pond: int, checksums: dict[Path, str]) -> PondAudit:
    """Run every per-pond audit step on one file and return its tables."""
    frame = load_pond(pond)
    raw = read_raw(pond)
    header = _raw_header(pond)

    stamped = frame.loc[frame["ts"].notna()]
    stamp_secs = _seconds(stamped["ts"])
    intervals = np.diff(stamp_secs)
    gap_idx = np.flatnonzero(intervals > GAP_SECONDS)
    rows_t = stamped["row"].to_numpy()
    ts_t = stamped["ts"].to_numpy()
    gaps = pd.DataFrame(
        {
            "pond": pond,
            "row_before": rows_t[gap_idx],
            "row_after": rows_t[gap_idx + 1],
            "ts_before": ts_t[gap_idx],
            "ts_after": ts_t[gap_idx + 1],
            "duration_h": intervals[gap_idx] / 3600.0,
        }
    ).sort_values(["pond", "ts_before", "row_before"], ignore_index=True)

    daily = _daily_labels(pond, frame)
    segments = _label_segments(pond, daily)
    slim = frame[["ts_raw", "ts", "ts_kind", "entry_id", *SENSOR_VARS]].copy()

    return PondAudit(
        pond=pond,
        summary=_summary(pond, frame, raw, intervals, gap_idx, checksums),
        schema=_schema(pond, raw, frame, header),
        ts_runs=_ts_runs(pond, frame),
        interval_hist=_interval_hist(pond, intervals),
        gaps=gaps,
        coverage=_coverage(pond, frame),
        ranges=_ranges(pond, frame, raw),
        label_changes=_label_changes(pond, frame),
        daily_labels=daily,
        label_segments=segments,
        grid=_grid(pond, frame, daily, segments),
        offgrid=_offgrid(pond, segments),
        trailing=_trailing(pond, raw),
        nonnumeric=_nonnumeric(pond, raw, frame, header),
        faults=_faults(pond, raw, frame, intervals, rows_t),
        slim=slim,
    )


# ---------------------------------------------------------------- cross-pond


def _cross_pond(slim: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Overlap between every pair of ponds. Rows are matched on the raw timestamp string."""
    rows = []
    for a, b in itertools.combinations(POND_IDS, 2):
        frame_a, frame_b = slim[a], slim[b]
        timed_a = frame_a.loc[frame_a["ts"].notna()]
        timed_b = frame_b.loc[frame_b["ts"].notna()]
        common_raw = pd.Index(timed_a["ts_raw"].unique()).intersection(
            pd.Index(timed_b["ts_raw"].unique())
        )
        dateonly_a = frame_a.loc[frame_a["ts_kind"] == "date_only", "ts_raw"]
        dateonly_b = frame_b.loc[frame_b["ts_kind"] == "date_only", "ts_raw"]
        common_dateonly = pd.Index(dateonly_a.unique()).intersection(pd.Index(dateonly_b.unique()))
        common_clock = np.intersect1d(timed_a["ts"].to_numpy(), timed_b["ts"].to_numpy())
        common_entry = pd.Index(frame_a["entry_id"].dropna().unique()).intersection(
            pd.Index(frame_b["entry_id"].dropna().unique())
        )
        left = timed_a[["ts_raw", "ts", *SENSOR_VARS]]
        right = timed_b[["ts_raw", *SENSOR_VARS]]
        matched = left.merge(right, on="ts_raw", suffixes=("_a", "_b"))
        identical = {}
        for var in SENSOR_VARS:
            va = matched[f"{var}_a"]
            vb = matched[f"{var}_b"]
            identical[var] = (va.notna() & vb.notna() & (va == vb)).to_numpy()
        all_six = np.logical_and.reduce([identical[var] for var in SENSOR_VARS]) if len(matched) else np.array([], dtype=bool)
        ordered = matched.sort_values("ts", kind="stable")
        row = {
            "pond_a": a,
            "pond_b": b,
            "n_rows_a": len(frame_a),
            "n_rows_b": len(frame_b),
            "n_common_ts_raw": len(common_raw),
            "n_common_clock_times": len(common_clock),
            "n_common_date_only_raw": len(common_dateonly),
            "n_common_entry_id": len(common_entry),
            "n_matched_pairs": len(matched),
        }
        for var in SENSOR_VARS:
            row[f"frac_identical_{var}"] = float(identical[var].mean()) if len(matched) else float("nan")
        row["n_all_six_identical"] = int(all_six.sum()) if len(matched) else 0
        row["first_common_ts_raw"] = ordered["ts_raw"].iloc[0] if len(matched) else ""
        row["last_common_ts_raw"] = ordered["ts_raw"].iloc[-1] if len(matched) else ""
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- notebook test series


@dataclass
class OriginalSeries:
    """Reproduction of the original notebook's pond 2 test series (cells 1 to 4).

    `length_file` and `sensor_cells_file` hold, per file row (before `dropna()`), the
    Fish_Length (cm) value and the number of the six sensor cells that are present. The
    notebook drops Fish_Length (cm) before `dropna()`, so length is not part of `test`.
    """

    n_rows_file: int
    n_retained: int
    train_size: int
    n_test: int
    test: pd.DataFrame
    file_ts: pd.Series
    length_file: np.ndarray
    sensor_cells_file: np.ndarray
    changes: pd.DataFrame
    largest: int


def original_series() -> OriginalSeries:
    """Replicate notebook cells 1 to 4 on IoTPond2.csv with pandas default parsing."""
    raw = pd.read_csv(raw_path(2))
    sensor_cols = ["Temperature (C)", "Turbidity (NTU)", "Dissolved Oxygen(g/ml)", "PH", "Ammonia(g/ml)", "Nitrate(g/ml)"]
    df = raw.drop(["entry_id", "Population", "Fish_Length (cm)"], axis=1)
    df = df.rename(
        columns={
            "created_at": "Collected_Time",
            "Temperature (C)": "Temp_(C)",
            "Turbidity (NTU)": "Turb_(NTU)",
            "Dissolved Oxygen(g/ml)": "DO_(g/ml)",
            "Ammonia(g/ml)": "Ammonia_(g/ml)",
            "Nitrate(g/ml)": "Nitrate_(g/ml)",
            "Fish_Weight (g)": "Fish_Weight_(g)",
        }
    )
    df = df.dropna()
    file_row = df.index.to_numpy()
    df = df.reset_index(drop=True)
    n = len(df)
    train_size = int(TRAIN_FRACTION * n)
    file_ts = pd.Series(
        pd.to_datetime(raw["created_at"].str[:19], format="%Y-%m-%d %H:%M:%S"),
    )
    retained_ts = file_ts.iloc[file_row].to_numpy()
    keep = pd.DataFrame(
        {
            "file_row": file_row,
            "ts_raw": df["Collected_Time"].to_numpy(),
            "ts": retained_ts,
            "weight": df["Fish_Weight_(g)"].to_numpy(dtype="float64"),
        }
    )
    # Notebook: range(len(df) - test_size, len(df) - n_steps) with test_size = len(df) - train_size.
    test = keep.iloc[train_size : n - N_STEPS].reset_index(drop=True)
    n_test = len(test)
    w = test["weight"].to_numpy()
    k_changes = np.flatnonzero(w[1:] != w[:-1]) + 1
    changes = []
    for k in k_changes:
        pos = train_size + k
        row_file = int(test.at[k, "file_row"])
        prev_retained_pos = pos - 1
        prev_file_row = row_file - 1
        ts_now = pd.Timestamp(test.at[k, "ts"])
        changes.append(
            {
                "k": int(k),
                "file_row": row_file,
                "ts_raw": test.at[k, "ts_raw"],
                "ts": ts_now,
                "weight_old": float(w[k - 1]),
                "weight_new": float(w[k]),
                "change_g": float(w[k] - w[k - 1]),
                "prev_file_row": prev_file_row,
                "prev_file_ts": file_ts.iloc[prev_file_row],
                "dt_prev_file_row_s": (ts_now - file_ts.iloc[prev_file_row]).total_seconds(),
                "dropped_file_rows_before": row_file - int(keep.iloc[prev_retained_pos]["file_row"]) - 1,
                "dt_prev_retained_row_s": (ts_now - pd.Timestamp(keep.iloc[prev_retained_pos]["ts"])).total_seconds(),
            }
        )
    changes_df = pd.DataFrame(changes)
    largest = int(changes_df["k"].iloc[np.argmax(np.abs(changes_df["change_g"].to_numpy()))]) if len(changes_df) else -1
    return OriginalSeries(
        n_rows_file=len(raw),
        n_retained=n,
        train_size=train_size,
        n_test=n_test,
        test=test,
        file_ts=file_ts,
        length_file=raw["Fish_Length (cm)"].to_numpy(dtype="float64"),
        sensor_cells_file=raw[sensor_cols].notna().sum(axis=1).to_numpy(),
        changes=changes_df,
        largest=largest,
    )


# ---------------------------------------------------------------- figures


def plot_pond_labels(pond: int, daily: pd.DataFrame) -> Path:
    fig, (ax, bar) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.step(daily["date"], daily["weight_mode"], where="post", color="#1f4e79", lw=1.0)
    ax.set_ylabel("weight (g), modal per day")
    twin = ax.twinx()
    twin.step(daily["date"], daily["length_mode"], where="post", color="#b5651d", lw=1.0)
    twin.set_ylabel("length (cm), modal per day")
    for grid_date in GRID_DATES:
        ax.axvline(grid_date, color="0.5", lw=0.4, alpha=0.7, zorder=0)
    bar.bar(daily["date"], daily["n_rows"], width=0.8, color="0.3")
    bar.set_ylabel("rows per day")
    bar.set_yscale("log")
    bar.minorticks_off()
    bar.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title(f"Pond {pond:02d}: modal daily labels; thin lines are grid dates 2021-06-19 + 14k")
    path = FIGURES / f"pond{pond:02d}_labels.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_coverage(coverage: pd.DataFrame) -> Path:
    first = coverage["date"].min()
    last = coverage["date"].max()
    n_days = (last - first).days + 1
    matrix = np.full((len(POND_IDS), n_days), np.nan)
    for i, pond in enumerate(POND_IDS):
        part = coverage.loc[coverage["pond"] == pond]
        matrix[i, (part["date"] - first).dt.days.to_numpy()] = part["n_rows"].to_numpy(dtype="float64")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.xaxis_date()
    extent = [mdates.date2num(first), mdates.date2num(last) + 1, len(POND_IDS), 0]
    image = ax.imshow(
        np.ma.masked_invalid(matrix),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=LogNorm(vmin=1, vmax=float(np.nanmax(matrix))),
        extent=extent,
    )
    ax.set_yticks(np.arange(len(POND_IDS)) + 0.5)
    ax.set_yticklabels([f"pond{p:02d}" for p in POND_IDS])
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.set_title("Rows per calendar day (white = no rows)")
    fig.colorbar(image, ax=ax, label="rows per day (log scale)")
    path = FIGURES / "coverage_overview.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_original_series(series: OriginalSeries) -> Path:
    test = series.test
    w = test["weight"].to_numpy()
    k = np.arange(len(test))
    largest = series.largest
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(14, 9))
    top.plot(k, w, lw=0.6, color="#1f4e79")
    top.axvline(largest, color="#b5651d", ls="--", lw=1)
    top.set_xlabel("test sample index k (notebook x-axis)")
    top.set_ylabel("Fish_Weight (g), target")
    top.set_title(f"Original test target series (pond 2); largest step at k={largest}")
    bottom.plot(pd.to_datetime(test["ts"]), w, lw=0.6, color="#1f4e79")
    bottom.axvline(pd.Timestamp(test.at[largest, "ts"]), color="#b5651d", ls="--", lw=1)
    bottom.set_xlabel("timestamp (clock time as written)")
    bottom.set_ylabel("Fish_Weight (g), target")
    path = FIGURES / "pond02_original_test_series.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


# ---------------------------------------------------------------- written outputs


def write_step_md(series: OriginalSeries, segments: pd.DataFrame) -> Path:
    """Write step_row12000.md. `segments` is pond 2's label_segments table."""
    test = series.test
    changes = series.changes
    w = test["weight"].to_numpy()
    largest = series.largest
    big = changes.loc[changes["k"] == largest].iloc[0]
    prev_date = pd.Timestamp(big["prev_file_ts"]).normalize()
    change_date = pd.Timestamp(big["ts"]).normalize()
    pair = segments.loc[(segments["date_prev"] == prev_date) & (segments["date"] == change_date)]
    calendar_days = (change_date - prev_date).days
    file_rows_between = int(big["file_row"]) - int(big["prev_file_row"]) - 1
    if len(pair):
        pair_row = pair.iloc[0]
        pair_text = (
            f"label_segments.csv (pond 02) has the pair {prev_date:%Y-%m-%d} to {change_date:%Y-%m-%d}: "
            f"days_between {int(pair_row['days_between'])}, slope_w {pair_row['slope_w']:.2f} g/day. "
            f"days_between x slope_w = {pair_row['days_between'] * pair_row['slope_w']:.2f} g, "
            f"against the change of {big['change_g']:+.2f} g above."
        )
    else:
        pair_text = (
            f"No label_segments.csv pair runs from {prev_date:%Y-%m-%d} to {change_date:%Y-%m-%d}."
        )
    row = int(big["file_row"])
    prev_row = int(big["prev_file_row"])
    around_rows = [prev_row, row, row + 1]
    stamps = series.file_ts
    around = pd.DataFrame(
        {
            "file_row": around_rows,
            "clock_time": [f"{stamps.iloc[r]:%Y-%m-%d %H:%M:%S}" for r in around_rows],
            "Fish_Length_cm": [f"{series.length_file[r]:.2f}" for r in around_rows],
            "sensor_cells_of_6": [int(series.sensor_cells_file[r]) for r in around_rows],
        }
    )
    gap_before_h = (stamps.iloc[row] - stamps.iloc[prev_row]).total_seconds() / 3600
    gap_after_h = (stamps.iloc[row + 1] - stamps.iloc[row]).total_seconds() / 3600
    timing_text = (
        f"What the timestamps show: file rows {prev_row} and {row} are {gap_before_h:.2f} h apart with "
        f"{file_rows_between} rows between them, so the change is the first row after a gap in the record. "
        f"File row {row + 1} is {gap_after_h:.2f} h after file row {row}. "
        "The file has no column that flags a fish sampling event."
    )
    table = changes.assign(
        k=changes["k"],
        file_row=changes["file_row"],
        timestamp=changes["ts_raw"],
        weight_old=changes["weight_old"].map(lambda v: f"{v:.2f}"),
        weight_new=changes["weight_new"].map(lambda v: f"{v:.2f}"),
        change_g=changes["change_g"].map(lambda v: f"{v:+.2f}"),
        dt_prev_file_row_s=changes["dt_prev_file_row_s"],
        dt_prev_retained_row_s=changes["dt_prev_retained_row_s"],
        dropped_file_rows_before=changes["dropped_file_rows_before"],
    )[
        [
            "k",
            "file_row",
            "timestamp",
            "weight_old",
            "weight_new",
            "change_g",
            "dt_prev_file_row_s",
            "dt_prev_retained_row_s",
            "dropped_file_rows_before",
        ]
    ]
    gap_rows = changes.loc[changes["dt_prev_file_row_s"] > GAP_SECONDS]
    lines = [
        "# Original test series, pond 2 (reviewer comment R4.7)",
        "",
        "Generated by `src/audit.py` from `data/raw/pond02.csv` (notebook cells 1 to 4 replicated",
        "with pandas default parsing: drop `entry_id`, `Population`, `Fish_Length (cm)`, then",
        "`dropna()` over the remaining columns).",
        "",
        "## Sizes",
        "",
        f"- Rows in the file (data rows, header excluded): {series.n_rows_file} (file rows 0 to {series.n_rows_file - 1}).",
        f"- Rows after `dropna()`: N = {series.n_retained}. Rows removed: {series.n_rows_file - series.n_retained}.",
        f"- `train_size = int(0.8 * N)` = {series.train_size}.",
        f"- `n_steps` = {N_STEPS}.",
        f"- Test samples: k = 0 to {series.n_test - 1}, target row `train_size + k`; count = N - n_steps - train_size = {series.n_test}.",
        f"- Notebook output check: X_test shape printed as (34298, 134, 7). Reproduced test samples = {series.n_test}.",
        f"- Notebook output check (cell 9): Actuals[0] = 305.80, reproduced = {w[0]:.2f}; Actuals[34293] = 394.66, reproduced = {w[34293]:.2f}.",
        "",
        "Test index k is the x-axis of the notebook plots (`plt.plot` with no x values).",
        "",
        "## Weight changes in the test series",
        "",
        f"There are {len(changes)} samples k (1 to {series.n_test - 1}) where the target differs from sample k-1.",
        f"Of these, {len(gap_rows)} follow a file gap longer than 600 s between the previous file row and the change row.",
        "`dt_prev_file_row_s` uses the previous row in the file. `dt_prev_retained_row_s` uses the previous row",
        "kept after `dropna()`. `dropped_file_rows_before` counts file rows removed by `dropna()` between them.",
        "",
        _md_table(table),
        "",
        "## Largest change",
        "",
        f"The largest weight change is at test sample k = {int(big['k'])} (file row {int(big['file_row'])}, timestamp {big['ts_raw']}).",
        f"Weight goes from {big['weight_old']:.2f} g to {big['weight_new']:.2f} g ({big['change_g']:+.2f} g).",
        f"The previous file row is {int(big['prev_file_row'])} at {pd.Timestamp(big['prev_file_ts']).strftime('%Y-%m-%d %H:%M:%S')}.",
        f"The time between them is {big['dt_prev_file_row_s'] / 3600:.2f} h ({int(big['dropped_file_rows_before'])} file rows removed by `dropna()` in between).",
        f"The previous kept row (file row {int(test.at[int(big['k']) - 1, 'file_row'])}) is {big['dt_prev_retained_row_s'] / 3600:.2f} h earlier.",
        f"File rows between the two rows: {file_rows_between}. Calendar days between their dates: {calendar_days}"
        f" ({prev_date:%Y-%m-%d} and {change_date:%Y-%m-%d}).",
        pair_text,
        "",
        "Rows around the change, from `data/raw/pond02.csv` (sensor_cells_of_6 = number of the six sensor cells present):",
        "",
        _md_table(around),
        "",
        timing_text,
        "",
    ]
    path = RESULTS / "step_row12000.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_report(
    audits: list[PondAudit],
    cross: pd.DataFrame,
    series: OriginalSeries,
    tables: dict[str, pd.DataFrame],
    run_info: dict[str, str],
) -> Path:
    summary = tables["summary"]
    daily = tables["daily_labels"]
    segments = tables["label_segments"]
    grid = tables["grid_check"]
    offgrid = tables["offgrid_knots"]
    changes = tables["label_changes"]
    gaps = tables["gaps"]
    hist = tables["interval_hist"]
    ranges = tables["ranges"]
    trailing = tables["trailing_cells"]
    nonnum = tables["nonnumeric_cells"]
    runs = tables["ts_runs"]
    schema = tables["schema"]

    per_pond = []
    for pond in POND_IDS:
        s = summary.loc[summary["pond"] == pond].iloc[0]
        d = daily.loc[daily["pond"] == pond]
        g = grid.loc[(grid["pond"] == pond)]
        c = changes.loc[changes["pond"] == pond]
        per_pond.append(
            {
                "pond": f"{pond:02d}",
                "rows": int(s["n_rows"]),
                "first_date": s["min_date"],
                "last_date": s["max_date"],
                "median_interval_s": s["interval_median_s"],
                "gaps_gt_10min": int(s["n_gaps_gt_10min"]),
                "label_changes": len(c),
                "distinct_daily_weight_modes": int(d["weight_mode"].dropna().nunique()),
                "grid_dates_with_rows": int(g["has_rows"].sum()),
                "population": s["population_unique"] or "absent",
            }
        )
    per_pond_df = pd.DataFrame(per_pond)

    units = pd.DataFrame(
        [
            {"pond": f"{p:02d}", **{var: UNITS_AS_LABELLED[p][var] for var in VARIABLES}}
            for p in POND_IDS
        ]
    )
    runs_text = []
    for pond in POND_IDS:
        part = runs.loc[runs["pond"] == pond]
        runs_text.append(
            f"- pond {pond:02d}: "
            + "; ".join(
                f"{r.ts_kind} rows {int(r.first_row)} to {int(r.last_row)} (n={int(r.n_rows)})"
                for r in part.itertuples()
            )
        )

    dateonly_lines = [
        f"- pond {int(r.pond):02d}: {int(r.n_date_only_dates)} calendar dates, from "
        f"{r.date_only_first_date} to {r.date_only_last_date}; spacing between those dates (days): "
        f"{r.date_only_spacing_days or 'none'}"
        for r in summary.loc[summary["n_date_only"] > 0].itertuples()
    ]

    hist_wide = hist.pivot(index="pond", columns="bin_s", values="count")[INTERVAL_LABELS].reset_index()
    hist_wide["pond"] = hist_wide["pond"].map(lambda p: f"{p:02d}")

    dup_table = summary[
        ["pond", "n_exact_duplicate_lines", "n_entry_id_repeated_rows", "n_intervals_nonpositive"]
    ].assign(pond=summary["pond"].map(lambda p: f"{p:02d}"))
    faults = tables["order_and_duplicate_faults"]
    nonpos_table = faults.loc[faults["fault"] == "non_positive_interval", [
        "pond", "row", "file_line", "other_row", "ts_raw", "entry_id"]].assign(
        pond=lambda d: d["pond"].map(lambda p: f"{p:02d}"))

    rng_flag = ranges.loc[ranges["n_out_of_audit_bounds"].fillna(0) > 0,
                          ["pond", "variable", "n_out_of_audit_bounds", "audit_lo", "audit_hi", "n_infinite"]]
    rng_flag = rng_flag.assign(pond=rng_flag["pond"].map(lambda p: f"{p:02d}"))
    neg127 = ranges.loc[ranges["variable"] == "temperature", ["pond", "n_equal_minus127"]]
    neg127 = neg127.assign(pond=neg127["pond"].map(lambda p: f"{p:02d}"))

    inf_counts = ranges.loc[ranges["variable"] == "ammonia", ["pond", "n_infinite"]]
    inf_counts = inf_counts.assign(pond=inf_counts["pond"].map(lambda p: f"{p:02d}"))

    nonnum_text = nonnum.assign(
        pond=nonnum["pond"].map(lambda p: f"{p:02d}")
    )[["pond", "row", "file_line", "column", "value"]] if len(nonnum) else pd.DataFrame()
    trailing_text = trailing.assign(
        pond=trailing["pond"].map(lambda p: f"{p:02d}")
    )[["pond", "row", "file_line", "column_name", "value"]] if len(trailing) else pd.DataFrame()

    cross_nonzero = cross.loc[cross["n_matched_pairs"] > 0, [
        "pond_a", "pond_b", "n_common_ts_raw", "n_common_clock_times", "n_common_date_only_raw",
        "n_common_entry_id", "n_matched_pairs", "frac_identical_temperature", "frac_identical_ammonia",
        "n_all_six_identical", "first_common_ts_raw", "last_common_ts_raw",
    ]]

    segs_cand = segments.loc[segments["candidate_knot"]] if len(segments) else segments
    seg_counts = (
        segs_cand.groupby("pond").size().reindex(POND_IDS, fill_value=0).reset_index(name="candidate_knots")
        if len(segs_cand) else pd.DataFrame({"pond": POND_IDS, "candidate_knots": 0})
    )
    seg_counts["pond"] = seg_counts["pond"].map(lambda p: f"{p:02d}")
    off_counts = (
        offgrid.groupby("pond").size().reindex(POND_IDS, fill_value=0).reset_index(name="off_grid_knots")
        if len(offgrid) else pd.DataFrame({"pond": POND_IDS, "off_grid_knots": 0})
    )
    off_counts["pond"] = off_counts["pond"].map(lambda p: f"{p:02d}")
    on_grid_knots = (
        grid.assign(knot=grid["candidate_knot_here"]).groupby("pond")["knot"].sum()
        .reindex(POND_IDS, fill_value=0).reset_index(name="candidate_knots_on_grid_dates")
    )
    on_grid_knots["pond"] = on_grid_knots["pond"].map(lambda p: f"{p:02d}")

    sha_table = summary[["pond", "sha256_matches_SHA256SUMS"]].assign(
        pond=summary["pond"].map(lambda p: f"{p:02d}")
    )

    lines: list[str] = [
        "# AUDIT_REPORT (Phase 1, aquacj-4561230)",
        "",
        "Facts only, generated by `src/audit.py` from `data/raw/`. Every number below is an output",
        "of that script; the file in brackets holds the full table. Row numbers are 0-based data",
        "rows (header excluded), so the file line is row + 2. No row is dropped in any table",
        "except where the text says so (`dropna()` in the notebook replication).",
        "",
        f"Run: {run_info['finished']}; runtime {run_info['seconds']} s; python {run_info['python']};",
        f"pandas {run_info['pandas']}; {run_info['platform']}; {run_info['cpus']} CPU cores.",
        "",
        "## 0. Inputs",
        "",
        "Raw files are read-only symlinks into `data/IoTPond_Dataset/`. SHA-256 is compared with `data/raw/SHA256SUMS`:",
        "",
        _md_table(sha_table.rename(columns={"sha256_matches_SHA256SUMS": "matches SHA256SUMS"})),
        "",
        "## 1. Schema, units as labelled, trailing columns (schema.csv, trailing_cells.csv)",
        "",
        f"The files have {len(schema.groupby('pond'))} pond files and {len(schema)} columns in total (schema.csv).",
        "Units are the text in the original header, unconverted (UNITS_AS_LABELLED in src/load.py):",
        "",
        _md_table(units),
        "",
        "Trailing unnamed columns (dropped by `load_pond`, every non-empty cell listed in trailing_cells.csv):",
        "",
        _md_table(trailing_text) if len(trailing_text) else "None.",
        "",
        "## 2. Row counts and timestamp kinds (summary.csv, ts_runs in this report)",
        "",
        "Contiguous runs of one timestamp kind, file order (first and last data row):",
        "",
        *runs_text,
        "",
        "Timezone labels as written: " + "; ".join(
            f"pond {int(r.pond):02d} {r.tz_label_counts or 'none'}" for r in summary.itertuples()
        ) + ".",
        "",
        "Date-only rows (no clock time, so they are outside the interval analysis):",
        "",
        *dateonly_lines,
        "",
        "## 3. Sampling intervals and gaps (summary.csv, interval_hist.csv, gaps.csv)",
        "",
        "Interval counts between consecutive timestamped rows (interval_hist.csv):",
        "",
        _md_table(hist_wide),
        "",
        "Gaps longer than 600 s between consecutive timestamped rows (gaps.csv):",
        "",
        _md_table(
            summary[["pond", "n_gaps_gt_10min", "total_gap_h", "longest_gap_h"]].assign(
                pond=summary["pond"].map(lambda p: f"{p:02d}")
            )
        ),
        "",
        "## 4. Missing values, non-numeric cells, infinite values (ranges.csv, nonnumeric_cells.csv)",
        "",
        "Non-numeric cells (non-empty text that did not parse as a number; NaN in the loaded frame):",
        "",
        _md_table(nonnum_text) if len(nonnum_text) else "None.",
        "",
        "Cells holding the literal `inf` parse as +inf and stay in the frame (n_infinite in ranges.csv):",
        "",
        _md_table(inf_counts),
        "",
        "Exact duplicate raw lines (a line identical in every cell to an earlier line) and entry IDs that repeat:",
        "",
        _md_table(dup_table),
        "",
        "Timestamped rows whose clock time is earlier than the row before (order_and_duplicate_faults.csv lists every row):",
        "",
        _md_table(nonpos_table) if len(nonpos_table) else "None.",
        "",
        "## 5. Ranges and audit bounds (ranges.csv)",
        "",
        "Temperature values equal to -127, by pond:",
        "",
        _md_table(neg127),
        "",
        "Variables with values outside the audit bounds (flags only, nothing removed):",
        "",
        _md_table(rng_flag) if len(rng_flag) else "None.",
        "",
        "## 6. Label changes (label_changes.csv)",
        "",
        "Rows where weight or length differs from the previous file row (missing against a value counts):",
        "",
        _md_table(
            pd.DataFrame(
                {
                    "pond": [f"{p:02d}" for p in POND_IDS],
                    "label_changes": [int((changes["pond"] == p).sum()) for p in POND_IDS],
                }
            )
        ),
        "",
        "## 7. Daily labels and segments (daily_labels.csv, label_segments.csv)",
        "",
        "Distinct daily modal weights, candidate knots (|delta slope| > 0.03 g/day and > 2 % of the previous slope):",
        "",
        _md_table(
            seg_counts.merge(
                pd.DataFrame(
                    {
                        "pond": [f"{p:02d}" for p in POND_IDS],
                        "distinct_daily_weight_modes": [
                            int(daily.loc[daily["pond"] == p, "weight_mode"].dropna().nunique())
                            for p in POND_IDS
                        ],
                    }
                ),
                on="pond",
            )
        ),
        "",
        "## 8. Grid check, 14-day grid from 2021-06-19 (grid_check.csv, offgrid_knots.csv)",
        "",
        "Grid dates with rows, and candidate knots that fall on a grid date (date_prev of the flagged pair):",
        "",
        _md_table(
            pd.DataFrame(
                {
                    "pond": [f"{p:02d}" for p in POND_IDS],
                    "grid_dates_with_rows": [
                        int(grid.loc[(grid["pond"] == p) & grid["has_rows"]].shape[0]) for p in POND_IDS
                    ],
                }
            ).merge(on_grid_knots, on="pond").merge(off_counts, on="pond")
        ),
        "",
        "## 9. Cross-pond overlaps (cross_pond_duplicates.csv)",
        "",
        "Pairs with at least one matched row on the raw timestamp string:",
        "",
        _md_table(cross_nonzero) if len(cross_nonzero) else "None.",
        "",
        "## 10. Original test series, pond 2 (step_row12000.md)",
        "",
        f"N = {series.n_retained} rows after `dropna()`, train_size = {series.train_size}, "
        f"test samples = {series.n_test}. The largest weight change in the test series is at "
        f"k = {series.largest}. Details and every change: step_row12000.md.",
        "",
        "## 11. Per-pond table",
        "",
        _md_table(per_pond_df),
        "",
        "Columns: rows; first and last calendar date; median positive interval between consecutive timestamped rows (s);",
        "gaps longer than 10 min; label changes; distinct daily modal weights; grid dates (of 17) with rows; population values.",
        "",
        "## 12. Definitions and deviations from the task file",
        "",
        "- Timestamps: the clock time is kept as written. Zone labels and `+01:00` offsets are not converted.",
        "- Intervals and gaps use consecutive rows that have a clock time. Date-only rows are not timestamped.",
        "- Interval percentiles and mean use positive intervals only. Non-positive intervals are counted separately.",
        "- `inf` cells: the loader keeps +inf (the token parses as a number). Quantiles use finite values; n_infinite and out-of-bounds counts include them.",
        "- Cross-pond overlap matches only rows with a clock time on the raw string; date-only overlaps are counted separately.",
        "- n_matched_pairs counts row pairs, so it can exceed n_common_ts_raw when a pond repeats a timestamp (see summary.csv n_rows_with_duplicate_ts_raw).",
        "- Candidate knot date is date_prev of the flagged pair (the junction between the two slopes).",
        "- Grid slopes: slope_w_before is the last pair ending on or before the grid date, slope_w_after the first pair starting on or after it.",
        "- Added files beyond the task list, same pointer convention: trailing_cells.csv, nonnumeric_cells.csv, order_and_duplicate_faults.csv.",
        "- Added summary columns: date-column agreement for ponds 7, 8, 9 (n_date_col_disagree_with_ts), exact duplicate lines, repeated entry IDs, sha256 check, date-only dates.",
        "- Population values: \"absent\" means the file has no population column (ponds 7, 8, 9).",
        "",
    ]
    path = RESULTS / "AUDIT_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------- main


def main() -> None:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    checksums = _published_sums()

    audits: list[PondAudit] = []
    for pond in POND_IDS:
        audits.append(analyse_pond(pond, checksums))
        print(f"audited pond {pond:02d}", flush=True)

    def stack(name: str) -> pd.DataFrame:
        return pd.concat([getattr(a, name) for a in audits], ignore_index=True)

    summary = pd.DataFrame([a.summary for a in audits])
    cross = _cross_pond({a.pond: a.slim for a in audits})
    tables = {
        "schema": stack("schema"),
        "summary": summary,
        "interval_hist": stack("interval_hist"),
        "gaps": stack("gaps"),
        "daily_coverage": stack("coverage"),
        "ranges": stack("ranges"),
        "label_changes": stack("label_changes"),
        "daily_labels": stack("daily_labels"),
        "label_segments": stack("label_segments"),
        "grid_check": stack("grid"),
        "offgrid_knots": stack("offgrid"),
        "trailing_cells": stack("trailing"),
        "nonnumeric_cells": stack("nonnumeric"),
        "order_and_duplicate_faults": stack("faults"),
        "ts_runs": stack("ts_runs"),
    }
    output_names = {
        "schema": "schema.csv",
        "summary": "summary.csv",
        "interval_hist": "interval_hist.csv",
        "gaps": "gaps.csv",
        "daily_coverage": "daily_coverage.csv",
        "ranges": "ranges.csv",
        "label_changes": "label_changes.csv",
        "daily_labels": "daily_labels.csv",
        "label_segments": "label_segments.csv",
        "grid_check": "grid_check.csv",
        "offgrid_knots": "offgrid_knots.csv",
        "trailing_cells": "trailing_cells.csv",
        "nonnumeric_cells": "nonnumeric_cells.csv",
        "order_and_duplicate_faults": "order_and_duplicate_faults.csv",
    }
    for key, filename in output_names.items():
        tables[key].to_csv(RESULTS / filename, index=False)
    cross.to_csv(RESULTS / "cross_pond_duplicates.csv", index=False)

    series = original_series()
    pond2_segments = next(a.label_segments for a in audits if a.pond == 2)
    write_step_md(series, pond2_segments)

    for audit in audits:
        plot_pond_labels(audit.pond, audit.daily_labels)
    plot_coverage(tables["daily_coverage"])
    plot_original_series(series)

    seconds = round(time.time() - started, 1)
    run_info = {
        "finished": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "seconds": f"{seconds}",
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "cpus": str(os.cpu_count()),
    }
    write_report(audits, cross, series, tables, run_info)
    print(f"done in {seconds} s", flush=True)


if __name__ == "__main__":
    main()
