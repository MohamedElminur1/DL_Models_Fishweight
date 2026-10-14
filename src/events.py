"""Step 1 of the Phase 4 data pipeline: the real measurement events of the cohort ponds.

Run from the repo root:  .venv/bin/python -I src/events.py

An event is a sampling date (day0 + 14 k, k = 0..16) on which the pond file has at least one
row with a weight. Its weight and length are the most frequent values among the rows dated
that day (dates come from `load_pond`, the file's own clock). Writes
data/processed/events.csv and results/events/grid_precision.csv. Stops without writing when
a grid date carries more than one distinct weight or length, when the event counts differ
from the Gate 1 table, or when an event differs from results/audit/grid_check.csv.
Specification: ANALYSIS_PLAN.md sections 2.2 to 2.4, tasks/phase4_experiments.md step 1.
"""

from __future__ import annotations

import datetime
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.load import SENSOR_VARS, load_pond  # noqa: E402
from src.pipeline_io import grid_dates, load_config, log_run, repo_path, write_csv  # noqa: E402

EVENT_COLUMNS = [
    "pond", "stream", "date", "t", "grid_k", "weight", "length", "n_rows", "source",
    "n_distinct_weight", "n_distinct_length",
]
PRECISION_COLUMNS = [
    "pond", "n_grid_values", "n_grid_values_2dec", "n_offgrid_days", "n_offgrid_days_2dec",
]


def daily_modal(frame: pd.DataFrame, column: str) -> pd.Series:
    """Most frequent value of `column` per date, over the rows dated that day.

    Ties resolve to the smallest value (as in src/audit.py). Rows without a date or without
    a value in the column are ignored, so a date with no value is absent from the result.
    """
    used = frame.loc[frame["date"].notna() & frame[column].notna(), ["date", column]]
    counts = used.groupby(["date", column]).size().rename("n").reset_index()
    counts = counts.sort_values(["date", "n", column], ascending=[True, False, True])
    return counts.drop_duplicates("date").set_index("date")[column]


def pond_events(frame: pd.DataFrame, pond: int, cfg: dict) -> pd.DataFrame:
    """Events of one pond from its `load_pond` frame, one row per sampling date with a weight.

    `n_rows` counts every row dated that day, `source` is `sensor` when any of those rows has
    a sensor value and `fish_only` otherwise, and the distinct counts are over non-missing
    values. `stream` is the pond's sensor stream from the config. No value is filled in: a
    sampling date without a weight is not an event.
    """
    day0 = pd.Timestamp(cfg["cohort"]["day0"])
    period = cfg["cohort"]["grid"]["period_days"]
    on_grid = frame.loc[frame["date"].isin(grid_dates(cfg))]
    has_sensor_value = on_grid[list(SENSOR_VARS)].notna().any(axis=1)
    grouped = on_grid.assign(has_sensor_value=has_sensor_value).groupby("date")
    table = pd.DataFrame(
        {
            "n_rows": grouped.size(),
            "any_sensor": grouped["has_sensor_value"].any(),
            "n_distinct_weight": grouped["weight"].nunique(),
            "n_distinct_length": grouped["length"].nunique(),
        }
    )
    table["weight"] = daily_modal(on_grid, "weight")
    table["length"] = daily_modal(on_grid, "length")
    events = table.loc[table["n_distinct_weight"] >= 1].reset_index()
    events["pond"] = pond
    events["stream"] = cfg["streams"][pond]
    events["t"] = (events["date"] - day0).dt.days
    events["grid_k"] = events["t"] // period
    events["source"] = np.where(events["any_sensor"], "sensor", "fish_only")
    return events[EVENT_COLUMNS].sort_values("date").reset_index(drop=True)


def two_decimals(values: np.ndarray, decimals: int, tolerance: float) -> np.ndarray:
    """True where a value equals its rounding to `decimals` places within `tolerance`."""
    return np.abs(values - np.round(values, decimals)) <= tolerance


def precision_row(frame: pd.DataFrame, pond: int, cfg: dict) -> dict:
    """Counts of per-date modal weights (all dated rows) with at most 2 decimals, grid vs off-grid."""
    modal = daily_modal(frame, "weight")
    two = two_decimals(
        modal.to_numpy(dtype="float64"),
        cfg["grid_precision"]["decimals"],
        cfg["grid_precision"]["tolerance"],
    )
    on_grid = modal.index.isin(grid_dates(cfg))
    return {
        "pond": pond,
        "n_grid_values": int(on_grid.sum()),
        "n_grid_values_2dec": int((on_grid & two).sum()),
        "n_offgrid_days": int((~on_grid).sum()),
        "n_offgrid_days_2dec": int((~on_grid & two).sum()),
    }


def read_grid_check(cfg: dict) -> pd.DataFrame:
    """results/audit/grid_check.csv (read-only), floats parsed exactly."""
    return pd.read_csv(
        repo_path(cfg["paths"]["grid_check"]),
        parse_dates=["grid_date"],
        float_precision="round_trip",
    )


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise equality where two missing values count as equal."""
    return (a == b) | (a.isna() & b.isna())


def check_events(events: pd.DataFrame, grid_check: pd.DataFrame, cfg: dict) -> list[str]:
    """Problems found in the event table; an empty list means every check passed.

    Checks: events per pond equal the Gate 1 table; no event has more than one distinct
    weight or length (or no length); the (pond, date) set, weight, length and row count equal
    the audit's grid_check.csv, and the audit's row source agrees with ours.
    """
    problems: list[str] = []
    ponds = cfg["cohort"]["ponds"]

    counts = events.groupby("pond").size().reindex(ponds, fill_value=0)
    for pond in ponds:
        expected = cfg["cohort"]["expected_events"][pond]
        if counts[pond] != expected:
            problems.append(f"pond {pond}: {counts[pond]} events, Gate 1 table says {expected}")

    ambiguous = events.loc[(events["n_distinct_weight"] != 1) | (events["n_distinct_length"] != 1)]
    for row in ambiguous.itertuples():
        problems.append(
            f"pond {row.pond} {row.date:%Y-%m-%d}: {row.n_distinct_weight} distinct weights, "
            f"{row.n_distinct_length} distinct lengths on a grid date"
        )

    audit = grid_check.loc[grid_check["pond"].isin(ponds) & grid_check["weight_mode"].notna()]
    merged = events.merge(
        audit, left_on=["pond", "date"], right_on=["pond", "grid_date"],
        how="outer", suffixes=("", "_audit"), indicator=True,
    )
    for row in merged.loc[merged["_merge"] == "left_only"].itertuples():
        problems.append(f"pond {row.pond} {row.date:%Y-%m-%d}: event not in grid_check.csv")
    for row in merged.loc[merged["_merge"] == "right_only"].itertuples():
        problems.append(f"pond {row.pond} {row.grid_date:%Y-%m-%d}: grid_check.csv has a weight, no event here")
    both = merged.loc[merged["_merge"] == "both"]
    wrong = both.loc[
        ~_same(both["weight"], both["weight_mode"])
        | ~_same(both["length"], both["length_mode"])
        | (both["n_rows"] != both["n_rows_audit"])
        | ((both["source"] == "fish_only") != (both["row_source"] == "fish_only"))
    ]
    for row in wrong.itertuples():
        problems.append(
            f"pond {row.pond} {row.date:%Y-%m-%d}: events weight/length/rows/source "
            f"{row.weight}/{row.length}/{row.n_rows}/{row.source} vs grid_check "
            f"{row.weight_mode}/{row.length_mode}/{row.n_rows_audit}/{row.row_source}"
        )
    return problems


def main() -> None:
    started = datetime.datetime.now(datetime.timezone.utc)
    clock = time.perf_counter()
    cfg = load_config()

    event_parts, precision_rows = [], []
    for pond in cfg["cohort"]["ponds"]:
        frame = load_pond(pond)
        event_parts.append(pond_events(frame, pond, cfg))
        precision_rows.append(precision_row(frame, pond, cfg))
    events = pd.concat(event_parts, ignore_index=True)
    precision = pd.DataFrame(precision_rows, columns=PRECISION_COLUMNS)

    problems = check_events(events, read_grid_check(cfg), cfg)
    if problems:
        raise SystemExit("event checks failed, nothing written:\n  " + "\n  ".join(problems))

    out = events.assign(date=events["date"].dt.strftime("%Y-%m-%d"))
    write_csv(out, repo_path(cfg["paths"]["events"]))
    write_csv(precision, repo_path(cfg["paths"]["grid_precision"]))

    print("events per pond:", events.groupby("pond").size().to_dict(), "total", len(events))
    print(precision.to_string(index=False))
    seconds = time.perf_counter() - clock
    log_run(cfg, "events.py", started, seconds)
    print(f"done in {seconds:.1f} s")


if __name__ == "__main__":
    main()
