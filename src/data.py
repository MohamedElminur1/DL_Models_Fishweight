"""Loading and checking the Phase 4 tables for the modelling scripts.

`load_tables` reads the T1 and T2 tables, the daily water-quality table and the features by
date (data/processed/, written by src/features.py) and merges into each task table the window
columns of the supplementary window lengths that its sensitivity models use (plan 8.3), taken
from wq_by_date.csv under their prefixes (`t1w07_...`). `validate_tables` stops on anything
that would make a model silently wrong: duplicate keys, a time axis that disagrees with the
dates, a horizon that is not 14 or 28 days, a stream column that disagrees with the design,
or a window feature that differs between a task table and wq_by_date.csv (which the
permutation test uses as the donor of water-quality inputs).

Dates stay ISO strings in the frames, as in the CSV files.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.artifacts import repo_path
from src.baselines import window_prefixes
from src.folds import sensor_streams

T1_COLUMNS = ["pond", "stream", "date", "t", "t2", "log_w", "w", "log_l", "l",
              "n_covered_window", "eligible_eval"]
T2_COLUMNS = ["pond", "stream", "origin_date", "target_date", "horizon", "t_k", "dt_prev",
              "dt_target", "log_w_k", "log_w_prev", "w_origin", "w_target", "y",
              "n_covered_window", "eligible_eval"]
KEY_COLUMN = {"T1": "date", "T2": "origin_date"}   # the date whose water quality a sample uses
WQ_PREFIX = {"T1": "t1", "T2": "t2"}               # prefix of the task's main window in wq_by_date


@dataclass
class Tables:
    """The four input tables. `table(task)` returns the T1 or T2 table."""

    t1: pd.DataFrame
    t2: pd.DataFrame
    daily: pd.DataFrame
    wq: pd.DataFrame

    def table(self, task: str) -> pd.DataFrame:
        return self.t1 if task == "T1" else self.t2


def read_csv(path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip")


def window_columns(wq: pd.DataFrame, prefix: str) -> list[str]:
    """Columns of wq_by_date that belong to a prefix (`t1`, `t1w07`, ...), in file order."""
    return [c for c in wq.columns if c.startswith(prefix + "_")]


def merge_window_columns(task: str, table: pd.DataFrame, wq: pd.DataFrame, prefixes: list[str]) -> pd.DataFrame:
    """Add the window columns of the given prefixes from wq_by_date to a task table.

    Each sample receives the row of its pond at the table's key date (T1 event date, T2
    origin date). The columns keep their prefixed names. Raises ValueError if any sample has
    no row. The rows and their order are those of `table`.
    """
    if not prefixes:
        return table
    key = KEY_COLUMN[task]
    wanted = [c for prefix in prefixes for c in window_columns(wq, prefix)
              if not c.endswith("n_covered_window")]
    if len(wanted) != 24 * len(prefixes):
        raise ValueError(f"wq_by_date lacks window columns for prefixes {prefixes}")
    join = table[["pond", key]].merge(
        wq[["pond", "date", *wanted]], left_on=["pond", key], right_on=["pond", "date"],
        how="left", validate="many_to_one", indicator=True)
    if (join["_merge"] != "both").any():
        raise ValueError("a sample date has no row in wq_by_date.csv")
    out = table.copy()
    for column in wanted:
        out[column] = join[column].to_numpy()
    return out


def load_tables(cfg: dict) -> Tables:
    """Read the tables named in `cfg["paths"]`, merge the window variants and validate."""
    paths = cfg["paths"]
    t1, t2 = read_csv(repo_path(paths["t1_table"])), read_csv(repo_path(paths["t2_table"]))
    daily, wq = read_csv(repo_path(paths["daily_wq"])), read_csv(repo_path(paths["wq_by_date"]))
    tables = Tables(
        merge_window_columns("T1", t1, wq, window_prefixes("T1", cfg)),
        merge_window_columns("T2", t2, wq, window_prefixes("T2", cfg)),
        daily, wq)
    validate_tables(tables, cfg)
    return tables


def _same(a: pd.Series, b: pd.Series) -> bool:
    return bool(np.allclose(a.to_numpy(dtype=float), b.to_numpy(dtype=float),
                            rtol=1e-12, atol=1e-12, equal_nan=True))


def validate_tables(tables: Tables, cfg: dict) -> None:
    """Raise ValueError unless the tables are consistent with the design (see module docstring)."""
    design = cfg["design"]
    cohort = set(design["cohort"])
    day0 = pd.Timestamp(design["day0"])
    for task, expected in (("T1", T1_COLUMNS), ("T2", T2_COLUMNS)):
        table = tables.table(task)
        missing = [c for c in expected if c not in table.columns]
        if missing:
            raise ValueError(f"{task} table lacks columns {missing}")
        if not set(table["pond"]) <= cohort:
            raise ValueError(f"{task} table has ponds outside the cohort: {sorted(set(table['pond']) - cohort)}")
        if table["eligible_eval"].dtype != bool:
            raise ValueError(f"{task} column eligible_eval is not boolean")
        streams = table.groupby("stream")["pond"].agg(lambda s: tuple(sorted(set(s))))
        want = {tuple(sorted(s)) for s in sensor_streams(sorted(set(table["pond"])), design["shared_streams"])}
        if set(streams) != want:
            raise ValueError(f"{task} stream column {sorted(set(streams))} disagrees with design.shared_streams")
    t1, t2 = tables.t1, tables.t2
    if t1.duplicated(["pond", "date"]).any() or t2.duplicated(["pond", "origin_date", "horizon"]).any():
        raise ValueError("duplicate sample keys in the T1 or T2 table")
    days = (pd.to_datetime(t1["date"]) - day0).dt.days
    if not (days == t1["t"]).all() or not (t1["t"] > 0).all():
        raise ValueError("T1 column t is not the days since day 0 of the date, or is not positive")
    if not np.allclose(t1["t2"], t1["t"] ** 2) or not np.allclose(t1["log_w"], np.log(t1["w"])):
        raise ValueError("T1 columns t2 or log_w disagree with t and w")
    if not (t1[["log_w", "log_l", "t", "t2"]].notna().all().all() and (t1["w"] > 0).all()):
        raise ValueError("T1 time, weight or length columns are missing or not positive")
    horizons = set(design["horizons"])
    if not set(t2["horizon"]) <= horizons:
        raise ValueError(f"T2 horizons {sorted(set(t2['horizon']) - horizons)} are not in the design")
    if not (t2["dt_target"] == design["horizon_days"] * t2["horizon"]).all():
        raise ValueError("a T2 target is not exactly horizon_days * horizon days after the origin")
    origin_t = (pd.to_datetime(t2["origin_date"]) - day0).dt.days
    target_t = (pd.to_datetime(t2["target_date"]) - day0).dt.days
    if not ((origin_t == t2["t_k"]).all() and (target_t - origin_t == t2["dt_target"]).all()):
        raise ValueError("T2 columns t_k or dt_target disagree with the dates")
    if not (t2["dt_prev"] > 0).all():
        raise ValueError("T2 column dt_prev is not positive")
    needed = ["log_w_k", "log_w_prev", "w_origin", "w_target", "y"]
    if t2[needed].isna().any().any() or not (t2[["w_origin", "w_target"]] > 0).all().all():
        raise ValueError("T2 weights or targets are missing or not positive")
    if not (np.allclose(t2["y"], np.log(t2["w_target"] / t2["w_origin"]))
            and np.allclose(t2["log_w_k"], np.log(t2["w_origin"]))):
        raise ValueError("T2 columns y or log_w_k disagree with the weights")
    wq = tables.wq
    for task, table in (("T1", t1), ("T2", t2)):
        prefix = WQ_PREFIX[task]
        rows = table[["pond", KEY_COLUMN[task]]].merge(
            wq, left_on=["pond", KEY_COLUMN[task]], right_on=["pond", "date"], how="left")
        if rows["date"].isna().any():
            raise ValueError(f"a {task} sample date has no row in wq_by_date.csv")
        for column in window_columns(wq, prefix):
            own = column[len(prefix) + 1:]
            if not _same(table[own], rows[column]):
                raise ValueError(f"{task} column {own} differs from wq_by_date column {column}")
    if not cohort <= set(tables.daily["pond"]):
        raise ValueError("daily_wq.csv lacks a cohort pond")
