"""Loaders shared by the figures that show the real weighings (fig_events, fig_coverage).

The event table joins `data/processed/events.csv` (one row per real weighing of the eight
cohort ponds) with the eligibility flag of `data/processed/t1_table.csv`. Counts are asserted
against the numbers of Gate 1 and ANALYSIS_PLAN.md 4.1 (70 weighings, 63 with t > 0, 35
eligible), so a data change stops the figure scripts instead of silently changing a figure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from reporting import figstyle as fs

EVENTS_CSV = fs.REPO / "data/processed/events.csv"
T1_CSV = fs.REPO / "data/processed/t1_table.csv"
DAILY_WQ_CSV = fs.REPO / "data/processed/daily_wq.csv"
DAILY_LABELS_CSV = fs.REPO / "results/audit/daily_labels.csv"

EXPECTED = {"events": 70, "events_t_gt_0": 63, "eligible": 35, "events_t0": 7}


def load_events() -> pd.DataFrame:
    """Real weighings with columns pond, date, t, weight, source and kind.

    kind is "t0" (stocking weighing, t = 0), "eligible" (T1 event scored in the evaluation) or
    "not_eligible" (T1 event with fewer than 7 covered days in the 14 days before it).
    """
    events = pd.read_csv(EVENTS_CSV, parse_dates=["date"])
    t1 = pd.read_csv(T1_CSV, parse_dates=["date"], usecols=["pond", "date", "eligible_eval"])
    merged = events.merge(t1, on=["pond", "date"], how="left", validate="one_to_one")
    later = merged["t"] > 0
    if merged.loc[later, "eligible_eval"].isna().any():
        raise ValueError("an event with t > 0 has no row in t1_table.csv")
    merged["kind"] = np.where(
        ~later, "t0", np.where(merged["eligible_eval"].astype("boolean").fillna(False), "eligible", "not_eligible")
    )
    found = {
        "events": len(merged),
        "events_t_gt_0": int(later.sum()),
        "eligible": int((merged["kind"] == "eligible").sum()),
        "events_t0": int((merged["kind"] == "t0").sum()),
    }
    if found != EXPECTED:
        raise ValueError(f"event counts changed: found {found}, expected {EXPECTED}")
    if set(merged["pond"]) != set(fs.COHORT_PONDS):
        raise ValueError("events.csv ponds differ from the cohort ponds of figstyle")
    return merged.drop(columns=["eligible_eval"])


def break_at_gaps(dates: pd.Series, values: pd.Series, max_step_days: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Date and value arrays with a NaN row wherever consecutive dates are more than `max_step_days` apart.

    A line drawn through them is broken at days without rows instead of being bridged.
    """
    x = pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")
    y = values.to_numpy(dtype=float)
    out_x, out_y = [], []
    for i in range(len(x)):
        if i and (x[i] - x[i - 1]) > np.timedelta64(max_step_days, "D"):
            out_x.append(x[i - 1] + np.timedelta64(1, "h"))
            out_y.append(np.nan)
        out_x.append(x[i])
        out_y.append(y[i])
    return np.array(out_x), np.array(out_y)
