"""fig_coverage: sensors with a daily value after quality control, per cohort pond and date.

Run from the repo root: `.venv/bin/python -I reporting/figs/fig_coverage.py`

Source: data/processed/daily_wq.csv (column status: observed, interpolated or missing for each
pond, date and sensor, as saved by the data pipeline). Event markers come from
data/processed/events.csv and data/processed/t1_table.csv, as in fig_events.
Each strip stacks, for every day, one cell per sensor with a value (at most six): observed
sensors from the baseline up, interpolated sensors (gaps of at most two days) above them in
grey. Output: figures/main/fig_coverage.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from reporting import figstyle as fs  # noqa: E402
from reporting.figs import _events  # noqa: E402

START, END = "2021-06-01", "2022-02-05"
N_SENSORS = 6
OBSERVED_COLOR = fs.OKABE_ITO["blue"]
MARKER_Y = -1.25


def daily_counts() -> pd.DataFrame:
    """Per pond and date, the number of sensors observed and interpolated (0 to 6 together)."""
    daily = pd.read_csv(_events.DAILY_WQ_CSV, parse_dates=["date"], usecols=["pond", "date", "sensor", "status"])
    if daily["sensor"].nunique() != N_SENSORS:
        raise ValueError("daily_wq.csv no longer has six sensors")
    counts = (
        daily.assign(observed=daily["status"] == "observed", interpolated=daily["status"] == "interpolated")
        .groupby(["pond", "date"])[["observed", "interpolated"]]
        .sum()
        .reset_index()
    )
    if (counts["observed"] + counts["interpolated"]).max() > N_SENSORS:
        raise ValueError("more than six sensors with a value on one day")
    return counts


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and stop index (exclusive) of every run of True in a boolean array."""
    padded = np.concatenate([[False], mask, [False]])
    change = np.flatnonzero(padded[1:] != padded[:-1])
    return change[0::2], change[1::2]


def main() -> None:
    counts = daily_counts()
    events = _events.load_events()

    n = len(fs.COHORT_PONDS)
    fig, axes = fs.subplots(n, 1, width_cm=fs.FULL_CM, height_cm=11.2, sharex=True)
    for ax, pond in zip(axes, fs.COHORT_PONDS):
        own = counts[counts["pond"] == pond].sort_values("date")
        edges = np.append(own["date"].to_numpy(), own["date"].iloc[-1] + pd.Timedelta(days=1))
        observed = np.append(own["observed"].to_numpy(), own["observed"].iloc[-1])
        total = np.append((own["observed"] + own["interpolated"]).to_numpy(),
                          (own["observed"] + own["interpolated"]).iloc[-1])
        ax.fill_between(edges, 0, total, step="post", color=fs.INTERP_COLOR, lw=0, zorder=2)
        ax.fill_between(edges, 0, observed, step="post", color=OBSERVED_COLOR, lw=0, zorder=2.1)
        for level in range(1, N_SENSORS):
            starts, stops = _runs(total[:-1] > level)
            ax.hlines(np.full(len(starts), level), edges[starts], edges[stops], color="white", lw=0.3, zorder=2.5)

        own_events = events[events["pond"] == pond]
        for kind in ("not_eligible", "eligible", "t0"):
            sel = own_events[own_events["kind"] == kind]
            if len(sel):
                ax.plot(sel["date"], np.full(len(sel), MARKER_Y), **fs.event_marker_kw(kind, pond, size=3.6),
                        clip_on=False, zorder=3)

        ax.set_ylim(-2.0, N_SENSORS + 0.3)
        ax.set_yticks([0, 3, 6])
        ax.set_ylabel(f"Pond {pond}", rotation=0, ha="right", va="center", labelpad=14)
        ax.spines["left"].set_bounds(0, N_SENSORS)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))

    bottom = axes[-1]
    bottom.spines["bottom"].set_visible(True)
    bottom.tick_params(axis="x", which="major", length=2.8)
    fs.month_axis(bottom, START, END, every=1)
    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="x", which="major", color=fs.GRID_COLOR, lw=0.4)
    fig.supylabel("Sensors with a value per day (0 to 6)", fontsize=fs.TEXT_PT)

    handles = [
        Patch(facecolor=OBSERVED_COLOR, label="Sensor with an observed daily value"),
        Patch(facecolor=fs.INTERP_COLOR, label="Sensor value interpolated (gap of at most 2 days)"),
        *fs.event_legend_handles(),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    for path in fs.save(fig, "fig_coverage", "main"):
        print(path)


if __name__ == "__main__":
    main()
