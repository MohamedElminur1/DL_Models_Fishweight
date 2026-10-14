"""fig_events: fish weight of the eight cohort ponds, real weighings against the daily file values.

Run from the repo root: `.venv/bin/python -I reporting/figs/fig_events.py`

Sources: data/processed/events.csv (real weighings), data/processed/t1_table.csv (column
eligible_eval), results/audit/daily_labels.csv (column weight_mode, the daily values found in
the pond files, which are straight-line interpolations between weighings).
Output: figures/main/fig_events.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

from reporting import figstyle as fs  # noqa: E402
from reporting.figs import _events  # noqa: E402

START, END = "2021-06-01", "2022-02-05"
WEIGHT_LIMITS = (2.0, 1500.0)
LETTERS = "abcdefgh"


def main() -> None:
    events = _events.load_events()
    labels = pd.read_csv(_events.DAILY_LABELS_CSV, parse_dates=["date"])

    fig, axes = fs.subplots(2, 4, width_cm=fs.FULL_CM, height_cm=9.4, sharex=True, sharey=True)
    for ax, letter, pond in zip(axes.flat, LETTERS, fs.COHORT_PONDS):
        daily = labels[labels["pond"] == pond].sort_values("date")
        x, y = _events.break_at_gaps(daily["date"], daily["weight_mode"])
        ax.plot(x, y, **fs.INTERP_LINE_KW, zorder=2)
        ax.axvline(fs.CUTOFF_DATE, **fs.CUTOFF_KW)

        own = events[events["pond"] == pond]
        for kind in ("not_eligible", "eligible", "t0"):
            sel = own[own["kind"] == kind]
            if len(sel):
                ax.plot(sel["date"], sel["weight"], **fs.event_marker_kw(kind, pond), zorder=3)

        ax.set_yscale("log")
        ax.set_ylim(*WEIGHT_LIMITS)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        fs.month_axis(ax, START, END, every=2, rotate=45)
        fs.panel_letter(ax, letter, label=f"Pond {pond}")
        n_eligible = int((own["kind"] == "eligible").sum())
        ax.annotate(f"{len(own)} weighings\n{n_eligible} eligible", xy=(1, 0), xycoords="axes fraction",
                    xytext=(-2, 3), textcoords="offset points", ha="right", va="bottom",
                    fontsize=fs.TEXT_PT, color="#4D4D4D")

    fig.supylabel("Fish weight (g)", fontsize=fs.TEXT_PT)
    handles = [
        Line2D([], [], **fs.INTERP_LINE_KW, label="Daily values in the source file (interpolated)"),
        *fs.event_legend_handles(),
        Line2D([], [], **{**fs.CUTOFF_KW, "zorder": None}, label=f"Hold-out cut-off, {fs.CUTOFF_DATE:%d %b %Y}"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    for path in fs.save(fig, "fig_events", "main"):
        print(path)


if __name__ == "__main__":
    main()
