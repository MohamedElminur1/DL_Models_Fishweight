"""fig_original_test: pond 2, the original test period, in calendar time.

Run from the repo root: `.venv/bin/python -I reporting/figs/fig_original_test.py`

Sources: results/phase2/predictions/ (S0_notebook persistence and CNN-LSTM seeds 0 to 2, and the
S2_random CNN-LSTM seed 0 restricted to targets inside the test period) and
data/processed/events.csv (the two real weighings of pond 2 inside the period).
Panel (a) shows the whole test period of the notebook design, panel (b) a zoom on
2021-10-12 to 2021-10-25. The label is the daily value of the source file (interpolated);
lines are broken, not bridged, where consecutive test rows are more than 24 h apart, and
rows that stand alone between two such gaps are drawn as single markers.
Output: figures/main/fig_original_test.{png,pdf}
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

PRED_DIR = fs.REPO / "results/phase2/predictions"
PERSISTENCE = "S0_notebook__none__persistence__sNA"
CNN_LSTM_S0 = [f"S0_notebook__notebook__cnn_lstm__s{seed}" for seed in (0, 1, 2)]
CNN_LSTM_S2 = "S2_random__notebook__cnn_lstm__s0"

GAP_MIN_HOURS = 24
MAIN_GAP = (pd.Timestamp("2021-10-14 06:11:11"), pd.Timestamp("2021-10-21 18:11:11"))
ZOOM = (pd.Timestamp("2021-10-12"), pd.Timestamp("2021-10-25"))
POND = 2
LABEL_COLOR = "#B0B0B0"
MAIN_GAP_COLOR = "#E4E4E4"
OTHER_GAP_COLOR = "#F0F0F0"


def load_predictions(name: str) -> pd.DataFrame:
    """Predictions file with `time` parsed from created_at (clock as labelled, CET)."""
    frame = pd.read_csv(PRED_DIR / f"{name}.csv.gz")
    if not frame["created_at"].str.endswith(" CET").all():
        raise ValueError(f"{name}: created_at is not labelled CET throughout")
    frame["time"] = pd.to_datetime(frame["created_at"].str.removesuffix(" CET"))
    return frame


def logging_gaps(time: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Intervals between consecutive rows that are longer than GAP_MIN_HOURS."""
    step = time.diff()
    ends = time[step > pd.Timedelta(hours=GAP_MIN_HOURS)]
    return [(time.loc[i - 1], end) for i, end in ends.items()]


def with_breaks(time: pd.Series, values: pd.Series, gaps: list[tuple[pd.Timestamp, pd.Timestamp]]):
    """Arrays with a NaN point just after the last row before each gap, so lines stop at the gap."""
    x = time.to_numpy(dtype="datetime64[ns]")
    y = values.to_numpy(dtype=float)
    for start, _ in reversed(gaps):
        i = int(np.searchsorted(x, np.datetime64(start), side="right"))
        x = np.insert(x, i, np.datetime64(start) + np.timedelta64(1, "s"))
        y = np.insert(y, i, np.nan)
    return x, y


def isolated_rows(time: pd.Series) -> np.ndarray:
    """Boolean mask of rows with a gap of more than GAP_MIN_HOURS on both sides."""
    limit = pd.Timedelta(hours=GAP_MIN_HOURS)
    before = time.diff() > limit
    after = time.diff(-1).abs() > limit
    return (before & after).to_numpy()


def plot_panel(ax, series: dict, weighings: pd.DataFrame, gaps):
    """Draw every series, the gap bands and the real weighings on `ax`."""
    for gap in gaps:
        ax.axvspan(*gap, color=MAIN_GAP_COLOR if gap == MAIN_GAP else OTHER_GAP_COLOR, lw=0, zorder=0.5)
    alone = isolated_rows(series["time"])
    model_colors = fs.MODEL_COLORS

    label_x, label_y = with_breaks(series["time"], series["label"], gaps)
    ax.plot(label_x, label_y, color=LABEL_COLOR, lw=2.4, drawstyle="steps-post", solid_capstyle="butt", zorder=1)
    ax.plot(series["time"][alone], series["label"][alone], ls="none", marker="o", ms=3.2, mew=0,
            color=LABEL_COLOR, zorder=1)

    ax.plot(series["random_time"], series["random_pred"], ls="none", marker="o", ms=0.9, mew=0,
            color=model_colors["cnn_lstm_random"], alpha=0.55, zorder=2)
    for seed_values in series["cnn_lstm_s0"]:
        x, y = with_breaks(series["time"], seed_values, gaps)
        ax.plot(x, y, color=model_colors["cnn_lstm_notebook"], lw=0.45, zorder=2.2)
        ax.plot(series["time"][alone], seed_values[alone], ls="none", marker="o", ms=2.2, mew=0,
                color=model_colors["cnn_lstm_notebook"], zorder=2.2)

    persistence_x, persistence_y = with_breaks(series["time"], series["persistence"], gaps)
    ax.plot(persistence_x, persistence_y, color=model_colors["persistence"], lw=0.8, drawstyle="steps-post", zorder=2.5)
    ax.plot(series["time"][alone], series["persistence"][alone], ls="none", marker="o", ms=2.4, mew=0,
            color=model_colors["persistence"], zorder=2.5)

    noon = weighings["date"] + pd.Timedelta(hours=12)
    ax.errorbar(noon, weighings["weight"], xerr=pd.Timedelta(hours=12), fmt="none",
                ecolor=fs.POND_COLORS[POND], elinewidth=1.0, capsize=0, zorder=3)
    ax.plot(noon, weighings["weight"], **fs.event_marker_kw("eligible", POND, size=4.6), zorder=3.2)


def main() -> None:
    persistence = load_predictions(PERSISTENCE)
    cnn_s0 = [load_predictions(name) for name in CNN_LSTM_S0]
    for frame in cnn_s0:
        if not frame["created_at"].equals(persistence["created_at"]) or not frame["y_true"].equals(persistence["y_true"]):
            raise ValueError("CNN-LSTM S0 test rows differ from the persistence rows")
    random_split = load_predictions(CNN_LSTM_S2)

    time = persistence["time"]
    gaps = logging_gaps(time)
    if MAIN_GAP not in gaps or max(gaps, key=lambda gap: gap[1] - gap[0]) != MAIN_GAP:
        raise ValueError(f"the longest logging gap is not {MAIN_GAP}: found {gaps}")
    in_period = (random_split["time"] >= time.iloc[0]) & (random_split["time"] <= time.iloc[-1])
    random_in = random_split[in_period]

    events = _events.load_events()
    weighings = events[(events["pond"] == POND) & (events["date"] >= time.iloc[0].normalize())
                       & (events["date"] <= time.iloc[-1].normalize())]
    if len(weighings) != 2:
        raise ValueError("expected two real weighings of pond 2 inside the test period")

    series = {
        "time": time,
        "label": persistence["y_true"],
        "persistence": persistence["y_pred"],
        "cnn_lstm_s0": [frame["y_pred"] for frame in cnn_s0],
        "random_time": random_in["time"],
        "random_pred": random_in["y_pred"],
    }

    fig, (ax_a, ax_b) = fs.subplots(2, 1, width_cm=fs.FULL_CM, height_cm=12.6)
    plot_panel(ax_a, series, weighings, gaps)
    plot_panel(ax_b, series, weighings, gaps)

    fs.day_axis(ax_a, str(time.iloc[0].normalize() - pd.Timedelta(days=0)), "2021-11-01", every_days=2)
    fs.day_axis(ax_b, str(ZOOM[0]), str(ZOOM[1]), every_days=1)
    ax_a.set_ylim(135, 420)
    window = (series["time"] >= ZOOM[0]) & (series["time"] <= ZOOM[1])
    low = min(series["label"][window].min(), *[s[window].min() for s in series["cnn_lstm_s0"]])
    high = max(series["label"][window].max(), *[s[window].max() for s in series["cnn_lstm_s0"]])
    margin = 0.05 * (high - low)
    ax_b.set_ylim(low - margin, high + margin)
    for ax in (ax_a, ax_b):
        ax.set_ylabel("Fish weight (g)")
        ax.set_xlabel("created_at (CET, 2021)")
    fs.panel_letter(ax_a, "a", outside=True)
    fs.panel_letter(ax_b, "b", outside=True)

    # Bracket under panel (a) marking the span shown in panel (b).
    y_bracket = 150
    ax_a.annotate("", xy=(ZOOM[0], y_bracket), xytext=(ZOOM[1], y_bracket),
                  arrowprops={"arrowstyle": "|-|,widthA=0.25,widthB=0.25", "lw": 0.6, "color": fs.INK,
                              "shrinkA": 0, "shrinkB": 0})
    ax_a.annotate("zoom in (b)", xy=(ZOOM[0] + (ZOOM[1] - ZOOM[0]) / 2, y_bracket), xycoords="data",
                  xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom", fontsize=fs.TEXT_PT)

    for ax, y_frac in ((ax_a, 0.62), (ax_b, 0.62)):
        ax.annotate("No rows logged\n(7.5 days)", xy=(MAIN_GAP[0] + (MAIN_GAP[1] - MAIN_GAP[0]) / 2, y_frac),
                    xycoords=("data", "axes fraction"), ha="center", va="center", fontsize=fs.TEXT_PT,
                    color="#4D4D4D")

    handles = [
        Line2D([], [], color=LABEL_COLOR, lw=2.4, label="Label: daily value in the source file (interpolated)"),
        Line2D([], [], **fs.event_marker_kw("eligible", POND, size=4.6), label="Real weighing (date only)"),
        Line2D([], [], color=fs.MODEL_COLORS["persistence"], lw=0.9, label="Persistence"),
        Line2D([], [], color=fs.MODEL_COLORS["cnn_lstm_notebook"], lw=0.9, label="CNN-LSTM, notebook design (seeds 0 to 2)"),
        Line2D([], [], ls="none", marker="o", ms=3, mew=0, color=fs.MODEL_COLORS["cnn_lstm_random"],
               label="CNN-LSTM, random split (seed 0)"),
        Patch(facecolor=MAIN_GAP_COLOR, edgecolor="#BDBDBD", lw=0.4, label="No rows logged (gaps of more than 24 h)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    for path in fs.save(fig, "fig_original_test", "main"):
        print(path)


if __name__ == "__main__":
    main()
