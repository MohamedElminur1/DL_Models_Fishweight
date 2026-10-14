"""Shared style for the publication figures of the aquacj-4561230 revision.

Every figure script under reporting/figs/ builds its figure through this module, so size,
fonts, colours and output formats are the same everywhere. Run a script from the repo root as
`.venv/bin/python -I reporting/figs/<name>.py`; `-I` removes the script directory from
sys.path, so each script first puts the repo root on sys.path (see any script).

Typical use:

    fig, axes = fs.subplots(2, 4, width_cm=fs.FULL_CM, height_cm=9.0, sharex=True)
    ...
    fs.save(fig, "fig_events", "main")                 # PNG (600 dpi) and PDF
    fs.save(fig, "fig_workflow", "main", svg=True)     # also SVG, for diagrams

Rules encoded here (tasks/phase6_figures_part1.md):
- 16 cm (FULL_CM) or 8 cm (HALF_CM) wide, sans serif text of 8 pt, panel letters of 9 pt, no
  title inside the image (the caption carries it);
- 600 dpi PNG and PDF for every figure, SVG as well for diagrams;
- Okabe-Ito palette; every cohort pond keeps one colour in every figure (POND_COLORS);
- real measurements are filled markers (event_marker_kw), interpolated values are grey
  (INTERP_COLOR) and named as interpolated in the legend;
- time axes show calendar dates (month_axis, day_axis), never sample indices.

Colour check (dataviz skill, validate_palette.py, white surface): the pond order below keeps
neighbouring ponds 1, 2, 3, 4, 6, 7, 8, 9 apart under protanopia and deuteranopia (worst
adjacent distance 19.3, target 8). Okabe-Ito black and yellow sit outside the lightness band
of that checker and orange, sky blue and yellow have under 3:1 contrast on white. Colour is
therefore never the only identity cue (panels and rows are labelled) and markers carry a dark
edge (event_marker_kw).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
FIG_DIR = REPO / "figures"

FULL_CM = 16.0
HALF_CM = 8.0
DPI = 600
TEXT_PT = 8.0
LETTER_PT = 9.0

OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# One colour per cohort pond, the same in every figure.
POND_COLORS = {
    1: OKABE_ITO["blue"],
    2: OKABE_ITO["vermillion"],
    3: OKABE_ITO["yellow"],
    4: OKABE_ITO["reddish_purple"],
    6: OKABE_ITO["orange"],
    7: OKABE_ITO["sky_blue"],
    8: OKABE_ITO["black"],
    9: OKABE_ITO["bluish_green"],
}
COHORT_PONDS = tuple(POND_COLORS)

# Colours of model roles in figures about one pond (not used where ponds are coloured).
MODEL_COLORS = {
    "persistence": OKABE_ITO["blue"],
    "cnn_lstm_notebook": OKABE_ITO["reddish_purple"],
    "cnn_lstm_random": OKABE_ITO["bluish_green"],
}

INK = "#1F1F1F"
EDGE = "#262626"
INTERP_COLOR = "#8C8C8C"      # interpolated values (daily labels, interpolated sensor days)
GAP_COLOR = "#BDBDBD"         # shaded stretches without data
GRID_COLOR = "#E3E3E3"

# Calendar hold-out (ANALYSIS_PLAN.md 7.4), read from the analysis config so the figures cannot
# disagree with it.
CUTOFF_DATE = pd.Timestamp(yaml.safe_load((REPO / "configs/phase4.yaml").read_text())["chrono"]["cutoff"])
CUTOFF_KW = {"color": "#3A3A3A", "lw": 0.6, "ls": (0, (3.0, 2.0)), "zorder": 1.5}
INTERP_LINE_KW = {"color": INTERP_COLOR, "lw": 0.7, "solid_capstyle": "butt"}

_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
    "font.size": TEXT_PT,
    "axes.labelsize": TEXT_PT,
    "axes.titlesize": TEXT_PT,
    "xtick.labelsize": TEXT_PT,
    "ytick.labelsize": TEXT_PT,
    "legend.fontsize": TEXT_PT,
    "figure.titlesize": TEXT_PT,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
    "mathtext.bf": "Liberation Sans:bold",
    "mathtext.cal": "Liberation Sans:italic",
    "mathtext.sf": "Liberation Sans",
    "mathtext.tt": "Liberation Mono",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "axes.labelpad": 2.0,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.8,
    "ytick.major.size": 2.8,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.major.pad": 2.0,
    "ytick.major.pad": 2.0,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.0,
    "legend.labelspacing": 0.3,
    "hatch.linewidth": 0.4,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "figure.dpi": 150,
    "savefig.dpi": DPI,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "path",
    "svg.hashsalt": "aquacj-4561230",
}


def apply() -> None:
    """Install the figure style; called by figure() and subplots(), safe to call twice."""
    plt.rcParams.update(_RC)
    # A glyph the font lacks must stop the run, not turn into an empty box in a figure.
    warnings.filterwarnings("error", message=r"Glyph \d+ .* missing from")


def figure(width_cm: float, height_cm: float, **kwargs):
    """New figure of the given size in cm with the style applied."""
    apply()
    return plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), **kwargs)


def subplots(nrows: int, ncols: int, width_cm: float, height_cm: float, **kwargs):
    """plt.subplots with the style applied and the size given in cm."""
    apply()
    kwargs.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width_cm / 2.54, height_cm / 2.54), **kwargs)


def save(fig, stem: str, group: str = "main", svg: bool = False) -> list[Path]:
    """Write figures/<group>/<stem>.png (600 dpi) and .pdf, and .svg when `svg`; close `fig`.

    Metadata that changes between runs (creation date, software version) is left out so that
    re-running a script reproduces the files.
    """
    out_dir = FIG_DIR / group
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "png": {"Software": None},
        "pdf": {"Creator": "aquacj-4561230 reporting/figs", "CreationDate": None},
        "svg": {"Date": None},
    }
    paths = []
    for ext in ("png", "pdf", *(("svg",) if svg else ())):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=DPI, metadata=metadata[ext])
        paths.append(path)
    plt.close(fig)
    return paths


def panel_letter(ax, letter: str, *, outside: bool = False, label: str = "") -> None:
    """Bold "(a)" at the top left of `ax`, inside the axes or just above its corner.

    `label` is regular text after the letter, such as "Pond 1".
    """
    text = rf"$\mathbf{{({letter})}}$" + (f" {label}" if label else "")
    if outside:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, 2),
                    textcoords="offset points", ha="left", va="bottom", fontsize=LETTER_PT)
    else:
        ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(3, -2.5),
                    textcoords="offset points", ha="left", va="top", fontsize=LETTER_PT,
                    bbox={"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none"}, zorder=6)


def month_axis(ax, start: str, end: str, every: int = 1, rotate: float = 0.0) -> None:
    """Calendar-date x axis, major ticks every `every` months labelled "Jun 2021", minor monthly."""
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=every))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    if rotate:
        ax.tick_params(axis="x", which="major", labelrotation=rotate)
        for label in ax.get_xticklabels():
            label.set_ha("right")
            label.set_rotation_mode("anchor")


def _day_label(value: float, _pos: int) -> str:
    stamp = mdates.num2date(value)
    return f"{stamp.day} {stamp:%b}"


def day_axis(ax, start: str, end: str, every_days: int = 2, minor_hours: int = 6) -> None:
    """Calendar-date x axis for a span of days: major ticks at midnight labelled "14 Oct"."""
    ax.set_xlim(pd.Timestamp(start), pd.Timestamp(end))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=every_days))
    ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=range(0, 24, minor_hours)))
    ax.xaxis.set_major_formatter(FuncFormatter(_day_label))


def event_marker_kw(kind: str, pond: int, size: float = 4.2) -> dict:
    """Marker keywords of a real weighing, in the colour of its pond.

    "eligible": a T1 event that is scored (filled circle); "not_eligible": a T1 event with too
    little sensor coverage (open circle); "t0": the stocking weighing at t = 0, not a T1 event
    (filled diamond).
    """
    color = POND_COLORS[pond]
    if kind == "eligible":
        return {"marker": "o", "ms": size, "mfc": color, "mec": EDGE, "mew": 0.45, "ls": "none"}
    if kind == "not_eligible":
        return {"marker": "o", "ms": size, "mfc": "white", "mec": color, "mew": 1.1, "ls": "none"}
    if kind == "t0":
        return {"marker": "D", "ms": size - 0.7, "mfc": color, "mec": EDGE, "mew": 0.45, "ls": "none"}
    raise ValueError(f"unknown event kind {kind!r}")


def event_legend_handles() -> list[Line2D]:
    """Neutral legend entries for the three event kinds (colour is the pond, shown by panel)."""
    neutral = {"mfc": "#4D4D4D", "mec": EDGE, "mew": 0.45, "marker": "o", "ms": 4.2, "ls": "none"}
    return [
        Line2D([], [], **neutral, label="Weighing, eligible T1 event"),
        Line2D([], [], marker="o", ms=4.2, mfc="white", mec="#4D4D4D", mew=1.1, ls="none",
               label="Weighing, not eligible"),
        Line2D([], [], marker="D", ms=3.5, mfc="#4D4D4D", mec=EDGE, mew=0.45, ls="none",
               label="Weighing at $t$ = 0 (stocking)"),
    ]
