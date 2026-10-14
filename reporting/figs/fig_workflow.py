"""fig_workflow: flow diagram of the new design (replaces Figures 2 and 4 of the original paper).

Run from the repo root: `.venv/bin/python -I reporting/figs/fig_workflow.py`

Every number in the diagram is read from a file, so the figure follows the analysis:
data/raw (file count), results/audit/summary.csv (logging interval), data/processed/events.csv
(weighings), results/qc/eligibility.csv (scored samples), configs/phase4.yaml and
configs/phase4_data.yaml (windows, gap rule, tuning budget, seeds, bootstrap, cut-off date).
The wording follows ANALYSIS_PLAN.md sections 2 to 9. Output: figures/main/fig_workflow.{png,pdf,svg}
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd  # noqa: E402
import yaml  # noqa: E402

from reporting import figstyle as fs  # noqa: E402
from reporting.figs import _events  # noqa: E402
from reporting.figs._diagram import Canvas, tint  # noqa: E402

WIDTH = 16.0
MARGIN = 0.12
LABEL_W = 2.2
CONTENT_X = MARGIN + LABEL_W + 0.22
CONTENT_W = WIDTH - MARGIN - CONTENT_X
ROW_GAP = 0.66

# One colour per stage family; content boxes use a light tint, the stage tab a stronger one.
FAMILY = {
    "data": fs.OKABE_ITO["blue"],
    "tasks": fs.OKABE_ITO["orange"],
    "models": fs.OKABE_ITO["bluish_green"],
    "validation": fs.OKABE_ITO["reddish_purple"],
    "statistics": fs.OKABE_ITO["vermillion"],
}


def read_facts() -> dict:
    """Numbers shown in the diagram, from the files named in the module docstring."""
    events = _events.load_events()
    eligibility = pd.read_csv(fs.REPO / "results/qc/eligibility.csv")
    summary = pd.read_csv(fs.REPO / "results/audit/summary.csv")
    config = yaml.safe_load((fs.REPO / "configs/phase4.yaml").read_text())
    data_config = yaml.safe_load((fs.REPO / "configs/phase4_data.yaml").read_text())
    if not summary["interval_median_s"].eq(20).all():
        raise ValueError("the median logging interval is no longer 20 s in every pond")
    return {
        "n_files": len(list((fs.REPO / "data/raw").glob("pond*.csv"))),
        "interval_s": int(summary["interval_median_s"].iloc[0]),
        "n_events": len(events),
        "n_ponds": int(events["pond"].nunique()),
        "t1_samples": int(eligibility["t1_eligible"].sum()),
        "t1_ponds": int((eligibility["t1_eligible"] > 0).sum()),
        "t2_h1": int(eligibility["t2_h1_eligible"].sum()),
        "t2_h2": int(eligibility["t2_h2_eligible"].sum()),
        "t1_days": data_config["windows"]["t1_days"],
        "t2_days": data_config["windows"]["t2_days"],
        "min_hours": data_config["daily"]["min_hours"],
        "gap_days": data_config["daily"]["max_gap_days"],
        "grid_days": data_config["cohort"]["grid"]["period_days"],
        "day0": data_config["cohort"]["day0"],
        "trials": config["budget"]["n_trials"],
        "seeds": len(config["seeds"]["final"]),
        "inner_folds": config["design"]["n_inner_folds"],
        "boot": config["evaluate"]["bootstrap"]["B"],
        "cutoff": config["chrono"]["cutoff"],
    }


def draw(canvas: Canvas, top: float, f: dict) -> float:
    """Draw the diagram from y = `top` downwards; return the y of the bottom edge of the last row."""
    streams = "Ponds 1 and 2 log the same readings for temperature, turbidity, DO, pH and nitrate."
    y = top

    def row_height(boxes: list[tuple[str, str, float]]) -> float:
        return max(canvas.measure(title, body, w) for title, body, w in boxes)

    def tab(number: int, name: str, family: str, y_top: float, height: float) -> None:
        canvas.box(MARGIN, y_top - height, LABEL_W, height, f"{number}  {name}", fc=tint(FAMILY[family], 0.42),
                   ec=tint(FAMILY[family], 0.9), lw=0.5)

    def fill(family: str) -> str:
        return tint(FAMILY[family], 0.12)

    # 1 Data
    data_boxes = [(f"{f['n_files']} raw pond files",
                   f"Udanor et al. (2022), ponds 1 to 4 and 6 to 12: six water-quality sensors logged about every "
                   f"{f['interval_s']} s; fish weight and length written once per day.", CONTENT_W)]
    h1 = row_height(data_boxes)
    b_data = canvas.box(CONTENT_X, y - h1, CONTENT_W, h1, *data_boxes[0][:2], fc=fill("data"), valign="top")
    tab(1, "Data", "data", y, h1)
    y -= h1 + ROW_GAP

    # 2 Audit: one frame with three findings
    third = (CONTENT_W - 2 * 0.1) / 3
    audit = [
        (f"Events on the {f['grid_days']}-day grid",
         f"Real weighings every {f['grid_days']} days from {f['day0']}; the daily values between them are "
         f"interpolated.", third),
        ("QC rules", "Sensor ranges and a flat-line test, fixed before any model of the new design was run.", third),
        ("Shared sensor stream", streams, third),
    ]
    h2 = row_height(audit) + 0.12
    frame = canvas.box(CONTENT_X, y - h2, CONTENT_W, h2, fc=fill("data"), valign="top")
    for i, (title, body, w) in enumerate(audit):
        canvas.box(CONTENT_X + i * (third + 0.1), y - h2, w, h2, title, body, fc="none", ec="none", valign="top")
    for i in (1, 2):
        x_sep = CONTENT_X + i * (third + 0.1) - 0.05
        canvas.arrow([(x_sep, y - 0.12), (x_sep, y - h2 + 0.12)], head=False, lw=0.4, color=tint(FAMILY["data"], 0.6))
    tab(2, "Audit", "data", y, h2)
    canvas.arrow([(b_data.cx, b_data.bottom), (b_data.cx, y)])
    y -= h2 + ROW_GAP

    # 3 Cohort and sensors
    gap_ab, gap_bc = 0.34, 0.5
    w_a, w_b = 4.3, 3.1
    w_c = CONTENT_W - w_a - w_b - gap_ab - gap_bc
    cohort = [
        (f"{f['n_events']} real weighings in {f['n_ponds']} cohort ponds",
         f"Ponds 1 to 4 and 6 to 9; day 0 is {f['day0']}.", w_a),
        ("Sensor QC", "A reading that fails a rule becomes missing, never clipped.", w_b),
        ("Daily aggregates",
         f"Mean, SD, minimum and maximum per sensor and day from valid readings in at least {f['min_hours']} clock hours; gaps of up "
         f"to {f['gap_days']} days interpolated and flagged (the {f['gap_days']}-day gap rule).", w_c),
    ]
    h3 = row_height(cohort)
    b_events = canvas.box(CONTENT_X, y - h3, w_a, h3, *cohort[0][:2], fc=fill("data"), valign="top")
    b_qc = canvas.box(CONTENT_X + w_a + gap_ab, y - h3, w_b, h3, *cohort[1][:2], fc=fill("data"), valign="top")
    b_daily = canvas.box(b_qc.right + gap_bc, y - h3, w_c, h3, *cohort[2][:2], fc=fill("data"), valign="top")
    tab(3, "Cohort and sensors", "data", y, h3)
    canvas.arrow([(b_events.cx, frame.bottom), (b_events.cx, b_events.top)])
    canvas.arrow([(b_qc.cx, frame.bottom), (b_qc.cx, b_qc.top)])
    canvas.arrow([(b_qc.right, b_qc.cy), (b_daily.left, b_daily.cy)])
    y -= h3 + ROW_GAP

    # 4 Tasks
    gap = 0.2
    w_t = (CONTENT_W - gap) / 2
    tasks = [
        ("T1, estimation",
         f"Weight at a weighing from time since stocking and the water quality of the {f['t1_days']} days before "
         f"it. {f['t1_samples']} scored weighings in {f['t1_ponds']} ponds.", w_t),
        ("T2, next weights",
         f"Weight {f['grid_days']} and {2 * f['grid_days']} days after a weighing from past weights, time and the "
         f"water quality of the {f['t2_days']} days before it. {f['t2_h1']} (14 days) and {f['t2_h2']} (28 days) scored samples.", w_t),
    ]
    h4 = row_height(tasks)
    b_t1 = canvas.box(CONTENT_X, y - h4, w_t, h4, *tasks[0][:2], fc=fill("tasks"), valign="top")
    b_t2 = canvas.box(CONTENT_X + w_t + gap, y - h4, w_t, h4, *tasks[1][:2], fc=fill("tasks"), valign="top")
    tab(4, "Tasks", "tasks", y, h4)
    canvas.bus([(b_events.cx, b_events.bottom), (b_daily.cx, b_daily.bottom)],
               [(b_t1.cx, b_t1.top), (b_t2.cx, b_t2.top)], y + ROW_GAP / 2)
    y -= h4 + ROW_GAP

    # 5 Models
    gap = 0.16
    w_m = (CONTENT_W - 3 * gap) / 4
    models = [
        ("Time-only baselines",
         "Growth curves, cohort mean, persistence, linear and growth-rate extrapolation, cohort growth.", w_m),
        ("Ridge and gradient boosting", "On time or past weights, with and without water quality.", w_m),
        ("Deep models", "LSTM, GRU, CNN-LSTM and TFT, plus a twin MLP that sees the static inputs only.", w_m),
        ("Allometric reference", "Weight from length (T1 only).", w_m),
    ]
    h5 = row_height(models)
    b_models = [canvas.box(CONTENT_X + i * (w_m + gap), y - h5, w_m, h5, title, body, fc=fill("models"), valign="top")
                for i, (title, body, _) in enumerate(models)]
    tab(5, "Models", "models", y, h5)
    canvas.bus([(b_t1.cx, b_t1.bottom), (b_t2.cx, b_t2.bottom)], [(b.cx, b.top) for b in b_models], y + ROW_GAP / 2)
    y -= h5 + ROW_GAP

    # 6 Validation
    w_v = (CONTENT_W - 0.2) / 2
    validation = [
        ("Leave-one-pond-out",
         f"Nested tuning: {f['inner_folds']} inner folds, {f['trials']} Optuna trials per model and fold, "
         f"{f['seeds']} seeds for the deep models. If pond 1 or 2 is held out, the other leaves the training set.",
         w_v),
        ("Calendar hold-out", f"Descriptive: train on weighings before {f['cutoff']}, test on those on or after it.",
         w_v),
    ]
    h6 = row_height(validation)
    b_v1 = canvas.box(CONTENT_X, y - h6, w_v, h6, *validation[0][:2], fc=fill("validation"), valign="top")
    b_v2 = canvas.box(CONTENT_X + w_v + 0.2, y - h6, w_v, h6, *validation[1][:2], fc=fill("validation"), valign="top")
    tab(6, "Validation", "validation", y, h6)
    canvas.bus([(b.cx, b.bottom) for b in b_models], [(b_v1.cx, b_v1.top), (b_v2.cx, b_v2.top)], y + ROW_GAP / 2)
    y -= h6 + ROW_GAP

    # 7 Statistics
    stats = [
        ("Primary statistic S", "Mean over ponds of the mean absolute error of log weight.", w_m),
        ("Wilcoxon test", "Exact one-sided signed-rank test in a fixed sequence: T1, then T2 at 14 days.", w_m),
        ("Permutation test", "Water-quality inputs shuffled among sensor streams within a sampling date.", w_m),
        ("Bootstrap", f"Pond-level resampling, B\u00a0=\u00a0{f['boot']}, for 95% intervals.", w_m),
    ]
    h7 = row_height(stats)
    b_stats = [canvas.box(CONTENT_X + i * (w_m + gap), y - h7, w_m, h7, title, body, fc=fill("statistics"), valign="top")
               for i, (title, body, _) in enumerate(stats)]
    tab(7, "Statistics", "statistics", y, h7)
    canvas.bus([(b_v1.cx, b_v1.bottom), (b_v2.cx, b_v2.bottom)], [(b.cx, b.top) for b in b_stats], y + ROW_GAP / 2)
    return y - h7


def main() -> None:
    facts = read_facts()
    probe = Canvas(WIDTH, 40.0)
    used = 40.0 - draw(probe, 40.0 - MARGIN, facts)
    fs.plt.close(probe.fig)
    height = used + MARGIN
    canvas = Canvas(WIDTH, height)
    draw(canvas, height - MARGIN, facts)
    print(f"canvas {WIDTH} x {height:.2f} cm")
    for path in canvas.save("fig_workflow", "main", svg=True):
        print(path)


if __name__ == "__main__":
    main()
