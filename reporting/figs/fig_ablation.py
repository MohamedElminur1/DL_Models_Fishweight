"""Prespecified ridge and gradient-boosting ablations, without sensitivity variants."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D

from reporting.figs import _part2 as p
from reporting.figs.fig_pond_scores import draw_rows

CAPTION = """Descriptive ablation scores on the eligible leave-one-pond-out samples, shown
for T1 and T2 at 14 and 28 days ahead. T1 includes time, water quality, time plus water
quality, and time plus water quality plus length, for ridge and gradient boosting. T2
includes past weights plus time, with and without water quality. Names follow the shared
model labels; T2's past-weight models also include time. Coloured dots are the saved
per-pond S and black diamonds the saved mean S over ponds. Model order follows the feature
sets of plan 8.1 within each estimator. T1's symmetric-logarithmic axis is linear from 0 to
0.5 and logarithmic above, with the change marked by a light vertical line at 0.5 and a note
below the axis; T2 axes are linear. The T2 ridge model with water quality stopped under plan 9.0 and
has no dots. These comparisons are descriptive; length is available only in the T1 ablation
and reference analyses, not as an input to the primary T1 models."""


def main() -> dict:
    ponds = p.csv("results/metrics/per_pond.csv")
    scores = p.csv("results/metrics/scores.csv")
    fig, axes = p.fs.subplots(3, 1, width_cm=p.fs.FULL_CM, height_cm=20,
                             gridspec_kw={"height_ratios": [9, 5, 5]})
    for ax, letter, (task, horizon) in zip(axes, "abc", p.STRATA):
        if task == "T1":
            models = [f"{estimator}_{features}" for estimator in ("ridge", "gbm")
                      for features in ("time", "wq", "time_wq", "time_wq_len")]
        else:
            models = [f"{estimator}_{features}" for estimator in ("ridge", "gbm")
                      for features in ("pw", "pw_wq")]
        draw_rows(ax, task, horizon, [(None, models)], ponds, scores)
        p.fs.panel_letter(ax, letter, outside=True, label=p.horizon_name(task, horizon))
    fig.legend(handles=[*p.pond_handles(), Line2D([], [], marker="D", color="black", ls="none",
                                               ms=4, label="Model S")],
               loc="outside lower center", ncol=4)
    return p.finish(fig, "fig_ablation", "supp", CAPTION)


if __name__ == "__main__":
    main()
