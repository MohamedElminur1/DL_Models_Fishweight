"""All prespecified LOPO model scores, including the stopped T2 ridge row.

Run from the repository root with nice -n 19 .venv/bin/python -I reporting/figs/fig_pond_scores.py.
"""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from matplotlib.lines import Line2D

from reporting.figs import _part2 as p

CAPTION = """Per-pond mean absolute log error S for every prespecified T1 model on the eligible
leave-one-pond-out samples. Models follow the configured order within the labeled groups;
sensitivity variants and post hoc results are excluded. Colored circles are ponds;
black diamonds are the saved model S, the mean over scored ponds. Neural S values are the
saved means over five seeds. The primary pair
is labeled from inner-validation selection. The symmetric-logarithmic horizontal
axis is linear from 0 to 0.5 and logarithmic above to retain the large ridge errors. A light
vertical line at 0.5 and a note below the axis mark this change. An infinite S would be
shown by an arrow at the right edge, annotated with the saved count of predictions at or below 0 g; no such
prediction occurs among the models plotted here. Lower S means smaller error."""

CAPTION_T2 = """Per-pond mean absolute log error S for every prespecified T2 model on the eligible
leave-one-pond-out samples at 14 days ahead (a) and 28 days ahead (b). Both panels use the
same model rows in configured order, with model labels and group headings shown only in
panel (a). Sensitivity variants and post hoc results are excluded. Colored circles are
ponds; black diamonds are the saved model S, the mean over scored ponds. Neural S values
are the saved means over five seeds. The primary pair is labeled from inner-validation
selection. Each panel has its own linear horizontal axis and range. Ridge
(past weights + WQ) stopped because a learned-model run produced a non-finite prediction under plan 9.0;
its row is labeled "stopped (plan 9.0)" in both panels, with no score or pond dots. An
infinite S would be shown by an arrow at the right edge, annotated with the saved count of
predictions at or below 0 g; no such prediction occurs among the models plotted here.
Lower S means smaller error."""


def draw_rows(ax, task: str, horizon: int, groups: list[tuple[str | None, list[str]]],
              ponds, scores, primary: bool = False, show_labels: bool = True) -> None:
    """Draw saved scores in configured order; optionally hide row labels and group headings."""
    own = p.part(ponds, task, horizon)
    totals = p.part(scores, task, horizon)
    selected = p.selection(task)
    ticks, labels, headings = [], [], []
    y = 0
    values = []
    for group, models in groups:
        if not models:
            continue
        if group is not None:
            headings.append((y, group))
            y += 1.15
        for model in models:
            ticks.append(y)
            label = p.model_name(model)
            if primary and model in (selected["primary_baseline"], selected["primary_deep"]):
                label += " [primary]"
            labels.append(label)
            subset = own[own.model == model]
            total = totals[totals.model == model]
            if subset.empty:
                assert (task, model) == ("T2", "ridge_pw_wq")
                assert p.prediction(task, "lopo", model) is None
                ax.text(0.03, y, "stopped (plan 9.0)", transform=ax.get_yaxis_transform(),
                        va="center", color=p.fs.INTERP_COLOR)
            else:
                assert len(total) == 1 and len(subset) == 7
                for row in subset.itertuples():
                    offset = (list(p.fs.POND_COLORS).index(int(row.pond)) - 3) * 0.035
                    if np.isfinite(row.S):
                        ax.plot(row.S, y + offset, marker="o", ms=3.9, ls="none",
                                mfc=p.fs.POND_COLORS[int(row.pond)], mec=p.fs.EDGE,
                                mew=0.4, zorder=3)
                        values.append(row.S)
                    else:
                        ax.annotate(f"{row.n_nonpositive:g}", xy=(0.99, y + offset),
                                    xytext=(0.86, y + offset), xycoords=ax.get_yaxis_transform(),
                                    textcoords=ax.get_yaxis_transform(), va="center",
                                    color=p.fs.POND_COLORS[int(row.pond)],
                                    arrowprops={"arrowstyle": "->", "color": p.fs.POND_COLORS[int(row.pond)]})
                aggregate = total.iloc[0]
                if np.isfinite(aggregate.S):
                    ax.plot(aggregate.S, y, "D", ms=4.2, color="black", zorder=4)
                    values.append(aggregate.S)
                else:
                    ax.annotate(f"S = infinity; n = {aggregate.n_nonpositive:g}",
                                xy=(0.99, y), xytext=(0.65, y),
                                xycoords=ax.get_yaxis_transform(), textcoords=ax.get_yaxis_transform(),
                                va="center", arrowprops={"arrowstyle": "->", "color": "black"})
            y += 1
        y += 0.25
    ax.set_yticks(ticks, labels)
    ax.tick_params(axis="y", length=0, labelleft=show_labels)
    ax.spines.left.set_visible(False)
    ax.set_ylim(y - 0.25, -0.6)
    for position, group in headings:
        if show_labels:
            ax.text(0, position, p.group_name(group), transform=ax.get_yaxis_transform(),
                    ha="left", va="center", weight="bold", fontsize=8)
        ax.axhline(position + 0.55, color=p.fs.GRID_COLOR, lw=0.4)
    p.score_axis(ax, values)


def main() -> list[dict]:
    p.SOURCES.clear()
    ponds = p.csv("results/metrics/per_pond.csv")
    scores = p.csv("results/metrics/scores.csv")
    p.csv("results/metrics/completeness.csv")
    metric_sources = p.SOURCES.copy()
    handles = [*p.pond_handles(), Line2D([], [], marker="D", color="black",
                                       ls="none", ms=4, label="Model S")]
    fig, ax = p.fs.subplots(1, 1, width_cm=p.fs.FULL_CM, height_cm=13)
    groups = [(group, p.CFG["models"]["T1"][group]) for group in p.GROUPS]
    draw_rows(ax, "T1", 0, groups, ponds, scores, primary=True)
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    records = [p.finish(fig, "fig_pond_scores", "main", CAPTION)]

    p.SOURCES.clear()
    p.SOURCES.update(metric_sources)
    fig, axes = p.fs.subplots(1, 2, width_cm=p.fs.FULL_CM, height_cm=11, sharey=True)
    groups = [(group, p.CFG["models"]["T2"][group]) for group in p.GROUPS]
    for ax, letter, horizon in zip(axes, "ab", (1, 2)):
        draw_rows(ax, "T2", horizon, groups, ponds, scores, primary=True,
                  show_labels=horizon == 1)
        p.fs.panel_letter(ax, letter, outside=True, label=p.horizon_name("T2", horizon))
    fig.legend(handles=handles, loc="outside lower center", ncol=4)
    records.append(p.finish(fig, "fig_pond_scores_t2", "main", CAPTION_T2))
    return records


if __name__ == "__main__":
    main()
