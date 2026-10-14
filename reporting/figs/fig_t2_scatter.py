"""T2 target-weight predictions, with pond colours and model-specific marker shapes."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D

from reporting.figs import _part2 as p

CAPTION = """T2 target-weight predictions against observed weight under leave-one-pond-out
validation, at 14 and 28 days ahead. Both axes show grams on a logarithmic scale with equal
limits; the dashed diagonal is equality. Pond colours and model-specific marker shapes
identify the eligible origin-target samples of the fixed shown-model set. Separate legend
blocks show models by marker and ponds by colour. CNN-LSTM and
Static MLP points are arithmetic means of predicted grams across the five seeds; the other
models are deterministic. The selected baseline is Ridge (past weights), whose inputs also
include time. Ridge (past weights + WQ) has no points because its LOPO run stopped under
plan 9.0, and its legend entry records that absence. T2 evaluates transfer of growth over
14 or 28 days between ponds during an overlapping calendar period. The 28-day comparison
is descriptive; neither panel is a test of forecasting into an unseen future."""


def main() -> dict:
    frames = {}
    for model in p.shown("T2"):
        frame = p.prediction("T2", "lopo", model)
        if frame is not None:
            frames[model] = p.seed_predictions(frame, "T2", model)
    reference = set(next(iter(frames.values())).row_id)
    assert all(set(frame.row_id) == reference for frame in frames.values())
    fig, axes = p.fs.subplots(1, 2, width_cm=p.fs.FULL_CM, height_cm=11.8, sharex=True, sharey=True)
    values = [v for frame in frames.values() for v in [*frame.w_true, *frame.w_pred]]
    assert min(values) > 0
    limits = (min(values) / 1.25, max(values) * 1.25)
    for ax, letter, horizon in zip(axes, "ab", (1, 2)):
        ax.plot(limits, limits, color=p.fs.INTERP_COLOR, lw=0.75, ls="--", zorder=1)
        for i, model in enumerate(p.shown("T2")):
            if model not in frames:
                continue
            own = frames[model]
            own = own[own.horizon == horizon]
            for pond, rows in own.groupby("pond"):
                ax.scatter(rows.w_true, rows.w_pred, marker=p.MARKERS[i], s=18,
                           facecolors=p.fs.POND_COLORS[int(pond)], edgecolors=p.fs.EDGE,
                           linewidths=0.35, alpha=0.85, zorder=3)
        p.log_weight_axis(ax, "x")
        p.log_weight_axis(ax, "y")
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed weight (g)")
        p.fs.panel_letter(ax, letter, outside=True, label=p.horizon_name("T2", horizon))
    axes[0].set_ylabel("Predicted weight (g)")
    fig.legend(handles=p.model_handles("T2", stopped=True) + [
        Line2D([], [], color=p.fs.INTERP_COLOR, lw=0.75, ls="--", label="Identity")],
        title="Models (marker)", loc="outside lower left")
    fig.legend(handles=p.pond_handles(), title="Ponds (colour)", loc="outside lower right", ncol=2)
    return p.finish(fig, "fig_t2_scatter", "main", CAPTION)


if __name__ == "__main__":
    main()
