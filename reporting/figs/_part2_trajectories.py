"""Pond panels for saved T1 event predictions, shared by LOPO and calendar figures."""

from __future__ import annotations

from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator

from reporting.figs import _part2 as p


def draw(scheme: str):
    """Draw event predictions on shared weighing limits, labelling outliers at the edge."""
    models = p.shown("T1")
    predictions = {
        model: p.seed_predictions(p.prediction("T1", scheme, model), "T1", model)
        for model in models
    }
    events = p.csv("data/processed/events.csv", parse_dates=["date"])
    scored = sorted(predictions[models[0]].pond.unique())
    assert scored == [1, 2, 3, 4, 6, 7, 9]
    reference = set(predictions[models[0]].row_id)
    assert all(set(frame.row_id) == reference for frame in predictions.values())
    fig, axes = p.fs.subplots(2, 4, width_cm=p.fs.FULL_CM, height_cm=12.5, sharex=True, sharey=True)
    end = p.pd.Timestamp("2021-10-31")
    start = p.pd.Timestamp("2021-06-01") if scheme == "lopo" else p.fs.CUTOFF_DATE - p.pd.Timedelta(days=16)
    visible = events[events.pond.isin(scored) & events.date.between(start, end)]
    assert visible.weight.gt(0).all()
    limits = (10 ** p.np.floor(p.np.log10(visible.weight.min())),
              10 ** p.np.ceil(p.np.log10(visible.weight.max())))
    for ax, letter, pond in zip(axes.flat, "abcdefgh", scored):
        color = p.fs.POND_COLORS[pond]
        own_events = events[(events.pond == pond) & (events.date >= start) & (events.date <= end)]
        for i, model in enumerate(models):
            frame = predictions[model]
            own = frame[frame.pond == pond].sort_values("date")
            neural = model in p.CFG["models"]["T1"]["deep"] + ["mlp_static"]
            if neural:
                ax.fill_between(own.date, own.low, own.high, color=color, alpha=0.10, linewidth=0)
                ax.vlines(own.date, own.low, own.high, color=color, lw=0.6, alpha=0.35)
            ax.plot(own.date, own.w_pred, color=color, ls=p.STYLES[i], lw=0.75, zorder=3)
            inside = own[own.w_pred.between(*limits)]
            ax.plot(inside.date, inside.w_pred, color=color, ls="none", marker=p.MARKERS[i],
                    ms=3.1, mfc="white", mec=p.fs.EDGE, mew=0.45, lw=0.75, zorder=3)
            outside = own[~own.w_pred.between(*limits)]
            for j, row in enumerate(outside.itertuples()):
                above = row.w_pred > limits[1]
                edge = limits[1] if above else limits[0]
                ax.plot(row.date, edge, marker="^" if above else "v", ls="none",
                        ms=4.5, mfc="white", mec=color, mew=0.9, clip_on=False, zorder=6)
                if row.w_pred >= 10000:
                    exponent = int(p.np.floor(p.np.log10(row.w_pred)))
                    label = rf"${row.w_pred / 10**exponent:.2g}\,\times\,10^{{{exponent}}}$ g"
                else:
                    label = f"{row.w_pred:.2g} g"
                text_position = (0.04, 0.92 - 0.12 * j) if above else (row.date, 0.08 + 0.09 * (j % 3))
                ax.annotate(label, (row.date, edge), xytext=text_position,
                            textcoords="axes fraction" if above else ax.get_xaxis_transform(),
                            ha="left" if above else "center",
                            va="top" if above else "bottom", fontsize=7, color=p.fs.INK,
                            bbox={"fc": "white", "ec": "none", "pad": 0.15},
                            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.4}, zorder=7)
        ax.plot(own_events.date, own_events.weight, marker="o", ls="none", ms=4.6,
                mfc=color, mec=p.fs.EDGE, mew=0.65, zorder=5)
        p.log_weight_axis(ax)
        ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(FixedLocator([10.0 ** exponent for exponent in
                                                range(int(p.np.log10(limits[0])), int(p.np.log10(limits[1])) + 1)]))
        ax.grid(axis="y", color=p.fs.GRID_COLOR, lw=0.4)
        p.fs.month_axis(ax, str(start.date()), str(end.date()), every=2, rotate=45)
        if scheme == "chrono":
            ax.axvline(p.fs.CUTOFF_DATE, **p.fs.CUTOFF_KW)
        p.fs.panel_letter(ax, letter, outside=True, label=f"Pond {pond}")
    axes.flat[-1].set_visible(False)
    handles = [Line2D([], [], color=p.fs.INK, marker="o", ls="none", ms=4.6,
                      mfc=p.fs.INK, label="Real weighing")]
    handles.append(Line2D([], [], color=p.fs.INK, marker="^", ls="none", ms=4.5,
                          mfc="white", label="Prediction outside y-range (value labeled)"))
    for i, model in enumerate(models):
        handles.append(Line2D([], [], color=p.fs.INK, ls=p.STYLES[i], marker=p.MARKERS[i],
                              ms=3.1, mfc="white", label=p.model_name(model)))
    if scheme == "chrono":
        handles.append(Line2D([], [], **{**p.fs.CUTOFF_KW, "zorder": None},
                              label=f"Cut-off, {p.fs.CUTOFF_DATE:%d %b %Y}"))
    fig.legend(handles=handles, loc="outside lower center", ncol=2)
    fig.supylabel("Fish weight (g, log scale)")
    return fig
