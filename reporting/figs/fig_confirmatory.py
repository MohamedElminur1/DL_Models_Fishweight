"""Primary-pair pond differences and saved intervals, with the fixed-sequence test status."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D

from reporting.figs import _part2 as p

CAPTION = """Differences in S between the primary deep model and primary baseline,
with descriptive pond-bootstrap 95% percentile intervals. Panels show T1, LSTM minus Cohort
mean; T2 at 14 days ahead, CNN-LSTM minus Ridge (past weights); and T2 at 28 days ahead,
the same pair. Black diamonds and horizontal bars are the saved mean difference and interval
from pairwise.csv. Each coloured dot is a pond's saved seed-averaged S for the primary deep
model minus its saved S for the primary baseline, read from per_pond.csv for LOPO, eligible
samples and the panel's task-horizon. Small vertical offsets separate the seven pond dots;
pond colours are fixed across figures. Negative values favour the deep model. One-sided exact Wilcoxon p-values
come from confirmatory.csv for T1 and T2 at 14 days, and pairwise.csv for the descriptive
28-day comparison. The T2 14-day step was not tested in the fixed sequence because T1 did
not reject; its saved nominal p-value is displayed with that status. P-values are displayed
to three decimals. The intervals
are descriptive and are not the fixed-sequence rejection rule."""


def pond_differences(ponds: p.pd.DataFrame, task: str, horizon: int) -> p.pd.Series:
    """Subtract the primary pair's saved per-pond S, already averaged over neural seeds."""
    selected = p.selection(task)
    own = p.part(ponds, task, horizon)
    deep = own[own.model == selected["primary_deep"]].set_index("pond")
    baseline = own[own.model == selected["primary_baseline"]].set_index("pond")
    assert deep.index.is_unique and baseline.index.is_unique
    assert set(deep.index) == set(baseline.index) == {1, 2, 3, 4, 6, 7, 9}
    assert deep.n_seeds.eq(len(p.CFG["seeds"]["final"])).all()
    return (deep.S - baseline.S).sort_index()


def main() -> dict:
    pairs = p.csv("results/stats/pairwise.csv")
    tests = p.csv("results/stats/confirmatory.csv")
    ponds = p.csv("results/metrics/per_pond.csv")
    fig, axes = p.fs.subplots(3, 1, width_cm=p.fs.FULL_CM, height_cm=12, sharex=True)
    all_differences = []
    for ax, letter, (task, horizon) in zip(axes, "abc", p.STRATA):
        selected = p.selection(task)
        subset = pairs[(pairs.task == task) & (pairs.horizon.astype(str) == str(horizon))
                       & (pairs.model == selected["primary_deep"])
                       & (pairs.comparator == selected["primary_baseline"])]
        assert len(subset) == 1
        row = subset.iloc[0]
        differences = pond_differences(ponds, task, horizon)
        all_differences.extend(differences)
        for i, (pond, difference) in enumerate(differences.items()):
            ax.plot(difference, (i - 3) * 0.045, marker="o", ms=4, ls="none",
                    mfc=p.fs.POND_COLORS[int(pond)], mec=p.fs.EDGE, mew=0.4, zorder=3)
        ax.axvline(0, color=p.fs.INTERP_COLOR, lw=0.7, ls="--")
        ax.errorbar(row["diff"], 0, xerr=[[row["diff"] - row.diff_ci_low],
                                       [row.diff_ci_high - row["diff"]]],
                    color="black", marker="D", ms=4.5, lw=1.1, capsize=3, zorder=4)
        if task == "T2" and horizon == 2:
            pvalue = row.p_one_sided_lower
            status = "Descriptive"
        else:
            test = tests[(tests.task == task) & (tests.horizon.astype(str) == str(horizon))].iloc[0]
            pvalue = test.p_one_sided_lower
            status = "T1: not rejected" if task == "T1" else "T2 step not tested: T1 did not reject"
        ax.text(0.02, 0.95, f"One-sided p = {pvalue:.3f}\n{status}",
                transform=ax.transAxes, va="top")
        ax.text(0.98, 0.95, f"Mean = {row['diff']:.3f}\n95% interval [{row.diff_ci_low:.3f}, {row.diff_ci_high:.3f}]",
                transform=ax.transAxes, ha="right", va="top")
        label = (f"{p.horizon_name(task, horizon)}: {p.model_name(row.model)} minus "
                 f"{p.model_name(row.comparator)}")
        p.fs.panel_letter(ax, letter, outside=True, label=label)
        ax.set_yticks([])
        ax.spines.left.set_visible(False)
        ax.set_ylim(-0.3, 0.65)
        ax.grid(axis="x", color=p.fs.GRID_COLOR, lw=0.4)
    axes[-1].set_xlim(min(all_differences) - 0.03, max(all_differences) + 0.03)
    axes[-1].set_xlabel("Difference in S (deep minus baseline; negative favours deep)")
    fig.legend(handles=[*p.pond_handles(), Line2D([], [], color="black", marker="D", ms=4,
                               label="Mean difference over ponds, pond-bootstrap 95 % interval")],
               loc="outside lower center", ncol=3)
    return p.finish(fig, "fig_confirmatory", "main", CAPTION)


if __name__ == "__main__":
    main()
