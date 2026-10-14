"""Prespecified permutation distributions read directly from the saved permutation runs."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from reporting.figs import _part2 as p

CAPTION = """Saved permutation distributions of S under leave-one-pond-out validation, with
the observed S as a black vertical line. Columns follow the configured water-quality tabular
models, then the selected primary deep model; rows show T1 and T2 at 14 days ahead.
Each histogram counts the saved permutation runs, using 25 equal-width bins. The p-values
are read from the saved permutation statistics and displayed to three decimals, with the
lower-tail rule of plan 8.2.
Tabular models use 999 permutations and deep models 199. The observed and permuted deep
runs use seed 0, rather than the five-seed mean used in the score figures. Water-quality
values and masks are reassigned jointly between stream units within a date, with model
settings fixed. T2 Ridge (past weights + WQ) has no saved permutation distribution because
its original LOPO run stopped under plan 9.0; panel (d) states that no permutation test was run.
These distributions concern pond-specific sensor information and do not establish a causal
effect of water quality on growth."""


def main() -> dict:
    stats = p.csv("results/stats/permutation.csv")
    fig, axes = p.fs.subplots(2, 3, width_cm=p.fs.FULL_CM, height_cm=14)
    for row_index, task in enumerate(("T1", "T2")):
        models = [*p.CFG["permutation"]["tabular_models"][task], p.selection(task)["primary_deep"]]
        for col, model in enumerate(models):
            ax = axes[row_index, col]
            label = (p.horizon_name(task, 0 if task == "T1" else 1)
                     + "\n" + p.model_name(model))
            relative = f"results/permutation/{task}/{model}.csv"
            if not (p.ROOT / relative).exists():
                assert (task, model) == ("T2", "ridge_pw_wq")
                assert p.prediction(task, "lopo", model) is None
                ax.text(0.5, 0.48, "No permutation test:\nrun stopped under plan 9.0",
                        transform=ax.transAxes, ha="center", va="center", color=p.fs.INTERP_COLOR)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                permuted = p.csv(relative)
                stat = stats[(stats.task == task) & (stats.model == model)].iloc[0]
                assert len(permuted) == stat.R
                ax.hist(permuted.S_perm, bins=25, color=p.fs.OKABE_ITO["sky_blue"],
                        edgecolor="white", linewidth=0.3)
                ax.axvline(stat.S_obs, color="black", lw=1.2)
                label += f"\np = {stat.p:.3f}, R = {int(stat.R)}"
                ax.set_xlim(min(permuted.S_perm.min(), stat.S_obs) * 0.95,
                            max(permuted.S_perm.max(), stat.S_obs) * 1.04)
                ax.set_xlabel("S" if task == "T1" else "S, 14 days ahead")
            if col == 0:
                ax.set_ylabel("Permutation runs")
            p.fs.panel_letter(ax, "abcdef"[3 * row_index + col], outside=True, label=label)
    fig.legend(handles=[Patch(fc=p.fs.OKABE_ITO["sky_blue"], label="Permuted S"),
                        Line2D([], [], color="black", label="Observed S")],
               loc="outside lower center", ncol=2)
    return p.finish(fig, "fig_permutation", "main", CAPTION)


if __name__ == "__main__":
    main()
