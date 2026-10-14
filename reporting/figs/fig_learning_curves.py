"""Chosen-trial validation curves and final training curves for all neural models."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from matplotlib.lines import Line2D

from reporting.figs import _part2 as p

CAPTION = """Saved learning curves for all four deep models and Static MLP under
leave-one-pond-out validation. Columns show T1 and T2; rows follow the configured neural
model order. Dashed lines are validation MSE from the chosen trial only, one line for each
outer-fold and inner-fold combination, using tuning seed 0. Thin solid lines are training
MSE for each final outer-fold and seed fit, using seeds 0 to 4. Colour identifies the pond
held out in the outer fold. Open squares mark the final training epoch for every final-fit
seed; short marks along the top show the configured final epoch count read from chosen.csv
for each outer fold. MSE is on the fold-standardized target, log weight for T1 and log target
to origin weight ratio for T2. No final-fit validation or outer-test loss is drawn. Axes
retain every saved epoch of these selected curves; loss and epoch limits vary by panel.
These are training diagnostics, not an additional model comparison or hypothesis test."""


def selected_curves(task: str, model: str):
    """Join chosen trials by outer fold; reject incomplete final fits or missing inner curves."""
    prefix = f"results/tuning/{task}/lopo/{model}"
    curves = p.csv(prefix + "/curves.csv.gz")
    chosen = p.csv(prefix + "/chosen.csv")
    assert not chosen.outer_fold.duplicated().any()
    selected = curves.merge(chosen[["outer_fold", "best_trial", "epochs", "test_ponds"]],
                            on="outer_fold", validate="many_to_one")
    inner = selected[(selected.stage == "inner") & (selected.trial == selected.best_trial)]
    final = selected[selected.stage == "final"]
    for row in chosen.itertuples():
        validation = inner[inner.outer_fold == row.outer_fold]
        assert sorted(validation.inner_fold.unique()) == [0, 1, 2]
        assert set(validation.seed) == {p.CFG["seeds"]["tuning"]}
        training = final[final.outer_fold == row.outer_fold]
        assert sorted(training.seed.unique()) == p.CFG["seeds"]["final"]
        assert (training.groupby("seed").epoch.max() == row.epochs).all()
        assert (training.trial == row.best_trial).all()
    assert inner.val_mse.notna().all() and final.train_mse.notna().all()
    return inner, final, chosen


def main() -> dict:
    models = p.CFG["models"]["T1"]["deep"] + ["mlp_static"]
    fig, axes = p.fs.subplots(5, 2, width_cm=p.fs.FULL_CM, height_cm=25)
    for col, task in enumerate(("T1", "T2")):
        for row, model in enumerate(models):
            ax = axes[row, col]
            inner, final, chosen = selected_curves(task, model)
            for _, curve in final.groupby(["outer_fold", "seed"]):
                curve = curve.sort_values("epoch")
                color = p.fs.POND_COLORS[int(curve.test_ponds.iloc[0])]
                ax.plot(curve.epoch, curve.train_mse, color=color, lw=0.4, alpha=0.35)
                endpoint = curve.iloc[-1]
                ax.plot(endpoint.epoch, endpoint.train_mse, marker="s", ms=2.1,
                        mfc="white", mec=color, mew=0.45, zorder=4)
            for _, curve in inner.groupby(["outer_fold", "inner_fold", "seed"]):
                curve = curve.sort_values("epoch")
                color = p.fs.POND_COLORS[int(curve.test_ponds.iloc[0])]
                ax.plot(curve.epoch, curve.val_mse, color=color, lw=0.6, ls="--", alpha=0.8)
            for fit in chosen.itertuples():
                ax.plot(fit.epochs, 0.99, marker="|", color=p.fs.POND_COLORS[int(fit.test_ponds)],
                        ms=4, transform=ax.get_xaxis_transform(), clip_on=False)
            ax.set_ylim(bottom=0)
            ax.set_xlim(left=1)
            ax.grid(axis="y", color=p.fs.GRID_COLOR, lw=0.35)
            p.fs.panel_letter(ax, "abcdefghij"[row * 2 + col], outside=True,
                              label=f"{p.horizon_name(task, 0)}, {p.model_name(model)}" if task == "T1"
                              else f"T2, {p.model_name(model)}")
            if col == 0:
                ax.set_ylabel("MSE")
            if row == len(models) - 1:
                ax.set_xlabel("Epoch")
    handles = [Line2D([], [], color=p.fs.INK, lw=0.6, ls="--", label="Chosen-trial validation"),
               Line2D([], [], color=p.fs.INK, lw=0.5, label="Final training"),
               Line2D([], [], color=p.fs.INK, marker="s", ms=3, mfc="white", ls="none",
                      label="Final epoch"), *p.pond_handles()]
    fig.legend(handles=handles, loc="outside lower center", ncol=3)
    return p.finish(fig, "fig_learning_curves", "supp", CAPTION)


if __name__ == "__main__":
    main()
