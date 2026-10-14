"""Print the saved schemas and plotting ranges without changing any analysis output."""

import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from reporting.figs import _part2 as p


def main() -> None:
    scores = p.csv("results/metrics/scores.csv")
    ponds = p.csv("results/metrics/per_pond.csv")
    for task, horizon in p.STRATA:
        rows = p.part(scores, task, horizon)
        print(task, horizon, rows[~rows.model.isin(p.CFG["models"][task]["sensitivity"])][
            ["model", "S", "n_nonpositive"]].to_string(index=False))
        print("Pond S range", p.part(ponds, task, horizon).S.min(), p.part(ponds, task, horizon).S.max())
    for task in ("T1", "T2"):
        for model in p.CFG["models"][task]["deep"] + ["mlp_static"]:
            curves = p.csv(f"results/tuning/{task}/lopo/{model}/curves.csv.gz")
            chosen = p.csv(f"results/tuning/{task}/lopo/{model}/chosen.csv")
            print(task, model, "stages", curves.stage.value_counts().to_dict(),
                  "final epochs", chosen.epochs.tolist(), "loss range",
                  curves[["train_mse", "val_mse"]].max().to_dict())
    for task in ("T1", "T2"):
        for model in p.shown(task):
            frame = p.prediction(task, "lopo", model)
            if frame is not None:
                print(task, model, "prediction range", frame.w_pred.min(), frame.w_pred.max())


if __name__ == "__main__":
    main()
