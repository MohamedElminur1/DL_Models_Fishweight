"""Post-hoc descriptive view of the Phase 2 random-split (S2) predictions.

Usage (from the repo root): .venv/bin/python -I src/phase2_posthoc.py

Added by the lead after the Phase 2 grid had finished, so it is labelled post hoc. It runs no
model. It reads the saved S2 prediction files and scores them only on the targets that fall
in the original notebook's test period (rows from T = int(0.8 N) of the dropna() frame on,
weights 305.8 to 394.66 g), so that the random-split error can be compared with the error the
manuscript reported for that period. Writes results/phase2/posthoc_s2_final_period.csv
(per file) and results/phase2/posthoc_s2_final_period_summary.csv (mean and SD over seeds).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PRED_DIR = ROOT / "results" / "phase2" / "predictions"
OUT_DIR = ROOT / "results" / "phase2"


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    err = y_pred - y_true
    return {
        "n": int(len(err)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mape": float(100 * np.mean(np.abs(err) / np.abs(y_true))),
    }


def main() -> None:
    checks = json.loads((OUT_DIR / "split_checks.json").read_text())
    t_start = int(checks["T"])
    rows = []
    for f in sorted(PRED_DIR.glob("S2_random__*.csv.gz")):
        split, input_set, model, seed = f.name.removesuffix(".csv.gz").split("__")
        d = pd.read_csv(f)
        sub = d[d["target_row"] >= t_start]
        if sub.empty:
            raise ValueError(f"{f.name}: no target in the original test period")
        rows.append({"file": f.name, "split": split, "input_set": input_set, "model": model,
                     "seed": seed.removeprefix("s"), "period_start_row": t_start,
                     "n_test_total": int(len(d)),
                     **scores(sub["y_true"].to_numpy(float), sub["y_pred"].to_numpy(float))})
    per_file = pd.DataFrame(rows)
    per_file.to_csv(OUT_DIR / "posthoc_s2_final_period.csv", index=False)
    summary = (per_file.groupby(["input_set", "model"])
               .agg(n_seeds=("seed", "size"), n_min=("n", "min"), n_max=("n", "max"),
                    mae_mean=("mae", "mean"), mae_sd=("mae", "std"),
                    rmse_mean=("rmse", "mean"), rmse_sd=("rmse", "std"),
                    mape_mean=("mape", "mean"), mape_sd=("mape", "std"))
               .reset_index())
    summary.to_csv(OUT_DIR / "posthoc_s2_final_period_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
