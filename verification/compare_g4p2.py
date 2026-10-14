"""Gate 4 part 2: the blind Codex rebuild of the pond 2 fold against the saved deep-model predictions.

Run from the repository root: .venv/bin/python -I verification/compare_g4p2.py
Writes verification/codex_g4p2/COMPARISON.md. The rebuild (verification/codex_g4p2/g4p2.py) was
written from SPEC.md alone; its log shows it read only SPEC.md and data/processed.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
KEY = ["seed", "pond", "date", "origin_date", "horizon"]
codex = pd.read_csv(ROOT / "verification/codex_g4p2/codex_g4p2_predictions.csv", dtype={"origin_date": str})
per_pond = pd.read_csv(ROOT / "results/metrics/per_pond.csv", dtype={"horizon": str})

rows, lines = [], []
for task, model in [("T1", "lstm"), ("T2", "cnn_lstm")]:
    saved = pd.read_csv(ROOT / f"results/predictions/{task}/lopo/{model}.csv", dtype={"origin_date": str})
    saved = saved[(saved.sample_set == "eligible") & (saved.pond == 2)]
    rebuilt = codex[(codex.task == task) & (codex.model == model)]
    frames = []
    for frame in (saved, rebuilt):
        frame = frame.assign(origin_date=frame.origin_date.fillna(""), horizon=frame.horizon.astype(str))
        frames.append(frame)
    both = frames[0].merge(frames[1], on=KEY, how="outer", suffixes=("_saved", "_codex"), indicator=True)
    unmatched = int((both._merge != "both").sum())
    both = both[both._merge == "both"]
    d_log = (both.y_pred_log_codex - both.y_pred_log_saved).abs()
    d_w = (both.w_pred_codex - both.w_pred_saved).abs()
    s_codex = both.assign(e=(both.y_pred_log_codex - both.y_true_log).abs()).groupby("seed").e.mean().mean()
    reference = per_pond[(per_pond.task == task) & (per_pond.scheme == "lopo") & (per_pond.sample_set == "eligible")
                         & (per_pond.model == model) & (per_pond.pond == 2)]
    s_saved = float(reference[reference.horizon == ("0" if task == "T1" else "all")].S.iloc[0])
    rows.append((task, model, sorted(saved.fold.unique()), len(saved), len(rebuilt), unmatched,
                 d_log.max(), d_w.max(), s_codex, s_saved, abs(s_codex - s_saved)))

out = pd.DataFrame(rows, columns=["task", "model", "saved_fold", "rows_saved", "rows_codex", "unmatched",
                                  "max_abs_d_log_pred", "max_abs_d_w_pred_g", "S_pond2_codex",
                                  "S_pond2_saved", "S_diff"])
lines = ["# Gate 4 part 2: blind rebuild of the pond 2 fold against the saved predictions", "",
         f"Every saved prediction is matched ({int(out.unmatched.sum())} unmatched); largest |difference| "
         f"{out.max_abs_d_log_pred.max():.2e} in log weight and {out.max_abs_d_w_pred_g.max():.2e} g; "
         f"largest pond 2 S difference {out.S_diff.max():.2e}.", "",
         "S_pond2 is the seed-averaged S of pond 2 (T2: both horizons, the `all` stratum of per_pond.csv).", "",
         "```", out.to_string(index=False, float_format=lambda v: f"{v:.6g}"), "```", ""]
(ROOT / "verification/codex_g4p2/COMPARISON.md").write_text("\n".join(lines))
print("\n".join(lines[:3]))
