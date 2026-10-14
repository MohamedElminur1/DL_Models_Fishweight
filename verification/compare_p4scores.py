"""Gate 4 part 1, scores: Codex's blind outer scores against results/metrics (lead, after scoring).

Run from the repository root: .venv/bin/python -I verification/compare_p4scores.py
Writes verification/codex_p4models/SCORES_COMPARISON.md. Read only after the selection was
committed and src/evaluate.py had run (plan 9.4).
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
codex = pd.read_csv(ROOT / "verification/codex_p4models/codex_scores.csv")
scores = pd.read_csv(ROOT / "results/metrics/scores.csv")
per_pond = pd.read_csv(ROOT / "results/metrics/per_pond.csv")
keep = (scores.scheme == "lopo") & (scores.sample_set == "eligible")
scores = scores[keep].astype({"horizon": str})
per_pond = per_pond[(per_pond.scheme == "lopo") & (per_pond.sample_set == "eligible")].astype({"horizon": str})
codex = codex.astype({"horizon": str})

rows = []
for r in codex.itertuples(index=False):
    ours = scores[(scores.task == r.task) & (scores.model == r.model) & (scores.horizon == r.horizon)]
    if ours.empty:
        rows.append((r.task, r.model, r.horizon, r.S, None, None, "no pipeline score (stopped under plan 9.0)"))
        continue
    pp = per_pond[(per_pond.task == r.task) & (per_pond.model == r.model) & (per_pond.horizon == r.horizon)]
    diffs = [abs(getattr(r, f"pond_{p}") - s) for p, s in zip(pp.pond, pp.S)]
    rows.append((r.task, r.model, r.horizon, r.S, float(ours.S.iloc[0]), max(diffs), ""))

out = pd.DataFrame(rows, columns=["task", "model", "horizon", "S_codex", "S_pipeline", "max_pond_diff", "note"])
out["S_diff"] = (out.S_codex - out.S_pipeline).abs()
compared = out[out.S_pipeline.notna()]
lines = ["# Codex part 1 outer scores against results/metrics", "",
         f"Compared {len(compared)} task-model-horizon scores; largest |S difference| "
         f"{compared.S_diff.max():.2e}, largest per-pond difference {compared.max_pond_diff.max():.2e}.", "",
         "```", out.to_string(index=False, float_format=lambda v: f"{v:.6g}"), "```", ""]
(ROOT / "verification/codex_p4models/SCORES_COMPARISON.md").write_text("\n".join(lines))
print("\n".join(lines[:3]))
