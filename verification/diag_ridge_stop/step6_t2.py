"""T2 ridge models with water quality: which inner fits overflow, and which column drives it.

Run from the repo root: nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -I -B /tmp/diag_w28/step6_t2.py
Inner splits only (plan.inner); plan.test_idx is never touched. Writes only under /tmp/diag_w28.
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/dev/development/publications")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_tables  # noqa: E402
from src.folds import load_config, plan_folds  # noqa: E402
from src.tuning import Context, NonFinitePredictionError, predict_checked, ridge_alphas  # noqa: E402

TASK, SCHEME, SAMPLE_SET = "T2", "lopo", "eligible"
MODELS = ["ridge_pw_wq", "ridge_pw_wq_noind", "ridge_pw_wq_w14", "ridge_pw_wq_w42"]

cfg = load_config()
table = load_tables(cfg).table(TASK)
plans = [p for p in plan_folds(TASK, SCHEME, table, SAMPLE_SET, cfg, None) if not p.skipped]
alphas = ridge_alphas(cfg)
seed = cfg["seeds"]["tuning"]
rows = []
for model in MODELS:
    ctx = Context(TASK, SCHEME, model, SAMPLE_SET, cfg, table, None, False)
    n_fit = n_bad = 0
    bad = []
    max_pred = (0.0, None)
    for plan in plans:
        for inner in plan.inner:
            train = ctx.table.loc[list(inner.train_idx)]
            val = ctx.table.loc[list(inner.val_idx)]
            first = None
            for ai, a in enumerate(alphas):
                lr = ctx.learner({"alpha": a}, seed)
                lr.fit(train)
                n_fit += 1
                pred = lr.predict_log(val)
                try:
                    predict_checked(ctx, lr, val, f"inner split {inner.index}")
                except NonFinitePredictionError:
                    n_bad += 1
                    k = int(np.argmax(np.where(np.isfinite(pred), np.abs(pred), np.inf)))
                    bad.append((plan.index, inner.index, ai, int(val.iloc[k]["pond"]), str(val.iloc[k]["origin_date"])[:10] + " h" + str(int(val.iloc[k]["horizon"])), float(pred[k])))
                fin = pred[np.isfinite(pred)]
                if fin.size and np.abs(fin).max() > max_pred[0]:
                    k = int(np.argmax(np.where(np.isfinite(pred), np.abs(pred), -1)))
                    max_pred = (float(np.abs(fin).max()), (plan.index, inner.index, ai, int(val.iloc[k]["pond"]), str(val.iloc[k]["origin_date"])[:10]))
                if first is None:
                    first = lr
            names = list(first.active_) + list(first.indicator_columns)
            z = (first._design(val) - first.mean_) / first.scale_
            dtr = first._design(train)
            for j, nm in enumerate(names):
                r = int(np.argmax(np.abs(z[:, j])))
                rows.append({"model": model, "fold": plan.index, "inner": inner.index, "column": nm,
                             "max_abs_z": float(np.abs(z[r, j])), "val_pond": int(val.iloc[r]["pond"]),
                             "val_date": str(val.iloc[r]["origin_date"])[:10], "val_raw": float(first._design(val)[r, j]),
                             "train_sd": float(dtr[:, j].std()), "train_distinct": int(len(np.unique(dtr[:, j]))),
                             "train_values": " ".join(f"{v:g}" for v in np.unique(dtr[:, j])[:4]),
                             "train_ponds": " ".join(str(p) for p in sorted(set(train["pond"])))})
    print("=" * 110)
    print(f"{model}: {n_fit} inner fits ({len(plans)} outer folds x inner splits x {len(alphas)} alphas), {n_bad} with a non-finite prediction")
    for b in bad:
        print(f"  non-finite: outer fold {b[0]} inner {b[1]} alpha[{b[2]}]={alphas[b[2]]:g}  row pond {b[3]} {b[4]}  log pred {b[5]:.6g}")
    print(f"  largest finite |log prediction| on inner validation rows: {max_pred[0]:.6g} at (fold, inner, alpha idx, pond, date) {max_pred[1]}")
    d = pd.DataFrame([r for r in rows if r["model"] == model]).sort_values("max_abs_z", ascending=False)
    print(d.head(6)[["fold", "inner", "column", "max_abs_z", "val_pond", "val_date", "val_raw", "train_sd", "train_distinct", "train_values", "train_ponds"]].to_string(index=False))

pd.DataFrame(rows).to_csv("/tmp/diag_w28/step6_t2_zmax_by_column.csv", index=False)
