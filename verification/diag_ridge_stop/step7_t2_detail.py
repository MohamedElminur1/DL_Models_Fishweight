"""T2 ridge_pw_wq, outer folds 2 and 3, inner split 1, alpha 1e-3: the overflowing row's largest coef*z terms.

Run from the repo root: .venv/bin/python -I -B /tmp/diag_w28/step7_t2_detail.py   (inner splits only)
"""
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/dev/development/publications")

import numpy as np  # noqa: E402

from src.data import load_tables  # noqa: E402
from src.folds import load_config, plan_folds  # noqa: E402
from src.tuning import Context, ridge_alphas  # noqa: E402

cfg = load_config()
table = load_tables(cfg).table("T2")
plans = {p.index: p for p in plan_folds("T2", "lopo", table, "eligible", cfg, None) if not p.skipped}
ctx = Context("T2", "lopo", "ridge_pw_wq", "eligible", cfg, table, None, False)
alpha = ridge_alphas(cfg)[0]
for fold in (2, 3):
    inner = [i for i in plans[fold].inner if i.index == 1][0]
    train, val = table.loc[list(inner.train_idx)], table.loc[list(inner.val_idx)]
    lr = ctx.learner({"alpha": alpha}, cfg["seeds"]["tuning"])
    lr.fit(train)
    pred, grams = lr.predict_log(val), lr.predict_grams(val)
    k = int(np.argmax(np.abs(pred)))
    names = list(lr.active_) + list(lr.indicator_columns)
    z = (lr._design(val) - lr.mean_) / lr.scale_
    coef = lr.model_.coef_ if hasattr(lr, "model_") else lr.ridge_.coef_
    contrib = coef * z[k]
    print(f"outer fold {fold} inner 1 alpha {alpha:g}: train ponds {sorted(set(train.pond))}, val ponds {sorted(set(val.pond))}")
    print(f"  row pond {int(val.iloc[k].pond)} origin {str(val.iloc[k].origin_date)[:10]} h{int(val.iloc[k].horizon)}: log-ratio pred {pred[k]:.6g}, grams {grams[k]}")
    for j in np.argsort(-np.abs(contrib))[:4]:
        print(f"    {names[j]:28s} z={z[k, j]:.6g}  coef={coef[j]:.6g}  coef*z={contrib[j]:.6g}")
