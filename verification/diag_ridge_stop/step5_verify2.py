import statistics
import sys

sys.path.insert(0, "/tmp/diag_w28")
from common import *  # noqa: F401,F403

cfg, tables, table, plans, alphas = setup()
seed = cfg["seeds"]["tuning"]
col = "t1w28_do_min_w"
ctx = make_ctx(cfg, table, "ridge_time_wq_w28")

# (a) independent z of the failing row, from the CSV values (no RidgeModel): fold 2 inner 1
plan = next(p for p in plans if p.index == 2)
inner = plan.inner[1]
tr = table.loc[list(inner.train_idx), col].tolist()
va = table.loc[list(inner.val_idx)]
row = va[va[col] > 1]
print("(a) failing row:", row[["pond", "date", "t", col]].to_dict("records"))
print("    train n=%d, values %s, pstdev=%r, mean=%r" % (
    len(tr), {v: tr.count(v) for v in sorted(set(tr))}, statistics.pstdev(tr), statistics.fmean(tr)))
print("    independent z = %.6f" % ((row[col].iloc[0] - statistics.fmean(tr)) / statistics.pstdev(tr)))

# (b) counterfactual over every alpha of the failing inner split: every validation row with the column's own term removed
for fold in (2, 3):
    p = next(q for q in plans if q.index == fold)
    inn = p.inner[1]
    train = ctx.table.loc[list(inn.train_idx)]
    val = ctx.table.loc[list(inn.val_idx)]
    worst_with, worst_without = 0.0, 0.0
    for a in alphas:
        lr = ctx.learner({"alpha": a}, seed)
        lr.fit(train)
        names = column_names(lr)
        j = names.index(col)
        z = standardised(lr, val)
        full = lr.predict_log(val)
        wo = full - z[:, j] * lr.model_.coef_[j]
        worst_with = max(worst_with, float(np.abs(full).max()))
        worst_without = max(worst_without, float(np.abs(wo).max()))
    print(f"(b) fold {fold} inner 1, all 13 alphas: max|log pred| with the column = {worst_with:.6g}; "
          f"without its term = {worst_without:.6g}")
