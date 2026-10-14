import statistics
import sys
sys.path.insert(0, "/tmp/diag_w28")
from common import *  # noqa: F401,F403

cfg, tables, table, plans, alphas = setup()
seed = cfg["seeds"]["tuning"]

# (a) independent recomputation of the failing column from the CSV via the fold plan's row labels, no RidgeModel involved
plan = next(p for p in plans if p.index == 2)
inner = plan.inner[1]
col = "t1w28_do_min_w"
tr = table.loc[list(inner.train_idx), col].tolist()
va = table.loc[list(inner.val_idx)]
print("(a) fold 2 inner 1 independent check")
print("   train n =", len(tr), "| distinct =", sorted(set(tr)), "| counts =", {v: tr.count(v) for v in sorted(set(tr))})
print("   pstdev (statistics module) =", repr(statistics.pstdev(tr)), "| mean =", repr(statistics.fmean(tr)))
v = va.loc[va["pond"] == 9, col]
print("   validation pond 9 value(s):", v.tolist(), "dates", va.loc[va["pond"] == 9, "date"].tolist())
print("   z =", (v.iloc[0] - statistics.fmean(tr)) / statistics.pstdev(tr))
print("   exp overflow threshold ln(max float) =", float(np.log(np.finfo(float).max)))

# (b) the column by pond over the eligible rows of the cohort (features only)
elig = table[table["eligible_eval"].astype(bool) & table["pond"].isin(cfg["design"]["cohort"])]
for c in ["t1w28_do_min_w", "do_min_w", "t1w07_do_min_w"]:
    print(f"(b) {c} by pond over eligible rows (n, distinct values, min, max):")
    for pond, g in elig.groupby("pond"):
        u = np.unique(g[c].dropna())
        print(f"   pond {pond}: n={len(g)}, NaN={int(g[c].isna().sum())}, distinct={len(u)}, min={g[c].min():.6g}, max={g[c].max():.6g}" + (f", values={u.tolist()}" if len(u) <= 4 else ""))

# (c) w28: per outer fold, inner split 1: training sd of the column, max |z|, and the largest finite |log pred| at alpha[0] (inner rows only)
ctx = make_ctx(cfg, table, "ridge_time_wq_w28")
print("(c) w28, inner split 1 of every outer fold, alpha[0]=0.001:")
for p in plans:
    inn = p.inner[1]
    train = ctx.table.loc[list(inn.train_idx)]
    val = ctx.table.loc[list(inn.val_idx)]
    lr = ctx.learner({"alpha": alphas[0]}, seed)
    lr.fit(train)
    names = column_names(lr)
    j = names.index(col)
    z = standardised(lr, val)
    pred = lr.predict_log(val)
    print(f"   fold {p.index}: train ponds {inn.train_ponds}, val ponds {inn.val_ponds}; train sd of {col} = {lr._design(train)[:, j].std():.6g}, max|z| of it = {np.abs(z[:, j]).max():.6g}, "
          f"coef = {lr.model_.coef_[j]:.4f}, max|log pred| = {np.abs(pred).max():.6g}, exceeds 709.78: {bool(np.abs(pred).max() > 709.78)}")
