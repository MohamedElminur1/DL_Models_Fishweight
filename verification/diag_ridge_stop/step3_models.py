import sys
sys.path.insert(0, "/tmp/diag_w28")
from common import *  # noqa: F401,F403

MODELS = ["ridge_time_wq_w28", "ridge_time_wq", "ridge_time_wq_w07", "ridge_time_wq_len", "ridge_time_wq_noind"]
cfg, tables, table, plans, alphas = setup()
seed = cfg["seeds"]["tuning"]
rows = []          # one record per (model, fold, inner, column, validation row): z, raw value, train stats (alpha-independent)
summ = {}
for model in MODELS:
    ctx = make_ctx(cfg, table, model)
    n_fit = n_bad = 0
    max_pred = (0.0, None)
    for plan in plans:
        for inner in plan.inner:
            train = ctx.table.loc[list(inner.train_idx)]
            val = ctx.table.loc[list(inner.val_idx)]
            lr0 = None
            for ai, a in enumerate(alphas):
                lr = ctx.learner({"alpha": a}, seed)
                lr.fit(train)
                n_fit += 1
                try:
                    predict_checked(ctx, lr, val, f"inner split {inner.index}")
                except NonFinitePredictionError:
                    n_bad += 1
                pred = lr.predict_log(val)
                fin = pred[np.isfinite(pred)]
                if fin.size and np.abs(fin).max() > max_pred[0]:
                    k = int(np.argmax(np.where(np.isfinite(pred), np.abs(pred), -1)))
                    max_pred = (float(np.abs(fin).max()), (plan.index, inner.index, ai, int(val.iloc[k]["pond"]), val.iloc[k]["date"]))
                if lr0 is None:
                    lr0 = lr
            # standardised validation matrix does not depend on alpha: take it from the first fit
            names = column_names(lr0)
            z = standardised(lr0, val)
            dtr = lr0._design(train)
            sd_tr = dtr.std(axis=0)
            for j, nm in enumerate(names):
                r = int(np.argmax(np.abs(z[:, j])))
                rows.append({"model": model, "fold": plan.index, "inner": inner.index, "column": nm,
                             "max_abs_z": float(np.abs(z[r, j])), "z": float(z[r, j]),
                             "val_pond": int(val.iloc[r]["pond"]), "val_date": val.iloc[r]["date"],
                             "train_sd": float(sd_tr[j]), "train_distinct": int(len(np.unique(dtr[:, j]))),
                             "train_mean": float(lr0.mean_[j]), "scale_used": float(lr0.scale_[j])})
    summ[model] = (n_fit, n_bad, max_pred)

df = pd.DataFrame(rows)
df.to_csv("/tmp/diag_w28/step3_zmax_by_column.csv", index=False)
pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_colwidth", 40)
for model in MODELS:
    n_fit, n_bad, mp = summ[model]
    d = df[df.model == model].sort_values("max_abs_z", ascending=False)
    top = d.iloc[0]
    print("=" * 120)
    print(f"{model}: {n_fit} inner fits ({len(plans)} outer folds x 3 inner splits x {len(alphas)} alphas), {n_bad} with a non-finite prediction")
    print(f"  largest |z| over all folds/inner splits/validation rows/columns: {top.max_abs_z:.6g}  column {top.column}  outer fold {top.fold} inner {top.inner}  "
          f"(val pond {top.val_pond}, {top.val_date}; train sd {top.train_sd:.6g}, mean {top.train_mean:.6g}, distinct {top.train_distinct})")
    print(f"  largest |log prediction| on inner validation rows (finite): {mp[0]:.6g} at (fold, inner, alpha idx, pond, date) = {mp[1]}")
    print("  top 6 (fold, inner, column) by max |z|:")
    print(d.head(6)[["fold", "inner", "column", "max_abs_z", "val_pond", "val_date", "train_sd", "train_distinct", "train_mean"]].to_string(index=False))
