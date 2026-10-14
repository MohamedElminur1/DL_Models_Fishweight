import sys
sys.path.insert(0, "/tmp/diag_w28")
from common import *  # noqa: F401,F403

MODEL = "ridge_time_wq_w28"
cfg, tables, table, plans, alphas = setup()
ctx = make_ctx(cfg, table, MODEL)
seed = cfg["seeds"]["tuning"]
np.set_printoptions(linewidth=200)

for plan in plans:
    if plan.index not in (2, 3):
        continue
    inner = plan.inner[1]
    train = ctx.table.loc[list(inner.train_idx)]
    val = ctx.table.loc[list(inner.val_idx)]
    print("=" * 110)
    print(f"outer fold {plan.index} (test pond {plan.test_ponds}, dropped {plan.dropped_ponds}); outer train ponds {plan.train_ponds}")
    print(f"inner split {inner.index}: train ponds {inner.train_ponds} (n={len(train)}), validation ponds {inner.val_ponds} (n={len(val)})")
    # per alpha: how many predictions are non-finite and the largest |log prediction| (inner validation rows only)
    print("per alpha, inner split 1: max |log pred| over the 10 validation rows, n grams non-finite")
    for ai, a in enumerate(alphas):
        lr = ctx.learner({"alpha": a}, seed)
        lr.fit(train)
        pred = lr.predict_log(val)
        grams = lr.predict_grams(val)
        print(f"   alpha[{ai:2d}]={a:10.6g}  max|pred|={np.nanmax(np.abs(pred)):.6g}  n pred non-finite={int((~np.isfinite(pred)).sum())}  n grams non-finite={int((~np.isfinite(grams)).sum())}")
    a0 = alphas[0]
    lr = ctx.learner({"alpha": a0}, seed)
    lr.fit(train)
    names = column_names(lr)
    z = standardised(lr, val)
    coef = lr.model_.coef_
    pred = lr.predict_log(val)
    grams = lr.predict_grams(val)
    bad_rows = np.where(~np.isfinite(grams) | ~np.isfinite(pred))[0]
    print(f"alpha[0]={a0}: bad validation row positions {bad_rows.tolist()}; n_dropped_all_missing={lr.n_dropped_}; design columns {len(names)}")
    design_tr = lr._design(train)
    std_tr = design_tr.std(axis=0)
    for r in bad_rows:
        row = val.iloc[r]
        print(f"  bad row: table index {val.index[r]}, pond {int(row['pond'])}, date {row['date']}, t={row['t']}; log-pred={pred[r]!r}; grams={grams[r]!r}; intercept={lr.model_.intercept_!r}")
        contrib = z[r] * coef
        order = np.argsort(-np.abs(contrib))[:6]
        print("  top columns by |coef*z| in this row:")
        for j in order:
            print(f"    {names[j]:34s} z={z[r, j]:.6e}  coef={coef[j]:.6e}  coef*z={contrib[j]:.6e}")
        j = int(order[0])
        col = names[j]
        raw_tr = train[col].to_numpy(dtype=float) if col in train.columns else None
        print(f"  column of interest: {col}")
        print(f"    training design column (after median imputation): n={len(design_tr)}, distinct={len(np.unique(design_tr[:, j]))}, "
              f"std as computed in RidgeModel (design.std(axis=0))={std_tr[j]!r}, mean_={lr.mean_[j]!r}, scale_ used={lr.scale_[j]!r}, median_ imputed={lr.median_[j] if j < len(lr.median_) else None!r}")
        if raw_tr is not None:
            fin = raw_tr[np.isfinite(raw_tr)]
            u, c = np.unique(fin, return_counts=True)
            print(f"    raw training values: non-NaN n={len(fin)}, NaN n={int(np.isnan(raw_tr).sum())}, distinct non-NaN={len(u)}")
            for uu, cc in zip(u, c):
                print(f"      value {uu!r} x{cc}")
            print(f"    validation raw value (this row)={row[col]!r}; all 10 validation raw values={val[col].to_numpy().tolist()}")
        print(f"    standardised validation value z={z[r, j]!r}")
        # counterfactual: the same prediction without this one column's term
        others = float(lr.model_.intercept_ + np.sum(contrib) - contrib[j])
        print(f"    log-pred of this row without the {col} term = {others!r}")
