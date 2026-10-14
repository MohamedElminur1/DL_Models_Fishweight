import sys
sys.path.insert(0, "/tmp/diag_w28")
from common import *  # noqa: F401,F403

MODEL = "ridge_time_wq_w28"
cfg, tables, table, plans, alphas = setup()
ctx = make_ctx(cfg, table, MODEL)
seed = cfg["seeds"]["tuning"]
print("model", MODEL, "| kind", ctx.kind, "| outer folds run:", [p.index for p in plans])
print("alpha grid:", [f"{a:.6g}" for a in alphas])
print("features:", __import__("src.baselines", fromlist=["x"]).feature_columns(TASK, MODEL, cfg))
print("indicators:", __import__("src.baselines", fromlist=["x"]).indicator_columns(TASK, MODEL, cfg))
print()

failures = []   # (fold, alpha_idx, inner, n_bad, n_val, pred_nonfinite, grams_nonfinite)
for plan in plans:
    for ai, a in enumerate(alphas):
        for inner in plan.inner:
            train = ctx.table.loc[list(inner.train_idx)]
            val = ctx.table.loc[list(inner.val_idx)]
            learner = ctx.learner({"alpha": a}, seed)
            learner.fit(train)
            try:
                predict_checked(ctx, learner, val, f"inner split {inner.index}")
            except NonFinitePredictionError as exc:
                pred = learner.predict_log(val)
                grams = learner.predict_grams(val)
                failures.append((plan.index, ai, inner.index, str(exc), int((~np.isfinite(pred)).sum()),
                                 int((~np.isfinite(grams)).sum()), len(val)))
print(f"{len(failures)} (outer fold, alpha, inner split) combinations raise NonFinitePredictionError")
for f in failures:
    print(f"  outer fold {f[0]} alpha[{f[1]}]={alphas[f[1]]:.6g} inner {f[2]}: {f[3]} | n pred non-finite {f[4]}, n grams non-finite {f[5]}")
print()
print("first failure per outer fold in the order run_trial evaluates (alpha-major, then inner split):")
for plan in plans:
    mine = [f for f in failures if f[0] == plan.index]
    if mine:
        first = min(mine, key=lambda f: (f[1], f[2]))
        print(f"  outer fold {plan.index} (test pond {plan.test_ponds}): alpha[{first[1]}]={alphas[first[1]]:.6g}, inner split {first[2]}: {first[3]}")
    else:
        print(f"  outer fold {plan.index} (test pond {plan.test_ponds}): no failure")
