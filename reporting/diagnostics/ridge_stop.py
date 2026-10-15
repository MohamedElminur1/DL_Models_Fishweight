"""Diagnose five frozen ridge models using inner training/validation splits only.

Run from /home/dev/development/publications:
    nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      .venv/bin/python -I -B reporting/diagnostics/ridge_stop.py

Writes results/diagnostics/ridge_stop.csv, one row per task, model, unskipped
outer fold, inner split and ridge alpha. Floats use repr, with no rounding.
The complete table replaces the CSV atomically; a failed calculation leaves
an existing CSV intact. No outer-test rows are selected or predicted, and no
scores are calculated. T1 predictions are log weight; T2 predictions are log
weight ratios, as returned by the repository learner.
"""

from __future__ import annotations

import csv
import math
import platform
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.baselines import RidgeModel  # noqa: E402
from src.data import load_tables  # noqa: E402
from src.folds import InnerPlan, load_config, plan_folds  # noqa: E402
from src.tuning import (  # noqa: E402
    Context,
    NonFinitePredictionError,
    predict_checked,
    ridge_alphas,
)

MODELS = (
    ("T1", "ridge_time_wq_w28"),
    ("T2", "ridge_pw_wq"),
    ("T2", "ridge_pw_wq_noind"),
    ("T2", "ridge_pw_wq_w14"),
    ("T2", "ridge_pw_wq_w42"),
)
SCHEME, SAMPLE_SET = "lopo", "eligible"
OUTPUT = ROOT / "results/diagnostics/ridge_stop.csv"
EXP_OVERFLOW_THRESHOLD = math.log(sys.float_info.max)
COLUMNS = (
    "task", "model", "outer_fold", "inner_split", "alpha", "train_ponds",
    "validation_ponds", "n_validation", "max_abs_log_pred", "n_nonfinite_grams",
    "stops_run", "row_pond", "row_date", "row_horizon", "row_log_pred",
    "top_column", "top_raw_value", "top_train_mean", "top_train_scale", "top_z",
    "top_coef", "top_contribution", "top_train_min", "top_train_max",
    "top_train_values", "exp_overflow_threshold",
)


def training_values(values: np.ndarray) -> str:
    """Summarize imputed design values, sorted, with at most ten distinct values."""
    distinct, counts = np.unique(values, return_counts=True)
    parts = [f"{repr(float(value))}x{int(count)}"
             for value, count in zip(distinct[:10], counts[:10])]
    if len(distinct) > 10:
        parts.append("...")
    return ";".join(parts)


def split_frames(ctx: Context, inner: InnerPlan) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select only the inner row labels, rejecting empty or overlapping splits."""
    if not inner.train_idx or not inner.val_idx:
        raise ValueError(f"{ctx.task} {ctx.model}: empty inner split {inner.index}")
    if set(inner.train_idx) & set(inner.val_idx):
        raise ValueError(f"{ctx.task} {ctx.model}: overlapping inner split {inner.index}")
    train = ctx.table.loc[list(inner.train_idx)]
    validation = ctx.table.loc[list(inner.val_idx)]
    for frame, ponds in ((train, inner.train_ponds), (validation, inner.val_ponds)):
        actual = tuple(sorted(int(p) for p in frame["pond"].unique()))
        if actual != ponds or not frame["eligible_eval"].all():
            raise ValueError(f"{ctx.task} {ctx.model}: inner split membership disagrees with plan")
    return train, validation


def candidate_row(ctx: Context, outer_fold: int, inner: InnerPlan, alpha: float,
                  train: pd.DataFrame, validation: pd.DataFrame) -> dict:
    """Fit the repository learner and record its most extreme validation prediction."""
    learner = ctx.learner({"alpha": alpha}, ctx.cfg["seeds"]["tuning"])
    if not isinstance(learner, RidgeModel):
        raise TypeError(f"{ctx.task} {ctx.model}: expected the repository RidgeModel")
    learner.fit(train)
    if learner.constant_ is not None:
        raise ValueError(f"{ctx.task} {ctx.model}: no design columns remain")

    stops_run = False
    try:
        pred, grams, _ = predict_checked(ctx, learner, validation, f"inner split {inner.index}")
    except NonFinitePredictionError:
        stops_run = True
        pred = learner.predict_log(validation)
        grams = learner.predict_grams(validation)
    if pred.shape != (len(validation),) or grams.shape != pred.shape:
        raise ValueError("unexpected prediction shape")
    if not np.isfinite(pred).all():
        raise ValueError("nonfinite log predictions prevent finite contribution diagnosis")
    n_nonfinite_grams = int((~np.isfinite(grams)).sum())
    if stops_run != (n_nonfinite_grams > 0):
        raise ValueError("predict_checked disagrees with gram-prediction finiteness")

    position = int(np.argmax(np.abs(pred)))
    row = validation.iloc[position]
    names = list(learner.active_) + list(learner.indicator_columns)
    design_train = learner._design(train)
    design_val = learner._design(validation)
    if (design_train.shape != (len(train), len(names))
            or design_val.shape != (len(validation), len(names))
            or not np.isfinite(design_train).all()
            or not np.isfinite(design_val).all()
            or not np.isfinite(learner.mean_).all()
            or not np.isfinite(learner.scale_).all()
            or not (learner.scale_ > 0).all()
            or not np.isfinite(learner.model_.coef_).all()):
        raise ValueError("unexpected ridge design or fitted parameters")
    z = (design_val - learner.mean_) / learner.scale_
    contributions = learner.model_.coef_ * z[position]
    if not np.isfinite(z).all() or not np.isfinite(contributions).all():
        raise ValueError("nonfinite standardized values or contributions")
    column_index = int(np.argmax(np.abs(contributions)))
    column = names[column_index]
    if column_index < len(learner.active_):
        raw = row[column]
        raw_value = "" if pd.isna(raw) else float(raw)
    else:
        # The appended design column is the missingness flag of this sensor.
        column = f"missing({column})"
        raw_value = float(pd.isna(row[names[column_index]]))
    training_column = design_train[:, column_index]
    return {
        "task": ctx.task, "model": ctx.model, "outer_fold": outer_fold,
        "inner_split": inner.index, "alpha": alpha,
        "train_ponds": ";".join(str(p) for p in inner.train_ponds),
        "validation_ponds": ";".join(str(p) for p in inner.val_ponds),
        "n_validation": len(validation),
        "max_abs_log_pred": float(abs(pred[position])),
        "n_nonfinite_grams": n_nonfinite_grams, "stops_run": stops_run,
        "row_pond": int(row["pond"]),
        "row_date": str(row["date" if ctx.task == "T1" else "origin_date"]),
        "row_horizon": "" if ctx.task == "T1" else int(row["horizon"]),
        "row_log_pred": float(pred[position]), "top_column": column,
        "top_raw_value": raw_value,
        "top_train_mean": float(learner.mean_[column_index]),
        "top_train_scale": float(learner.scale_[column_index]),
        "top_z": float(z[position, column_index]),
        "top_coef": float(learner.model_.coef_[column_index]),
        "top_contribution": float(contributions[column_index]),
        "top_train_min": float(training_column.min()),
        "top_train_max": float(training_column.max()),
        "top_train_values": training_values(training_column),
        "exp_overflow_threshold": EXP_OVERFLOW_THRESHOLD,
    }


def main() -> None:
    """Write all candidates deterministically, continuing past diagnosed gram overflows."""
    started = time.perf_counter()
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"run from the repository root: {ROOT}")
    cfg = load_config()
    tables = load_tables(cfg)
    alphas = ridge_alphas(cfg)
    if not alphas or len(set(alphas)) != len(alphas) or any(
            not math.isfinite(a) or a <= 0 for a in alphas):
        raise ValueError("ridge alpha grid must contain distinct positive finite values")
    rows = []
    for task, model in MODELS:
        table = tables.table(task)
        if not table.index.is_unique:
            raise ValueError(f"{task} has duplicate row labels")
        ctx = Context(task, SCHEME, model, SAMPLE_SET, cfg, table, None, False)
        if ctx.kind != "ridge":
            raise ValueError(f"{task} {model}: expected a configured ridge model")
        plans = [p for p in plan_folds(task, SCHEME, table, SAMPLE_SET, cfg, None) if not p.skipped]
        if not plans or len({p.index for p in plans}) != len(plans):
            raise ValueError(f"{task}: empty or duplicate outer folds")
        first = len(rows)
        for plan in plans:
            if not plan.inner or len({i.index for i in plan.inner}) != len(plan.inner):
                raise ValueError(f"{task} fold {plan.index}: empty or duplicate inner splits")
            for inner in plan.inner:
                if not (set(inner.train_idx) | set(inner.val_idx)) <= set(plan.train_idx):
                    raise ValueError("inner row labels are outside outer training labels")
                train, validation = split_frames(ctx, inner)
                for alpha in alphas:
                    rows.append(candidate_row(ctx, plan.index, inner, alpha, train, validation))
        print(f"{task} {model}: {len(rows) - first} candidates, "
              f"{sum(r['stops_run'] for r in rows[first:])} stop candidates")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: repr(value) if isinstance(value, float) else value
                             for key, value in row.items()})
    temporary.replace(OUTPUT)
    print(f"{len(rows)} rows -> {OUTPUT.relative_to(ROOT)}")
    print(f"runtime_seconds={time.perf_counter() - started!r}; "
          f"platform={platform.platform()}; python={platform.python_version()}")


if __name__ == "__main__":
    main()
