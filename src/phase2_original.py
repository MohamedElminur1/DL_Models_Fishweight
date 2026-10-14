#!/usr/bin/env python3
"""Phase 2: reproduce and diagnose the original pond-2 analysis (tasks/phase2_diagnostics.md).

Run from the repo root:

    .venv/bin/python -I src/phase2_original.py --config configs/phase2.yaml
        Split checks, persistence on every split, the 18-step persistence diagnostic, then the
        epoch timing with the projected-time guard, then the CNN-LSTM grid. Resumable: a
        CNN-LSTM run whose predictions file already exists is skipped. Persistence, the split
        checks and every summary are rebuilt on each invocation.

    ... --time-epoch
        Only time training epochs (training windows only, no validation or test rows) and write
        epoch_timing.json with the projection of the whole grid.

The script imports nothing from src/ (python -I does not put the script directory on sys.path).
Outputs go to the results directory named in the config; nothing else is written.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402
from keras import layers  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

EVAL_SETS = ("all", "change_rows", "grid_first_rows")
NO_INPUT_SET = "none"  # persistence does not use an input set
SPLIT_ORDER = ("S0_notebook", "S1_chrono", "S2_random")
MODEL_ORDER = ("persistence", "cnn_lstm")
INPUT_ORDER = (NO_INPUT_SET, "notebook", "manuscript", "wq_only")

METRIC_COLS = ["split", "input_set", "model", "seed", "lookback", "eval_set",
               "n", "mae", "rmse", "mape", "r2"]
RUN_COLS = ["split", "input_set", "seed", "n_train", "n_val", "n_test", "epochs_run",
            "best_epoch", "best_val_loss", "train_seconds", "predict_seconds",
            "cpu_model", "n_cores", "tensorflow_version",
            "model", "lookback", "wall_seconds", "keras_version", "started_utc",
            "n_nonfinite_pred"]
PRED_COLS = ["target_row", "created_at", "y_true", "y_pred", "is_change_row", "is_grid_first_row"]
INT_COLS = ["seed", "lookback", "n", "n_train", "n_val", "n_test", "epochs_run", "best_epoch",
            "n_cores", "n_nonfinite_pred"]


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cpu_model() -> str:
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def hardware_info() -> dict:
    return {
        "cpu_model": cpu_model(),
        "n_cores": len(os.sched_getaffinity(0)),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
    }


# ---------------------------------------------------------------------------------------------
# Data preparation (notebook cells 1-2, protocol "Data preparation")
# ---------------------------------------------------------------------------------------------

@dataclass
class Data:
    """Pond-2 frame after the notebook's preparation, with the arrays every run reads."""
    df: pd.DataFrame
    N: int
    weight: np.ndarray        # float64 grams, file order
    created_at: np.ndarray    # str, as in the file
    date_str: np.ndarray      # str, clock date taken from created_at
    is_grid_date: np.ndarray  # bool, date is anchor + period * k with k >= 0
    checks: dict              # data-preparation facts for split_checks.json


def load_data(cfg: dict) -> Data:
    """read_csv, drop entry_id, dropna, keep file order; asserts N and the Population/Length claim."""
    dcfg = cfg["data"]
    raw = pd.read_csv(REPO / cfg["paths"]["raw_csv"])
    assert list(raw.columns) == dcfg["raw_columns"], list(raw.columns)
    assert len(raw) == dcfg["expected_raw_rows"], len(raw)

    no_id = raw.drop(columns=["entry_id"])
    missing = no_id.isna().sum()
    assert int(missing["Ammonia(g/ml)"]) == dcfg["expected_missing_ammonia"], missing.to_dict()
    kept = no_id.dropna()
    notebook_kept = raw.drop(columns=["entry_id"] + dcfg["notebook_dropped"]).dropna()
    assert kept.index.equals(notebook_kept.index), \
        "dropna() keeps different rows with and without Population and Fish_Length"
    assert len(kept) == dcfg["expected_rows"], f"N = {len(kept)}, expected {dcfg['expected_rows']}"

    df = kept.reset_index(drop=True)
    df.columns = dcfg["columns"]
    numeric = [c for c in df.columns if c != "created_at"]
    df[numeric] = df[numeric].astype("float64")

    date_str = df["created_at"].str.slice(0, 10)
    day_index = (pd.to_datetime(date_str, format="%Y-%m-%d")
                 - pd.Timestamp(cfg["grid_dates"]["anchor"])).dt.days.to_numpy()
    is_grid_date = (day_index >= 0) & (day_index % cfg["grid_dates"]["period_days"] == 0)
    ts = pd.to_datetime(df["created_at"].str.slice(0, 19), format="%Y-%m-%d %H:%M:%S")

    checks = {
        "raw_rows": int(len(raw)),
        "rows_after_dropna": int(len(kept)),
        "rows_dropped": int(len(raw) - len(kept)),
        "missing_per_column": {c: int(v) for c, v in missing.items()},
        "rows_after_dropna_without_population_and_length": int(len(notebook_kept)),
        "dropna_same_rows_with_and_without_population_and_length": True,
        "population_unique_values": sorted(raw["Population"].unique().tolist()),
        "timestamps_strictly_increasing": bool((ts.diff().dropna() > pd.Timedelta(0)).all()),
        "first_created_at": str(df["created_at"].iloc[0]),
        "last_created_at": str(df["created_at"].iloc[-1]),
        "n_grid_dates_with_rows": int(len(set(date_str.to_numpy()[is_grid_date]))),
        # 0-based data-row index in the raw file (header excluded) of each row removed by dropna();
        # target_row r in the predictions files is file row r + (number of dropped rows <= file row).
        "dropped_file_rows": (no_id.index.difference(kept.index)).to_numpy().tolist(),
    }
    assert checks["timestamps_strictly_increasing"], "created_at is not strictly increasing in file order"
    return Data(df=df, N=len(df), weight=df["weight"].to_numpy(dtype=np.float64),
                created_at=df["created_at"].to_numpy(dtype=object),
                date_str=date_str.to_numpy(dtype=object), is_grid_date=is_grid_date, checks=checks)


# ---------------------------------------------------------------------------------------------
# Splits (protocol "Splits")
# ---------------------------------------------------------------------------------------------

@dataclass
class Partition:
    """Target-row indices of one split (sample i = rows [i - lookback, i) -> weight[i]) and the
    row range [fit_lo, fit_hi) the scalers are fitted on. Test indices are sorted ascending."""
    split: str
    seed: int | None            # the shuffle seed for S2_random, otherwise None
    lookback: int
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    fit_lo: int
    fit_hi: int
    row_bounds: dict | None     # chrono only: input-row range each subset's windows must stay in


def make_partition(name: str, scfg: dict, N: int, seed: int | None) -> Partition:
    L, kind = scfg["lookback"], scfg["kind"]
    if kind == "notebook":
        T = int(scfg["train_frac"] * N)
        n_val = int(scfg["val_frac"] * N)
        return Partition(name, None, L, np.arange(L, T), np.arange(T, T + n_val), np.arange(T, N - L),
                         0, N, None)
    if kind == "chrono":
        a, b = int(scfg["train_end_frac"] * N), int(scfg["val_end_frac"] * N)
        return Partition(name, None, L, np.arange(L, a), np.arange(a + L, b), np.arange(b + L, N),
                         0, a, {"train": (0, a), "val": (a, b), "test": (b, N)})
    if kind == "random":
        order = np.random.default_rng(seed).permutation(np.arange(L, N))
        n = len(order)
        n_tr, n_va = int(scfg["train_frac"] * n), int(scfg["val_frac"] * n)
        return Partition(name, seed, L, order[:n_tr], np.sort(order[n_tr:n_tr + n_va]),
                         np.sort(order[n_tr + n_va:]), 0, N, None)
    raise ValueError(f"unknown split kind {kind!r}")


def _sha(indices: np.ndarray) -> str:
    return hashlib.sha256(np.sort(indices).astype("<i8").tobytes()).hexdigest()


def eval_flags(data: Data, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(is_change_row, is_grid_first_row) for sorted test targets. change: weight[i] != weight[i-1].
    grid first: the earliest test target on each calendar date of the 14-day grid."""
    assert np.all(np.diff(test) > 0)
    w = data.weight
    change = w[test] != w[test - 1]
    cand = test[data.is_grid_date[test]]
    first = (pd.Series(cand).groupby(data.date_str[cand]).min().to_numpy()
             if len(cand) else np.array([], dtype=np.int64))
    return change, np.isin(test, first)


def partition_report(data: Data, cfg: dict, p: Partition) -> dict:
    """Facts and assertions for one partition (goes to split_checks.json). Raises on a violation."""
    N, L = data.N, p.lookback
    subsets = {"train": p.train, "val": p.val, "test": p.test}
    rep: dict = {"split": p.split, "seed": p.seed, "lookback": L,
                 "scaler_fit_rows_half_open": [p.fit_lo, p.fit_hi]}
    for k, t in subsets.items():
        s = np.sort(t)
        assert s[0] >= L and s[-1] < N
        rep[k] = {"n": int(len(t)), "target_min": int(s[0]), "target_max": int(s[-1]),
                  "target_range_half_open": [int(s[0]), int(s[-1]) + 1],
                  "contiguous": bool(s[-1] - s[0] + 1 == len(s)),
                  "first_input_row": int(s[0] - L), "last_input_row": int(s[-1] - 1),
                  "sha256_sorted_targets": _sha(t)}

    inter = lambda x, y: int(len(np.intersect1d(x, y)))  # noqa: E731
    checks: dict = {}
    if p.split == "S0_notebook":
        checks["val_is_subset_of_test"] = bool(np.isin(p.val, p.test).all())
        checks["train_and_test_disjoint"] = inter(p.train, p.test) == 0
        counts = cfg["data"]["notebook_cell4_counts"]
        checks["counts_match_notebook_cell4"] = [len(p.train), len(p.val), len(p.test)] == counts
        checks["test_first_actual_matches_notebook_cell9"] = bool(
            data.weight[p.test[0]] == cfg["data"]["notebook_cell9_first_actual"])
        checks["test_last_actual_matches_notebook_cell9"] = bool(
            data.weight[p.test[-1]] == cfg["data"]["notebook_cell9_last_actual"])
        checks["T"] = int(p.test[0])
        checks["val_end_exclusive"] = int(p.val[-1]) + 1
    else:
        checks["train_val_test_pairwise_disjoint"] = (
            inter(p.train, p.val) == 0 and inter(p.train, p.test) == 0 and inter(p.val, p.test) == 0)
        if p.row_bounds is not None:
            checks["no_window_crosses_a_boundary"] = all(
                rep[k]["first_input_row"] >= lo and rep[k]["target_max"] < hi
                for k, (lo, hi) in p.row_bounds.items())
            checks["row_bounds_half_open"] = {k: list(v) for k, v in p.row_bounds.items()}
            checks["a"], checks["b"] = p.row_bounds["val"]
        else:
            checks["train_val_test_cover_all_targets"] = bool(np.array_equal(
                np.sort(np.concatenate([p.train, p.val, p.test])), np.arange(L, N)))
    rep["checks"] = checks
    flags = [v for v in checks.values() if isinstance(v, bool)]
    assert all(flags), f"split check failed for {p.split} (seed {p.seed}): {checks}"

    change, grid_first = eval_flags(data, p.test)
    rep["eval_sets"] = {"all": int(len(p.test)), "change_rows": int(change.sum()),
                        "grid_first_rows": int(grid_first.sum()),
                        "grid_first_rows_detail": [
                            {"target_row": int(r), "date": str(data.date_str[r])} for r in p.test[grid_first]]}
    return rep


def build_split_checks(cfg: dict, data: Data) -> dict:
    out: dict = {"N": data.N, "data": data.checks, "splits": {}}
    for name, scfg in cfg["splits"].items():
        if scfg["kind"] == "random":
            out["splits"][name] = {"seeds": {
                str(s): partition_report(data, cfg, make_partition(name, scfg, data.N, s))
                for s in cfg["seeds"]}}
        else:
            out["splits"][name] = partition_report(data, cfg, make_partition(name, scfg, data.N, None))
    s1 = out["splits"]["S1_chrono"]["checks"]
    out["a"], out["b"] = s1["a"], s1["b"]
    out["T"] = out["splits"]["S0_notebook"]["checks"]["T"]
    return out


def standardize(data: Data, columns: list[str], fit_lo: int, fit_hi: int):
    """StandardScaler on the inputs and on the weight target, both fitted on rows [fit_lo, fit_hi).
    Returns float32 scaled inputs (N, F), float32 scaled target (N, 1), target mean and scale."""
    X = data.df[columns].to_numpy(dtype=np.float64)
    y = data.df[["weight"]].to_numpy(dtype=np.float64)
    xs = StandardScaler().fit(X[fit_lo:fit_hi])
    ys = StandardScaler().fit(y[fit_lo:fit_hi])
    return (xs.transform(X).astype(np.float32), ys.transform(y).astype(np.float32),
            float(ys.mean_[0]), float(ys.scale_[0]))


# ---------------------------------------------------------------------------------------------
# Metrics and result files
# ---------------------------------------------------------------------------------------------

def metric_values(y: np.ndarray, p: np.ndarray) -> dict:
    """n, MAE, RMSE, MAPE (%) and R2 (only for n >= 3 and a non-constant target). Non-finite
    predictions give NaN metrics; they are not dropped."""
    n = int(len(y))
    if n == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "mape": np.nan, "r2": np.nan}
    err = p - y
    sse = float(np.sum(err * err))
    sst = float(np.sum((y - y.mean()) ** 2))
    return {"n": n, "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(sse / n)),
            "mape": float(100.0 * np.mean(np.abs(err) / np.abs(y))),
            "r2": float(1.0 - sse / sst) if n >= 3 and sst > 0 else np.nan}


def _sort_for_reading(df: pd.DataFrame) -> pd.DataFrame:
    """Stable, human-friendly row order: split, model, input set, seed, evaluation set
    (listed from the most significant key; columns the frame lacks are skipped)."""
    def ranked(col: str, order: tuple) -> np.ndarray:
        return df[col].map({v: i for i, v in enumerate(order)}).fillna(len(order)).to_numpy(dtype=float)

    keys = []
    if "split" in df.columns:
        keys.append(ranked("split", SPLIT_ORDER))
    if "model" in df.columns:
        keys.append(ranked("model", MODEL_ORDER))
    if "input_set" in df.columns:
        keys.append(ranked("input_set", INPUT_ORDER))
    if "seed" in df.columns:
        keys.append(df["seed"].astype("float64").fillna(-1).to_numpy())
    if "eval_set" in df.columns:
        keys.append(ranked("eval_set", EVAL_SETS))
    return df.iloc[np.lexsort(keys[::-1])].reset_index(drop=True)


def replace_rows(path: Path, new: pd.DataFrame, key_cols: list[str], columns: list[str]) -> None:
    """Merge `new` into the CSV at `path`, dropping old rows that share a key with a new row;
    the file is replaced atomically so a partial write never survives an interruption."""
    new = new[columns]
    if path.exists():
        old = pd.read_csv(path)
        new_keys = set(map(tuple, new[key_cols].astype(object).to_numpy().tolist()))
        keep = [tuple(r) not in new_keys for r in old[key_cols].astype(object).to_numpy().tolist()]
        out = pd.concat([old[keep], new], ignore_index=True)
    else:
        out = new.copy()
    for c in INT_COLS:
        if c in out.columns:
            out[c] = out[c].astype("Int64")
    out = _sort_for_reading(out[columns])
    tmp = path.with_name(path.name + ".tmp")
    out.to_csv(tmp, index=False)
    os.replace(tmp, path)


def write_predictions(df: pd.DataFrame, path: Path) -> None:
    """Atomic write of a gzip CSV; mtime 0 keeps identical predictions byte-identical."""
    tmp = path.with_name(path.name + ".tmp")
    df.to_csv(tmp, index=False, compression={"method": "gzip", "mtime": 0})
    os.replace(tmp, path)


def pred_path(out_dir: Path, split: str, input_set: str, model: str, seed: int | None) -> Path:
    s = "NA" if seed is None else str(seed)
    return out_dir / "predictions" / f"{split}__{input_set}__{model}__s{s}.csv.gz"


def build_records(data: Data, part: Partition, input_set: str, model: str, seed: int | None,
                  y_pred: np.ndarray, info: dict, hw: dict):
    """Metrics rows (one per evaluation set), the runs.csv row and the predictions frame of one run."""
    test = part.test
    y_true = data.weight[test]
    change, grid_first = eval_flags(data, test)
    masks = {"all": np.ones(len(test), dtype=bool), "change_rows": change, "grid_first_rows": grid_first}
    metrics = pd.DataFrame([
        {"split": part.split, "input_set": input_set, "model": model, "seed": seed,
         "lookback": part.lookback, "eval_set": name, **metric_values(y_true[m], y_pred[m])}
        for name, m in masks.items()])
    run = {"split": part.split, "input_set": input_set, "seed": seed, "n_train": info.get("n_train"),
           "n_val": info.get("n_val"), "n_test": int(len(test)), "epochs_run": info.get("epochs_run"),
           "best_epoch": info.get("best_epoch"), "best_val_loss": info.get("best_val_loss"),
           "train_seconds": info.get("train_seconds"), "predict_seconds": info.get("predict_seconds"),
           "cpu_model": hw["cpu_model"], "n_cores": hw["n_cores"],
           "tensorflow_version": hw["tensorflow_version"], "model": model, "lookback": part.lookback,
           "wall_seconds": info.get("wall_seconds"), "keras_version": hw["keras_version"],
           "started_utc": info.get("started_utc"), "n_nonfinite_pred": int((~np.isfinite(y_pred)).sum())}
    preds = pd.DataFrame({"target_row": test, "created_at": data.created_at[test], "y_true": y_true,
                          "y_pred": y_pred, "is_change_row": change, "is_grid_first_row": grid_first})
    return metrics, pd.DataFrame([run]), preds[PRED_COLS]


# ---------------------------------------------------------------------------------------------
# Persistence and the 18-step diagnostic (no training)
# ---------------------------------------------------------------------------------------------

def run_persistence(cfg: dict, data: Data, out_dir: Path, hw: dict) -> None:
    """Prediction for target i is weight[i - 1], in grams, on every split's test targets (S2: once
    per seed's test set). Always recomputed; replaces the old persistence rows wholesale."""
    metric_parts, run_parts = [], []
    for name in SPLIT_ORDER:
        scfg = cfg["splits"][name]
        for seed in (cfg["seeds"] if scfg["kind"] == "random" else [None]):
            t0 = time.time()
            part = make_partition(name, scfg, data.N, seed)
            y_pred = data.weight[part.test - 1]
            info = {"predict_seconds": time.time() - t0, "wall_seconds": time.time() - t0,
                    "started_utc": utc_now()}
            m, r, p = build_records(data, part, NO_INPUT_SET, "persistence", seed, y_pred, info, hw)
            write_predictions(p, pred_path(out_dir, name, NO_INPUT_SET, "persistence", seed))
            metric_parts.append(m)
            run_parts.append(r)
            row = m.set_index("eval_set").loc["all"]
            log(f"persistence {name} seed={seed}: n={int(row.n)} MAE={row.mae:.5g} g "
                f"RMSE={row.rmse:.5g} g MAPE={row.mape:.5g} %")
    replace_rows(out_dir / "metrics.csv", pd.concat(metric_parts), ["model"], METRIC_COLS)
    replace_rows(out_dir / "runs.csv", pd.concat(run_parts), ["model"], RUN_COLS)


def forecast18_persistence(cfg: dict, data: Data, out_dir: Path) -> None:
    """Forecast rows i .. i + H - 1 with weight[i - 1] for every test origin i with i + H - 1 < N;
    pooled over origins and steps, and per step h = 1..H."""
    H = cfg["forecast18"]["horizon"]
    pooled, by_step = [], []
    for name in cfg["forecast18"]["splits"]:
        part = make_partition(name, cfg["splits"][name], data.N, None)
        origins = part.test[part.test + H - 1 < data.N]
        if part.row_bounds is not None:
            lo, hi = part.row_bounds["test"]
            assert origins.min() - 1 >= lo and origins.max() + H - 1 < hi
        y = data.weight[origins[:, None] + np.arange(H)[None, :]]
        p = np.broadcast_to(data.weight[origins - 1][:, None], y.shape)
        pooled.append({"split": name, "horizon": H, "n_origins": int(len(origins)), "n_obs": int(y.size),
                       **{k: v for k, v in metric_values(y.ravel(), p.ravel()).items()
                          if k in ("mae", "rmse", "mape")}})
        for h in range(H):
            mv = metric_values(y[:, h], p[:, h])
            by_step.append({"split": name, "step": h + 1, "n": mv["n"], "mae": mv["mae"],
                            "rmse": mv["rmse"], "mape": mv["mape"]})
        log(f"forecast18 persistence {name}: origins={len(origins)} MAE={pooled[-1]['mae']:.5g} g "
            f"RMSE={pooled[-1]['rmse']:.5g} g MAPE={pooled[-1]['mape']:.5g} %")
    pd.DataFrame(pooled).to_csv(out_dir / "forecast18_persistence.csv", index=False)
    pd.DataFrame(by_step).to_csv(out_dir / "forecast18_persistence_by_step.csv", index=False)


# ---------------------------------------------------------------------------------------------
# CNN-LSTM
# ---------------------------------------------------------------------------------------------

def window_dataset(X_s: np.ndarray, y_s: np.ndarray, targets: np.ndarray, L: int, batch_size: int,
                   shuffle: bool, seed: int, with_y: bool = True):
    """Lazy windows: a tf.data pipeline over target-row indices that gathers rows [i - L, i) from the
    scaled (N, F) array for each batch. At most one batch of windows exists at a time."""
    X_t, y_t = tf.constant(X_s), tf.constant(y_s)
    offsets = tf.range(-L, 0, dtype=tf.int64)
    ds = tf.data.Dataset.from_tensor_slices(np.asarray(targets, dtype=np.int64))
    if shuffle:
        ds = ds.shuffle(len(targets), seed=seed, reshuffle_each_iteration=True)

    def gather(idx):
        x = tf.gather(X_t, idx[:, None] + offsets[None, :])
        return (x, tf.gather(y_t, idx)) if with_y else x

    return ds.batch(batch_size).map(gather).prefetch(1)


def build_model(L: int, F: int, mcfg: dict) -> keras.Model:
    """Input(L, F) -> Conv1D -> MaxPooling1D -> LSTM -> LSTM -> Flatten -> Dense(1); Adam, MSE."""
    inp = keras.Input(shape=(L, F))
    x = layers.Conv1D(mcfg["conv_filters"], mcfg["conv_kernel"], padding="same", activation="relu")(inp)
    x = layers.MaxPooling1D(mcfg["pool_size"])(x)
    x = layers.LSTM(mcfg["lstm_units"][0], return_sequences=True)(x)
    x = layers.LSTM(mcfg["lstm_units"][1], return_sequences=True)(x)
    x = layers.Flatten()(x)
    model = keras.Model(inp, layers.Dense(1)(x))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=mcfg["learning_rate"]), loss="mse",
                  jit_compile=mcfg["jit_compile"])
    return model


class EpochLog(keras.callbacks.Callback):
    """One log line per epoch: losses (standardized target), seconds, peak RSS."""

    def on_epoch_begin(self, epoch, logs=None):
        self._t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        log(f"    epoch {epoch + 1:>2}  loss={logs['loss']:.6f}  val_loss={logs['val_loss']:.6f}  "
            f"{time.time() - self._t0:.1f}s  rss_max={rss_mb():.0f}MB")


def run_cnn_lstm(cfg: dict, data: Data, out_dir: Path, hw: dict, split: str, input_set: str,
                 seed: int) -> None:
    """Train one CNN-LSTM, predict the test targets, write runs.csv and metrics.csv rows and, last,
    the predictions file (its existence marks the run as done)."""
    path = pred_path(out_dir, split, input_set, "cnn_lstm", seed)
    if path.exists():
        log(f"skip {split} {input_set} seed={seed}: {path.name} exists")
        return
    t_wall, started = time.time(), utc_now()
    scfg, mcfg = cfg["splits"][split], cfg["model"]
    L, columns, bs = scfg["lookback"], cfg["input_sets"][input_set], mcfg["batch_size"]
    part = make_partition(split, scfg, data.N, seed)
    X_s, y_s, y_mean, y_scale = standardize(data, columns, part.fit_lo, part.fit_hi)
    log(f"run {split} {input_set} seed={seed}: L={L} F={len(columns)} train={len(part.train)} "
        f"val={len(part.val)} test={len(part.test)} load1={os.getloadavg()[0]:.2f}")

    keras.backend.clear_session()
    gc.collect()
    keras.utils.set_random_seed(seed)
    model = build_model(L, len(columns), mcfg)
    train_ds = window_dataset(X_s, y_s, part.train, L, bs, True, seed)
    val_ds = window_dataset(X_s, y_s, part.val, L, bs, False, seed)
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=mcfg["patience"],
                                            restore_best_weights=True)
    t0 = time.time()
    hist = model.fit(train_ds, validation_data=val_ds, epochs=mcfg["max_epochs"],
                     callbacks=[stopper, EpochLog()], shuffle=False, verbose=0)
    train_seconds = time.time() - t0

    t0 = time.time()
    scaled = model.predict(window_dataset(X_s, y_s, part.test, L, bs, False, seed, with_y=False), verbose=0)
    y_pred = scaled[:, 0].astype(np.float64) * y_scale + y_mean
    predict_seconds = time.time() - t0

    info = {"n_train": len(part.train), "n_val": len(part.val), "epochs_run": len(hist.history["loss"]),
            "best_epoch": int(stopper.best_epoch) + 1,
            "best_val_loss": float(stopper.best) if stopper.best is not None else np.nan,
            "train_seconds": train_seconds, "predict_seconds": predict_seconds,
            "wall_seconds": time.time() - t_wall, "started_utc": started}
    metrics, runs, preds = build_records(data, part, input_set, "cnn_lstm", seed, y_pred, info, hw)
    key = ["model", "split", "input_set", "seed"]
    replace_rows(out_dir / "runs.csv", runs, key, RUN_COLS)
    replace_rows(out_dir / "metrics.csv", metrics, key + ["eval_set"], METRIC_COLS)
    write_predictions(preds, path)

    row = metrics.set_index("eval_set").loc["all"]
    log(f"done {split} {input_set} seed={seed}: epochs={info['epochs_run']} best_epoch={info['best_epoch']} "
        f"best_val_loss={info['best_val_loss']:.6f} train={train_seconds:.0f}s predict={predict_seconds:.1f}s "
        f"| test all: MAE={row.mae:.5g} g RMSE={row.rmse:.5g} g MAPE={row.mape:.5g} % "
        f"nonfinite_pred={runs.n_nonfinite_pred.iloc[0]}")
    keras.backend.clear_session()
    gc.collect()


# ---------------------------------------------------------------------------------------------
# Epoch timing and the projected-time guard
# ---------------------------------------------------------------------------------------------

def _timed(fn) -> float:
    t0 = time.time()
    fn()
    return time.time() - t0


def time_epochs(cfg: dict, data: Data, out_dir: Path, hw: dict) -> dict:
    """Time a first and a second training epoch for the S0 shape (L=134) and the S1/S2 shape (L=36),
    plus forward passes sized like the real validation and test sets. Uses training windows only, so
    no validation or test row is touched. Projects the whole grid assuming every run hits the epoch
    cap, which is the upper bound (early stopping can only shorten it). Writes epoch_timing.json."""
    mcfg = cfg["model"]
    bs, E = mcfg["batch_size"], mcfg["max_epochs"]
    shapes = {}
    for name in ("S0_notebook", "S1_chrono"):
        scfg = cfg["splits"][name]
        L, columns = scfg["lookback"], cfg["input_sets"]["notebook"]
        part = make_partition(name, scfg, data.N, None)
        X_s, y_s, _, _ = standardize(data, columns, part.fit_lo, part.fit_hi)
        keras.backend.clear_session()
        gc.collect()
        keras.utils.set_random_seed(0)
        model = build_model(L, len(columns), mcfg)
        train_ds = window_dataset(X_s, y_s, part.train, L, bs, True, 0)
        first = _timed(lambda: model.fit(train_ds, epochs=1, shuffle=False, verbose=0))
        steady = _timed(lambda: model.fit(train_ds, epochs=1, shuffle=False, verbose=0))
        n_val, n_test = len(part.val), len(part.test)
        val_pass = _timed(lambda: model.evaluate(
            window_dataset(X_s, y_s, part.train[:n_val], L, bs, False, 0), verbose=0))
        predict_pass = _timed(lambda: model.predict(
            window_dataset(X_s, y_s, part.train[:n_test], L, bs, False, 0, with_y=False), verbose=0))
        shapes[name] = {"lookback": L, "n_features": len(columns), "n_train": len(part.train),
                        "batches_per_epoch": int(np.ceil(len(part.train) / bs)),
                        "first_epoch_seconds": first, "steady_epoch_seconds": steady,
                        "n_val_timed": n_val, "val_pass_seconds": val_pass,
                        "n_test_timed": n_test, "predict_pass_seconds": predict_pass,
                        "rss_max_mb": rss_mb()}
        log(f"timing {name}: L={L} first epoch {first:.1f}s, second epoch {steady:.1f}s, "
            f"val-sized pass {val_pass:.1f}s, test-sized predict {predict_pass:.1f}s")
        keras.backend.clear_session()
        gc.collect()

    per_split, total = {}, 0.0
    for name in cfg["run_order"]:
        scfg = cfg["splits"][name]
        ref = shapes["S0_notebook" if name == "S0_notebook" else "S1_chrono"]
        seed0 = cfg["seeds"][0]
        part = make_partition(name, scfg, data.N, seed0 if scfg["kind"] == "random" else None)
        one_run = ((ref["first_epoch_seconds"] - ref["steady_epoch_seconds"])
                   + E * ref["steady_epoch_seconds"] * len(part.train) / ref["n_train"]
                   + E * ref["val_pass_seconds"] * len(part.val) / ref["n_val_timed"]
                   + ref["predict_pass_seconds"] * len(part.test) / ref["n_test_timed"])
        n_runs = len(cfg["seeds"]) * len(scfg["input_sets"])
        per_split[name] = {"n_runs": n_runs, "seconds_per_run_at_epoch_cap": one_run,
                           "seconds_total": one_run * n_runs}
        total += one_run * n_runs
    timing = {"measured_utc": utc_now(), "hardware": hw, "loadavg_at_end": list(os.getloadavg()),
              "epochs_assumed_per_run": E, "shapes": shapes,
              "projection": {"basis": "every run trains max_epochs epochs (upper bound); S2 timed with the "
                                      "S1 shape (same lookback, train size within 0.01 %)",
                             "per_split": per_split, "total_seconds": total, "total_hours": total / 3600,
                             "limit_hours": cfg["max_projected_hours"],
                             "exceeds_limit": bool(total / 3600 > cfg["max_projected_hours"])}}
    (out_dir / "epoch_timing.json").write_text(json.dumps(timing, indent=2) + "\n")
    log(f"projection: {total / 3600:.2f} h for 21 runs at {E} epochs each "
        f"(limit {cfg['max_projected_hours']} h)")
    return timing


# ---------------------------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------------------------

def summarize_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    """Mean and sample SD (ddof=1) over seeds per split, input_set, model, eval_set. NaN in any run
    makes the mean NaN (a diverged run is not averaged away); SD is blank for a single run."""
    rows = []
    for (split, input_set, model, eval_set), g in metrics.groupby(
            ["split", "input_set", "model", "eval_set"], sort=False):
        row = {"split": split, "input_set": input_set, "model": model, "eval_set": eval_set,
               "n_seeds": int(len(g)), "n_min": int(g["n"].min()), "n_max": int(g["n"].max())}
        for m in ("mae", "rmse", "mape", "r2"):
            v = g[m].to_numpy(dtype=np.float64)
            row[f"{m}_mean"] = float(np.mean(v))
            row[f"{m}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else np.nan
        rows.append(row)
    return _sort_for_reading(pd.DataFrame(rows))


def _pm(mean: float, sd: float, fmt: str = ".4g") -> str:
    if pd.isna(mean):
        return "NA"
    return f"{mean:{fmt}}" if pd.isna(sd) else f"{mean:{fmt}} ± {sd:{fmt}}"


def _md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def write_summaries(cfg: dict, out_dir: Path) -> None:
    """metrics_summary.csv and SUMMARY.md, generated from the other result files."""
    metrics = pd.read_csv(out_dir / "metrics.csv")
    summary = summarize_metrics(metrics)
    summary.to_csv(out_dir / "metrics_summary.csv", index=False)

    pers = summary[summary.model == "persistence"].set_index(["split", "eval_set"])
    head = []
    for _, r in summary[summary.model == "cnn_lstm"].iterrows():
        p = pers.loc[(r.split, r.eval_set)] if (r.split, r.eval_set) in pers.index else None
        n = str(r.n_min) if r.n_min == r.n_max else f"{r.n_min}-{r.n_max}"
        head.append([r.split, r.input_set, r.eval_set, n, r.n_seeds,
                     _pm(r.mae_mean, r.mae_sd), _pm(r.rmse_mean, r.rmse_sd),
                     _pm(r.mape_mean, r.mape_sd), _pm(r.r2_mean, r.r2_sd, ".4f"),
                     "NA" if p is None else _pm(p.mae_mean, p.mae_sd),
                     "NA" if p is None else _pm(p.rmse_mean, p.rmse_sd),
                     "NA" if p is None else _pm(p.mape_mean, p.mape_sd)])
    md = ["# Phase 2 summary", "",
          "Generated by `src/phase2_original.py` from the other files in this directory. Do not edit.", "",
          "## Single-step results on test targets", "",
          _md(["split", "input_set", "eval_set", "n", "seeds", "cnn_lstm MAE (g)", "cnn_lstm RMSE (g)",
               "cnn_lstm MAPE (%)", "cnn_lstm R2", "persistence MAE (g)", "persistence RMSE (g)",
               "persistence MAPE (%)"], head), "",
          "Columns: `n` is the number of test targets in the evaluation set (a range when it differs "
          "between seeds). cnn_lstm entries are mean ± sample SD (ddof 1) over the listed seeds. "
          "Persistence predicts target i with weight[i-1] and is scored on the same test targets "
          "(S2_random: on each seed's own test set, so its entry is also a mean ± SD over seeds; "
          "S0_notebook and S1_chrono have one value, shown without ±). `all` is every test target, "
          "`change_rows` the targets with weight[i] != weight[i-1], `grid_first_rows` the first test "
          "target on each date of the 14-day grid from 2021-06-19. R2 is NA when n < 3. "
          "Source: `metrics_summary.csv` (from `metrics.csv`).", ""]

    runs = pd.read_csv(out_dir / "runs.csv")
    runs = runs[runs.model == "cnn_lstm"]
    if len(runs):
        rrows = [[r.split, r.input_set, int(r.seed), int(r.n_train), int(r.n_val), int(r.n_test),
                  int(r.epochs_run), int(r.best_epoch), f"{r.best_val_loss:.5g}",
                  f"{r.train_seconds:.0f}", f"{r.predict_seconds:.1f}", int(r.n_nonfinite_pred)]
                 for r in runs.itertuples()]
        md += ["## CNN-LSTM runs", "",
               _md(["split", "input_set", "seed", "n_train", "n_val", "n_test", "epochs_run", "best_epoch",
                    "best_val_loss (standardized MSE)", "train_seconds", "predict_seconds",
                    "nonfinite_pred"], rrows), "",
               "Columns: `best_epoch` is 1-based; `train_seconds` covers `fit` including validation "
               "passes; hardware and software versions are in `runs.csv`. Source: `runs.csv`.", ""]

    f18, f18s = out_dir / "forecast18_persistence.csv", out_dir / "forecast18_persistence_by_step.csv"
    if f18.exists() and f18s.exists():
        pooled, steps = pd.read_csv(f18), pd.read_csv(f18s)
        md += ["## 18-step persistence forecast (forecast rows i .. i+17 with weight[i-1])", "",
               _md(["split", "origins", "observations", "MAE (g)", "RMSE (g)", "MAPE (%)"],
                   [[r.split, r.n_origins, r.n_obs, f"{r.mae:.5g}", f"{r.rmse:.5g}", f"{r.mape:.5g}"]
                    for r in pooled.itertuples()]), ""]
        for split in pooled["split"]:
            s = steps[steps.split == split]
            md += [f"Per step, {split}:", "",
                   _md(["step h", "n", "MAE (g)", "RMSE (g)", "MAPE (%)"],
                       [[int(r.step), int(r.n), f"{r.mae:.5g}", f"{r.rmse:.5g}", f"{r.mape:.5g}"]
                        for r in s.itertuples()]), ""]
        ref = cfg["manuscript_reported"]
        md += ["Values reported in the manuscript, copied from the task file for comparison (not computed "
               "here):", "",
               _md(["quantity", "MAE (g)", "RMSE (g)", "MAPE (%)"],
                   [["single-step CNN-LSTM, pond 2", *ref["single_step_cnn_lstm"].values()],
                    ["18-step TFT", *ref["forecast18_tft"].values()],
                    ["18-step CNN-LSTM", *ref["forecast18_cnn_lstm"].values()]]), "",
               "Columns: pooled rows average over all origins and steps h = 1..18; `n` in the per-step "
               "tables is the number of origins. Source: `forecast18_persistence.csv`, "
               "`forecast18_persistence_by_step.csv`.", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(md))


def check_grid_complete(cfg: dict, out_dir: Path) -> list[str]:
    """Names of grid cells (split, input set, seed) that have no cnn_lstm rows in metrics.csv."""
    m = pd.read_csv(out_dir / "metrics.csv")
    m = m[m.model == "cnn_lstm"]
    have = set(zip(m.split, m.input_set, m.seed.astype(int), m.eval_set))
    return [f"{s}/{i}/seed{sd}" for s, scfg in cfg["splits"].items() for i in scfg["input_sets"]
            for sd in cfg["seeds"] if any((s, i, sd, e) not in have for e in EVAL_SETS)]


# ---------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--time-epoch", action="store_true", help="only time epochs and project the grid")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load((cfg_path if cfg_path.is_absolute() else Path.cwd() / cfg_path).read_text())
    out_dir = REPO / cfg["paths"]["results_dir"]
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    hw = hardware_info()
    log(f"hardware: {hw}")
    log(f"python {sys.version.split()[0]}, numpy {np.__version__}, pandas {pd.__version__}, "
        f"keras backend {keras.backend.backend()}, load1={os.getloadavg()[0]:.2f}")
    data = load_data(cfg)
    log(f"data: N={data.N} rows after dropna ({data.checks['rows_dropped']} dropped), "
        f"dropna identical with and without Population/Fish_Length: "
        f"{data.checks['dropna_same_rows_with_and_without_population_and_length']}")

    if args.time_epoch:
        time_epochs(cfg, data, out_dir, hw)
        return 0

    checks = build_split_checks(cfg, data)
    (out_dir / "split_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    log("split checks passed; split_checks.json written")

    run_persistence(cfg, data, out_dir, hw)
    forecast18_persistence(cfg, data, out_dir)
    write_summaries(cfg, out_dir)

    timing_path = out_dir / "epoch_timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else time_epochs(cfg, data, out_dir, hw)
    proj = timing["projection"]
    log(f"projected grid time {proj['total_hours']:.2f} h (limit {cfg['max_projected_hours']} h)")
    if proj["total_hours"] > cfg["max_projected_hours"]:
        log("STOP: projected time exceeds the limit; persistence results are written, no CNN-LSTM run started")
        return 3

    for split in cfg["run_order"]:
        scfg = cfg["splits"][split]
        for seed in cfg["seeds"]:
            for input_set in scfg["input_sets"]:
                run_cnn_lstm(cfg, data, out_dir, hw, split, input_set, seed)
    write_summaries(cfg, out_dir)

    missing = check_grid_complete(cfg, out_dir)
    if missing:
        log(f"INCOMPLETE: no metrics rows for {missing}")
        return 4
    log("grid complete: every cell has metrics rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
