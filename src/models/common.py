"""Shared infrastructure of the sequence models (plan 5.2, 6.3, 7.2, 13).

Data path: `DailyStore` holds the daily sensor aggregates of every pond as arrays;
`DailyStore.visible_window` applies the causal gap rule of plan 2.7 (an interpolated
sensor-day whose right anchor is on or after the reference date is missing); `build_sampleset`
cuts one sequence per T1 or T2 sample with that rule (T1: day 0 to d_e - 1, left-padded with
zeros to the longest T1 sequence; T2: the 28 days before the origin date), optionally from
another pond's file (the donor of the permutation test, plan 8.2); `fit_scalers` computes
standardization statistics from the training samples only; `make_batch` turns samples into
model inputs (24 standardized aggregates plus 6 sensor masks per day, statics, padding mask,
standardized target).

Training path: `train_network` is the one shared loop (AdamW, MSE on the standardized
target, at most `max_epochs` epochs, early stopping on the validation MSE with the best
weights restored, or a fixed number of epochs without validation). It records, for every
epoch, the training MSE, the validation MSE and the seconds, which the caller stores as the
learning curves. `DeepRegressor` wraps all of it behind the `Learner` interface of
src/baselines.py.

Process setup: `configure_torch` makes training single-threaded and deterministic;
`set_seed` seeds random, numpy and torch.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

from src.baselines import FitInfo, Learner, grams_from_log

_THREAD_LIMIT = None  # keeps the threadpoolctl limit alive for the life of the process


def configure_torch() -> None:
    """One thread, deterministic algorithms where available (call once per process)."""
    global _THREAD_LIMIT
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already fixed in this process
    torch.use_deterministic_algorithms(True, warn_only=True)
    _THREAD_LIMIT = threadpool_limits(limits=1)


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------------------
# Padding helpers used by the architectures
# ----------------------------------------------------------------------------------------

def last_real_index(pad: torch.Tensor) -> torch.Tensor:
    """Per row, the index of the last step that is not padding. `pad` is [B, L], True = padded."""
    steps = torch.arange(pad.shape[1], device=pad.device).expand_as(pad)
    return torch.where(pad, torch.full_like(steps, -1), steps).amax(dim=1).clamp(min=0)


def gather_last(seq: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
    """Pick the last real step of `seq` [B, L, D] for every row; returns [B, D]."""
    return seq[torch.arange(seq.shape[0], device=seq.device), last_real_index(pad)]


# ----------------------------------------------------------------------------------------
# Daily store and per-sample sequences
# ----------------------------------------------------------------------------------------

class DailyStore:
    """Daily sensor aggregates as arrays, built from the rows of daily_wq.csv.

    `vals[pond_index, day, sensor, stat]` holds the aggregates (NaN where the sensor-day has
    no value) and `valid[pond_index, day, sensor]` is True where it has one (status observed
    or interpolated). `right_anchor[pond_index, day, sensor]` is, for an interpolated
    sensor-day, the day index of the observed day that closes its gap, and -1 for every other
    sensor-day. `day` counts from the first date in the table (`start`).
    """

    def __init__(self, daily: pd.DataFrame, sensors: list[str], stats: list[str]):
        needed = {"pond", "date", "sensor", "status", "right_anchor_date", *stats}
        if needed - set(daily.columns):
            raise ValueError(f"daily table lacks columns {sorted(needed - set(daily.columns))}")
        unknown = set(daily["sensor"]) - set(sensors)
        if unknown:
            raise ValueError(f"daily table has unknown sensors {sorted(unknown)}")
        bad_status = set(daily["status"]) - {"observed", "interpolated", "missing"}
        if bad_status:
            raise ValueError(f"daily table has unknown status values {sorted(bad_status)}")
        if daily.duplicated(["pond", "date", "sensor"]).any():
            raise ValueError("daily table has more than one row for a pond, date and sensor")
        dates = pd.to_datetime(daily["date"])
        self.sensors, self.stats = list(sensors), list(stats)
        self.start = dates.min().normalize()
        self.n_days = int((dates.max().normalize() - self.start).days) + 1
        self.ponds = sorted(int(p) for p in daily["pond"].unique())
        self.pond_index = {p: i for i, p in enumerate(self.ponds)}

        values = daily[stats].to_numpy(dtype=float)
        present = (daily["status"] != "missing").to_numpy()
        if not np.isfinite(values[present]).all():
            raise ValueError("a sensor-day with status observed or interpolated has a non-finite statistic")
        if np.isfinite(values[~present]).any():
            raise ValueError("a sensor-day with status missing carries values")
        interpolated = (daily["status"] == "interpolated").to_numpy()
        anchors = pd.to_datetime(daily["right_anchor_date"])
        if anchors[interpolated].isna().any():
            raise ValueError("an interpolated sensor-day has no right anchor date")
        if anchors[~interpolated].notna().any():
            raise ValueError("a sensor-day that is not interpolated has a right anchor date")
        if (anchors[interpolated] <= dates[interpolated]).any():
            raise ValueError("a right anchor date is not after the sensor-day it closes")
        pond_all = daily["pond"].map(self.pond_index).to_numpy()
        day_all = (dates - self.start).dt.days.to_numpy()
        sensor_all = daily["sensor"].map({s: i for i, s in enumerate(sensors)}).to_numpy()
        self.vals = np.full((len(self.ponds), self.n_days, len(sensors), len(stats)), np.nan)
        self.valid = np.zeros((len(self.ponds), self.n_days, len(sensors)), dtype=bool)
        self.right_anchor = np.full((len(self.ponds), self.n_days, len(sensors)), -1, dtype=int)
        self.vals[pond_all[present], day_all[present], sensor_all[present]] = values[present]
        self.valid[pond_all[present], day_all[present], sensor_all[present]] = True
        anchor_day = (anchors[interpolated] - self.start).dt.days.to_numpy()
        self.right_anchor[pond_all[interpolated], day_all[interpolated], sensor_all[interpolated]] = anchor_day

    def visible_window(self, pond_index: int, first_day: int, reference_day: int):
        """Aggregates and validity of the days `first_day` .. `reference_day - 1` of one pond.

        The causal gap rule of plan 2.7: an interpolated sensor-day whose right anchor is on or
        after `reference_day` (the day index of the T1 event date or the T2 origin date) is
        missing for that sample, because its value was interpolated from an observation the
        sample could not have had. Such entries come back invalid and NaN. Returns
        `vals[day, sensor, stat]` and `valid[day, sensor]`, both copies. The one function
        for every sequence: samples cut from their own file and from a donor's file alike.
        """
        window = slice(first_day, reference_day)
        valid = self.valid[pond_index, window] & ~(self.right_anchor[pond_index, window] >= reference_day)
        vals = np.where(valid[..., None], self.vals[pond_index, window], np.nan)
        return vals, valid


@dataclass
class SampleSet:
    """Raw (unscaled) sequences, statics and targets of a set of T1 or T2 samples.

    Row i belongs to the table row labelled `ids[i]`. `vals` is [N, L, sensors, stats],
    `valid` is [N, L, sensors], `pad` is [N, L] (True on the left padding of T1), `day` is
    the store day index of every step (-1 on padding), `statics` is [N, n_static] and `y` the
    log-scale target. `pond_idx` is the sample's own pond and `data_pond_idx` the pond whose
    daily file the sequence was cut from (the same, except in the permutation test).
    """

    task: str
    store: DailyStore
    ids: np.ndarray
    pond_idx: np.ndarray
    data_pond_idx: np.ndarray
    vals: np.ndarray
    valid: np.ndarray
    pad: np.ndarray
    day: np.ndarray
    statics: np.ndarray
    y: np.ndarray
    row_of: dict

    @property
    def length(self) -> int:
        return self.vals.shape[1]

    def rows(self, ids) -> np.ndarray:
        """Row positions of the given table labels; raises KeyError for an unknown label."""
        try:
            return np.fromiter((self.row_of[int(i)] for i in ids), dtype=int, count=len(ids))
        except KeyError as exc:
            raise KeyError(f"sample {exc} is not in the sample set") from exc


def build_sampleset(task: str, table: pd.DataFrame, store: DailyStore, cfg: dict,
                    donors=None) -> SampleSet:
    """Cut one sequence per row of `table` (T1 or T2 rows, labelled by the table index).

    T1: days 0 (2021-06-19) to d_e - 1, so `t` real steps, left-padded with zeros (masks 0)
    to the longest sequence in `table`. T2: the `t2_window_days` days before the origin date,
    no padding. The causal gap rule of plan 2.7 is applied (`DailyStore.visible_window`).
    `donors`, if given, is one pond id per row: the sequence is cut from that pond's daily
    file for the same dates instead of the sample's own (statics and targets stay the
    sample's own). Raises ValueError when a window falls outside the daily table or when `t`
    disagrees with the date.
    """
    design = cfg["design"]
    n = len(table)
    if task == "T1":
        dates = pd.to_datetime(table["date"])
        t = table["t"].to_numpy(dtype=int)
        if not (((dates - pd.Timestamp(design["day0"])).dt.days.to_numpy()) == t).all():
            raise ValueError("T1 column t does not equal the days since day 0 of the date")
        lengths = t
        length = int(lengths.max())
    else:
        dates = pd.to_datetime(table["origin_date"])
        lengths = np.full(n, int(design["t2_window_days"]))
        length = int(design["t2_window_days"])
    ends = (dates - store.start).dt.days.to_numpy()
    starts = ends - lengths
    if (starts < 0).any() or (ends > store.n_days).any():
        raise ValueError("a sequence window falls outside the daily table")
    n_sensors, n_stats = len(store.sensors), len(store.stats)
    vals = np.full((n, length, n_sensors, n_stats), np.nan)
    valid = np.zeros((n, length, n_sensors), dtype=bool)
    pad = np.ones((n, length), dtype=bool)
    day = np.full((n, length), -1, dtype=int)
    pond = table["pond"].to_numpy(dtype=int)
    pond_idx = np.array([store.pond_index[p] for p in pond], dtype=int)
    source = pond if donors is None else np.asarray(donors, dtype=int)
    if source.shape != pond.shape:
        raise ValueError("donors needs one pond id per table row")
    data_pond_idx = np.array([store.pond_index[p] for p in source], dtype=int)
    for i in range(n):
        keep = slice(length - lengths[i], length)
        vals[i, keep], valid[i, keep] = store.visible_window(data_pond_idx[i], starts[i], ends[i])
        pad[i, keep] = False
        day[i, keep] = np.arange(starts[i], ends[i])
    statics = table[cfg["statics"][task]].to_numpy(dtype=float)
    y = table["log_w" if task == "T1" else "y"].to_numpy(dtype=float)
    ids = table.index.to_numpy()
    return SampleSet(task, store, ids, pond_idx, data_pond_idx, vals, valid, pad, day, statics, y,
                     {int(label): row for row, label in enumerate(ids)})


# ----------------------------------------------------------------------------------------
# Standardization (training samples only)
# ----------------------------------------------------------------------------------------

@dataclass
class Scalers:
    """Statistics fitted on the training samples: channels [sensors, stats], statics, target."""

    ch_mean: np.ndarray
    ch_std: np.ndarray
    st_mean: np.ndarray
    st_std: np.ndarray
    y_mean: float
    y_std: float


def _safe_std(values: np.ndarray) -> float:
    std = float(np.std(values)) if len(values) else 1.0
    return std if std > 1e-12 else 1.0


def fit_scalers(ss: SampleSet, ids) -> Scalers:
    """Mean and SD (ddof 0) from the training samples `ids` only.

    Channel statistics use every sensor-day that a training sample's sequence can see, counted
    once per pond file, day and sensor, so overlapping windows do not weight a day twice. A
    channel with no value or no spread gets mean 0 (no value) and SD 1.
    """
    rows = ss.rows(ids)
    seen = np.zeros(ss.store.valid.shape, dtype=bool)  # [pond file, day, sensor]
    for i in rows:
        real = ~ss.pad[i]
        seen[ss.data_pond_idx[i], ss.day[i][real]] |= ss.valid[i][real]
    n_sensors, n_stats = ss.store.vals.shape[2], ss.store.vals.shape[3]
    ch_mean = np.zeros((n_sensors, n_stats))
    ch_std = np.ones((n_sensors, n_stats))
    for s in range(n_sensors):
        x = ss.store.vals[:, :, s, :][seen[:, :, s]]  # [cells, stats]
        for k in range(n_stats):
            if len(x):
                ch_mean[s, k] = float(x[:, k].mean())
                ch_std[s, k] = _safe_std(x[:, k])
    statics = ss.statics[rows]
    y = ss.y[rows]
    return Scalers(
        ch_mean, ch_std,
        statics.mean(axis=0), np.array([_safe_std(statics[:, j]) for j in range(statics.shape[1])]),
        float(y.mean()), _safe_std(y),
    )


@dataclass
class Batch:
    """Model inputs: x [n, L, sensors * stats + sensors], static [n, S], pad [n, L], y [n]."""

    x: torch.Tensor
    static: torch.Tensor
    pad: torch.Tensor
    y: torch.Tensor


def make_batch(ss: SampleSet, ids, sc: Scalers) -> Batch:
    """Standardize the samples `ids` with `sc`; missing values and padding become 0, masks 1/0."""
    rows = ss.rows(ids)
    vals, valid = ss.vals[rows], ss.valid[rows]
    z = np.where(valid[..., None], (vals - sc.ch_mean) / sc.ch_std, 0.0)
    n, length = z.shape[0], z.shape[1]
    x = np.concatenate([z.reshape(n, length, -1), valid.astype(float)], axis=-1)
    static = (ss.statics[rows] - sc.st_mean) / sc.st_std
    y = (ss.y[rows] - sc.y_mean) / sc.y_std
    return Batch(
        torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(static, dtype=torch.float32),
        torch.as_tensor(ss.pad[rows]), torch.as_tensor(y, dtype=torch.float32),
    )


# ----------------------------------------------------------------------------------------
# The shared training loop
# ----------------------------------------------------------------------------------------

@dataclass
class TrainResult:
    """Outcome of `train_network`; `curves` rows are (epoch, train_mse, val_mse, seconds)."""

    best_epoch: int
    epochs_run: int
    best_val_mse: float
    curves: list[tuple[int, float, float, float]]


def train_network(
    net: torch.nn.Module, train: Batch, val: Batch | None, *, lr: float, weight_decay: float,
    batch_size: int, max_epochs: int, patience: int, fixed_epochs: int | None, seed: int,
) -> TrainResult:
    """Train `net` in place with AdamW on the MSE of the standardized target.

    With `val`: at most `max_epochs` epochs, stop when the validation MSE has not improved
    for `patience` epochs, then restore the weights of the best epoch (`best_epoch` is
    1-based). Without `val`: exactly `fixed_epochs` epochs, no early stopping, the final
    weights are kept and the validation MSE of every epoch is NaN. Batches are drawn from a
    generator seeded with `seed`. `train_mse` is the mean batch loss of the epoch with
    dropout active.
    """
    if (val is None) == (fixed_epochs is None):
        raise ValueError("give a validation set (early stopping) or fixed_epochs (no validation), not both")
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator().manual_seed(seed)
    n = train.y.shape[0]
    total = max_epochs if val is not None else fixed_epochs
    curves: list[tuple[int, float, float, float]] = []
    best_val, best_epoch, best_state = math.inf, 0, None
    for epoch in range(1, total + 1):
        started = time.perf_counter()
        net.train()
        order = torch.randperm(n, generator=generator)
        running = 0.0
        for first in range(0, n, batch_size):
            idx = order[first:first + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(net(train.x[idx], train.static[idx], train.pad[idx]), train.y[idx])
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(idx)
        val_mse = math.nan
        if val is not None:
            net.eval()
            with torch.no_grad():
                val_mse = float(F.mse_loss(net(val.x, val.static, val.pad), val.y))
            if val_mse < best_val:
                best_val, best_epoch = val_mse, epoch
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        curves.append((epoch, running / n, val_mse, time.perf_counter() - started))
        if val is not None and epoch - best_epoch >= patience:
            break
    if best_state is not None:
        net.load_state_dict(best_state)
    else:  # no validation set, or its loss never became finite: keep the final weights
        best_epoch, best_val = len(curves), math.nan
    return TrainResult(best_epoch, len(curves), best_val, curves)


# ----------------------------------------------------------------------------------------
# Learner wrapper
# ----------------------------------------------------------------------------------------

class DeepRegressor(Learner):
    """A sequence model behind the `Learner` interface.

    `fit(train, val)` early-stops on `val`; `fit(train, epochs=n)` trains n epochs on all of
    `train`. Scalers are fitted on `train` inside `fit`. `train`, `val` and prediction frames
    are row subsets of the table the SampleSet was built from (their index labels select the
    samples). `hp` holds the architecture parameters and dropout, lr, weight_decay and
    batch_size.
    """

    def __init__(self, task: str, arch: str, hp: dict, sampleset: SampleSet, budget: dict, seed: int):
        self.task, self.arch, self.hp, self.ss = task, arch, dict(hp), sampleset
        self.max_epochs, self.patience, self.seed = int(budget["max_epochs"]), int(budget["patience"]), seed
        self.scalers: Scalers | None = None
        self.net: torch.nn.Module | None = None

    def _fit(self, train: pd.DataFrame, val: pd.DataFrame | None, epochs: int | None) -> FitInfo:
        from src.models import build_network  # imported here: the architectures import this module

        ids = train.index.to_numpy()
        self.scalers = fit_scalers(self.ss, ids)
        train_batch = make_batch(self.ss, ids, self.scalers)
        val_batch = make_batch(self.ss, val.index.to_numpy(), self.scalers) if val is not None else None
        set_seed(self.seed)
        self.net = build_network(
            self.arch, self.hp, n_static=train_batch.static.shape[1],
            n_sensors=len(self.ss.store.sensors), n_stats=len(self.ss.store.stats),
        )
        result = train_network(
            self.net, train_batch, val_batch, lr=self.hp["lr"], weight_decay=self.hp["weight_decay"],
            batch_size=int(self.hp["batch_size"]), max_epochs=self.max_epochs,
            patience=self.patience, fixed_epochs=epochs, seed=self.seed,
        )
        return FitInfo(epochs_run=result.epochs_run, best_epoch=result.best_epoch,
                       best_val_mse=result.best_val_mse, curves=result.curves)

    def predict_log(self, df: pd.DataFrame) -> np.ndarray:
        batch = make_batch(self.ss, df.index.to_numpy(), self.scalers)
        self.net.eval()
        with torch.no_grad():
            standardized = self.net(batch.x, batch.static, batch.pad).numpy().astype(float)
        return standardized * self.scalers.y_std + self.scalers.y_mean

    def predict_grams(self, df: pd.DataFrame) -> np.ndarray:
        return grams_from_log(self.task, self.predict_log(df), df)
