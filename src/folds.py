"""Fold construction for the Phase 4 validation design (ANALYSIS_PLAN.md sections 4 and 7).

`lopo_folds` gives the outer leave-one-pond-out folds over the cohort ponds, with the stream
partner of the held-out pond removed from training (Gate 1: ponds 1 and 2 share one sensor
stream). `inner_folds` splits the training ponds of one outer fold into the stream-grouped
folds used for tuning. `select_samples` applies the sample rule of plan 4.3. `plan_folds`
turns a table into the complete fold plans of a scheme, as lists of table row labels:
`plan_lopo` for the leave-one-pond-out scheme and `plan_chrono` for the calendar hold-out of
plan 7.4. `suggest_cutoff` applies the plan's rule for the date c of that hold-out.
`load_config` reads configs/phase4.yaml for every Phase 4 script.

Run as a script, this file prints the cut-off date the rule gives and the sample counts of the
chrono scheme. It reads the eligibility columns of the tables and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "phase4.yaml"

SAMPLE_SETS = ("eligible", "all")
SCHEMES = ("lopo", "chrono")


def load_config(path: Path | str | None = None) -> dict:
    """The Phase 4 modelling configuration (configs/phase4.yaml unless `path` is given)."""
    with Path(path or DEFAULT_CONFIG).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass(frozen=True)
class OuterFold:
    """One outer fold: `test_ponds` are scored, `train_ponds` fit, `dropped_ponds` do neither."""

    index: int
    test_ponds: tuple[int, ...]
    train_ponds: tuple[int, ...]
    dropped_ponds: tuple[int, ...]


@dataclass(frozen=True)
class InnerFold:
    """One inner fold of a tuning run: fit on `train_ponds`, validate on `val_ponds`."""

    index: int
    train_ponds: tuple[int, ...]
    val_ponds: tuple[int, ...]


@dataclass(frozen=True)
class InnerPlan:
    """An inner split as table row labels: fit on `train_idx`, validate on `val_idx`."""

    index: int
    train_idx: tuple[int, ...]
    val_idx: tuple[int, ...]
    train_ponds: tuple[int, ...]
    val_ponds: tuple[int, ...]


@dataclass(frozen=True)
class FoldPlan:
    """One outer fold of a scheme as table row labels.

    `train_idx` and `test_idx` select rows of the T1 or T2 table, `inner` are the tuning
    splits of the training rows. `excluded_idx` are rows used for neither (the calendar
    hold-out of T2, plan 7.4). A fold without test samples has `skipped` set to the reason
    and no rows; it is recorded and not run (plan 4.3, 7.1).
    """

    index: int
    label: str
    scheme: str
    test_ponds: tuple[int, ...]
    train_ponds: tuple[int, ...]
    dropped_ponds: tuple[int, ...]
    train_idx: tuple[int, ...] = ()
    test_idx: tuple[int, ...] = ()
    inner: tuple[InnerPlan, ...] = ()
    skipped: str = ""
    excluded_idx: tuple[int, ...] = ()


def sensor_streams(
    ponds: Iterable[int], shared_streams: Sequence[Sequence[int]]
) -> list[tuple[int, ...]]:
    """Group `ponds` into sensor streams, sorted by the smallest pond id of each stream.

    Ponds listed together in `shared_streams` form one stream (restricted to the ponds given);
    every other pond is its own stream.
    """
    remaining = sorted(set(ponds))
    streams: list[tuple[int, ...]] = []
    seen: set[int] = set()
    for pond in remaining:
        if pond in seen:
            continue
        group = {pond}
        for shared in shared_streams:
            if pond in shared:
                group |= set(shared)
        stream = tuple(sorted(group & set(remaining)))
        seen |= set(stream)
        streams.append(stream)
    return sorted(streams, key=lambda stream: stream[0])


def lopo_folds(
    cohort: Sequence[int], shared_streams: Sequence[Sequence[int]]
) -> list[OuterFold]:
    """Outer leave-one-pond-out folds, one per cohort pond in increasing pond order.

    The held-out pond is the test set. Any pond that shares its sensor stream is dropped from
    the training ponds of that fold (plan 7.1 and 2.8).
    """
    ordered = sorted(set(cohort))
    folds = []
    for index, pond in enumerate(ordered):
        stream = next(s for s in sensor_streams(ordered, shared_streams) if pond in s)
        dropped = tuple(p for p in stream if p != pond)
        train = tuple(p for p in ordered if p != pond and p not in dropped)
        folds.append(OuterFold(index, (pond,), train, dropped))
    return folds


def inner_folds(
    train_ponds: Sequence[int],
    shared_streams: Sequence[Sequence[int]],
    n_folds: int = 3,
) -> list[InnerFold]:
    """Stream-grouped inner folds (plan 7.2).

    `train_ponds` are the training ponds that have at least one training sample (the caller
    removes the others, plan 7.2). They are grouped into sensor streams, the streams are
    sorted by their smallest pond id and assigned round-robin to `n_folds` folds. Fold j
    validates on the streams assigned to j and fits on all the others. Raises ValueError when
    there are fewer streams than folds.
    """
    streams = sensor_streams(train_ponds, shared_streams)
    if len(streams) < n_folds:
        raise ValueError(f"{len(streams)} sensor streams cannot fill {n_folds} inner folds")
    folds = []
    for j in range(n_folds):
        val = tuple(sorted(p for i, s in enumerate(streams) if i % n_folds == j for p in s))
        train = tuple(sorted(p for p in train_ponds if p not in val))
        folds.append(InnerFold(j, train, val))
    return folds


def select_samples(table: pd.DataFrame, ponds: Iterable[int], sample_set: str) -> pd.DataFrame:
    """Rows of `table` for `ponds` under the sample rule of plan 4.3.

    `eligible` keeps the rows with `eligible_eval` true (the evaluation set and the training
    set of every model); `all` keeps every row (the secondary time-only table of plan 4.1).
    The row labels of `table` are kept so rows can be traced back.
    """
    if sample_set not in SAMPLE_SETS:
        raise ValueError(f"sample_set must be one of {SAMPLE_SETS}, got {sample_set!r}")
    rows = table[table["pond"].isin(list(ponds))]
    if sample_set == "eligible":
        rows = rows[rows["eligible_eval"].astype(bool)]
    return rows


def sample_date_column(task: str) -> str:
    """The date that places a sample on the calendar: T1 event date, T2 target date (plan 7.4)."""
    return "date" if task == "T1" else "target_date"


def _labels(rows: pd.DataFrame) -> tuple[int, ...]:
    return tuple(int(i) for i in rows.index)


def plan_lopo(task: str, table: pd.DataFrame, sample_set: str, cfg: dict) -> list[FoldPlan]:
    """Outer LOPO folds with their inner splits, as row labels of `table` (plan 7.1, 7.2).

    One plan per cohort pond in pond order. A pond without a sample in `sample_set` has no
    test set: its plan is marked skipped and carries no rows. The inner folds are built over
    the training ponds that have a sample (plan 7.2), grouped by sensor stream.
    """
    design = cfg["design"]
    shared = design["shared_streams"]
    rows = select_samples(table, design["cohort"], sample_set)
    plans = []
    for outer in lopo_folds(design["cohort"], shared):
        label = f"pond{outer.test_ponds[0]:02d}"
        test = rows[rows["pond"].isin(outer.test_ponds)]
        if test.empty:
            plans.append(FoldPlan(outer.index, label, "lopo", outer.test_ponds, outer.train_ponds,
                                  outer.dropped_ponds, skipped="no test samples"))
            continue
        train = rows[rows["pond"].isin(outer.train_ponds)]
        inner = []
        for fold in inner_folds(sorted(int(p) for p in train["pond"].unique()), shared,
                                design["n_inner_folds"]):
            inner.append(InnerPlan(
                fold.index,
                _labels(train[train["pond"].isin(fold.train_ponds)]),
                _labels(train[train["pond"].isin(fold.val_ponds)]),
                fold.train_ponds, fold.val_ponds,
            ))
        plans.append(FoldPlan(outer.index, label, "lopo", outer.test_ponds, outer.train_ponds,
                              outer.dropped_ponds, _labels(train), _labels(test), tuple(inner)))
    return plans


def plan_chrono(task: str, table: pd.DataFrame, sample_set: str, cfg: dict,
                cutoff: str | pd.Timestamp) -> list[FoldPlan]:
    """The calendar hold-out of plan 7.4 as a single outer fold.

    Training: samples dated before `cutoff` (T1 event date, T2 target date), so nothing dated
    `cutoff` or later is in training. Test: T1 events dated `cutoff` or later; T2 samples with
    target on or after `cutoff` and origin on or after `cutoff` minus
    `chrono.test_origin_margin_days`. T2 samples with a late target and an earlier origin are
    used for neither and are listed in `excluded_idx`. Tuning has one inner split: validation
    is the training samples at the latest training date, the inner training set is every
    training sample before that date. All cohort ponds are in both sets, no pond is dropped.
    """
    cohort = sorted(cfg["design"]["cohort"])
    rows = select_samples(table, cohort, sample_set)
    column = sample_date_column(task)
    dates = pd.to_datetime(rows[column])
    cut = pd.Timestamp(cutoff)
    train = rows[dates < cut]
    late = rows[dates >= cut]
    if task == "T2":
        margin = pd.Timedelta(days=int(cfg["chrono"]["test_origin_margin_days"]))
        recent_origin = pd.to_datetime(late["origin_date"]) >= cut - margin
        test, excluded = late[recent_origin], late[~recent_origin]
    else:
        test, excluded = late, late.iloc[0:0]
    if train.empty or test.empty:
        raise ValueError(f"cut-off {cut:%Y-%m-%d} leaves {len(train)} training and {len(test)} test samples")
    train_dates = pd.to_datetime(train[column])
    last = train_dates.max()
    validation, inner_train = train[train_dates == last], train[train_dates < last]
    if inner_train.empty:
        raise ValueError("the training samples have a single date, so no inner split exists")

    def ponds_of(frame: pd.DataFrame) -> tuple[int, ...]:
        return tuple(sorted(int(p) for p in frame["pond"].unique()))

    inner = (InnerPlan(0, _labels(inner_train), _labels(validation), ponds_of(inner_train),
                       ponds_of(validation)),)
    return [FoldPlan(0, "chrono", "chrono", ponds_of(test), ponds_of(train), (), _labels(train),
                     _labels(test), inner, excluded_idx=_labels(excluded))]


def plan_folds(task: str, scheme: str, table: pd.DataFrame, sample_set: str, cfg: dict,
               cutoff: str | pd.Timestamp | None = None) -> list[FoldPlan]:
    """The fold plans of `scheme` (lopo or chrono) for one task table."""
    if scheme == "lopo":
        return plan_lopo(task, table, sample_set, cfg)
    if scheme == "chrono":
        if cutoff is None:
            raise ValueError("the chrono scheme needs a cut-off date")
        return plan_chrono(task, table, sample_set, cfg, cutoff)
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def submission_order(plans: Iterable[FoldPlan]) -> list[FoldPlan]:
    """The order in which the runner starts the outer folds: largest training set first.

    The longest jobs start first, so that with several workers the run ends as early as it
    can; equal sizes keep their fold order. src/projection.py schedules the folds the same way.
    """
    return sorted(plans, key=lambda plan: -len(plan.train_idx))


def suggest_cutoff(t1: pd.DataFrame, min_events: int, min_ponds: int) -> str:
    """The date c of plan 7.4: the latest date with enough eligible T1 events on or after it.

    "Enough" is at least `min_events` eligible events in at least `min_ponds` ponds. Only the
    pond, date and eligibility columns are read. Events lie on grid dates, so the answer is
    the date of an event. Raises ValueError when no date qualifies.
    """
    eligible = t1[t1["eligible_eval"].astype(bool)]
    dates = pd.to_datetime(eligible["date"])
    for candidate in sorted(dates.unique(), reverse=True):
        later = eligible[dates >= candidate]
        if len(later) >= min_events and later["pond"].nunique() >= min_ponds:
            return pd.Timestamp(candidate).strftime("%Y-%m-%d")
    raise ValueError("no date has enough eligible T1 events on or after it")


def main() -> None:
    """Print the cut-off the plan's rule gives and the sample counts it implies."""
    cfg = load_config()
    t1 = pd.read_csv(REPO_ROOT / cfg["paths"]["t1_table"])
    cutoff = suggest_cutoff(t1, cfg["chrono"]["min_events"], cfg["chrono"]["min_ponds"])
    print(f"cut-off from the rule of plan 7.4: {cutoff} (configs/phase4.yaml chrono.cutoff is "
          f"{cfg['chrono']['cutoff']!r})")
    for task, name in (("T1", "t1_table"), ("T2", "t2_table")):
        table = t1 if task == "T1" else pd.read_csv(REPO_ROOT / cfg["paths"][name])
        plan = plan_chrono(task, table, "eligible", cfg, cutoff)[0]
        print(f"{task}: {len(plan.train_idx)} training and {len(plan.test_idx)} test samples "
              f"({len(plan.excluded_idx)} used for neither); test ponds {list(plan.test_ponds)}; "
              f"inner split {len(plan.inner[0].train_idx)} fit / {len(plan.inner[0].val_idx)} validation")


if __name__ == "__main__":
    main()
