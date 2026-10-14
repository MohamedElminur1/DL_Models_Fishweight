"""Chooses the primary baseline and the primary deep model of each task (plan 9.4).

    .venv/bin/python -I src/select_primary.py [--task T1 T2]

Run after the LOPO runs of the candidate models and before src/evaluate.py; commit the two
files it writes (results/selection/primary_T1.json and primary_T2.json) before scoring, the
scorer refuses to run otherwise (plan 12 (d)).

The choice uses inner-validation scores only. The one input of a candidate is
results/tuning/{task}/lopo/{model}/inner_scores.csv; no prediction file, outer score or
chosen-hyperparameter file is opened. A model's score is the mean over the outer folds of the
mean over that fold's inner folds of S (T2: S is already the mean of the h = 1 and h = 2
values, plan 9.4). A tuned model enters with the inner scores of its best trial, as written
by src/train.py. The primary baseline is the candidate with the lowest score among the
time-only models of plan 6.1, the primary deep model the one with the lowest score among
LSTM, GRU, CNN-LSTM and TFT. Ties go to the model listed first in the config, which is the
order of plans 6.1 and 6.3. A candidate whose score is +infinity cannot be primary.

The output is deterministic: running it again on the same inner-score files gives the same
bytes.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src.artifacts import Layout, sha256_file, write_json_atomic  # noqa: E402
from src.folds import load_config  # noqa: E402

SCHEME = "lopo"
POOLS = {"primary_baseline": "time_only", "primary_deep": "deep"}


class SelectionError(RuntimeError):
    """The inner scores do not allow a selection (missing, malformed or inconsistent files)."""


def inner_scores_path(layout: Layout, task: str, model: str) -> Path:
    return layout.tuning(task, SCHEME, model) / "inner_scores.csv"


def read_inner_scores(path: Path) -> dict[int, float]:
    """Mean over the inner folds of S, per outer fold, from one inner_scores.csv.

    Raises SelectionError for a missing file, missing columns, no rows, or an S that is NaN
    (what a --dev run writes). S = +infinity is kept.
    """
    if not path.is_file():
        raise SelectionError(f"missing inner scores: {path}")
    frame = pd.read_csv(path)
    missing = {"outer_fold", "inner_fold", "S"} - set(frame.columns)
    if missing or frame.empty:
        raise SelectionError(f"{path} lacks columns {sorted(missing)} or has no rows")
    if frame["S"].isna().any():
        raise SelectionError(f"{path} has an S that is not a number (a --dev file?)")
    return {int(k): float(v) for k, v in frame.groupby("outer_fold")["S"].mean().items()}


def format_score(value: float):
    """Floats as they are; +infinity as the string "inf" (JSON has no infinity)."""
    return "inf" if math.isinf(value) else value


def select_task(cfg: dict, layout: Layout, task: str) -> dict:
    """The selection record of one task (see the module docstring for the rule)."""
    pools = {name: list(cfg["models"][task][group]) for name, group in POOLS.items()}
    names = [m for pool in pools.values() for m in pool]
    missing = [m for m in names if not inner_scores_path(layout, task, m).is_file()]
    if missing:
        raise SelectionError(f"{task}: no inner scores for {missing}")
    per_fold = {m: read_inner_scores(inner_scores_path(layout, task, m)) for m in names}
    folds = {m: sorted(per_fold[m]) for m in names}
    reference = folds[names[0]]
    odd = [m for m in names if folds[m] != reference]
    if odd:
        raise SelectionError(f"{task}: outer folds of {odd} differ from those of {names[0]} {reference}")
    scores = {m: sum(per_fold[m].values()) / len(per_fold[m]) for m in names}
    record: dict = {
        "task": task, "scheme": SCHEME,
        "rule": ("mean over the outer folds of the mean over the inner folds of S, lowest wins, ties to the "
                 "model listed first (plan 9.4)"),
        "outer_folds": reference,
        "pools": pools,
        "mean_inner_S": {m: format_score(scores[m]) for m in names},
        "inner_S_by_outer_fold": {m: {str(k): format_score(v) for k, v in per_fold[m].items()} for m in names},
        "inputs": {inner_scores_path(layout, task, m).relative_to(layout.root).as_posix():
                   sha256_file(inner_scores_path(layout, task, m)) for m in names},
    }
    for field, pool in pools.items():
        best = min(pool, key=lambda m: (scores[m], pool.index(m)))
        if math.isinf(scores[best]):
            raise SelectionError(f"{task}: every candidate of {POOLS[field]} has an infinite inner score")
        record[field] = best
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", nargs="+", choices=["T1", "T2"], default=["T1", "T2"])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    layout = Layout(cfg)
    status = 0
    for task in args.task:
        try:
            record = select_task(cfg, layout, task)
        except SelectionError as exc:
            print(f"{task}: refused: {exc}", file=sys.stderr)
            status = 2
            continue
        write_json_atomic(layout.selection(task), record)
        print(f"{task}: primary baseline {record['primary_baseline']}, primary deep model {record['primary_deep']} "
              f"-> {layout.selection(task)}")
    return status


if __name__ == "__main__":
    sys.exit(main())
