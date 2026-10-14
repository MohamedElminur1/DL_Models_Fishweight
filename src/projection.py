"""Projects the wall-clock time of the full runs from the --dev timings (timings only).

    .venv/bin/python -I src/projection.py                        # lopo, 4 workers
    .venv/bin/python -I src/projection.py --scheme chrono --workers 4

`src/train.py --dev` times the inner splits of the first outer fold of a model and writes
results/dev/tuning/{task}/{scheme}/{model}/timing.json. This script scales those timings to
every outer fold of the full run by the number of training rows, and schedules the folds on
`workers` processes as the runner does (`submission_order`: largest training set first, each
job to the first free worker).

  tuning seconds of a fold    = trials x seconds per trial and row x inner training rows of the fold
  final-fit seconds of a fold = final seeds x estimated seconds per row x training rows of the fold

`trials` is the Optuna budget (gradient boosting, neural models), the alpha grid (ridge) or 1
(fixed baselines). The low figure takes the mean dev trial time, the high figure the longest
dev trial. The permutation test of a model (plan 8.2) costs one repeated observed fit and R
permutations, each the final fits of all folds with one seed, shared over the workers.
Writes results/dev/projection.csv. Dev timings were taken with one worker and the machine's
load at that time; the projection assumes no slowdown from running `workers` folds at once.
"""

from __future__ import annotations

import argparse
import heapq
import json
import sys
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.artifacts import Layout, model_key, write_csv_atomic  # noqa: E402
from src.data import load_tables  # noqa: E402
from src.folds import FoldPlan, load_config, plan_folds, submission_order  # noqa: E402
from src.train import models_of  # noqa: E402
from src.tuning import model_kind, ridge_alphas  # noqa: E402


def schedule(durations: Sequence[float], workers: int) -> float:
    """Wall-clock seconds of jobs started in the given order on `workers` processes.

    Each job goes to the process that is free first, as with the runner's process pool.
    """
    free = [0.0] * max(1, min(workers, len(durations)))
    for seconds in durations:
        heapq.heappush(free, heapq.heappop(free) + seconds)
    return max(free)


def inner_rows(plan: FoldPlan) -> int:
    """Training rows summed over the inner splits of an outer fold: the size of one trial."""
    return sum(len(inner.train_idx) for inner in plan.inner)


def trials_of(task: str, model: str, cfg: dict) -> int:
    """Trials per outer fold in a full run: the alpha grid for ridge, one for fixed baselines."""
    kind = model_kind(task, model, cfg)
    if kind == "ridge":
        return len(ridge_alphas(cfg))
    return 1 if kind == "fixed" else int(cfg["budget"]["n_trials"])


def permutations_of(task: str, model: str, cfg: dict) -> int:
    """R of the permutation test of the model: tabular models, or any deep model (the primary one is not known yet)."""
    perm = cfg["permutation"]
    if model in perm["tabular_models"][task]:
        return int(perm["r_tabular"])
    return int(perm["r_deep"]) if model in cfg["models"][task]["deep"] else 0


def project_model(cfg: dict, task: str, model: str, entries: list[dict], plans: list[FoldPlan],
                  workers: int) -> dict:
    """Projected seconds of the full run of `model`.

    `entries` are the fold records of its dev timing.json, `plans` the non-skipped outer folds
    of the full run. Returns the dev timings, the projected tuning and final-fit totals, the
    wall-clock of the fold run on `workers` processes (low and high) and the cost of the
    permutation test.
    """
    by_index = {p.index: p for p in plans}
    timed = [(e, by_index[e["fold"]]) for e in entries]
    trial_rate = np.mean([e["trial_seconds_mean"] / inner_rows(p) for e, p in timed])
    trial_rate_high = np.mean([e["trial_seconds_max"] / inner_rows(p) for e, p in timed])
    final_rate = np.mean([e["est_final_fit_seconds"] / len(p.train_idx) for e, p in timed])
    kind = model_kind(task, model, cfg)
    trials = trials_of(task, model, cfg)
    seeds = len(cfg["seeds"]["final"]) if kind == "deep" else 1
    ordered = submission_order(plans)
    final = np.array([seeds * final_rate * len(p.train_idx) for p in ordered])
    low = np.array([trials * trial_rate * inner_rows(p) for p in ordered]) + final
    high = np.array([trials * trial_rate_high * inner_rows(p) for p in ordered]) + final
    n_perm = permutations_of(task, model, cfg)
    one_permutation = float(final_rate * sum(len(p.train_idx) for p in plans))
    return {
        "task": task, "model": model, "kind": kind, "n_folds": len(plans), "trials": trials, "seeds": seeds,
        "dev_s_per_trial": float(np.mean([e["trial_seconds_mean"] for e, _ in timed])),
        "dev_s_per_trial_max": float(np.mean([e["trial_seconds_max"] for e, _ in timed])),
        "dev_s_per_final_fit": float(np.mean([e["est_final_fit_seconds"] for e, _ in timed])),
        "tuning_s_total": float(low.sum() - final.sum()), "final_fit_s_total": float(final.sum()),
        "sequential_s_low": float(low.sum()), "sequential_s_high": float(high.sum()),
        "wall_s_low": schedule(low, workers), "wall_s_high": schedule(high, workers),
        "workers": workers, "perm_R": n_perm,
        "perm_s_per_permutation": one_permutation if n_perm else float("nan"),
        "perm_wall_s": one_permutation * (n_perm / workers + 1) if n_perm else float("nan"),
    }


def project_all(cfg: dict, scheme: str, workers: int) -> tuple[pd.DataFrame, list[str]]:
    """One row per (task, model, sample set) that has a dev timing, and the names of those that have none."""
    tables = load_tables(cfg)
    layout = Layout(cfg, dev=True)
    cutoff = cfg["chrono"]["cutoff"] if scheme == "chrono" else None
    rows, missing = [], []
    for task in ("T1", "T2"):
        sets = [("eligible", models_of(task, cfg)), ("all", cfg["models"][task]["all_sample_set"])]
        for sample_set, models in sets:
            plans = [p for p in plan_folds(task, scheme, tables.table(task), sample_set, cfg, cutoff) if not p.skipped]
            for model in models:
                path = layout.tuning(task, scheme, model_key(model, sample_set)) / "timing.json"
                if not path.exists():
                    missing.append(f"{task} {model_key(model, sample_set)}")
                    continue
                entries = json.loads(path.read_text(encoding="utf-8"))["folds"]
                rows.append({"scheme": scheme, "sample_set": sample_set,
                             **project_model(cfg, task, model, entries, plans, workers)})
    return pd.DataFrame(rows), missing


def hours(seconds: float) -> str:
    return "" if pd.isna(seconds) else f"{seconds / 3600:.2f}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scheme", default="lopo", choices=["lopo", "chrono"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    table, missing = project_all(cfg, args.scheme, args.workers)
    if table.empty:
        print(f"no dev timings under {Layout(cfg, dev=True).root} for scheme {args.scheme}", file=sys.stderr)
        return 2
    write_csv_atomic(table, Layout(cfg, dev=True).root / f"projection_{args.scheme}.csv")
    shown = pd.DataFrame({
        "task": table["task"], "model": table["model"] + table["sample_set"].map({"eligible": "", "all": " (all)"}),
        "trials": table["trials"], "s/trial": table["dev_s_per_trial"].round(2),
        "s/final fit": table["dev_s_per_final_fit"].round(2),
        "h sequential": [f"{hours(a)}-{hours(b)}" for a, b in zip(table["sequential_s_low"], table["sequential_s_high"])],
        f"h on {args.workers} workers": [f"{hours(a)}-{hours(b)}" for a, b in zip(table["wall_s_low"], table["wall_s_high"])],
        "perm R": table["perm_R"], "perm h": table["perm_wall_s"].map(hours)})
    print(shown.to_string(index=False))
    print(f"\nall listed runs one after another on {args.workers} workers: "
          f"{hours(table['wall_s_low'].sum())}-{hours(table['wall_s_high'].sum())} h")
    if missing:
        print(f"no dev timing for: {', '.join(missing)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
