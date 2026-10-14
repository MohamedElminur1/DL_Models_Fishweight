"""Within-date permutation test of the water-quality inputs (plan 8.2).

    .venv/bin/python -I src/permute.py --task T1 --model ridge_time_wq [--workers 4]
    .venv/bin/python -I src/permute.py --task T1 --describe          (data structure only, fits nothing)

Run after the LOPO run of the model and after src/select_primary.py (committed). The model is
one of `permutation.tabular_models[task]` (R = permutation.r_tabular) or the primary deep
model of the task in results/selection/primary_{task}.json (R = permutation.r_deep, seed
permutation.deep_seed in the observed and the permuted runs).

For each permutation r = 0..R-1 and each date d (T1 event date, T2 origin date) on which at
least two sensor-stream units have eligible samples, the units' water-quality inputs for d
are reassigned among those units by a uniformly random derangement drawn from
numpy.random.default_rng(permutation.seed + r), units in the order of their smallest pond id
and dates in increasing order. A unit takes the inputs of the unit assigned to it: the
tabular features of the donor pond's row in wq_by_date.csv, and the sequence cut from the
donor pond's daily file with the causal gap rule at the sample's own date (the donor pond of a
shared stream is `permutation.donor_pond`, pond 2 for the stream of ponds 1 and 2). The
missing-sensor pattern moves with the values. A date with one unit stays as it is and is
counted. Training and test samples of every outer fold are permuted alike; targets, time and
past-weight inputs, eligibility, folds and the hyperparameters and epoch counts chosen in the
observed run (results/tuning/{task}/lopo/{model}/chosen.csv) stay fixed. Each fold is fitted
again on the permuted training samples without tuning and predicts its permuted test samples.
The statistic is S (T2: S at horizon 1) over the test samples of all folds.

Before the first permutation the observed fits are repeated on the unpermuted data with the
chosen hyperparameters and compared with the saved predictions; the run stops if they differ,
so that the permuted runs and the observed run are known to be the same procedure.

Output: results/permutation/{task}/{model}.csv (r, S_perm and the counts of dates and rows
changed) and {model}.json (hashes, the check of the observed run). p-values come from
src/evaluate.py. A model whose output exists with the same hash is not run again; unfinished
runs resume from the per-permutation files in {model}.parts/. The freeze guard is called
before every permutation.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import math
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import guard, scoring, tuning  # noqa: E402
from src.artifacts import (  # noqa: E402
    Layout, append_runtime_rows, repo_path, runtime_row, sha256_file, sha256_text, write_csv_atomic,
    write_json_atomic,
)
from src.baselines import wq_columns  # noqa: E402
from src.data import KEY_COLUMN, WQ_PREFIX, Tables, load_tables  # noqa: E402
from src.folds import FoldPlan, load_config, plan_folds, select_samples, sensor_streams  # noqa: E402
from src.models import NETWORK_MODELS  # noqa: E402
from src.models.common import DailyStore, build_sampleset, configure_torch  # noqa: E402
from src.tuning import Context, search_space  # noqa: E402

INPUT_FILES = ("t1_table", "t2_table", "daily_wq", "wq_by_date")
PART_COLUMNS = ["r", "S_perm", "n_dates", "n_dates_unchanged", "n_rows_changed"]
CHECK_TOLERANCE = 1e-8


class PermutationError(RuntimeError):
    """The permutation test cannot be run or checked as asked."""


# ----------------------------------------------------------------------------------------
# Drawing the permutations
# ----------------------------------------------------------------------------------------

def derange(rng: np.random.Generator, k: int) -> np.ndarray:
    """A uniformly random derangement of range(k): permutations are drawn until none keeps its place."""
    if k < 2:
        raise ValueError("a derangement needs at least two items")
    identity = np.arange(k)
    while True:
        order = rng.permutation(k)
        if not (order == identity).any():
            return order


def unit_lookup(cfg: dict) -> dict[int, tuple[int, ...]]:
    """The sensor-stream unit (tuple of ponds, smallest first) of every cohort pond."""
    design = cfg["design"]
    return {p: unit for unit in sensor_streams(design["cohort"], design["shared_streams"]) for p in unit}


def donor_of(cfg: dict, unit: tuple[int, ...]) -> int:
    """The pond whose daily file stands for a unit: `permutation.donor_pond`, else the pond itself."""
    donors = {int(k): int(v) for k, v in cfg["permutation"]["donor_pond"].items()}
    chosen = {donors.get(p, p) for p in unit}
    if len(chosen) != 1:
        raise PermutationError(f"the ponds of unit {unit} name different donor ponds {sorted(chosen)}")
    return chosen.pop()


def permuted_donors(rows: pd.DataFrame, task: str, cfg: dict, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """The pond whose water quality each eligible sample receives, and counts of the dates.

    `rows` are the eligible samples (any order). Returns an array aligned with `rows`: the
    sample's own pond where its date has a single unit (or where the donor is itself), else the
    donor pond of the unit assigned to the sample's unit. Dates are visited in increasing order
    and the rng is used only for dates with two or more units.
    """
    key = KEY_COLUMN[task]
    unit_of = unit_lookup(cfg)
    donors = rows["pond"].to_numpy().copy()
    position = pd.Series(np.arange(len(rows)), index=rows.index)
    n_dates = n_unchanged = n_rows = 0
    for _date, part in rows.groupby(key, sort=True):
        n_dates += 1
        units = sorted({unit_of[int(p)] for p in part["pond"]})
        if len(units) < 2:
            n_unchanged += 1
            continue
        order = derange(rng, len(units))
        receives = {unit: donor_of(cfg, units[order[i]]) for i, unit in enumerate(units)}
        for label, pond in zip(part.index, part["pond"]):
            donors[position[label]] = receives[unit_of[int(pond)]]
            n_rows += 1
    return donors, {"n_dates": n_dates, "n_dates_unchanged": n_unchanged, "n_rows_changed": n_rows}


def permuted_table(table: pd.DataFrame, rows: pd.DataFrame, donors: np.ndarray, wq: pd.DataFrame,
                   task: str, cfg: dict) -> pd.DataFrame:
    """A copy of `table` whose water-quality features, for the samples of `rows`, are those of their donor ponds.

    The donor's features at the sample's own date come from wq_by_date.csv (columns
    `t1_...` or `t2_...`); a feature that is missing for the donor is missing in the copy.
    Samples whose donor is their own pond keep their values. No other column changes.
    """
    columns = wq_columns(task, cfg)
    prefix = WQ_PREFIX[task]
    donor_columns = [f"{prefix}_{c}" for c in columns]
    changed = donors != rows["pond"].to_numpy()
    out = table.copy()
    if not changed.any():
        return out
    wanted = pd.DataFrame({"pond": donors[changed], "date": rows[KEY_COLUMN[task]].to_numpy()[changed]})
    found = wanted.merge(wq[["pond", "date", *donor_columns]], on=["pond", "date"], how="left",
                         validate="many_to_one", indicator=True)
    if (found["_merge"] != "both").any():
        raise PermutationError("a donor pond has no wq_by_date row at a sample date")
    labels = rows.index[changed]
    out.loc[labels, columns] = found[donor_columns].to_numpy(dtype=float)
    return out


def permutation_statistic(task: str, predictions: pd.DataFrame) -> float:
    """S of the plan (T2: S at horizon 1) over the pooled test predictions of all folds."""
    frame = predictions[predictions["horizon"] == 1] if task == "T2" else predictions
    errors = scoring.abs_log_error(frame["y_pred_log"].to_numpy(), frame["y_true_log"].to_numpy())
    return scoring.s_value(scoring.pond_scores(frame["pond"].to_numpy(), errors))


# ----------------------------------------------------------------------------------------
# What the observed run chose
# ----------------------------------------------------------------------------------------

def cast(value, spec: dict):
    """A hyperparameter read from chosen.csv, in the type of its search-space entry."""
    if spec["type"] == "float":
        return float(value)
    if spec["type"] == "int":
        return int(round(float(value)))
    choices = spec["choices"]
    if all(isinstance(c, int) for c in choices):
        return int(round(float(value)))
    if all(isinstance(c, float) for c in choices):
        return float(value)
    return type(choices[0])(value)


def load_chosen(layout: Layout, cfg: dict, task: str, model: str, plans: list[FoldPlan]) -> dict[int, tuple[dict, int | None]]:
    """Per outer fold: the hyperparameters and the epoch count of the observed run (from chosen.csv)."""
    path = layout.tuning(task, "lopo", model) / "chosen.csv"
    if not path.is_file():
        raise PermutationError(f"missing {path}: run the LOPO scheme of {model} first")
    frame = pd.read_csv(path, float_precision="round_trip")
    kind = tuning.model_kind(task, model, cfg)
    spec = {"alpha": {"type": "float"}} if kind == "ridge" else search_space(task, model, cfg)
    chosen = {}
    for plan in plans:
        row = frame[frame["outer_fold"] == plan.index]
        if len(row) != 1:
            raise PermutationError(f"{path} has {len(row)} rows for outer fold {plan.index}")
        row = row.iloc[0]
        hp = {name: cast(row[f"hp_{name}"], s) for name, s in spec.items()}
        epochs = int(row["epochs"]) if kind == "deep" else None
        chosen[plan.index] = (hp, epochs)
    return chosen


def fold_predictions(ctx: Context, plans: list[FoldPlan], chosen: dict, seed: int) -> pd.DataFrame:
    """Fit every outer fold of `ctx` with its chosen hyperparameters and return the pooled test predictions.

    The caller has passed the freeze guard (`tuning.fit_and_predict`).
    """
    frames = []
    for plan in plans:
        hp, epochs = chosen[plan.index]
        frames.append(tuning.fit_and_predict(ctx, plan, hp, epochs, [seed])[0].predictions)
    return pd.concat(frames, ignore_index=True)


# ----------------------------------------------------------------------------------------
# The runs
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class PermJob:
    """A chunk of permutations for one worker."""

    cfg: dict
    task: str
    model: str
    rs: tuple[int, ...]
    chosen: dict
    seed: int
    run_hash: str
    parts_dir: str
    guard_check: Callable[[], object] = field(default=guard.assert_test_allowed)


def eligible_rows(tables: Tables, task: str, cfg: dict) -> pd.DataFrame:
    return select_samples(tables.table(task), cfg["design"]["cohort"], "eligible")


def prepare(cfg: dict, task: str, model: str):
    """Tables, eligible rows, LOPO plans and the daily store shared by all permutations of a worker."""
    tables = load_tables(cfg)
    rows = eligible_rows(tables, task, cfg)
    plans = [p for p in plan_folds(task, "lopo", tables.table(task), "eligible", cfg) if not p.skipped]
    store = None
    if model in NETWORK_MODELS:
        design = cfg["design"]
        store = DailyStore(tables.daily, design["sensors"], design["stats"])
    return tables, rows, plans, store


def permuted_context(cfg: dict, tables: Tables, rows: pd.DataFrame, store, task: str, model: str,
                     rng: np.random.Generator) -> tuple[Context, dict]:
    """The context of one permutation: permuted table (and sequences for the neural models)."""
    donors, counts = permuted_donors(rows, task, cfg, rng)
    table = permuted_table(tables.table(task), rows, donors, tables.wq, task, cfg)
    sampleset = build_sampleset(task, rows, store, cfg, donors=donors) if store is not None else None
    return Context(task, "lopo", model, "eligible", cfg, table, sampleset, False), counts


def run_chunk(job: PermJob) -> list[int]:
    """Run the permutations `job.rs`; each result is written to its own file as soon as it exists."""
    configure_torch()
    tables, rows, plans, store = prepare(job.cfg, job.task, job.model)
    seed0 = int(job.cfg["permutation"]["seed"])
    done = []
    for r in job.rs:
        job.guard_check()
        started = time.perf_counter()
        ctx, counts = permuted_context(job.cfg, tables, rows, store, job.task, job.model,
                                       np.random.default_rng(seed0 + r))
        predictions = fold_predictions(ctx, plans, job.chosen, job.seed)
        write_json_atomic(Path(job.parts_dir) / f"r{r:04d}.json", {
            "r": r, "S_perm": permutation_statistic(job.task, predictions), **counts,
            "hash": job.run_hash, "seconds": round(time.perf_counter() - started, 4), "pid": os.getpid()})
        done.append(r)
    return done


def check_observed(cfg: dict, layout: Layout, task: str, model: str, chosen: dict, seed: int) -> dict:
    """Repeat the observed fits on the unpermuted data and compare them with the saved predictions."""
    configure_torch()                                   # one thread, as in the observed run
    tables, rows, plans, store = prepare(cfg, task, model)
    started = time.perf_counter()
    ctx = Context(task, "lopo", model, "eligible", cfg, tables.table(task),
                  build_sampleset(task, rows, store, cfg) if store is not None else None, False)
    again = fold_predictions(ctx, plans, chosen, seed)
    saved = pd.read_csv(layout.predictions(task, "lopo", model), float_precision="round_trip")
    saved = saved[saved["seed"] == seed].sort_values("row_id")
    again = again.sort_values("row_id")
    if list(saved["row_id"]) != list(again["row_id"]):
        raise PermutationError("the repeated observed fits cover different samples than the saved predictions")
    same = np.allclose(saved["y_pred_log"].to_numpy(dtype=float), again["y_pred_log"].to_numpy(dtype=float),
                       rtol=0.0, atol=CHECK_TOLERANCE, equal_nan=True)
    if not same:
        raise PermutationError(
            f"{task} {model}: repeating the observed fits with the chosen hyperparameters does not reproduce the saved "
            "predictions, so the permuted runs would not be the same procedure as the observed run")
    return {"observed_seed": seed, "observed_S": permutation_statistic(task, again),
            "n_predictions": int(len(again)), "seconds": round(time.perf_counter() - started, 3)}


def model_kind_for_permutation(cfg: dict, layout: Layout, task: str, model: str) -> tuple[str, int, int]:
    """('tabular' | 'deep', R, seed) for a model that the plan permutes; PermutationError otherwise."""
    perm = cfg["permutation"]
    if model in perm["tabular_models"][task]:
        return "tabular", int(perm["r_tabular"]), 0
    path = layout.selection(task)
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("primary_deep") == model:
        return "deep", int(perm["r_deep"]), int(perm["deep_seed"])
    raise PermutationError(f"{model} is neither a permutation model of {task} nor its primary deep model ({path.name})")


def assert_selection_committed(layout: Layout, task: str) -> None:
    guard.assert_files_committed([os.path.relpath(layout.selection(task), guard.REPO_ROOT)])


@contextlib.contextmanager
def run_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise PermutationError(f"another process is running this permutation test ({path.name})") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_hash_of(cfg: dict, layout: Layout, task: str, model: str, n_perm: int, seed: int) -> str:
    inputs = {name: sha256_file(repo_path(cfg["paths"][name])) for name in INPUT_FILES}
    inputs["chosen"] = sha256_file(layout.tuning(task, "lopo", model) / "chosen.csv")
    inputs["predictions"] = sha256_file(layout.predictions(task, "lopo", model))
    return sha256_text(json.dumps({"cfg": cfg, "inputs": inputs, "task": task, "model": model, "R": n_perm,
                                   "seed": seed}, sort_keys=True, default=str))


def read_parts(parts_dir: Path, run_hash: str, n_perm: int) -> dict[int, dict]:
    """The finished permutations in `parts_dir`; parts of another run or beyond R are refused."""
    parts = {}
    for path in sorted(parts_dir.glob("r*.json")) if parts_dir.is_dir() else []:
        part = json.loads(path.read_text(encoding="utf-8"))
        if part["hash"] != run_hash or not (0 <= part["r"] < n_perm):
            raise PermutationError(f"{path} belongs to another run (config, inputs or R changed); move {parts_dir.name} away")
        parts[int(part["r"])] = part
    return parts


def run_permutations(cfg: dict, task: str, model: str, *, n_perm: int | None = None, workers: int = 1,
                     guard_check: Callable[[], object] = guard.assert_test_allowed,
                     selection_check: Callable[[Layout, str], None] = assert_selection_committed,
                     log=print) -> dict:
    """Run (or resume) the permutation test of one model; returns {"status", "meta"}.

    `n_perm` replaces R of the config (Python API only, for tests). `guard_check` is the freeze
    guard; it is called here and before every permutation in the workers (the command line
    always uses the real one).
    """
    layout = Layout(cfg)
    kind, n_config, seed = model_kind_for_permutation(cfg, layout, task, model)
    n_perm = n_config if n_perm is None else n_perm
    selection_check(layout, task)
    provenance = guard_check()
    out_csv = layout.permutation(task, model)
    out_meta = out_csv.with_suffix(".json")
    parts_dir = out_csv.with_suffix(".parts")
    with run_lock(out_csv.with_suffix(".lock")):
        run_hash = run_hash_of(cfg, layout, task, model, n_perm, seed)
        if out_meta.exists():
            meta = json.loads(out_meta.read_text(encoding="utf-8"))
            if meta["hash"] != run_hash:
                raise PermutationError(f"{out_meta} comes from another config or other inputs; move it away to run again")
            log(f"{task} {model}: permutation test already complete; nothing to do")
            return {"status": "no-op", "meta": meta}
        plans = [p for p in plan_folds(task, "lopo", load_tables(cfg).table(task), "eligible", cfg) if not p.skipped]
        chosen = load_chosen(layout, cfg, task, model, plans)
        check = check_observed(cfg, layout, task, model, chosen, seed)
        log(f"{task} {model}: the repeated observed fits reproduce the saved predictions")
        parts = read_parts(parts_dir, run_hash, n_perm)
        todo = [r for r in range(n_perm) if r not in parts]
        size = max(1, min(25, math.ceil(len(todo) / (4 * workers)))) if todo else 1
        jobs = [PermJob(cfg, task, model, tuple(todo[i:i + size]), chosen, seed, run_hash, str(parts_dir), guard_check)
                for i in range(0, len(todo), size)]
        started = time.perf_counter()
        if workers > 1 and len(jobs) > 1:
            context = multiprocessing.get_context("spawn")
            with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=context) as pool:
                futures = [pool.submit(run_chunk, job) for job in jobs]
                try:
                    for future in concurrent.futures.as_completed(futures):
                        done = future.result()
                        log(f"{task} {model}: permutations {done[0]}..{done[-1]} finished")
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
        else:
            for job in jobs:
                done = run_chunk(job)
                log(f"{task} {model}: permutations {done[0]}..{done[-1]} finished")
        parts = read_parts(parts_dir, run_hash, n_perm)
        if sorted(parts) != list(range(n_perm)):
            raise PermutationError(f"{task} {model}: permutations {sorted(set(range(n_perm)) - set(parts))} are missing")
        table = pd.DataFrame([{c: parts[r][c] for c in PART_COLUMNS} for r in range(n_perm)])
        write_csv_atomic(table, out_csv)
        seconds = sum(parts[r]["seconds"] for r in parts)
        meta = {"task": task, "model": model, "kind": kind, "R": n_perm, "seed": int(cfg["permutation"]["seed"]),
                "fit_seed": seed, "statistic": "S" if task == "T1" else "S_h1", "hash": run_hash,
                "observed_check": check, "guard": provenance, "complete": True,
                "processes": sorted({int(parts[r]["pid"]) for r in parts}),
                "n_dates": int(table["n_dates"].iloc[0]), "n_dates_unchanged": int(table["n_dates_unchanged"].iloc[0]),
                "n_rows_changed": int(table["n_rows_changed"].iloc[0])}
        write_json_atomic(out_meta, meta)
        append_runtime_rows(layout.runtime_log(), [
            runtime_row(task, "permutation", model, "", "check", seed, seconds=check["seconds"]),
            runtime_row(task, "permutation", model, "", "permutation", f"R={n_perm}", seconds=seconds,
                        notes=f"workers={workers}; mean_seconds_per_permutation={seconds / n_perm:.3f}")],
            task, "permutation", model)
        shutil.rmtree(parts_dir, ignore_errors=True)
        log(f"{task} {model}: {n_perm} permutations written to {out_csv}; wall clock {time.perf_counter() - started:.1f} s")
    return {"status": "complete", "meta": meta}


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------

def describe(cfg: dict, task: str) -> dict:
    """Counts that describe the permutation of one task (no model is fitted, no outcome is read)."""
    tables = load_tables(cfg)
    rows = eligible_rows(tables, task, cfg)
    unit_of = unit_lookup(cfg)
    per_date = rows.groupby(KEY_COLUMN[task])["pond"].agg(lambda s: len({unit_of[int(p)] for p in s}))
    _donors, counts = permuted_donors(rows, task, cfg, np.random.default_rng(int(cfg["permutation"]["seed"])))
    return {"task": task, "n_eligible_samples": int(len(rows)), "n_dates": int(len(per_date)),
            "units_per_date": {int(k): int(v) for k, v in per_date.value_counts().sort_index().items()},
            "n_dates_unchanged": counts["n_dates_unchanged"], "n_rows_changed_in_permutation_0": counts["n_rows_changed"]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=["T1", "T2"])
    parser.add_argument("--model")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--describe", action="store_true", help="print the data-structure counts and stop")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.describe:
        print(json.dumps(describe(cfg, args.task), indent=2, sort_keys=True))
        return 0
    if not args.model or args.workers < 1:
        print("refused: --model is required (and --workers must be at least 1)", file=sys.stderr)
        return 2
    try:
        run_permutations(cfg, args.task, args.model, workers=args.workers, log=lambda m: print(m, flush=True))
    except (PermutationError, guard.GuardError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except tuning.NonFinitePredictionError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
