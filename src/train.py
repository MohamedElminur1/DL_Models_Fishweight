"""Runs one model of the plan on one validation scheme (ANALYSIS_PLAN.md sections 7, 12, 13).

    .venv/bin/python -I src/train.py --task T1 --scheme lopo --model ridge_time --dev
    .venv/bin/python -I src/train.py --task T1 --scheme lopo --model lstm --workers 4

Full run (no --dev): for every outer fold with test samples, tune the model on the inner
splits of the fold (src/tuning.py), fit the chosen model on all training rows (neural models:
five seeds, for the median best epoch of the best trial) and predict the test rows. The run
refuses to start, and the final fit refuses to run, unless the freeze guard passes
(src/guard.py). Folds are the unit of parallelism (`--workers`): each fold has its own Optuna
study in its own SQLite file and its own directory, and a fold that finished is skipped on a
re-run. A model that finished with identical config and inputs is a no-op; a different
config or input hash is refused.

Outputs (`results/`, or `results/dev/` with --dev):
  predictions/{task}/{scheme}/{model}.csv         one row per test sample, seed and fold
  tuning/{task}/{scheme}/{model}/
      inner_scores.csv    inner-validation score of the best trial: outer_fold, inner_fold,
                          n_samples, S (and S_h1, S_h2 in T2)
      chosen.csv          best trial and hyperparameters per outer fold
      curves.csv.gz       train and validation loss per epoch of every inner and final fit
      timing.json         seconds per trial and per fit, estimated final-fit time
      run_meta.json       hashes, folds, skipped folds, guard provenance
      fold{NN}/           study.db, trials.csv and the fold's part files
  logs/runtime.csv       task, scheme, model, fold, stage, trial_or_seed, epochs, seconds, cpu,
                          python, torch, notes

--dev runs the inner splits of the first outer fold only, no final fit, no prediction. It
never prints, logs or writes an error value, objective value or loss: the columns that would
hold them are NaN. What it records is timings, counts and the epochs the neural models ran.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import fcntl
import json
import multiprocessing
import os
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

from src import guard  # noqa: E402
from src.artifacts import (  # noqa: E402
    Layout, append_runtime_rows, hardware, model_key, repo_path, runtime_row, sha256_file,
    sha256_text, write_csv_atomic, write_json_atomic,
)
from src.data import Tables, load_tables  # noqa: E402
from src.folds import (  # noqa: E402
    FoldPlan, SAMPLE_SETS, SCHEMES, load_config, plan_folds, select_samples, submission_order, suggest_cutoff,
)
from src.models import NETWORK_MODELS  # noqa: E402
from src.models.common import DailyStore, build_sampleset, configure_torch  # noqa: E402
from src.tuning import Context, NonFinitePredictionError, final_fits, tune_fold  # noqa: E402

MODEL_GROUPS = ("time_only", "water_quality", "reference", "deep", "static_only", "sensitivity")
CURVE_COLUMNS = ["task", "scheme", "model", "outer_fold", "stage", "trial", "inner_fold", "seed",
                 "epoch", "train_mse", "val_mse", "seconds"]
INPUT_FILES = ("t1_table", "t2_table", "daily_wq", "wq_by_date")


class RunRefused(RuntimeError):
    """The request cannot be run as asked (bad option, or results from a different config)."""


# ----------------------------------------------------------------------------------------
# Request checks
# ----------------------------------------------------------------------------------------

def models_of(task: str, cfg: dict) -> list[str]:
    """Every model name of a task in the config."""
    return [m for group in MODEL_GROUPS for m in cfg["models"][task][group]]


def resolve_cutoff(cfg: dict, tables: Tables, scheme: str, dev: bool, cli_cutoff: str | None) -> str | None:
    """The cut-off date of a chrono run; None for lopo.

    A full run uses the date in the config, which must be set and must equal the date the
    rule of plan 7.4 gives for the tables. --dev may name another date or use the rule's.
    """
    if scheme != "chrono":
        return None
    rule = suggest_cutoff(tables.t1, cfg["chrono"]["min_events"], cfg["chrono"]["min_ponds"])
    configured = cfg["chrono"]["cutoff"]
    if dev:
        return cli_cutoff or configured or rule
    if configured is None:
        raise RunRefused("chrono.cutoff is not set in the config; a full chrono run needs the date c (plan 7.4)")
    if str(configured) != rule:
        raise RunRefused(f"chrono.cutoff {configured} differs from the date the rule of plan 7.4 gives ({rule})")
    return str(configured)


def check_request(cfg: dict, task: str, scheme: str, model: str, sample_set: str, dev: bool,
                  n_trials: int | None, cutoff: str | None, workers: int = 1, dev_folds: int = 1) -> None:
    if scheme not in SCHEMES or sample_set not in SAMPLE_SETS:
        raise RunRefused(f"scheme must be one of {SCHEMES} and sample set one of {SAMPLE_SETS}")
    if model not in models_of(task, cfg):
        raise RunRefused(f"{model!r} is not a {task} model of the config")
    if sample_set == "all" and model not in cfg["models"][task]["all_sample_set"]:
        raise RunRefused(f"the all-events sample set is only for {cfg['models'][task]['all_sample_set']}")
    if n_trials is not None and not dev:
        raise RunRefused("--n-trials changes the budget of the plan; it is only allowed with --dev")
    if cutoff is not None and not (dev and scheme == "chrono"):
        raise RunRefused("--cutoff is only allowed with --dev and --scheme chrono")
    if dev_folds != 1 and not dev:
        raise RunRefused("--dev-folds is only allowed with --dev")
    if workers < 1 or dev_folds < 1:
        raise RunRefused("--workers and --dev-folds must be at least 1")


def build_context(cfg: dict, tables: Tables, task: str, scheme: str, model: str, sample_set: str,
                  dev: bool) -> Context:
    """Table, config and (for the neural models) the sequence sample set of the eligible rows."""
    table = tables.table(task)
    sampleset = None
    if model in NETWORK_MODELS:
        design = cfg["design"]
        store = DailyStore(tables.daily, design["sensors"], design["stats"])
        sampleset = build_sampleset(task, select_samples(table, design["cohort"], "eligible"), store, cfg)
    return Context(task, scheme, model, sample_set, cfg, table, sampleset, dev)


def compute_run_hash(cfg: dict, task: str, scheme: str, model: str, sample_set: str, cutoff: str | None) -> str:
    """Hash of the config, the input files and the request; results are reused only if it matches."""
    inputs = {name: sha256_file(repo_path(cfg["paths"][name])) for name in INPUT_FILES}
    payload = {"cfg": cfg, "inputs": inputs, "task": task, "scheme": scheme, "model": model,
               "sample_set": sample_set, "cutoff": cutoff}
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))


# ----------------------------------------------------------------------------------------
# One fold (runs in a worker process or in this one)
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class FoldJob:
    cfg: dict
    task: str
    scheme: str
    model: str
    sample_set: str
    cutoff: str | None
    fold: int
    dev: bool
    n_trials: int
    run_hash: str
    fold_dir: str


def fold_is_done(fold_dir: Path, run_hash: str) -> bool:
    """True if the fold finished with this hash; RunRefused if it finished with another."""
    marker = fold_dir / "done.json"
    if not marker.exists():
        return False
    found = json.loads(marker.read_text(encoding="utf-8"))["hash"]
    if found != run_hash:
        raise RunRefused(f"{fold_dir} was finished with a different config or inputs (hash {found[:12]}); "
                         "move it away to start again")
    return True


def _curve_frame(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=CURVE_COLUMNS)


def run_fold_job(job: FoldJob) -> dict:
    """Tune one outer fold and, unless dev, fit the chosen model and predict its test rows.

    Writes the fold's part files and, last, done.json. A fold that is not marked done is
    restarted from scratch, with a fresh study.
    """
    configure_torch()
    fold_dir = Path(job.fold_dir)
    if not job.dev and fold_is_done(fold_dir, job.run_hash):
        return {"fold": job.fold, "resumed": True}
    if fold_dir.exists():
        shutil.rmtree(fold_dir)
    fold_dir.mkdir(parents=True)
    tables = load_tables(job.cfg)
    ctx = build_context(job.cfg, tables, job.task, job.scheme, job.model, job.sample_set, job.dev)
    plan = next(p for p in plan_folds(job.task, job.scheme, tables.table(job.task), job.sample_set,
                                      job.cfg, job.cutoff) if p.index == job.fold)
    started = time.perf_counter()
    outcome = tune_fold(ctx, plan, job.n_trials, fold_dir / "study.db")
    runtime = [runtime_row(
        job.task, job.scheme, ctx.key, plan.index, "plan",
        notes=(f"label={plan.label}; n_train={len(plan.train_idx)}; n_test={len(plan.test_idx)}; "
               f"n_excluded={len(plan.excluded_idx)}; test_ponds={list(plan.test_ponds)}; "
               f"dropped_ponds={list(plan.dropped_ponds)}"))] + list(outcome.runtime)
    curves = list(outcome.curves)
    chosen = {"outer_fold": plan.index, "label": plan.label, "test_ponds": list(plan.test_ponds),
              "n_train": len(plan.train_idx), "n_test": len(plan.test_idx),
              "n_excluded": len(plan.excluded_idx), "n_trials": job.n_trials}
    if not job.dev:
        seeds = list(job.cfg["seeds"]["final"])
        fits = final_fits(ctx, plan, outcome.hp, outcome.epochs, seeds)
        for fit in fits:
            runtime.append(runtime_row(
                job.task, job.scheme, ctx.key, plan.index, "final", fit.seed,
                epochs=fit.info.epochs_run or "", seconds=fit.info.seconds,
                notes="; ".join(n for n in (fit.notes, f"n_nonpositive_pred={fit.n_nonpositive}") if n)))
            curves += [(job.task, job.scheme, ctx.key, plan.index, "final", outcome.best_trial, "", fit.seed,
                        epoch, train_mse, val_mse, seconds)
                       for epoch, train_mse, val_mse, seconds in fit.info.curves]
        write_csv_atomic(pd.concat([f.predictions for f in fits], ignore_index=True),
                         fold_dir / "predictions.csv")
        chosen.update({"best_trial": outcome.best_trial, "inner_value": outcome.best_value,
                       "epochs": outcome.epochs, "hp": outcome.hp,
                       "n_nonpositive_inner": outcome.n_nonpositive})
    if len(outcome.trials):
        write_csv_atomic(outcome.trials, fold_dir / "trials.csv")
    write_csv_atomic(outcome.inner_scores, fold_dir / "inner_scores.csv")
    if curves:
        write_csv_atomic(_curve_frame(curves), fold_dir / "curves.csv.gz", float_format="%.7g")
    write_csv_atomic(pd.DataFrame(runtime), fold_dir / "runtime.csv")
    write_json_atomic(fold_dir / "chosen.json", chosen)
    write_json_atomic(fold_dir / "timing.json", {**outcome.timing, "fold": plan.index, "label": plan.label})
    write_json_atomic(fold_dir / "done.json", {"hash": job.run_hash, "fold": plan.index, "pid": os.getpid(),
                                               "seconds": round(time.perf_counter() - started, 3)})
    return {"fold": job.fold, "resumed": False}


# ----------------------------------------------------------------------------------------
# One model: folds in parallel, then the model-level files
# ----------------------------------------------------------------------------------------

def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=True)


def assemble(layout: Layout, ctx: Context, plans: list[FoldPlan], ran: list[FoldPlan], run_hash: str,
             cutoff: str | None, provenance: dict | None, seconds: float) -> dict:
    """Combine the fold part files into the model-level files and the runtime log."""
    tdir = layout.tuning(ctx.task, ctx.scheme, ctx.key)
    parts = [tdir / f"fold{p.index:02d}" for p in ran]
    runtime, chosen, timing = [], [], []
    for plan, part in zip(ran, parts):
        runtime += _read(part / "runtime.csv").fillna("").astype(str).to_dict("records")
        chosen.append(json.loads((part / "chosen.json").read_text(encoding="utf-8")))
        timing.append(json.loads((part / "timing.json").read_text(encoding="utf-8")))
    for plan in plans:
        if plan.skipped:
            runtime.append(runtime_row(ctx.task, ctx.scheme, ctx.key, plan.index, "skip",
                                       notes=f"{plan.label}: {plan.skipped}"))
    inner = pd.concat([_read(p / "inner_scores.csv") for p in parts], ignore_index=True)
    write_csv_atomic(inner, tdir / "inner_scores.csv")
    curve_parts = [_read(p / "curves.csv.gz") for p in parts if (p / "curves.csv.gz").exists()]
    if curve_parts:
        curves = pd.concat(curve_parts, ignore_index=True)
        write_csv_atomic(curves, tdir / "curves.csv.gz", float_format="%.7g")
    write_json_atomic(tdir / "timing.json", {"folds": timing})
    meta = {
        "task": ctx.task, "scheme": ctx.scheme, "model": ctx.model, "sample_set": ctx.sample_set,
        "hash": run_hash, "cutoff": cutoff, "dev": ctx.dev, "complete": True,
        "folds_run": [p.index for p in ran],
        "skipped_folds": [{"fold": p.index, "label": p.label, "reason": p.skipped} for p in plans if p.skipped],
        "seconds": round(seconds, 3), "hardware": hardware(), "guard": provenance,
        "values_withheld": ctx.dev,
        "n_excluded": {str(p.index): len(p.excluded_idx) for p in ran if p.excluded_idx},
    }
    if not ctx.dev:
        rows = []
        for entry in chosen:
            row = {k: v for k, v in entry.items() if k != "hp"}
            row["test_ponds"] = " ".join(str(p) for p in entry["test_ponds"])
            row.update({f"hp_{k}": v for k, v in entry.get("hp", {}).items()})
            rows.append(row)
        write_csv_atomic(pd.DataFrame(rows), tdir / "chosen.csv")
        predictions = pd.concat([_read(p / "predictions.csv") for p in parts], ignore_index=True)
        expected = sum(len(p.test_idx) for p in ran) * (len(ctx.cfg["seeds"]["final"]) if ctx.kind == "deep" else 1)
        if len(predictions) != expected:
            raise RuntimeError(f"{len(predictions)} predictions written, {expected} expected")
        write_csv_atomic(predictions.sort_values(["fold", "seed", "row_id"], kind="stable"),
                         layout.predictions(ctx.task, ctx.scheme, ctx.key))
        meta["n_predictions"] = len(predictions)
    write_json_atomic(tdir / "run_meta.json", meta)
    append_runtime_rows(layout.runtime_log(), runtime, ctx.task, ctx.scheme, ctx.key)
    return meta


@contextlib.contextmanager
def run_lock(path: Path):
    """Hold an exclusive lock on `path` while a model runs; RunRefused if another process holds it.

    The lock is released by the operating system if the process dies, so a crashed run never
    blocks the next one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RunRefused(f"another process is running this model ({path.name})") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_jobs(jobs: list[FoldJob], workers: int, log) -> None:
    """Run the fold jobs in this process, or in `workers` spawned processes (one fold per task).

    The first failure stops the run: folds that have not started are cancelled, folds that
    are running finish (their results are kept and skipped on the next run).
    """
    def report(result: dict) -> None:
        log(f"fold {result['fold']} {'resumed' if result['resumed'] else 'finished'}")

    if workers > 1 and len(jobs) > 1:
        context = multiprocessing.get_context("spawn")
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, len(jobs)),
                                                    mp_context=context) as pool:
            futures = [pool.submit(run_fold_job, job) for job in jobs]
            try:
                for future in concurrent.futures.as_completed(futures):
                    report(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    else:
        for job in jobs:
            report(run_fold_job(job))


def run_model(cfg: dict, task: str, scheme: str, model: str, *, sample_set: str = "eligible",
              dev: bool = False, workers: int = 1, n_trials: int | None = None,
              cutoff: str | None = None, dev_folds: int = 1, log=print) -> dict:
    """Run one model on one scheme and write its files; returns {"status", "meta"}.

    `status` is "complete", or "no-op" when the model had already finished with this config
    and these inputs. Raises RunRefused for a request or an existing result that does not
    fit, GuardError when the freeze guard fails, NonFinitePredictionError when a learned
    model predicts a value that is not finite (a runtime row records it first).
    """
    check_request(cfg, task, scheme, model, sample_set, dev, n_trials, cutoff, workers, dev_folds)
    provenance = None if dev else guard.assert_test_allowed()
    started = time.perf_counter()
    tables = load_tables(cfg)
    cutoff = resolve_cutoff(cfg, tables, scheme, dev, cutoff)
    plans = plan_folds(task, scheme, tables.table(task), sample_set, cfg, cutoff)
    ran = [p for p in plans if not p.skipped]
    if dev:
        ran = ran[:dev_folds]
    key = model_key(model, sample_set)
    layout = Layout(cfg, dev)
    tdir = layout.tuning(task, scheme, key)
    run_hash = compute_run_hash(cfg, task, scheme, model, sample_set, cutoff)
    prefix = f"{task} {scheme} {key}"
    with run_lock(tdir.with_name(tdir.name + ".lock")):
        meta_path = tdir / "run_meta.json"
        if not dev and meta_path.exists():
            found = json.loads(meta_path.read_text(encoding="utf-8"))
            if found["hash"] != run_hash:
                raise RunRefused(f"{tdir} holds a finished run with a different config or inputs "
                                 f"(hash {found['hash'][:12]}); move it away to run again")
            log(f"{prefix}: already complete with this config and inputs; nothing to do")
            return {"status": "no-op", "meta": found}
        budget = n_trials or (cfg["dev"]["n_trials"] if dev else cfg["budget"]["n_trials"])
        jobs = [FoldJob(cfg, task, scheme, model, sample_set, cutoff, p.index, dev, budget, run_hash,
                        str(tdir / f"fold{p.index:02d}"))
                for p in submission_order(ran)]
        try:
            run_jobs(jobs, workers, lambda message: log(f"{prefix}: {message}"))
        except NonFinitePredictionError as exc:
            append_runtime_rows(layout.runtime_log(), [runtime_row(task, scheme, key, "", "error", notes=str(exc))],
                                task, scheme, key)
            log(f"{prefix}: ERROR {exc}")
            raise
        ctx = Context(task, scheme, model, sample_set, cfg, tables.table(task), None, dev)
        meta = assemble(layout, ctx, plans, ran, run_hash, cutoff, provenance, time.perf_counter() - started)
    return {"status": "complete", "meta": meta}


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------

def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--task", required=True, choices=["T1", "T2"])
    parser.add_argument("--scheme", required=True, choices=list(SCHEMES))
    parser.add_argument("--model", required=True)
    parser.add_argument("--sample-set", default="eligible", choices=list(SAMPLE_SETS))
    parser.add_argument("--dev", action="store_true", help="inner splits of the first outer fold only; no error values")
    parser.add_argument("--dev-folds", type=int, default=1, help="--dev only: number of outer folds to cover")
    parser.add_argument("--workers", type=int, default=1, help="outer folds run in parallel")
    parser.add_argument("--n-trials", type=int, default=None, help="--dev only: trials per search")
    parser.add_argument("--cutoff", default=None, help="--dev --scheme chrono only: cut-off date")
    parser.add_argument("--config", default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    try:
        result = run_model(cfg, args.task, args.scheme, args.model, sample_set=args.sample_set,
                           dev=args.dev, workers=args.workers, n_trials=args.n_trials,
                           cutoff=args.cutoff, dev_folds=args.dev_folds, log=lambda m: print(m, flush=True))
    except (RunRefused, guard.GuardError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except NonFinitePredictionError:
        return 1
    meta = result["meta"]
    if args.dev:
        tdir = Layout(cfg, True).tuning(args.task, args.scheme, model_key(args.model, args.sample_set))
        for timing in json.loads((tdir / "timing.json").read_text(encoding="utf-8"))["folds"]:
            paths = ", ".join(f"{k}={v}" for k, v in sorted(timing["path_counts"].items()))
            print(f"dev {args.task} {args.scheme} {args.model}: outer fold {timing['fold']} ({timing['label']}), "
                  f"{timing['n_trials']} trials x {timing['n_inner_fits'] // timing['n_trials']} inner fits, "
                  f"{timing['trial_seconds_mean']:.2f} s per trial, estimated final fit "
                  f"{timing['est_final_fit_seconds']:.2f} s" + (f"; paths: {paths}" if paths else ""))
        print(f"dev run {meta['seconds']:.1f} s in total (no error values are printed, logged or written)")
    else:
        print(f"{args.task} {args.scheme} {args.model}: {result['status']} in {meta['seconds']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
