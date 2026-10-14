"""D-003 ridge tuning: a non-finite inner candidate loses with an infinite score.

The frozen functions are imported and wrapped in this process, including in each spawned
worker. No frozen modelling or scoring logic is copied. The wrappers are scoped and restore
the imported modules on exit; final and permuted prediction checks remain the frozen ones.
Run this entry point from the repository root with Python -I.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import sys
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True  # -I ignores PYTHONDONTWRITEBYTECODE; imports must stay read-only.
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src import artifacts, evaluate, guard, permute, train, tuning  # noqa: E402
from src.data import load_tables  # noqa: E402
from src.folds import FoldPlan, load_config, plan_folds  # noqa: E402
from src.models.common import configure_torch  # noqa: E402
from src.tuning import Context, NonFinitePredictionError, TuneOutcome  # noqa: E402

IDENTITY = {"posthoc": "ridge_candidate_loses", "decision": "DECISIONS.md D-003"}
LOSS_COLUMNS = ["outer_fold", "inner_split", "alpha", "count"]
RULE_TEXT = (
    "In ridge tuning, an alpha with a non-finite inner-validation prediction receives an "
    "infinite trial score and loses.\n"
    "The run still stops if every alpha has an infinite score, or if a final-fit or "
    "permuted prediction is non-finite.\n"
    "Everything else follows the frozen procedure; all outputs are post hoc and "
    "Supplementary only under DECISIONS.md D-003.\n"
)
_FROZEN_TUNE = tuning.tune_fold
_FROZEN_INNER = tuning.evaluate_inner
_FROZEN_FOLD_JOB = train.run_fold_job
_FROZEN_ASSEMBLE = train.assemble
_FROZEN_CHUNK = permute.run_chunk


def write_json(path: Path, payload: dict) -> None:
    """Write provenance on every JSON produced by this procedure, including part files."""
    artifacts.write_json_atomic(path, {**payload, **IDENTITY})


def variant_config(cfg: dict) -> dict:
    """Redirect results and distinguish resume hashes without changing model settings."""
    out = copy.deepcopy(cfg)
    out["paths"]["results"] = str(artifacts.Layout(cfg).root / "posthoc" / IDENTITY["posthoc"])
    out.update(IDENTITY)
    return out


def assert_ready(cfg: dict, *, scoring: bool = False) -> dict:
    """Check the freeze, committed main selections and D-003's completed-main-run gate."""
    provenance = guard.assert_test_allowed()
    main = artifacts.Layout(cfg)
    evaluate.assert_selection_committed(main)
    log = main.root / "logs" / "autopilot.log"
    if not log.is_file() or not any(line.strip().endswith("DONE evaluate")
                                 for line in log.read_text(encoding="utf-8").splitlines()):
        raise guard.GuardError("D-003 requires DONE evaluate in results/logs/autopilot.log")
    if scoring:
        scores = main.metrics("scores.csv")
        if not scores.is_file():
            raise guard.GuardError("main results/metrics/scores.csv is missing; score the frozen run first")
        guard.assert_files_committed([os.path.relpath(scores, guard.REPO_ROOT)])
    return provenance


def tune_fold(ctx: Context, plan: FoldPlan, n_trials: int, db_path=None,
              *, losses: list[dict] | None = None) -> TuneOutcome:
    """Apply D-003 only to ridge inner predictions; retain the frozen tuner otherwise."""
    if ctx.kind != "ridge":
        return _FROZEN_TUNE(ctx, plan, n_trials, db_path)
    failures = losses if losses is not None else []

    def losing_inner(context, hp, split, seed):
        if context.kind != "ridge":
            return _FROZEN_INNER(context, hp, split, seed)
        factory = context.learner
        fitted = []
        learners = []
        with contextlib.ExitStack() as stack:
            def capture_learner(parameters, learner_seed):
                learner = factory(parameters, learner_seed)
                learners.append(learner)
                fit = learner.fit

                def capture_fit(*args, **kwargs):
                    info = fit(*args, **kwargs)
                    fitted.append(info)
                    return info

                stack.enter_context(patch.object(learner, "fit", capture_fit))
                return learner

            stack.enter_context(patch.object(context, "learner", capture_learner))
            try:
                return _FROZEN_INNER(context, hp, split, seed)
            except NonFinitePredictionError as exc:
                # Parse the frozen check's count, without predicting a second time or
                # swallowing a non-finite error raised during fit or another stage.
                hit = re.search(r": (\d+) of \d+ predictions are not finite \(inner split (\d+)\);", str(exc))
                if hit is None or int(hit[2]) != split.index or not fitted:
                    raise
                failures.append({"outer_fold": plan.index, "inner_split": split.index,
                                 "alpha": hp["alpha"], "count": int(hit[1])})
                return tuning.InnerResult(
                    split.index, len(split.val_idx), float("inf"),
                    {h: float("inf") for h in context.cfg["design"]["horizons"]},
                    fitted[0], 0, learners[0].notes())

    with patch.object(tuning, "evaluate_inner", losing_inner):
        outcome = _FROZEN_TUNE(ctx, plan, n_trials, db_path)
    if np.isposinf(outcome.best_value):
        raise NonFinitePredictionError(
            f"{ctx.task} {ctx.scheme} {ctx.key}: every alpha in outer fold {plan.index} "
            "has an infinite inner score; the run stops under D-003")
    return outcome


def inner(cfg: dict, task: str, scheme: str, model: str,
          *, sample_set: str = "eligible", out: Path | None = None) -> dict:
    """Tune all non-skipped outer folds without any outer fit or outer prediction."""
    train.check_request(cfg, task, scheme, model, sample_set, False, None, None)
    if tuning.model_kind(task, model, cfg) != "ridge":
        raise train.RunRefused("inner alpha reports require a ridge model")
    configure_torch()
    tables = load_tables(cfg)
    cutoff = train.resolve_cutoff(cfg, tables, scheme, False, None)
    plans = plan_folds(task, scheme, tables.table(task), sample_set, cfg, cutoff)
    ctx = train.build_context(cfg, tables, task, scheme, model, sample_set, False)
    folds = []
    for plan in plans:
        if plan.skipped:
            folds.append({"outer_fold": plan.index, "label": plan.label, "skipped": plan.skipped})
            continue
        losses: list[dict] = []
        outcome = tune_fold(ctx, plan, cfg["budget"]["n_trials"], losses=losses)
        folds.append({"outer_fold": plan.index, "label": plan.label,
                      "chosen_alpha": outcome.hp["alpha"], "best_value": outcome.best_value,
                      "trials": outcome.trials.to_dict("records"),
                      "inner_scores": outcome.inner_scores.to_dict("records"),
                      "nonfinite_candidates": losses})
    report = {**IDENTITY, "task": task, "scheme": scheme, "model": model,
              "sample_set": sample_set, "inner_only": True, "folds": folds}
    if out is not None:
        write_json(Path(out), report)
    return report


def run_fold_job(job: train.FoldJob) -> dict:
    """Install the tuning and JSON wrappers inside each frozen runner's worker."""
    losses: list[dict] = []

    def variant_tune(ctx, plan, n_trials, db_path=None):
        return tune_fold(ctx, plan, n_trials, db_path, losses=losses)

    def fold_json(path, payload):
        if Path(path).name == "done.json":
            artifacts.write_csv_atomic(pd.DataFrame(losses, columns=LOSS_COLUMNS),
                                       Path(job.fold_dir) / "nonfinite_candidates.csv")
        write_json(path, payload)

    with patch.object(train, "tune_fold", variant_tune), patch.object(train, "write_json_atomic", fold_json):
        return _FROZEN_FOLD_JOB(job)


def assemble(layout: artifacts.Layout, ctx: Context, plans: list[FoldPlan], ran: list[FoldPlan],
             run_hash: str, cutoff: str | None, provenance: dict | None, seconds: float) -> dict:
    """Combine candidate failures before the frozen assembly marks the run complete."""
    tdir = layout.tuning(ctx.task, ctx.scheme, ctx.key)
    losses = pd.concat([train._read(tdir / f"fold{p.index:02d}" / "nonfinite_candidates.csv")
                        for p in ran], ignore_index=True)
    artifacts.write_csv_atomic(losses, tdir / "nonfinite_candidates.csv")

    def frozen_csv(frame, path, float_format=None):
        # The frozen assembler flattens chosen.json. Keep its CSV schema despite
        # the provenance added to that JSON, which otherwise becomes extra columns.
        if Path(path).name == "chosen.csv":
            frame = frame.drop(columns=list(IDENTITY), errors="ignore")
        artifacts.write_csv_atomic(frame, path, float_format=float_format)

    with patch.object(train, "write_csv_atomic", frozen_csv):
        return _FROZEN_ASSEMBLE(layout, ctx, plans, ran, run_hash, cutoff, provenance, seconds)


def run(cfg: dict, task: str, scheme: str, model: str, *, sample_set: str = "eligible",
        workers: int = 1, log=print) -> dict:
    """Reuse the frozen full runner under the post hoc root after all main-run gates."""
    train.check_request(cfg, task, scheme, model, sample_set, False, None, None, workers)
    if tuning.model_kind(task, model, cfg) != "ridge":
        raise train.RunRefused("D-003 full runs require a ridge model")
    assert_ready(cfg)
    post_cfg = variant_config(cfg)
    with (patch.object(train, "run_fold_job", run_fold_job),
          patch.object(train, "assemble", assemble),
          patch.object(train, "write_json_atomic", write_json)):
        result = train.run_model(post_cfg, task, scheme, model, sample_set=sample_set,
                                 workers=workers, log=log)
    artifacts.write_text_atomic(artifacts.Layout(post_cfg).root / "README.md", RULE_TEXT)
    return result


def run_chunk(job: permute.PermJob) -> list[int]:
    """Tag permutation parts in each worker without relaxing prediction checks."""
    with patch.object(permute, "write_json_atomic", write_json):
        return _FROZEN_CHUNK(job)


def permutations(cfg: dict, task: str, model: str, *, workers: int = 1, log=print) -> dict:
    """Reuse the frozen derangements, fixed alphas, observed check and permutation count."""
    assert_ready(cfg)
    if model not in cfg["permutation"]["tabular_models"][task]:
        raise permute.PermutationError("only permutation.tabular_models may be permuted post hoc")
    if workers < 1:
        raise permute.PermutationError("--workers must be at least 1")
    post_cfg = variant_config(cfg)
    layout = artifacts.Layout(post_cfg)
    meta_path = layout.tuning(task, "lopo", model) / "run_meta.json"
    if not meta_path.is_file():
        raise permute.PermutationError("a complete post hoc LOPO run is required before permutation")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not meta.get("complete") or any(meta.get(k) != v for k, v in IDENTITY.items()):
        raise permute.PermutationError("the observed run is not a complete D-003 post hoc run")
    for path in (layout.tuning(task, "lopo", model) / "chosen.csv",
                 layout.predictions(task, "lopo", model)):
        if not path.is_file():
            raise permute.PermutationError(f"missing observed-run artifact {path}")
    with (patch.object(permute, "run_chunk", run_chunk),
          patch.object(permute, "write_json_atomic", write_json)):
        return permute.run_permutations(
            post_cfg, task, model, workers=workers,
            selection_check=lambda _layout, _task: evaluate.assert_selection_committed(artifacts.Layout(cfg)),
            log=log)


def codex_agreement(layout: artifacts.Layout) -> pd.DataFrame:
    """Compare T2 ridge alphas and keyed predictions with the independent rebuild."""
    chosen_path = layout.tuning("T2", "lopo", "ridge_pw_wq") / "chosen.csv"
    columns = ["outer_fold", "held_out_pond", "alpha", "codex_alpha", "alpha_equal",
               "n_predictions", "max_abs_y_pred_log", "max_abs_w_pred"]
    if not chosen_path.is_file():
        return pd.DataFrame(columns=columns)
    chosen = train._read(chosen_path)
    reference_root = REPO_ROOT / "verification" / "codex_p4models"
    ref_inner = train._read(reference_root / "codex_inner.csv")
    ref_pred = train._read(reference_root / "codex_predictions.csv")
    ref_inner = ref_inner[(ref_inner["task"] == "T2") & (ref_inner["model"] == "ridge_pw_wq")]
    ref_pred = ref_pred[(ref_pred["task"] == "T2") & (ref_pred["model"] == "ridge_pw_wq")]
    predictions = train._read(layout.predictions("T2", "lopo", "ridge_pw_wq"))
    rows = []
    keys = ["pond", "origin_date", "date", "horizon"]
    for fold in chosen.itertuples(index=False):
        pond = int(fold.test_ponds)
        alphas = ref_inner.loc[ref_inner["held_out_pond"] == pond, "alpha"].unique()
        if len(alphas) != 1:
            raise evaluate.EvaluationError(f"Codex has {len(alphas)} chosen alphas for pond {pond}")
        left = predictions[predictions["fold"] == fold.outer_fold]
        right = ref_pred[ref_pred["held_out_pond"] == pond]
        matched = left.merge(right, on=keys, how="outer", validate="one_to_one",
                             suffixes=("", "_codex"), indicator=True)
        if matched.empty or (matched["_merge"] != "both").any():
            raise evaluate.EvaluationError(f"Codex and post hoc predictions cover different rows for pond {pond}")
        rows.append({"outer_fold": fold.outer_fold, "held_out_pond": pond,
                     "alpha": fold.hp_alpha, "codex_alpha": alphas[0],
                     "alpha_equal": fold.hp_alpha == alphas[0], "n_predictions": len(matched),
                     **{f"max_abs_{name}": float((matched[name] - matched[f"{name}_codex"]).abs().max())
                        for name in ("y_pred_log", "w_pred")}})
    return pd.DataFrame(rows, columns=columns)


def score(cfg: dict) -> dict[str, pd.DataFrame]:
    """Score only saved post hoc predictions after the frozen scores are committed."""
    assert_ready(cfg, scoring=True)
    post_cfg = variant_config(cfg)
    layout = artifacts.Layout(post_cfg)
    frame = evaluate.load_predictions(layout)
    evaluate.check_same_samples(frame, cfg)
    by_seed = evaluate.per_pond_seed_table(frame)
    per_pond = evaluate.per_pond_table(by_seed)
    selections = {task: evaluate.read_selection(artifacts.Layout(cfg), task) for task in evaluate.TASKS}
    tables = {"per_pond_seed": by_seed, "per_pond": per_pond,
              "scores": evaluate.scores_table(cfg, by_seed, per_pond),
              "codex_agreement": codex_agreement(layout),
              "permutation": evaluate.permutation_table(cfg, layout, by_seed, selections)}
    for name, table in tables.items():
        table = table.assign(**IDENTITY)
        tables[name] = table
        folder = layout.stats if name == "permutation" else layout.metrics
        artifacts.write_csv_atomic(table, folder(f"{name}.csv"))
    return tables


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("inner", "run", "permute"):
        sub = commands.add_parser(name)
        sub.add_argument("--task", required=True, choices=["T1", "T2"])
        sub.add_argument("--model", required=True)
        if name != "permute":
            sub.add_argument("--scheme", required=True, choices=["lopo", "chrono"])
            sub.add_argument("--sample-set", default="eligible", choices=["eligible", "all"])
        if name == "inner":
            sub.add_argument("--out", type=Path)
        else:
            sub.add_argument("--workers", type=int, default=1)
    commands.add_parser("score")
    args = parser.parse_args(argv)
    cfg = load_config()
    try:
        if args.command == "inner":
            report = inner(cfg, args.task, args.scheme, args.model, sample_set=args.sample_set, out=args.out)
            print(json.dumps(report, indent=2, sort_keys=True))
        elif args.command == "run":
            run(cfg, args.task, args.scheme, args.model, sample_set=args.sample_set, workers=args.workers)
        elif args.command == "permute":
            permutations(cfg, args.task, args.model, workers=args.workers)
        else:
            tables = score(cfg)
            print(f"wrote {len(tables)} post hoc metric and agreement tables")
    except (guard.GuardError, train.RunRefused, permute.PermutationError, evaluate.EvaluationError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    except NonFinitePredictionError as exc:
        print(f"ERROR {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
