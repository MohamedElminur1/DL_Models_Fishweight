"""End-to-end tests of src/train.py on the synthetic tables of tests/synth_p4.py.

Every test runs in its own temporary directory with its own tables and results, so nothing is
written to the repository. Full runs (no --dev) call the freeze guard; the `frozen` fixture
stands in for it, and one test runs the real guard against a directory that is not a git
repository. Budgets are tiny: two trials, six epochs.
"""

import copy
import fcntl
import json
import os
import re

import numpy as np
import pandas as pd
import pytest
import yaml

from src import guard, train, tuning
from src.artifacts import RUNTIME_COLUMNS
from src.folds import load_config, plan_folds, submission_order
from src.tuning import NonFinitePredictionError
from tests.p4env import make_env, quiet


def plans_of(env, task, scheme="lopo", sample_set="eligible", cutoff=None):
    plans = plan_folds(task, scheme, env.tables.table(task), sample_set, env.cfg, cutoff)
    return [p for p in plans if not p.skipped]


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setattr(guard, "assert_test_allowed", lambda: {"tag": "analysis-plan-frozen", "commit": "0" * 40})


@pytest.fixture
def env(tmp_path):
    return make_env(tmp_path)


# ----------------------------------------------------------------------------------------
# Request checks
# ----------------------------------------------------------------------------------------

@pytest.mark.parametrize("task,scheme,model,extra,message", [
    ("T1", "lopo", "persistence", {}, "not a T1 model"),
    ("T2", "lopo", "gc", {}, "not a T2 model"),
    ("T1", "lopo", "lstm", {"sample_set": "all"}, "only for"),
    ("T2", "lopo", "persistence", {"sample_set": "all"}, "only for"),
    ("T1", "lopo", "ridge_time", {"n_trials": 3}, "only allowed with --dev"),
    ("T1", "lopo", "ridge_time", {"cutoff": "2021-08-28"}, "only allowed with --dev"),
    ("T1", "lopo", "ridge_time", {"dev": True, "cutoff": "2021-08-28"}, "--scheme chrono"),
    ("T1", "lopo", "ridge_time", {"dev_folds": 2}, "only allowed with --dev"),
    ("T1", "lopo", "ridge_time", {"workers": 0}, "at least 1"),
    ("T1", "weird", "ridge_time", {}, "scheme must be one of"),
])
def test_bad_requests_are_refused_before_anything_runs(env, task, scheme, model, extra, message):
    with pytest.raises(train.RunRefused, match=message):
        train.run_model(env.cfg, task, scheme, model, log=quiet, **extra)
    assert not env.live.root.joinpath("tuning").exists() and not env.live.root.joinpath("predictions").exists()


def test_every_model_of_the_config_is_known_to_the_runner():
    cfg = load_config()
    for task in ("T1", "T2"):
        names = train.models_of(task, cfg)
        assert len(names) == len(set(names)) and {"ridge_time" if task == "T1" else "ridge_pw", "lstm", "mlp_static"} <= set(names)


def test_a_full_chrono_run_needs_the_cutoff_of_the_rule(env, frozen):
    rule = env.cfg["chrono"]["cutoff"]
    for bad, message in ((None, "is not set"), ("2021-01-01", "differs from the date the rule")):
        cfg = copy.deepcopy(env.cfg)
        cfg["chrono"]["cutoff"] = bad
        with pytest.raises(train.RunRefused, match=message):
            train.run_model(cfg, "T1", "chrono", "ridge_time", log=quiet)
    assert train.resolve_cutoff(env.cfg, env.tables, "chrono", False, None) == rule
    assert train.resolve_cutoff(env.cfg, env.tables, "lopo", False, None) is None
    other = copy.deepcopy(env.cfg)
    other["chrono"]["cutoff"] = None
    assert train.resolve_cutoff(other, env.tables, "chrono", True, "2021-08-14") == "2021-08-14"   # dev may choose
    assert train.resolve_cutoff(other, env.tables, "chrono", True, None) == rule                   # else the rule's


def test_the_real_guard_stops_a_full_run_before_any_file_is_written(env, tmp_path_factory, monkeypatch):
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path_factory.mktemp("not_a_repo"))
    with pytest.raises(guard.GuardError):
        train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    assert not env.live.root.exists()


# ----------------------------------------------------------------------------------------
# A complete LOPO run
# ----------------------------------------------------------------------------------------

def test_a_lopo_run_writes_predictions_inner_scores_chosen_and_runtime_rows(env, frozen):
    result = train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    assert result["status"] == "complete"
    plans = plans_of(env, "T1")
    tdir = env.live.tuning("T1", "lopo", "ridge_time")

    pred = pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time"))
    assert len(pred) == sum(len(p.test_idx) for p in plans) and (pred["seed"] == 0).all()
    assert sorted(pred["fold"].unique()) == [p.index for p in plans]
    table = env.tables.t1
    np.testing.assert_allclose(pred["y_true_log"], table.loc[pred["row_id"], "log_w"])
    np.testing.assert_allclose(pred["w_true"], table.loc[pred["row_id"], "w"])
    assert list(pred["pond"]) == list(table.loc[pred["row_id"], "pond"])
    for plan in plans:                                             # each row is predicted by its own pond's fold
        assert set(pred.loc[pred["fold"] == plan.index, "pond"]) == set(plan.test_ponds)

    inner = pd.read_csv(tdir / "inner_scores.csv")
    assert list(inner.columns) == ["outer_fold", "inner_fold", "n_samples", "S"]
    assert len(inner) == 3 * len(plans) and np.isfinite(inner["S"]).all()
    chosen = pd.read_csv(tdir / "chosen.csv")
    assert list(chosen["outer_fold"]) == [p.index for p in plans] and chosen["hp_alpha"].notna().all()
    assert chosen["inner_value"].tolist() == pytest.approx(inner.groupby("outer_fold")["S"].mean().tolist())

    log = pd.read_csv(env.live.runtime_log(), dtype=str, keep_default_na=False)
    assert list(log.columns) == RUNTIME_COLUMNS
    assert (log["stage"] == "plan").sum() == len(plans) and (log["stage"] == "final").sum() == len(plans)
    assert (log["stage"] == "trial").sum() == 13 * len(plans)
    meta = json.loads((tdir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["complete"] and meta["folds_run"] == [p.index for p in plans] and meta["guard"]["commit"] == "0" * 40
    assert meta["n_predictions"] == len(pred) and meta["cutoff"] is None


def test_a_pond_without_eligible_samples_is_skipped_and_recorded(tmp_path, frozen):
    env = make_env(tmp_path, ponds=(1, 2, 3, 4, 6, 8), sparse=(8,))
    train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    meta = json.loads((env.live.tuning("T1", "lopo", "ridge_time") / "run_meta.json").read_text(encoding="utf-8"))
    assert [(s["label"], s["reason"]) for s in meta["skipped_folds"]] == [("pond08", "no test samples")]
    log = pd.read_csv(env.live.runtime_log(), dtype=str, keep_default_na=False)
    skip = log[log["stage"] == "skip"]
    assert len(skip) == 1 and "pond08" in skip["notes"].iloc[0]
    pred = pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time"))
    assert 8 not in set(pred["pond"])
    everything = train.run_model(env.cfg, "T1", "lopo", "ridge_time", sample_set="all", log=quiet)
    all_pred = pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time__all"))
    assert 8 in set(all_pred["pond"]) and len(all_pred) > len(pred) and (all_pred["sample_set"] == "all").all()
    assert everything["meta"]["skipped_folds"] == []


def test_the_runtime_log_keeps_the_rows_of_other_models_and_is_not_duplicated_by_a_rerun(env, frozen):
    train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    train.run_model(env.cfg, "T1", "lopo", "gc", log=quiet)
    before = env.live.runtime_log().read_bytes()
    assert set(pd.read_csv(env.live.runtime_log())["model"]) == {"ridge_time", "gc"}
    assert train.run_model(env.cfg, "T1", "lopo", "gc", log=quiet)["status"] == "no-op"
    assert env.live.runtime_log().read_bytes() == before


# ----------------------------------------------------------------------------------------
# Re-running: no-op, refusal, resume
# ----------------------------------------------------------------------------------------

def test_a_finished_run_is_a_no_op_and_a_changed_config_is_refused(env, frozen):
    train.run_model(env.cfg, "T2", "lopo", "persistence", log=quiet)
    files = {p: p.read_bytes() for p in env.root.joinpath("results").rglob("*") if p.is_file() and p.suffix != ".lock"}
    assert train.run_model(env.cfg, "T2", "lopo", "persistence", log=quiet)["status"] == "no-op"
    assert {p: p.read_bytes() for p in files} == files
    changed = copy.deepcopy(env.cfg)
    changed["budget"]["patience"] = 4
    with pytest.raises(train.RunRefused, match="different config or inputs"):
        train.run_model(changed, "T2", "lopo", "persistence", log=quiet)
    assert {p: p.read_bytes() for p in files} == files


def test_the_run_hash_covers_config_inputs_and_request(env):
    base = train.compute_run_hash(env.cfg, "T1", "lopo", "ridge_time", "eligible", None)
    assert base == train.compute_run_hash(copy.deepcopy(env.cfg), "T1", "lopo", "ridge_time", "eligible", None)
    assert base != train.compute_run_hash(env.cfg, "T1", "lopo", "gbm_time", "eligible", None)
    assert base != train.compute_run_hash(env.cfg, "T1", "lopo", "ridge_time", "all", None)
    assert base != train.compute_run_hash(env.cfg, "T1", "chrono", "ridge_time", "eligible", "2021-08-28")
    changed = copy.deepcopy(env.cfg)
    changed["ridge"]["alpha_log10"]["num"] = 7
    assert base != train.compute_run_hash(changed, "T1", "lopo", "ridge_time", "eligible", None)
    table = env.root / "t1_table.csv"
    table.write_text(table.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert base != train.compute_run_hash(env.cfg, "T1", "lopo", "ridge_time", "eligible", None)


def test_a_run_that_died_halfway_resumes_and_gives_the_files_of_a_clean_run(tmp_path, frozen, monkeypatch):
    clean = make_env(tmp_path / "clean")
    train.run_model(clean.cfg, "T1", "lopo", "ridge_time", log=quiet)
    env = make_env(tmp_path / "interrupted")
    plans = plans_of(env, "T1")
    order = submission_order(plans)                                # the order in which the runner starts the folds
    doomed = order[2].index
    real = train.tune_fold

    def dying(ctx, plan, n_trials, db_path=None):
        if plan.index == doomed:
            raise RuntimeError("the machine went away")
        return real(ctx, plan, n_trials, db_path)

    monkeypatch.setattr(train, "tune_fold", dying)
    with pytest.raises(RuntimeError, match="went away"):
        train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    tdir = env.live.tuning("T1", "lopo", "ridge_time")
    assert not (tdir / "run_meta.json").exists() and not env.live.predictions("T1", "lopo", "ridge_time").exists()
    assert not (tdir / f"fold{doomed:02d}" / "done.json").exists()
    finished = sorted(p.parent.name for p in tdir.glob("fold*/done.json"))
    assert finished == sorted(f"fold{p.index:02d}" for p in order[:2])
    monkeypatch.setattr(train, "tune_fold", real)
    stale = tdir / f"fold{doomed:02d}" / "stale.txt"
    stale.parent.mkdir(exist_ok=True)
    stale.write_text("left by the interrupted attempt", encoding="utf-8")
    messages = []
    train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=messages.append)
    assert sum("resumed" in m for m in messages) == 2 and sum("finished" in m for m in messages) == len(plans) - 2
    assert not stale.exists()                                       # an unfinished fold restarts from nothing
    for name in ("inner_scores.csv", "chosen.csv"):
        pd.testing.assert_frame_equal(pd.read_csv(tdir / name), pd.read_csv(clean.live.tuning("T1", "lopo", "ridge_time") / name))
    pd.testing.assert_frame_equal(pd.read_csv(env.live.predictions("T1", "lopo", "ridge_time")),
                                  pd.read_csv(clean.live.predictions("T1", "lopo", "ridge_time")))


def test_a_fold_is_marked_done_only_after_all_its_files_are_written(env, frozen, monkeypatch):
    real = train.write_json_atomic

    def failing(path, payload):
        if path.name == "timing.json" and path.parent.name.startswith("fold"):
            raise OSError("disk full")
        return real(path, payload)

    monkeypatch.setattr(train, "write_json_atomic", failing)
    with pytest.raises(OSError, match="disk full"):
        train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    tdir = env.live.tuning("T1", "lopo", "ridge_time")
    assert len(list(tdir.glob("fold*/predictions.csv"))) == 1 and not list(tdir.glob("fold*/done.json"))
    monkeypatch.setattr(train, "write_json_atomic", real)
    assert train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)["status"] == "complete"


def test_a_fold_finished_with_another_config_is_never_reused(env, frozen):
    train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    tdir = env.live.tuning("T1", "lopo", "ridge_time")
    (tdir / "run_meta.json").unlink()
    changed = copy.deepcopy(env.cfg)
    changed["ridge"]["alpha_log10"]["num"] = 7
    with pytest.raises(train.RunRefused, match="finished with a different config"):
        train.run_model(changed, "T1", "lopo", "ridge_time", log=quiet)


def test_a_damaged_part_file_is_caught_when_the_model_files_are_assembled(env, frozen):
    train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    tdir = env.live.tuning("T1", "lopo", "ridge_time")
    (tdir / "run_meta.json").unlink()
    part = tdir / "fold00" / "predictions.csv"
    frame = pd.read_csv(part)
    frame.iloc[1:].to_csv(part, index=False)
    with pytest.raises(RuntimeError, match="predictions written"):
        train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)


def test_two_processes_cannot_run_the_same_model_at_once(env, frozen):
    lock = env.live.tuning("T1", "lopo", "ridge_time").with_name("ridge_time.lock")
    lock.parent.mkdir(parents=True)
    with lock.open("w") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        with pytest.raises(train.RunRefused, match="another process"):
            train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)
    assert train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)["status"] == "complete"


# ----------------------------------------------------------------------------------------
# Non-finite predictions
# ----------------------------------------------------------------------------------------

def test_a_non_finite_prediction_stops_the_model_with_a_log_row_and_no_prediction_file(env, frozen, monkeypatch):
    original = tuning.Context.learner

    def broken(self, hp, seed):
        learner = original(self, hp, seed)
        learner.predict_log = lambda df: np.full(len(df), np.nan)
        return learner

    monkeypatch.setattr(tuning.Context, "learner", broken)
    messages = []
    with pytest.raises(NonFinitePredictionError, match="T1 lopo ridge_time"):
        train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=messages.append)
    assert any("ERROR" in m for m in messages)
    log = pd.read_csv(env.live.runtime_log(), dtype=str, keep_default_na=False)
    assert list(log["stage"]) == ["error"] and "not finite" in log["notes"].iloc[0]
    assert not env.live.predictions("T1", "lopo", "ridge_time").exists()
    assert not (env.live.tuning("T1", "lopo", "ridge_time") / "run_meta.json").exists()
    monkeypatch.setattr(tuning.Context, "learner", original)
    assert train.run_model(env.cfg, "T1", "lopo", "ridge_time", log=quiet)["status"] == "complete"
    assert "error" not in set(pd.read_csv(env.live.runtime_log())["stage"])


# ----------------------------------------------------------------------------------------
# Neural models: seeds, epochs, learning curves
# ----------------------------------------------------------------------------------------

def test_a_neural_run_fits_five_seeds_and_keeps_every_learning_curve(env, frozen):
    train.run_model(env.cfg, "T1", "lopo", "gru", log=quiet)
    plans = plans_of(env, "T1")
    tdir = env.live.tuning("T1", "lopo", "gru")
    pred = pd.read_csv(env.live.predictions("T1", "lopo", "gru"))
    assert sorted(pred["seed"].unique()) == [0, 1, 2, 3, 4]
    assert len(pred) == 5 * sum(len(p.test_idx) for p in plans)
    chosen = pd.read_csv(tdir / "chosen.csv")
    assert (chosen["epochs"].between(1, env.cfg["budget"]["max_epochs"])).all()
    assert {"hp_hidden", "hp_layers", "hp_dropout", "hp_lr", "hp_weight_decay", "hp_batch_size"} <= set(chosen.columns)

    curves = pd.read_csv(tdir / "curves.csv.gz")
    assert list(curves.columns) == train.CURVE_COLUMNS
    log = pd.read_csv(env.live.runtime_log())
    inner, final = curves[curves["stage"] == "inner"], curves[curves["stage"] == "final"]
    assert len(inner) == log.loc[log["stage"] == "trial", "epochs"].sum()
    assert len(final) == log.loc[log["stage"] == "final", "epochs"].sum()
    assert inner["val_mse"].notna().all() and final["val_mse"].isna().all() and final["train_mse"].notna().all()
    assert set(final["seed"]) == {0, 1, 2, 3, 4} and set(inner["seed"]) == {env.cfg["seeds"]["tuning"]}
    assert set(inner["inner_fold"]) == {0, 1, 2} and final["inner_fold"].isna().all()
    assert (final.groupby(["outer_fold", "seed"])["epoch"].max().values == chosen["epochs"].repeat(5).values).all()


# ----------------------------------------------------------------------------------------
# The calendar hold-out
# ----------------------------------------------------------------------------------------

def test_the_chrono_run_predicts_the_hold_out_and_records_the_excluded_samples(env, frozen):
    cut = env.cfg["chrono"]["cutoff"]
    for task, model in (("T1", "ridge_time"), ("T2", "ridge_pw")):
        train.run_model(env.cfg, task, "chrono", model, log=quiet)
        plan = plans_of(env, task, "chrono", cutoff=cut)[0]
        pred = pd.read_csv(env.live.predictions(task, "chrono", model))
        assert len(pred) == len(plan.test_idx) and (pred["fold"] == 0).all()
        assert (pd.to_datetime(pred["date"]) >= pd.Timestamp(cut)).all()
        if task == "T2":
            margin = pd.Timedelta(days=env.cfg["chrono"]["test_origin_margin_days"])
            assert (pd.to_datetime(pred["origin_date"]) >= pd.Timestamp(cut) - margin).all()
        log = pd.read_csv(env.live.runtime_log(), dtype=str, keep_default_na=False)
        row = log[(log["model"] == model) & (log["stage"] == "plan")].iloc[0]
        assert f"n_excluded={len(plan.excluded_idx)}" in row["notes"] and f"n_test={len(plan.test_idx)}" in row["notes"]
        meta = json.loads((env.live.tuning(task, "chrono", model) / "run_meta.json").read_text(encoding="utf-8"))
        assert meta["cutoff"] == cut
        inner = pd.read_csv(env.live.tuning(task, "chrono", model) / "inner_scores.csv")
        assert len(inner) == 1                                      # one inner split


# ----------------------------------------------------------------------------------------
# --dev: timings and counts only
# ----------------------------------------------------------------------------------------

VALUE_COLUMNS = {"S", "S_h1", "S_h2", "value", "train_mse", "val_mse", "y_pred_log", "w_pred", "y_true_log", "w_true"}


@pytest.mark.parametrize("task,model", [("T1", "ridge_time"), ("T1", "gbm_time_wq"), ("T2", "persistence"),
                                        ("T1", "lstm"), ("T2", "tft"), ("T2", "linear")])
def test_a_dev_run_leaves_only_timings_and_counts(env, monkeypatch, capsys, task, model):
    monkeypatch.setattr(guard, "assert_test_allowed", lambda: pytest.fail("the guard is not used in dev"))
    result = train.run_model(env.cfg, task, "lopo", model, dev=True, log=print)
    assert result["status"] == "complete" and result["meta"]["values_withheld"]
    assert not env.live.root.joinpath("predictions").exists() and not env.live.tuning(task, "lopo", model).exists()
    root = env.dev.root
    assert not list(root.rglob("*.db")) and not list(root.rglob("predictions*"))
    assert not (env.dev.tuning(task, "lopo", model) / "chosen.csv").exists()
    seen = 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix in {".csv", ".gz", ".json"}):
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if path.name == "chosen.json":
                assert not {"hp", "inner_value", "best_trial", "epochs"} & set(payload)
            assert "inner_value" not in json.dumps(payload) and "best_value" not in json.dumps(payload)
        else:
            frame = pd.read_csv(path)
            for column in VALUE_COLUMNS & set(frame.columns):
                assert frame[column].isna().all(), f"{path.name}:{column}"
            assert not [c for c in frame.columns if re.search(r"inner\d+_S$", c)]
        seen += 1
    assert seen >= 5
    captured = capsys.readouterr()
    assert re.fullmatch(r"(?:T[12] lopo \S+: fold \d+ finished\n)+", captured.out) and captured.err == ""


def test_a_dev_run_covers_the_first_fold_unless_asked_for_more(env):
    one = train.run_model(env.cfg, "T1", "lopo", "ridge_time", dev=True, log=quiet)["meta"]
    assert one["folds_run"] == [plans_of(env, "T1")[0].index]
    two = train.run_model(env.cfg, "T1", "lopo", "ridge_time", dev=True, dev_folds=2, log=quiet)["meta"]
    assert two["folds_run"] == [p.index for p in plans_of(env, "T1")[:2]]
    timing = json.loads((env.dev.tuning("T1", "lopo", "ridge_time") / "timing.json").read_text(encoding="utf-8"))["folds"]
    assert [t["fold"] for t in timing] == two["folds_run"] and all(t["trial_seconds_mean"] > 0 for t in timing)


def test_dev_chrono_can_use_another_cutoff_and_the_n_trials_override(env):
    cfg = copy.deepcopy(env.cfg)
    cfg["chrono"]["cutoff"] = None
    cut = pd.Timestamp(env.cfg["chrono"]["cutoff"]) - pd.Timedelta(days=14)
    meta = train.run_model(cfg, "T1", "chrono", "gbm_time", dev=True, cutoff=cut.strftime("%Y-%m-%d"),
                           n_trials=3, log=quiet)["meta"]
    assert meta["cutoff"] == cut.strftime("%Y-%m-%d")
    trials = pd.read_csv(env.dev.tuning("T1", "chrono", "gbm_time") / "fold00" / "trials.csv")
    assert len(trials) == 3 and trials["value"].isna().all()


def test_the_fold_pool_runs_the_same_folds_as_the_main_process(env):
    train.run_model(env.cfg, "T1", "lopo", "gbm_time", dev=True, dev_folds=3, workers=1, log=quiet)
    tdir = env.dev.tuning("T1", "lopo", "gbm_time")
    assert {json.loads(p.read_text(encoding="utf-8"))["pid"] for p in tdir.glob("fold*/done.json")} == {os.getpid()}
    single = {p.parent.name: pd.read_csv(p).filter(like="params_") for p in tdir.glob("fold*/trials.csv")}
    inner_single = pd.read_csv(tdir / "inner_scores.csv")
    train.run_model(env.cfg, "T1", "lopo", "gbm_time", dev=True, dev_folds=3, workers=2, log=quiet)
    pids = {json.loads(p.read_text(encoding="utf-8"))["pid"] for p in tdir.glob("fold*/done.json")}
    assert pids and os.getpid() not in pids                           # the folds ran in worker processes
    pooled = {p.parent.name: pd.read_csv(p).filter(like="params_") for p in tdir.glob("fold*/trials.csv")}
    assert sorted(pooled) == sorted(single) and len(pooled) == 3
    for name in single:
        pd.testing.assert_frame_equal(single[name], pooled[name])
    pd.testing.assert_frame_equal(inner_single, pd.read_csv(tdir / "inner_scores.csv"))


# ----------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------

def write_config(env):
    path = env.root / "phase4_test.yaml"
    path.write_text(yaml.safe_dump(env.cfg), encoding="utf-8")
    return str(path)


def test_the_command_line_returns_zero_two_and_one(env, frozen, capsys, monkeypatch):
    config = write_config(env)
    base = ["--task", "T1", "--scheme", "lopo", "--config", config]
    assert train.main([*base, "--model", "ridge_time", "--dev"]) == 0
    out = capsys.readouterr().out
    lines = out.strip().splitlines()
    assert re.fullmatch(r"dev T1 lopo ridge_time: outer fold \d+ \(pond\d\d\), 13 trials x 3 inner fits, [\d.]+ s per "
                        r"trial, estimated final fit [\d.]+ s(?:; paths: .*)?", lines[-2])
    assert re.fullmatch(r"dev run [\d.]+ s in total \(no error values are printed, logged or written\)", lines[-1])

    assert train.main([*base, "--model", "persistence"]) == 2
    assert "refused:" in capsys.readouterr().err
    assert train.main([*base, "--model", "ridge_time"]) == 0
    assert "complete in" in capsys.readouterr().out
    assert train.main([*base, "--model", "ridge_time"]) == 0
    assert "no-op in" in capsys.readouterr().out

    original = tuning.Context.learner

    def broken(self, hp, seed):
        learner = original(self, hp, seed)
        learner.predict_log = lambda df: np.full(len(df), np.inf)
        return learner

    monkeypatch.setattr(tuning.Context, "learner", broken)
    assert train.main([*base, "--model", "gbm_time"]) == 1
    assert "ERROR" in capsys.readouterr().out
