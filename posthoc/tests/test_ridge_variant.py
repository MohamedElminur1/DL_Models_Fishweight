"""D-003 checks on real tables, inner rows only; all writes use tmp_path."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from posthoc import ridge_variant as variant
from src import artifacts, guard, train, tuning
from src.baselines import FitInfo, Learner, RidgeModel
from src.data import load_tables
from src.folds import load_config, plan_folds


@pytest.fixture(autouse=True)
def prohibit_outer_work(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("an outer fit, prediction or score was attempted")

    monkeypatch.setattr(tuning, "fit_and_predict", forbidden)
    monkeypatch.setattr(variant.evaluate, "load_predictions", forbidden)
    monkeypatch.setattr(variant.permute, "check_observed", forbidden)


@pytest.fixture(scope="module")
def world():
    cfg = load_config()
    return cfg, load_tables(cfg)


@pytest.fixture(params=[("T1", "ridge_time_wq"), ("T2", "ridge_pw_wq"), ("T1", "ridge_time_wq_w28")])
def report(request, world, tmp_path):
    cfg, _tables = world
    task, model = request.param
    path = tmp_path / "inner.json"
    result = variant.inner(cfg, task, "lopo", model, out=path)
    assert json.loads(path.read_text()) == result
    assert result["posthoc"] == "ridge_candidate_loses"
    assert result["decision"] == "DECISIONS.md D-003"
    return result


def test_real_inner_results(report):
    folds = [f for f in report["folds"] if "skipped" not in f]
    assert len(folds) == 7
    if report["model"] == "ridge_time_wq":
        root = artifacts.Layout(load_config()).tuning("T1", "lopo", "ridge_time_wq")
        chosen = pd.read_csv(root / "chosen.csv", float_precision="round_trip")
        scores = pd.read_csv(root / "inner_scores.csv", float_precision="round_trip")
        got = pd.DataFrame([{"outer_fold": f["outer_fold"], "hp_alpha": f["chosen_alpha"]} for f in folds])
        pd.testing.assert_frame_equal(got, chosen[["outer_fold", "hp_alpha"]], check_exact=True)
        actual = pd.DataFrame([r for f in folds for r in f["inner_scores"]])
        pd.testing.assert_frame_equal(actual, scores, check_exact=False, rtol=0, atol=1e-12)
        assert not any(f["nonfinite_candidates"] for f in folds)
    else:
        alphas = tuning.ridge_alphas(load_config())
        bad_alphas = alphas[:1] if report["task"] == "T2" else alphas[:7]
        expected = [{"outer_fold": fold, "inner_split": 1, "alpha": alpha, "count": 1}
                    for fold in (2, 3) for alpha in bad_alphas]
        assert [r for f in folds for r in f["nonfinite_candidates"]] == expected
        for fold in folds:
            assert len(fold["trials"]) == 13
            for trial in fold["trials"]:
                loses = fold["outer_fold"] in (2, 3) and trial["params_alpha"] in bad_alphas
                assert np.isposinf(trial["value"]) == loses
        print(f"chosen {report['task']} {report['model']}: "
              + ", ".join(f"fold {f['outer_fold']}={f['chosen_alpha']:.17g}" for f in folds))


class InfiniteLearner(Learner):
    task = "T1"

    def _fit(self, train, val, epochs):
        return FitInfo()

    def predict_log(self, df):
        return np.full(len(df), np.inf)


def test_all_infinite_candidates_still_stop(world, monkeypatch):
    cfg, tables = world
    ctx = train.build_context(cfg, tables, "T1", "lopo", "ridge_time_wq", "eligible", False)
    plan = next(p for p in plan_folds("T1", "lopo", tables.t1, "eligible", cfg) if not p.skipped)
    monkeypatch.setattr(ctx, "learner", lambda hp, seed: InfiniteLearner())
    with pytest.raises(tuning.NonFinitePredictionError):
        variant.tune_fold(ctx, plan, cfg["budget"]["n_trials"])


def test_inner_never_predicts_outer_test_rows(world, monkeypatch, tmp_path):
    cfg, tables = world
    predict = RidgeModel.predict_log
    tune = variant.tune_fold
    calls = []

    def checked_tune(ctx, plan, *args, **kwargs):
        allowed = set(plan.train_idx)
        forbidden = set(plan.test_idx)

        def predict_inner(self, rows):
            assert set(rows.index) <= allowed
            assert set(rows.index).isdisjoint(forbidden)
            calls.append((plan.index, tuple(rows.index)))
            return predict(self, rows)

        with monkeypatch.context() as scoped:
            scoped.setattr(RidgeModel, "predict_log", predict_inner)
            return tune(ctx, plan, *args, **kwargs)

    monkeypatch.setattr(variant, "tune_fold", checked_tune)
    for task, model in (("T2", "ridge_pw_wq"), ("T1", "ridge_time_wq_w28")):
        variant.inner(cfg, task, "lopo", model, out=tmp_path / f"{task}.json")
    assert len(calls) == 2 * 7 * 13 * 3 * 2


def invoke(command, cfg):
    if command == "run":
        return variant.run(cfg, "T2", "lopo", "ridge_pw_wq")
    if command == "permute":
        return variant.permutations(cfg, "T2", "ridge_pw_wq")
    return variant.score(cfg)


@pytest.mark.parametrize("command", ["run", "permute", "score"])
@pytest.mark.parametrize("gate", ["freeze", "selections_missing", "selections_uncommitted", "main_not_done"])
def test_main_run_preconditions_fail_closed(world, tmp_path, monkeypatch, command, gate):
    cfg = copy.deepcopy(world[0])
    cfg["paths"]["results"] = str(tmp_path / "results")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    if gate != "freeze":
        monkeypatch.setattr(guard, "assert_test_allowed", lambda: {"tag": "frozen"})
    if gate == "selections_uncommitted":
        for task in ("T1", "T2"):
            variant.write_json(artifacts.Layout(cfg).selection(task), {"primary_baseline": "ridge_time"})
    if gate == "main_not_done":
        monkeypatch.setattr(variant.evaluate, "assert_selection_committed", lambda layout: None)
    with pytest.raises(guard.GuardError):
        invoke(command, cfg)
    assert not (tmp_path / "results" / "posthoc").exists()


@pytest.mark.parametrize("present", [False, True])
def test_score_needs_existing_committed_main_scores(world, tmp_path, monkeypatch, present):
    cfg = copy.deepcopy(world[0])
    cfg["paths"]["results"] = str(tmp_path / "results")
    monkeypatch.setattr(guard, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(guard, "assert_test_allowed", lambda: {"tag": "frozen"})
    monkeypatch.setattr(variant.evaluate, "assert_selection_committed", lambda layout: None)
    artifacts.write_text_atomic(artifacts.Layout(cfg).root / "logs" / "autopilot.log", "17:00 DONE evaluate\n")
    if present:
        artifacts.write_text_atomic(artifacts.Layout(cfg).metrics("scores.csv"), "committed prerequisite only\n")
    with pytest.raises(guard.GuardError, match="not tracked|missing"):
        variant.score(cfg)
    assert not (tmp_path / "results" / "posthoc").exists()


def test_worker_installs_rule_and_tags_fold_records_before_completion(world, tmp_path, monkeypatch):
    cfg, tables = world
    ctx = train.build_context(cfg, tables, "T2", "lopo", "ridge_pw_wq", "eligible", False)
    plan = next(p for p in plan_folds("T2", "lopo", tables.t2, "eligible", cfg) if p.index == 2)
    job = train.FoldJob(cfg, "T2", "lopo", "ridge_pw_wq", "eligible", None, 2, False,
                        cfg["budget"]["n_trials"], "inner-only-test", str(tmp_path / "fold02"))
    original_tune, original_json = train.tune_fold, train.write_json_atomic

    def inner_only_job(received):
        assert received == job
        result = train.tune_fold(ctx, plan, job.n_trials)
        train.write_json_atomic(tmp_path / "fold02" / "chosen.json", {"hp": result.hp})
        train.write_json_atomic(tmp_path / "fold02" / "done.json", {"hash": job.run_hash})
        return {"fold": 2, "resumed": False}

    monkeypatch.setattr(variant, "_FROZEN_FOLD_JOB", inner_only_job)
    assert variant.run_fold_job(job) == {"fold": 2, "resumed": False}
    assert train.tune_fold is original_tune
    assert train.write_json_atomic is original_json
    failures = pd.read_csv(tmp_path / "fold02" / "nonfinite_candidates.csv")
    assert failures.to_dict("records") == [{"outer_fold": 2, "inner_split": 1, "alpha": 0.001, "count": 1}]
    for name in ("chosen.json", "done.json"):
        payload = json.loads((tmp_path / "fold02" / name).read_text())
        assert all(payload[k] == v for k, v in variant.IDENTITY.items())


def test_fit_failure_is_not_an_inner_prediction_failure(world, monkeypatch):
    cfg, tables = world
    ctx = train.build_context(cfg, tables, "T1", "lopo", "ridge_time_wq", "eligible", False)
    plan = next(p for p in plan_folds("T1", "lopo", tables.t1, "eligible", cfg) if not p.skipped)

    class FailedFit(InfiniteLearner):
        def _fit(self, train, val, epochs):
            raise tuning.NonFinitePredictionError("non-finite during fitting")

    monkeypatch.setattr(ctx, "learner", lambda hp, seed: FailedFit())
    with pytest.raises(tuning.NonFinitePredictionError, match="during fitting"):
        variant.tune_fold(ctx, plan, cfg["budget"]["n_trials"])
