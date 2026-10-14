"""Unit tests for src/projection.py: the projection of dev timings onto the full runs.

The timings are made up for the purpose; the fold plans come from the synthetic tables of
tests/p4env.py.
"""

import dataclasses

import pandas as pd
import pytest
import yaml

from src import projection, train
from src.folds import plan_folds, submission_order
from tests.p4env import make_env, quiet

ENTRY = {"fold": 0, "trial_seconds_mean": 2.0, "trial_seconds_max": 3.0, "est_final_fit_seconds": 4.0}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    return make_env(tmp_path_factory.mktemp("projection"))


def plans_of(env, task="T1"):
    return [p for p in plan_folds(task, "lopo", env.tables.table(task), "eligible", env.cfg) if not p.skipped]


def scaled(plan, factor, index):
    """A copy of `plan` with `factor` times the training rows (rows repeated; only their number is used)."""
    inner = tuple(dataclasses.replace(i, train_idx=i.train_idx * factor) for i in plan.inner)
    return dataclasses.replace(plan, index=index, train_idx=plan.train_idx * factor, inner=inner)


@pytest.mark.parametrize("durations,workers,expected", [
    ([10, 10, 5, 5], 2, 15),
    ([9, 5, 4, 3], 2, 12),        # 4 goes to the worker free at 5, 3 to whichever is free at 9
    ([9, 5, 4, 3], 1, 21),
    ([9, 5], 4, 9),               # more workers than jobs
    ([], 4, 0),
])
def test_jobs_go_to_the_first_free_worker_in_the_given_order(durations, workers, expected):
    assert projection.schedule(durations, workers) == expected


def test_a_one_fold_run_costs_trials_times_the_dev_trial_plus_the_final_fits(env):
    plan = plans_of(env)[:1]
    budget = env.cfg["budget"]["n_trials"]
    ridge = projection.project_model(env.cfg, "T1", "ridge_time", [ENTRY], plan, workers=4)
    alphas = projection.trials_of("T1", "ridge_time", env.cfg)
    assert alphas > budget                                           # ridge runs its alpha grid, not the Optuna budget
    assert ridge["sequential_s_low"] == pytest.approx(alphas * 2.0 + 4.0)
    assert ridge["sequential_s_high"] == pytest.approx(alphas * 3.0 + 4.0)
    assert ridge["wall_s_low"] == pytest.approx(ridge["sequential_s_low"])
    assert (ridge["dev_s_per_trial"], ridge["dev_s_per_final_fit"]) == (2.0, 4.0)
    gbm = projection.project_model(env.cfg, "T1", "gbm_time", [ENTRY], plan, workers=4)
    assert gbm["sequential_s_low"] == pytest.approx(budget * 2.0 + 4.0)
    deep = projection.project_model(env.cfg, "T1", "lstm", [ENTRY], plan, workers=4)
    assert deep["seeds"] == 5 and deep["sequential_s_low"] == pytest.approx(budget * 2.0 + 5 * 4.0)
    fixed = projection.project_model(env.cfg, "T1", "gc", [ENTRY], plan, workers=4)
    assert fixed["trials"] == 1 and fixed["sequential_s_low"] == pytest.approx(2.0 + 4.0)


def test_time_scales_with_training_rows_and_folds_are_shared_between_workers(env):
    first = plans_of(env)[0]
    plans = [first, scaled(first, 2, 1)]                              # the second fold has twice the rows
    one = projection.project_model(env.cfg, "T1", "gbm_time", [ENTRY], [first], workers=1)["sequential_s_low"]
    two = projection.project_model(env.cfg, "T1", "gbm_time", [ENTRY], plans, workers=2)
    assert two["sequential_s_low"] == pytest.approx(3 * one)
    assert two["wall_s_low"] == pytest.approx(2 * one)                # the larger fold alone on one worker
    assert projection.project_model(env.cfg, "T1", "gbm_time", [ENTRY], plans, workers=1)["wall_s_low"] == pytest.approx(3 * one)
    assert two["tuning_s_total"] + two["final_fit_s_total"] == pytest.approx(two["sequential_s_low"])
    three = [first, scaled(first, 1, 1), scaled(first, 2, 2)]         # given smallest first; the runner starts the largest first
    assert projection.project_model(env.cfg, "T1", "gbm_time", [ENTRY], three, workers=2)["wall_s_low"] == pytest.approx(2 * one)


def test_the_timed_fold_may_be_any_fold_of_the_run(env):
    first = plans_of(env)[0]
    plans = [first, scaled(first, 2, 1)]
    from_small = projection.project_model(env.cfg, "T1", "gbm_time", [ENTRY], plans, workers=1)
    doubled = {**ENTRY, "fold": 1, "trial_seconds_mean": 4.0, "trial_seconds_max": 6.0, "est_final_fit_seconds": 8.0}
    from_large = projection.project_model(env.cfg, "T1", "gbm_time", [doubled], plans, workers=1)
    assert from_large["sequential_s_low"] == pytest.approx(from_small["sequential_s_low"])


def test_the_permutation_test_costs_r_times_the_final_fits_of_all_folds_over_the_workers(env):
    first = plans_of(env)[0]
    plans = [first, scaled(first, 2, 1)]
    perm = env.cfg["permutation"]
    tabular = projection.project_model(env.cfg, "T1", "ridge_time_wq", [ENTRY], plans, workers=4)
    assert tabular["perm_R"] == perm["r_tabular"]
    assert tabular["perm_s_per_permutation"] == pytest.approx(4.0 * 3)             # one seed, 1 + 2 folds' worth of rows
    assert tabular["perm_wall_s"] == pytest.approx(12.0 * (perm["r_tabular"] / 4 + 1))   # plus the repeated observed fit
    deep = projection.project_model(env.cfg, "T1", "gru", [ENTRY], plans, workers=4)
    assert deep["perm_R"] == perm["r_deep"] and deep["perm_s_per_permutation"] == pytest.approx(4.0 * 3)
    none = projection.project_model(env.cfg, "T1", "gc", [ENTRY], plans, workers=4)
    assert none["perm_R"] == 0 and pd.isna(none["perm_wall_s"])


def test_folds_are_scheduled_in_the_order_the_runner_starts_them(env):
    plans = plans_of(env)
    sizes = [len(p.train_idx) for p in submission_order(plans)]
    assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) > 1


def test_the_table_lists_the_models_that_were_timed_and_names_the_others(tmp_path, capsys):
    env = make_env(tmp_path, ponds=(1, 2, 3, 4, 6, 8), sparse=(8,))        # pond 8 has no samples: its fold is skipped
    config = tmp_path / "phase4.yaml"
    config.write_text(yaml.safe_dump(env.cfg), encoding="utf-8")
    assert projection.main(["--config", str(config)]) == 2                   # nothing timed yet
    train.run_model(env.cfg, "T1", "lopo", "ridge_time", dev=True, log=quiet)
    train.run_model(env.cfg, "T1", "lopo", "lstm", dev=True, log=quiet)
    capsys.readouterr()
    assert projection.main(["--config", str(config), "--workers", "2"]) == 0
    out = capsys.readouterr()
    assert "ridge_time" in out.out and "lstm" in out.out and "gbm_time" not in out.out
    assert "T2 persistence" in out.err and "T1 ridge_time," not in out.err
    table = pd.read_csv(env.dev.root / "projection_lopo.csv")
    assert list(table["model"]) == ["ridge_time", "lstm"] and set(table["workers"]) == {2}
    assert table["wall_s_low"].notna().all() and (table["wall_s_low"] <= table["wall_s_high"]).all()
    assert set(table["n_folds"]) == {len(plans_of(env))} and len(plans_of(env)) < len(env.cfg["design"]["cohort"])
