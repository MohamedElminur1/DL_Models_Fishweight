"""Unit tests for src/select_primary.py: the choice of the primary baseline and primary deep model.

The inner-score files are written by the tests; their values are made up for the purpose.
"""

import json
import math

import pandas as pd
import pytest

from src import select_primary as sp
from src.artifacts import Layout
from src.folds import load_config

CFG = load_config()


def write_scores(layout, task, model, by_fold, extra=None):
    """Write an inner_scores.csv; `by_fold` maps an outer fold to its list of inner S values."""
    rows = [{"outer_fold": fold, "inner_fold": i, "n_samples": 5, "S": s}
            for fold, values in by_fold.items() for i, s in enumerate(values)]
    path = layout.tuning(task, "lopo", model) / "inner_scores.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def populate(layout, task, scores, folds=(0, 1)):
    """Every candidate of the task gets one value per inner fold from `scores[model]` (default 1.0)."""
    pools = [m for group in ("time_only", "deep") for m in CFG["models"][task][group]]
    for model in pools:
        value = scores.get(model, 1.0)
        write_scores(layout, task, model, {f: [value] * 3 for f in folds})


@pytest.fixture
def layout(tmp_path):
    cfg = {**CFG, "paths": {**CFG["paths"], "results": str(tmp_path / "results")}}
    return cfg, Layout(cfg)


def test_the_lowest_score_in_each_pool_wins_and_other_models_are_ignored(layout):
    cfg, lay = layout
    populate(lay, "T1", {"ridge_time": 0.30, "gbm_time": 0.35, "gc": 0.50, "gru": 0.20, "tft": 0.25})
    write_scores(lay, "T1", "ridge_time_wq", {0: [0.01] * 3, 1: [0.01] * 3})     # not a candidate
    write_scores(lay, "T1", "mlp_static", {0: [0.01] * 3, 1: [0.01] * 3})        # not a candidate
    record = sp.select_task(cfg, lay, "T1")
    assert record["primary_baseline"] == "ridge_time" and record["primary_deep"] == "gru"
    assert record["mean_inner_S"]["ridge_time"] == pytest.approx(0.30)
    assert set(record["mean_inner_S"]) == set(CFG["models"]["T1"]["time_only"]) | set(CFG["models"]["T1"]["deep"])
    assert record["pools"]["primary_baseline"] == ["gc", "cohort_mean", "ridge_time", "gbm_time"]


def test_t2_pools_are_the_t2_baselines_and_the_same_four_deep_models(layout):
    cfg, lay = layout
    populate(lay, "T2", {"persistence": 0.5, "sgr": 0.4, "cohort_growth": 0.3, "gbm_pw": 0.35, "lstm": 0.2})
    record = sp.select_task(cfg, lay, "T2")
    assert record["primary_baseline"] == "cohort_growth" and record["primary_deep"] == "lstm"
    assert record["pools"]["primary_baseline"][:2] == ["persistence", "linear"]


def test_a_score_is_the_mean_of_the_outer_fold_means_not_the_mean_of_all_rows(layout):
    cfg, lay = layout
    populate(lay, "T1", {})
    # fold 0 has three inner values, fold 1 one: mean of means = (0.2 + 0.6) / 2 = 0.4, mean of rows = 0.3
    write_scores(lay, "T1", "gbm_time", {0: [0.2, 0.2, 0.2], 1: [0.6]})
    assert sp.select_task(cfg, lay, "T1")["mean_inner_S"]["gbm_time"] == pytest.approx(0.4)


def test_ties_go_to_the_model_listed_first(layout):
    cfg, lay = layout
    populate(lay, "T1", {"cohort_mean": 0.3, "ridge_time": 0.3, "gbm_time": 0.3, "gc": 0.3,
                         "gru": 0.2, "lstm": 0.2, "tft": 0.2, "cnn_lstm": 0.2})
    record = sp.select_task(cfg, lay, "T1")
    assert record["primary_baseline"] == "gc" and record["primary_deep"] == "lstm"
    populate(lay, "T1", {"cohort_mean": 0.3, "ridge_time": 0.3, "gbm_time": 0.3, "gc": 0.4,
                         "gru": 0.2, "lstm": 0.3, "tft": 0.2, "cnn_lstm": 0.2})
    record = sp.select_task(cfg, lay, "T1")
    assert record["primary_baseline"] == "cohort_mean" and record["primary_deep"] == "gru"


def test_an_infinite_score_never_wins_and_a_pool_of_infinities_is_refused(layout):
    cfg, lay = layout
    populate(lay, "T2", {"linear": float("inf"), "persistence": 0.9})
    record = sp.select_task(cfg, lay, "T2")
    assert record["mean_inner_S"]["linear"] == "inf" and record["primary_baseline"] != "linear"
    populate(lay, "T2", {m: float("inf") for m in CFG["models"]["T2"]["time_only"]})
    with pytest.raises(sp.SelectionError, match="infinite"):
        sp.select_task(cfg, lay, "T2")


def test_missing_nan_and_inconsistent_inputs_are_refused(layout):
    cfg, lay = layout
    populate(lay, "T1", {})
    (lay.tuning("T1", "lopo", "tft") / "inner_scores.csv").unlink()
    with pytest.raises(sp.SelectionError, match="no inner scores for \\['tft'\\]"):
        sp.select_task(cfg, lay, "T1")
    write_scores(lay, "T1", "tft", {0: [float("nan")] * 3, 1: [float("nan")] * 3})
    with pytest.raises(sp.SelectionError, match="not a number"):
        sp.select_task(cfg, lay, "T1")
    write_scores(lay, "T1", "tft", {0: [0.3] * 3})                              # one outer fold instead of two
    with pytest.raises(sp.SelectionError, match="outer folds of \\['tft'\\] differ"):
        sp.select_task(cfg, lay, "T1")
    path = lay.tuning("T1", "lopo", "tft") / "inner_scores.csv"
    pd.DataFrame({"outer_fold": [0], "S": [0.3]}).to_csv(path, index=False)
    with pytest.raises(sp.SelectionError, match="lacks columns"):
        sp.select_task(cfg, lay, "T1")


def test_only_the_inner_score_files_are_read(layout):
    cfg, lay = layout
    populate(lay, "T1", {"ridge_time": 0.3, "gru": 0.2})
    first = sp.select_task(cfg, lay, "T1")
    for model in ("ridge_time", "gru", "gc"):                                   # files a choice must not depend on
        for path in (lay.predictions("T1", "lopo", model), lay.tuning("T1", "lopo", model) / "chosen.csv"):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\xff\xfe not a csv at all")
    (lay.tuning("T1", "lopo", "gc") / "curves.csv.gz").write_bytes(b"junk")
    lay.tuning("T1", "chrono", "gru").mkdir(parents=True)                       # other schemes are not read either
    write_scores(lay, "T1", "gru", {0: [0.2] * 3, 1: [0.2] * 3})
    assert sp.select_task(cfg, lay, "T1") == first


def test_the_record_lists_its_inputs_by_hash_and_is_repeatable(layout):
    cfg, lay = layout
    populate(lay, "T1", {"ridge_time": 0.3})
    record = sp.select_task(cfg, lay, "T1")
    assert record == sp.select_task(cfg, lay, "T1")
    key = "tuning/T1/lopo/ridge_time/inner_scores.csv"
    assert key in record["inputs"] and len(record["inputs"][key]) == 64
    assert set(record["inner_S_by_outer_fold"]["ridge_time"]) == {"0", "1"} and record["outer_folds"] == [0, 1]
    before = record["inputs"][key]
    write_scores(lay, "T1", "ridge_time", {0: [0.31] * 3, 1: [0.3] * 3})
    assert sp.select_task(cfg, lay, "T1")["inputs"][key] != before


def test_the_command_line_writes_both_selection_files_and_reports_a_refusal(layout, tmp_path, capsys):
    cfg, lay = layout
    populate(lay, "T1", {"gbm_time": 0.2, "tft": 0.1})
    populate(lay, "T2", {"sgr": 0.2, "gru": 0.1})
    import yaml
    config = tmp_path / "cfg.yaml"
    config.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    assert sp.main(["--config", str(config)]) == 0
    out = capsys.readouterr().out
    assert "T1: primary baseline gbm_time, primary deep model tft" in out and "T2: primary baseline sgr" in out
    assert json.loads(lay.selection("T1").read_text(encoding="utf-8"))["primary_deep"] == "tft"
    first = lay.selection("T2").read_bytes()
    assert sp.main(["--config", str(config), "--task", "T2"]) == 0 and lay.selection("T2").read_bytes() == first
    (lay.tuning("T2", "lopo", "gru") / "inner_scores.csv").unlink()
    assert sp.main(["--config", str(config), "--task", "T2"]) == 2
    assert "refused" in capsys.readouterr().err and lay.selection("T2").read_bytes() == first


def test_scores_with_a_nonfinite_average_are_written_as_text(layout):
    cfg, lay = layout
    populate(lay, "T2", {"linear": math.inf})
    record = sp.select_task(cfg, lay, "T2")
    sp.write_json_atomic(lay.selection("T2"), record)
    assert json.loads(lay.selection("T2").read_text(encoding="utf-8"))["mean_inner_S"]["linear"] == "inf"
