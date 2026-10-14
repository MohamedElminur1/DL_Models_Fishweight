"""A throwaway Phase 4 environment for tests: synthetic tables on disk, a config that points at them.

`make_env` writes the four tables of tests/synth_p4.py under a temporary directory, returns
the config with its paths and results directory pointing there, the loaded tables and the
live and dev result layouts. Budgets are tiny (two trials, six epochs). Nothing here touches
data/ or results/ of the repository.
"""

import copy
from types import SimpleNamespace

from src.artifacts import Layout
from src.data import load_tables
from src.folds import load_config, suggest_cutoff
from tests.synth_p4 import make_inputs, write_inputs


def make_env(tmp_path, ponds=(1, 2, 3, 4, 6), sparse=(), n_events=7, seed=4, edit=None):
    """`edit(inputs)` may change the synthetic tables (a NamedTuple of t1, t2, daily, wq) before they are written."""
    cfg = copy.deepcopy(load_config())
    cfg["budget"] = {**cfg["budget"], "n_trials": 2, "max_epochs": 6, "patience": 3}
    cfg["dev"]["n_trials"] = 2
    cfg["gbm"]["space"]["max_iter"] = {"type": "int", "low": 5, "high": 8}
    cfg["design"]["cohort"] = list(ponds)
    cfg["chrono"].update(min_events=6, min_ponds=3)
    inputs = make_inputs(ponds, n_events, seed, sparse_ponds=sparse)
    if edit is not None:
        inputs = edit(inputs) or inputs
    cfg = write_inputs(tmp_path, inputs, cfg)
    tables = load_tables(cfg)
    cfg["chrono"]["cutoff"] = suggest_cutoff(tables.t1, cfg["chrono"]["min_events"], cfg["chrono"]["min_ponds"])
    return SimpleNamespace(cfg=cfg, tables=tables, live=Layout(cfg), dev=Layout(cfg, True), root=tmp_path)


def quiet(message):
    pass
