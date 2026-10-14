"""Shared setup for the w28 diagnosis: the repository's own tables, folds, learner and alpha grid.

Run with: nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -I -B /tmp/diag_w28/<script>.py   (cwd = repo root)
Nothing is written under the repository. Only inner splits (plan.inner) are used; plan.test_idx is never touched.
"""
import sys

sys.dont_write_bytecode = True
ROOT = "/home/dev/development/publications"
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.data import load_tables  # noqa: E402
from src.folds import load_config, plan_folds  # noqa: E402
from src.tuning import Context, NonFinitePredictionError, predict_checked, ridge_alphas  # noqa: E402

TASK, SCHEME, SAMPLE_SET = "T1", "lopo", "eligible"


def setup():
    cfg = load_config()
    tables = load_tables(cfg)
    table = tables.table(TASK)
    plans = [p for p in plan_folds(TASK, SCHEME, table, SAMPLE_SET, cfg, None) if not p.skipped]
    return cfg, tables, table, plans, ridge_alphas(cfg)


def make_ctx(cfg, table, model):
    return Context(TASK, SCHEME, model, SAMPLE_SET, cfg, table, None, False)


def column_names(learner):
    return list(learner.active_) + list(learner.indicator_columns)


def standardised(learner, df):
    """The matrix RidgeModel.predict_log feeds to Ridge: (design - mean_) / scale_."""
    return (learner._design(df) - learner.mean_) / learner.scale_
