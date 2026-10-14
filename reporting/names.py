"""Display names for the manuscript, one source for tables, figures and text.

Model codes are the keys of `configs/phase4.yaml` (`models`); everything a reader sees uses
the names below. Use `model_name(code)`, `group_name(group)`, `horizon_name(task, horizon)`;
an unknown code raises KeyError, so a new model cannot slip into a table under its code.
Abbreviations used here (define them in the manuscript): WQ, water quality; GBM, gradient
boosting machine (scikit-learn HistGradientBoostingRegressor); SGR, specific growth rate;
MLP, multilayer perceptron.
"""

MODEL_NAMES = {
    # T1 time-only baselines
    "gc": "Growth curve",
    "cohort_mean": "Cohort mean",
    "ridge_time": "Ridge (time)",
    "gbm_time": "GBM (time)",
    # T1 tabular models with water quality
    "ridge_wq": "Ridge (WQ)",
    "gbm_wq": "GBM (WQ)",
    "ridge_time_wq": "Ridge (time + WQ)",
    "gbm_time_wq": "GBM (time + WQ)",
    "ridge_time_wq_len": "Ridge (time + WQ + length)",
    "gbm_time_wq_len": "GBM (time + WQ + length)",
    # T1 reference that uses length
    "allometric": "Allometric (length)",
    # deep models and their static-input twin (both tasks)
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn_lstm": "CNN-LSTM",
    "tft": "TFT",
    "mlp_static": "Static MLP",
    # T1 sensitivity variants
    "ridge_time_wq_noind": "Ridge (time + WQ), no missing indicators",
    "ridge_time_wq_w07": "Ridge (time + WQ), 7-day window",
    "gbm_time_wq_w07": "GBM (time + WQ), 7-day window",
    "ridge_time_wq_w28": "Ridge (time + WQ), 28-day window",
    "gbm_time_wq_w28": "GBM (time + WQ), 28-day window",
    # T2 baselines (past weights and time, no water quality)
    "persistence": "Persistence",
    "linear": "Linear extrapolation",
    "sgr": "Constant SGR",
    "anchored_gc": "Anchored growth curve",
    "cohort_growth": "Cohort growth",
    "ridge_pw": "Ridge (past weights)",
    "gbm_pw": "GBM (past weights)",
    # T2 tabular models with water quality
    "ridge_pw_wq": "Ridge (past weights + WQ)",
    "gbm_pw_wq": "GBM (past weights + WQ)",
    # T2 sensitivity variants
    "ridge_pw_wq_noind": "Ridge (past weights + WQ), no missing indicators",
    "ridge_pw_wq_w14": "Ridge (past weights + WQ), 14-day window",
    "gbm_pw_wq_w14": "GBM (past weights + WQ), 14-day window",
    "ridge_pw_wq_w42": "Ridge (past weights + WQ), 42-day window",
    "gbm_pw_wq_w42": "GBM (past weights + WQ), 42-day window",
}

GROUP_NAMES = {
    "time_only": "Baselines without water quality",
    "water_quality": "Tabular models with water quality",
    "reference": "Reference model using length",
    "deep": "Deep models",
    "static_only": "Static-input network",
    "sensitivity": "Sensitivity variants",
    "all_sample_set": "All weighing events (descriptive)",
}

TASK_NAMES = {"T1": "T1, weight estimation", "T2": "T2, weight prediction"}
SCHEME_NAMES = {"lopo": "leave-one-pond-out", "chrono": "calendar hold-out"}
SAMPLE_SET_NAMES = {"eligible": "eligible events", "all": "all weighing events"}


def model_name(code: str) -> str:
    return MODEL_NAMES[code]


def group_name(group: str) -> str:
    return GROUP_NAMES[group]


def horizon_name(task: str, horizon: int) -> str:
    """'T1' for task T1; 'T2, 14 days ahead' or 'T2, 28 days ahead' for horizons 1 and 2."""
    if task == "T1":
        return "T1"
    return {1: "T2, 14 days ahead", 2: "T2, 28 days ahead"}[int(horizon)]
