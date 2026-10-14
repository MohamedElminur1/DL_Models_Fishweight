"""Generate deep and tabular search-space tables from the frozen configuration and plan."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from names import model_name
from shared import Sources, literal, write_table


MODELS = ("lstm", "gru", "cnn_lstm", "tft", "mlp_static")
PARAMETERS = (
    ("Hidden size", "hidden"), ("Recurrent layers", "layers"),
    ("Convolution filters", "filters"), ("Kernel", "kernel"), ("Pooling", "pool"),
    ("Attention heads", "heads"), ("Dropout", "dropout"), ("Learning rate", "lr"),
    ("Weight decay", "weight_decay"), ("Batch size", "batch_size"),
)


def number_text(value: float) -> str:
    """A configured number as printed: exact powers of ten below 0.001 as 10^{n}, else the shortest form."""
    if 0 < abs(value) < 1e-3 and float(f"{value:e}".split("e")[0]) == 1:
        return f"10^{{{int(f'{value:e}'.split('e')[1])}}}"
    return f"{value:g}"


def range_text(spec: dict) -> str:
    """Show categorical choices and distinguish uniform, logarithmic and integer ranges."""
    if spec["type"] == "categorical":
        return ", ".join(number_text(value) for value in spec["choices"])
    if spec["type"] not in {"int", "float"}:
        raise ValueError(f"unknown search type: {spec['type']}")
    scale = "integer" if spec["type"] == "int" else "log-uniform" if spec.get("log") else "uniform"
    return f"{number_text(spec['low'])} to {number_text(spec['high'])} ({scale})"


def main() -> None:
    sources = Sources()
    sources.path("reporting/names.py")
    cfg = sources.config("configs/phase4.yaml")
    plan = " ".join(sources.text("ANALYSIS_PLAN.md").split())
    for required in ("AdamW", "mean squared error on the standardized target", "median best epoch", "1 (encoder)"):
        if required not in plan:
            raise ValueError(f"frozen-plan setting missing: {required}")
    architecture, common = cfg["deep"]["arch_space"], cfg["deep"]["common_space"]
    if set(architecture) != set(MODELS):
        raise ValueError("unexpected deep architecture list")
    rows = []
    for label, key in PARAMETERS:
        values = []
        for model in MODELS:
            spec = (architecture[model] | common).get(key)
            value = range_text(spec) if spec else ""
            if model == "tft" and key == "layers":
                value = "1 (encoder)"
            values.append(value)
        rows.append([label, *values])
    settings = [
        ("Trials per outer fold and sampler", f"{cfg['budget']['n_trials']} trials; Optuna TPE, seed {cfg['seeds']['sampler']}"),
        ("Inner folds", f"{cfg['design']['n_inner_folds']} (grouped by sensor stream)"),
        ("Maximum epochs", str(cfg["budget"]["max_epochs"])),
        ("Early-stopping patience", f"{cfg['budget']['patience']} epochs"),
        ("Final-fit epochs", "Median best epoch of the chosen trial's inner folds"),
        ("Final-fit seeds", ", ".join(map(str, cfg["seeds"]["final"]))),
        ("Optimizer", "AdamW"),
        ("Loss", "Mean squared error on the standardized target"),
    ]
    rows.extend([label, *([value] * len(MODELS))] for label, value in settings)
    write_table("search_spaces_deep", ["Parameter / setting", "LSTM", "GRU", "CNN-LSTM", "TFT", model_name("mlp_static")],
                rows, sources, [
        "All ranges, budgets, inner-fold count and seeds come from the full-run configs/phase4.yaml, not dev settings. Empty cells mean the parameter does not apply. The hidden size of CNN-LSTM is the LSTM part; pooling 1 means no pooling.",
        "Plan constants: Optuna TPE and grouping by sensor stream, ANALYSIS_PLAN.md section 7.2; final-fit epoch rule, section 7.2; AdamW and loss on the standardized target, section 6.3; TFT's fixed single encoder layer, section 13.",
        f"Tuning-fit seed is {cfg['seeds']['tuning']}; final fits use the listed seeds. Early stopping uses validation MSE (plan 6.3 and 13).",
        "Settings describe LOPO. The descriptive calendar hold-out uses one inner split (plan 7.4), with the same budgets and seeds.",
    ])
    tabular_sources = Sources()
    tabular_cfg = tabular_sources.config("configs/phase4.yaml")
    rows = [["GBM", literal(key), range_text(spec)] for key, spec in tabular_cfg["gbm"]["space"].items()]
    rows.extend([
        ["GBM", "Trials per outer fold", str(tabular_cfg["budget"]["n_trials"])],
        ["GBM", literal("random_state"), str(tabular_cfg["seeds"]["gbm"])],
    ])
    grid = tabular_cfg["ridge"]["alpha_log10"]
    if grid["num"] < 2:
        raise ValueError("ridge grid must have at least two candidates")
    exponents = [grid["start"] + i * (grid["stop"] - grid["start"]) / (grid["num"] - 1) for i in range(grid["num"])]
    rows.extend([
        ["Ridge", "Alpha grid", ", ".join(f"10^{{{value:g}}}" for value in exponents)],
        ["Ridge", "Tie rule", "Choose the larger alpha"],
    ])
    write_table("search_spaces_tabular", ["Model", "Parameter / setting", "Range / value"], rows, tabular_sources, [
        "GBM is the HistGradientBoostingRegressor of scikit-learn. Its search ranges, full-run trial budget and random_state come from configs/phase4.yaml. Integer and continuous ranges include their configured endpoints; logarithmic ranges use log-uniform sampling.",
        "Ridge alpha candidates are computed from alpha_log10 start, stop and num; the larger-alpha tie rule is stated in the adjacent frozen YAML comment.",
        "The same spaces apply to each task, feature-set ablation and sensitivity variant using that model family; there is no task-specific range override.",
    ])


if __name__ == "__main__":
    main()
