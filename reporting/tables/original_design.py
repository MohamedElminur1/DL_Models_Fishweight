"""Generate the compact Gate 2 comparison and every supplementary diagnostic row."""

import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

from shared import Sources, number, write_table

# These tables write NA for unavailable values, as their footers define it (shared.MISSING is the
# dash of the Phase 4 tables).
MISSING = "NA"


INPUTS = {
    "none": "none",
    "notebook": "notebook inputs (water quality and the lagged weight)",
    "manuscript": "manuscript inputs (water quality, population, length)",
    "wq_only": "water quality only",
}
DESIGNS = {
    "S0_notebook": "Notebook design (scaler fitted on all rows; last 20% of rows as test)",
    "S1_chrono": "Chronological split (scaler fitted on training rows)",
    "S2_random": "Random split of windows",
}
MODELS = {"persistence": "Persistence", "cnn_lstm": "CNN-LSTM"}
EVALUATIONS = {
    "all": "All test windows",
    "change_rows": "Test rows where weight changes",
    "grid_first_rows": "First test row on each 14-day measurement-grid date",
}


def formatted(value: str | float | None, kind: str) -> str:
    """Preserve tiny diagnostic errors with two significant digits in decimal notation."""
    if value is None or value == "":
        return MISSING
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"non-finite diagnostic metric: {value}")
    if kind == "r2":
        return f"{numeric:.5f}" if f"{numeric:.3f}" == "1.000" else f"{numeric:.3f}"
    if 0 < abs(numeric) < 1:
        places = max(0, 1 - math.floor(math.log10(abs(numeric))))
        return f"{numeric:.{places}f}"
    return number(numeric, kind)


def metric(row: dict[str, str], key: str, kind: str) -> str:
    """Format saved seed means and SDs; absent metrics remain NA."""
    mean = formatted(row.get(key + "_mean"), kind)
    if int(row["n_seeds"]) > 1 and mean != MISSING:
        sd = row.get(key + "_sd")
        if sd is None or sd == "":
            raise ValueError(f"missing SD for multi-seed {key}")
        return f"{mean} ± {formatted(sd, kind)}"
    return mean


def saved_row(row: dict[str, str], design: str | None = None) -> list[str]:
    """Project a saved all-window summary into the compact comparison columns."""
    return [design or DESIGNS[row["split"]], INPUTS[row["input_set"]], MODELS[row["model"]],
            row["n_seeds"], metric(row, "mae", "grams"), metric(row, "rmse", "grams"),
            metric(row, "mape", "percent"), metric(row, "r2", "r2")]


def reported(design: str, model: str, mae: str, rmse: str, mape: str) -> list[str]:
    """Include only the submitted values explicitly authorized by the lead revision."""
    return [design, INPUTS["manuscript"], model, MISSING, mae, rmse, mape, MISSING]


def main() -> None:
    sources = Sources()
    saved = sources.csv("results/phase2/metrics_summary.csv")
    precision = "Non-R2 values below 1 use two significant digits in decimal notation; other gram and percentage values use the standard precision. R2 uses three decimals, or five when three decimals would give 1.000. The same rules apply to saved SDs."
    seed_note = "Mean ± SD is the saved summary over seeds where Seeds > 1, including random-split persistence; no model is retrained. Phase 2 CNN-LSTM uses three seeds. NA means unavailable, never zero."
    detail = [[DESIGNS[row["split"]], INPUTS[row["input_set"]], MODELS[row["model"]],
               EVALUATIONS[row["eval_set"]], row["n_seeds"], row["n_min"], row["n_max"],
               metric(row, "mae", "grams"), metric(row, "rmse", "grams"),
               metric(row, "mape", "percent"), metric(row, "r2", "r2")] for row in saved]
    write_table("original_design_detail", ["Design", "Inputs", "Model", "Evaluation set", "Seeds", "n min", "n max",
                                           "MAE (g)", "RMSE (g)", "MAPE (%)", "R^{2}"], detail, sources, [
        "All 30 rows of metrics_summary.csv are retained in source order with readable labels; no configuration is selected by its score.",
        "n min and n max are the saved minimum and maximum evaluation counts across seeds. Grid-first rows are the first test target on each available 14-day grid date; undefined R2 is NA.",
        "S0 uses all-row scaling and overlapping validation/test targets. S1 scales training rows only. S2 scales all rows before the random window split.", seed_note, precision,
    ])
    rows = [reported("Reported single-step (submitted manuscript, Table 3)", "CNN-LSTM", "0.55", "1.95", "0.20")]
    rows.extend(saved_row(row) for row in saved if row["eval_set"] == "all")
    posthoc = sources.csv("results/phase2/posthoc_s2_final_period_summary.csv")
    selected = [row for row in posthoc if row["input_set"] == "notebook" and row["model"] == "cnn_lstm"]
    if len(selected) != 1:
        raise ValueError("expected one notebook-input post hoc summary")
    rows.append(saved_row(selected[0], "Random split, original test period only (post hoc)"))
    rows.extend([reported("Reported 18-step (submitted manuscript, Table 4)", "TFT", "1.78", "2.25", "3.70"),
                 reported("Reported 18-step (submitted manuscript, Table 4)", "CNN-LSTM", "12.75", "18.41", "26.60")])
    forecast = sources.csv("results/phase2/forecast18_persistence.csv")
    selected_forecast = [row for row in forecast if row["split"] == "S0_notebook" and row["horizon"] == "18"]
    if len(selected_forecast) != 1:
        raise ValueError("expected one notebook-design 18-step persistence summary")
    row = selected_forecast[0]
    rows.append(["Notebook design, 18-step persistence", INPUTS["none"], "Persistence", "1",
                 formatted(row["mae"], "grams"), formatted(row["rmse"], "grams"),
                 formatted(row["mape"], "percent"), MISSING])
    if len(rows) != 15:
        raise ValueError("expected 15 compact comparison rows")
    write_table("original_design", ["Design", "Inputs", "Model", "Seeds", "MAE (g)", "RMSE (g)", "MAPE (%)", "R^{2}"],
                rows, sources, [
        "The diagnostic grid contributes only evaluation set all; every other grid row is in original_design_detail.",
        "Constants: submitted manuscript, Table 3, single-step CNN-LSTM MAE 0.55 g, RMSE 1.95 g, MAPE 0.20%; submitted manuscript, Table 4, 18-step TFT 1.78 g, 2.25 g, 3.70 %, and CNN-LSTM 12.75 g, 18.41 g, 26.60 %. These are reported claims, not recomputed results. Their input label describes the manuscript inputs, not verified training code; seeds and R2 are unavailable.",
        "Submitted metrics retain their reported precision; recomputed diagnostics follow the precision rules below.",
        "The post hoc notebook-input CNN-LSTM row is scored on the original final test period only. Its saved summary does not supply R2.",
        "The single 18-step persistence row uses S0_notebook, horizon 18, pooled over origins and steps, for comparison with the submitted claims. Persistence has no fitted model; Seeds = 1 denotes one deterministic evaluation. S1_chrono is not substituted or pooled with S0.",
        "Random split of windows used an all-row scaler. The source order fixes notebook, chronological and random diagnostics; rows are not ranked by error.", seed_note, precision,
    ])


if __name__ == "__main__":
    main()
