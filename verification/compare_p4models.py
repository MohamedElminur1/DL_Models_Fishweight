"""Compare Codex's blind rebuild of the folds and baseline models with the project pipeline.

Usage (from the repo root): .venv/bin/python -I verification/compare_p4models.py [--task T1|T2]

Reads verification/codex_p4models/codex_{folds,inner,predictions,curves}.csv (Codex, built
from data/processed/t1_table.csv, t2_table.csv and verification/codex_p4models/SPEC.md
only) and the project outputs of the leave-one-pond-out runs: results/predictions/,
results/tuning/{task}/lopo/{model}/{chosen,inner_scores}.csv and the plan and final rows of
results/logs/runtime.csv. Writes verification/codex_p4models/COMPARISON.md with, per check,
the keys matched, keys on one side only, and the largest difference per column.

It compares fold plans, inner-validation scores, chosen ridge alphas and growth curves, and
predicted values. It computes no outer-test error: plan 9.4 scores outer predictions only
after the primary-model selection is committed, so codex_scores.csv is not read here.
Exit code 1 if anything differs beyond the tolerance.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CODEX = ROOT / "verification" / "codex_p4models"
RESULTS = ROOT / "results"
OUT = CODEX / "COMPARISON.md"
RTOL, ATOL = 1e-6, 1e-9  # curve fits may stop at slightly different points; strict count shown too
STRICT = 1e-9

MODELS = {
    "T1": ["gc", "cohort_mean", "ridge_time", "ridge_time_wq", "allometric"],
    "T2": ["persistence", "linear", "sgr", "anchored_gc", "cohort_growth", "ridge_pw", "ridge_pw_wq"],
}
CURVE_MODELS = {"T1": "gc", "T2": "anchored_gc"}


def numeric_columns(name: str, both: pd.DataFrame, cols: list[str], lines: list[str]) -> bool:
    ok = True
    lines.append("| column | cells | beyond tolerance | beyond 1e-9 | max abs diff |")
    lines.append("|---|---|---|---|---|")
    for c in cols:
        a = pd.to_numeric(both[f"{c}_ours"], errors="coerce").astype(float)
        b = pd.to_numeric(both[f"{c}_codex"], errors="coerce").astype(float)
        nan_mismatch = a.isna() != b.isna()
        close = np.isclose(a, b, rtol=RTOL, atol=ATOL) | (a.isna() & b.isna())
        strict = np.isclose(a, b, rtol=STRICT, atol=1e-12) | (a.isna() & b.isna())
        bad = int((~close | nan_mismatch).sum())
        diff = (a - b).abs().max(skipna=True)
        diff_txt = "0" if pd.isna(diff) or diff == 0 else f"{diff:.3g}"
        lines.append(f"| {c} | {len(both)} | {bad} | {int((~strict | nan_mismatch).sum())} | {diff_txt} |")
        ok &= bad == 0
    lines.append("")
    return ok


def merge_report(name: str, ours: pd.DataFrame, theirs: pd.DataFrame, keys: list[str],
                 num_cols: list[str], text_cols: list[str], lines: list[str]) -> bool:
    m = ours.merge(theirs, on=keys, how="outer", suffixes=("_ours", "_codex"), indicator=True)
    only_ours = int((m["_merge"] == "left_only").sum())
    only_codex = int((m["_merge"] == "right_only").sum())
    both = m[m["_merge"] == "both"]
    lines.append(f"## {name}\n")
    lines.append(f"Keys {keys}: {len(both)} matched, {only_ours} only in ours, {only_codex} only in Codex.\n")
    ok = only_ours == 0 and only_codex == 0
    for c in text_cols:
        diff = both[both[f"{c}_ours"].astype(str) != both[f"{c}_codex"].astype(str)]
        lines.append(f"- `{c}`: {len(diff)} of {len(both)} differ.")
        if len(diff):
            lines.append("  " + diff[keys + [f"{c}_ours", f"{c}_codex"]].head(10).to_string(index=False).replace("\n", "\n  "))
        ok &= diff.empty
    if text_cols:
        lines.append("")
    if num_cols:
        ok &= numeric_columns(name, both, num_cols, lines)
    return ok


def pond_list(text) -> str:
    """'[1]', '1 2', '[1, 2]' or NaN as a space-separated sorted list of pond ids."""
    if pd.isna(text):
        return ""
    return " ".join(str(p) for p in sorted(int(x) for x in re.findall(r"\d+", str(text))))


def chosen(task: str, model: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / "tuning" / task / "lopo" / model / "chosen.csv")
    frame["held_out_pond"] = frame["test_ponds"].map(pond_list).astype(int)
    return frame


def runtime_rows(task: str, stage: str) -> pd.DataFrame:
    rt = pd.read_csv(RESULTS / "logs" / "runtime.csv", dtype=str)
    return rt[(rt["task"] == task) & (rt["scheme"] == "lopo") & (rt["stage"] == stage)]


def note(notes: str, key: str) -> str:
    found = re.search(rf"(?:^|; ){key}=([^;]*)", notes or "")
    return found.group(1).strip() if found else ""


NOT_COMPARED: list[str] = []


def stopped_check(task: str, model: str, path: Path, lines: list[str]) -> bool:
    """False (and a note) when the pipeline has no output for a model that stopped under plan 9.0."""
    if path.exists():
        return True
    lines.append(f"## {task} {model}: not compared\n")
    lines.append(f"The pipeline has no `{path.relative_to(ROOT)}`: the run of this model stopped on a "
                 "non-finite inner-validation prediction (plan 9.0). Codex's values are kept for reference.\n")
    NOT_COMPARED.append(f"{task} {model}")
    return False


def check_folds(task: str, lines: list[str]) -> bool:
    model = MODELS[task][0]
    ours = chosen(task, model)[["held_out_pond", "outer_fold", "n_train", "n_test"]].copy()
    plan = runtime_rows(task, "plan")
    plan = plan[plan["model"] == model].copy()
    plan["outer_fold"] = plan["fold"].astype(int)
    plan["dropped_ponds"] = plan["notes"].map(lambda s: pond_list(note(s, "dropped_ponds")))
    ours = ours.merge(plan[["outer_fold", "dropped_ponds"]], on="outer_fold", how="left")
    codex = pd.read_csv(CODEX / "codex_folds.csv", dtype={"dropped_ponds": str})
    codex = codex[codex["task"] == task].copy()
    codex["dropped_ponds"] = codex["dropped_ponds"].map(pond_list)
    return merge_report(f"{task} outer folds (ours: {model} chosen.csv and runtime plan rows)",
                        ours.drop(columns="outer_fold"), codex.drop(columns="task")[
                            ["held_out_pond", "n_train", "n_test", "dropped_ponds"]],
                        ["held_out_pond"], ["n_train", "n_test"], ["dropped_ponds"], lines)


def check_inner(task: str, lines: list[str]) -> bool:
    ok = True
    codex_all = pd.read_csv(CODEX / "codex_inner.csv")
    for model in MODELS[task]:
        if not stopped_check(task, model, RESULTS / "tuning" / task / "lopo" / model / "inner_scores.csv", lines):
            continue
        ours = pd.read_csv(RESULTS / "tuning" / task / "lopo" / model / "inner_scores.csv")
        fold_pond = chosen(task, model)[["outer_fold", "held_out_pond"]]
        ours = ours.merge(fold_pond, on="outer_fold").rename(columns={"n_samples": "n_val"})
        codex = codex_all[(codex_all["task"] == task) & (codex_all["model"] == model)]
        cols = ["n_val", "S"] + [c for c in ("S_h1", "S_h2") if c in ours.columns]
        ok &= merge_report(f"{task} {model}: inner-validation scores", ours[["held_out_pond", "inner_fold"] + cols],
                           codex[["held_out_pond", "inner_fold"] + cols], ["held_out_pond", "inner_fold"],
                           cols, [], lines)
        if model.startswith("ridge"):
            alphas = chosen(task, model)[["held_out_pond", "hp_alpha"]].rename(columns={"hp_alpha": "alpha"})
            codex_alpha = codex.groupby("held_out_pond", as_index=False)["alpha"].first()
            ok &= merge_report(f"{task} {model}: chosen alpha", alphas, codex_alpha, ["held_out_pond"],
                               ["alpha"], [], lines)
    return ok


def check_curves(task: str, lines: list[str]) -> bool:
    model = CURVE_MODELS[task]
    final = runtime_rows(task, "final")
    final = final[final["model"] == model].copy()
    final["outer_fold"] = final["fold"].astype(int)
    final["chosen_curve"] = final["notes"].map(lambda s: note(s, "curve"))
    final["n_failed"] = final["notes"].map(lambda s: note(s, "curve_failures")).astype(int)
    ours = final.merge(chosen(task, model)[["outer_fold", "held_out_pond"]], on="outer_fold")
    codex = pd.read_csv(CODEX / "codex_curves.csv")
    codex = codex[(codex["task"] == task) & (codex["model"] == model)].copy()
    codex["n_failed"] = codex["failed_curves"].map(lambda s: 0 if pd.isna(s) or not str(s).strip()
                                                   else len(re.split(r"[ ;,]+", str(s).strip())))
    return merge_report(f"{task} {model}: chosen growth curve per outer fold",
                        ours[["held_out_pond", "chosen_curve", "n_failed"]],
                        codex[["held_out_pond", "chosen_curve", "n_failed"]],
                        ["held_out_pond"], [], ["chosen_curve", "n_failed"], lines)


def check_predictions(task: str, lines: list[str]) -> bool:
    ok = True
    codex_all = pd.read_csv(CODEX / "codex_predictions.csv", dtype={"date": str, "origin_date": str})
    keys = ["pond", "date"] if task == "T1" else ["pond", "origin_date", "horizon"]
    for model in MODELS[task]:
        if not stopped_check(task, model, RESULTS / "predictions" / task / "lopo" / f"{model}.csv", lines):
            continue
        ours = pd.read_csv(RESULTS / "predictions" / task / "lopo" / f"{model}.csv",
                           dtype={"date": str, "origin_date": str})
        ours = ours[(ours["scheme"] == "lopo") & (ours["sample_set"] == "eligible")].copy()
        codex = codex_all[(codex_all["task"] == task) & (codex_all["model"] == model)].copy()
        for frame in (ours, codex):
            frame["date"] = frame["date"].str[:10]
            if task == "T2":
                frame["origin_date"] = frame["origin_date"].str[:10]
                frame["horizon"] = frame["horizon"].astype(int)
        if task == "T2" and len(ours) and len(codex):
            # Codex writes y = log(w_target / w_origin) in the log columns; convert if ours hold log W.
            probe = ours.merge(codex, on=keys, suffixes=("_ours", "_codex"))
            gap = (probe["y_true_log_ours"] - probe["y_true_log_codex"]).abs().median()
            if gap > 1e-6:
                lines.append(f"Note ({model}): our T2 log columns are log W, Codex's are y; Codex's are "
                             "converted with log w_origin before comparing.\n")
                codex = codex.merge(ours[keys + ["w_origin"]], on=keys, how="left")
                codex["y_true_log"] = codex["y_true_log"] + np.log(codex["w_origin"])
                codex["y_pred_log"] = codex["y_pred_log"] + np.log(codex["w_origin"])
                codex = codex.drop(columns="w_origin")
        cols = ["w_true", "w_pred", "y_true_log", "y_pred_log"]
        ok &= merge_report(f"{task} {model}: outer test predictions", ours[keys + cols],
                           codex[keys + cols], keys, cols, [], lines)
    return ok


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["T1", "T2"], action="append")
    args = parser.parse_args(argv)
    tasks = args.task or ["T1", "T2"]
    lines = ["# Codex blind rebuild of folds and baseline models against the pipeline\n",
             f"Tolerance: relative {RTOL:g}, absolute {ATOL:g}; the strict column counts differences "
             f"beyond 1e-9. Tasks compared: {', '.join(tasks)}. No outer-test error is computed here.\n"]
    ok = True
    for task in tasks:
        for check in (check_folds, check_inner, check_curves, check_predictions):
            ok &= check(task, lines)
    lines.append(f"Result: {'all compared values agree' if ok else 'DIFFERENCES FOUND'}."
                 + (f" Not compared (no pipeline output, plan 9.0 stop): {', '.join(sorted(set(NOT_COMPARED)))}."
                    if NOT_COMPARED else ""))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{'agree' if ok else 'DIFFER'}: {OUT.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
