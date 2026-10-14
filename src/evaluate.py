"""Scores the saved predictions and runs the statistics of plans 8.2 and 9.

    .venv/bin/python -I src/evaluate.py

Reads results/predictions/{task}/{scheme}/{model}.csv, results/selection/primary_{task}.json
and, if present, results/permutation/{task}/{model}.csv. Nothing else: every file below is
regenerated from these inputs alone and written atomically, so a second run on the same
inputs gives the same bytes. The scorer refuses to run unless the two selection files are
tracked by git and unmodified (plan 12 (d)); commit them after src/select_primary.py and
before scoring.

Outputs
  results/metrics/per_pond_seed.csv   S, MAE (g), MAPE (%) per pond and seed
  results/metrics/per_pond.csv        the same, averaged over seeds
  results/metrics/scores.csv          S per model (seed-averaged per pond, mean over ponds), its SD over
                                      seeds and the SD of the per-pond S and MAE over ponds
  results/metrics/pooled_by_seed.csv  MAE, RMSE (g), MAPE (%), R2 over all test samples, per seed
  results/metrics/pooled.csv          the same, mean and SD over seeds
  results/metrics/completeness.csv    which model of the config has predictions, per scheme
  results/stats/bootstrap_S.csv       pond-cluster bootstrap interval of S
  results/stats/pairwise.csv          per-pond differences, bootstrap interval, exact Wilcoxon p-values
  results/stats/influence.csv         S difference with each pond removed in turn
  results/stats/confirmatory.csv      the fixed sequence of plan 9.5
  results/stats/merged_units.csv      the confirmatory test with ponds 1 and 2 as one unit
  results/stats/permutation.csv       permutation p-values (plan 8.2), if the permutation files exist

Strata. Every table is computed per task, scheme, sample set and horizon: T1 has horizon "0";
T2 has "1", "2" and "all" (both horizons together, descriptive). S is the statistic of plan
9.0: the mean absolute error of the log target per pond, averaged over seeds, then the mean
over ponds; a prediction of 0 g or less has an infinite log error. The statistics (bootstrap,
Wilcoxon, influence) are computed for the LOPO scheme on the eligible samples only; the
calendar hold-out and the all-events table are descriptive (plans 7.4 and 4.1). A stratum
with fewer than `evaluate.min_ponds` scored ponds is descriptive: its tests are computed and
flagged, and the fixed sequence treats it as not rejected.

Conventions fixed here (the plan leaves them open): differences and |differences| that agree
to 12 decimals are ties and zeros; a pond on which both models have an infinite S counts as a
zero difference; the bootstrap draws one matrix of pond indices per stratum from one
generator seeded with `seeds.bootstrap`, strata in the order T1, then T2 horizons 1, 2, all,
so that every model and every pair of a stratum is resampled jointly; percentile intervals use
linear interpolation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from src import guard  # noqa: E402
from src.artifacts import Layout, model_key, write_csv_atomic  # noqa: E402
from src.folds import load_config  # noqa: E402
from src.scoring import abs_log_error, permutation_p  # noqa: E402

PREDICTION_COLUMNS = ["task", "scheme", "model", "sample_set", "fold", "seed", "row_id", "pond", "stream",
                      "horizon", "date", "origin_date", "t", "w_true", "w_pred", "y_true_log", "y_pred_log",
                      "w_origin"]
HORIZONS = {"T1": ["0"], "T2": ["1", "2", "all"]}
TASKS = ("T1", "T2")
SENSITIVITY_SUFFIX = re.compile(r"_(noind|w\d{2})$")
TIE_DIGITS = 12


class EvaluationError(RuntimeError):
    """The predictions or selection files do not allow a valid evaluation."""


# ----------------------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Wilcoxon:
    """Exact Wilcoxon signed-rank test on differences `d` = model minus comparator.

    `n` differences were non-zero and entered the test, `n_zero` were dropped. `t_plus` and
    `t_minus` are the rank sums of the positive and negative differences (midranks of |d|).
    `p_lower` is P(T+ <= observed) over all 2^n sign patterns: the one-sided p-value for the
    alternative that the model's values are lower. `p_two_sided` is 2 min(p_lower, p_upper)
    capped at 1. With no non-zero difference every p-value is 1.
    """

    n: int
    n_zero: int
    t_plus: float
    t_minus: float
    p_lower: float
    p_upper: float
    p_two_sided: float


def wilcoxon_exact(d) -> Wilcoxon:
    """Exact signed-rank test of the differences `d` (see `Wilcoxon`). NaN differences are refused."""
    d = np.round(np.asarray(d, dtype=float), TIE_DIGITS)
    if np.isnan(d).any():
        raise ValueError("differences contain NaN")
    nonzero = d[d != 0]
    n, n_zero = len(nonzero), len(d) - len(nonzero)
    if n == 0:
        return Wilcoxon(0, n_zero, 0.0, 0.0, 1.0, 1.0, 1.0)
    if n > 50:
        raise ValueError("the exact distribution is only enumerated for up to 50 non-zero differences")
    doubled = np.rint(2 * rankdata(np.abs(nonzero), method="average")).astype(np.int64)   # integers: exact sums
    total = int(doubled.sum())
    counts = np.zeros(total + 1, dtype=object)
    counts[0] = 1
    for rank in doubled:
        shifted = counts.copy()
        shifted[rank:] += counts[:total + 1 - rank]
        counts = shifted
    observed = int(doubled[nonzero > 0].sum())
    patterns = 2 ** n
    p_lower = float(sum(counts[:observed + 1])) / patterns
    p_upper = float(sum(counts[observed:])) / patterns
    return Wilcoxon(n, n_zero, observed / 2.0, (total - observed) / 2.0, p_lower, p_upper,
                    min(1.0, 2.0 * min(p_lower, p_upper)))


def percentile_interval(values, level: float) -> tuple[float, float]:
    """Percentile interval of `values` with linear interpolation; infinities are kept.

    Returns (nan, nan) if any value is NaN.
    """
    v = np.sort(np.asarray(values, dtype=float))
    if np.isnan(v).any() or len(v) == 0:
        return float("nan"), float("nan")

    def quantile(q: float) -> float:
        position = q * (len(v) - 1)
        low, high = int(math.floor(position)), int(math.ceil(position))
        if v[low] == v[high]:
            return float(v[low])
        return float(v[low] + (v[high] - v[low]) * (position - low))

    tail = (1.0 - level) / 2.0
    return quantile(tail), quantile(1.0 - tail)


def pooled_metrics(w_true, w_pred) -> dict[str, float]:
    """MAE and RMSE in grams, MAPE in percent and R2 over pooled samples (plan 9.1)."""
    w_true, w_pred = np.asarray(w_true, dtype=float), np.asarray(w_pred, dtype=float)
    error = w_pred - w_true
    sst = float(np.sum((w_true - w_true.mean()) ** 2))
    return {"mae_g": float(np.mean(np.abs(error))), "rmse_g": float(np.sqrt(np.mean(error ** 2))),
            "mape_pct": float(np.mean(np.abs(error) / w_true) * 100.0),
            "r2": float(1.0 - np.sum(error ** 2) / sst) if sst > 0 else float("nan")}


def pond_differences(model_s, comparator_s) -> np.ndarray:
    """Per-pond differences model minus comparator; both infinite counts as zero."""
    a, b = np.asarray(model_s, dtype=float), np.asarray(comparator_s, dtype=float)
    both_infinite = np.isinf(a) & np.isinf(b) & (np.sign(a) == np.sign(b))
    with np.errstate(invalid="ignore"):
        d = a - b
    return np.where(both_infinite, 0.0, d)


def merge_units(ponds, values, groups) -> np.ndarray:
    """Per-unit values: the members of each group in `groups` that were scored are averaged into one unit."""
    ponds = [int(p) for p in ponds]
    values = np.asarray(values, dtype=float)
    used, out = set(), []
    for group in groups:
        members = [i for i, p in enumerate(ponds) if p in group]
        if members:
            out.append(float(np.mean(values[members])))
            used.update(members)
    out.extend(float(values[i]) for i in range(len(ponds)) if i not in used)
    return np.array(out)


def bootstrap_indices(rng: np.random.Generator, n_boot: int, n_units: int) -> np.ndarray:
    """The (B, n) matrix of pond indices drawn with replacement for one stratum."""
    return rng.integers(0, n_units, size=(n_boot, n_units))


# ----------------------------------------------------------------------------------------
# Reading the inputs
# ----------------------------------------------------------------------------------------

def load_predictions(layout: Layout) -> pd.DataFrame:
    """All prediction files under results/predictions/, with the derived error columns."""
    files = sorted((layout.root / "predictions").glob("*/*/*.csv"))
    if not files:
        raise EvaluationError(f"no prediction files under {layout.root / 'predictions'}")
    frames = []
    for path in files:
        task, scheme = path.parent.parent.name, path.parent.name
        frame = pd.read_csv(path, float_precision="round_trip")
        missing = [c for c in PREDICTION_COLUMNS if c not in frame.columns]
        if missing:
            raise EvaluationError(f"{path} lacks columns {missing}")
        keys = {model_key(m, s) for m, s in zip(frame["model"], frame["sample_set"])}
        if keys != {path.stem} or set(frame["task"]) != {task} or set(frame["scheme"]) != {scheme}:
            raise EvaluationError(f"{path} holds rows that belong to another task, scheme or model")
        frames.append(frame[PREDICTION_COLUMNS])
    out = pd.concat(frames, ignore_index=True)
    if not np.isfinite(out[["w_true", "w_pred"]].to_numpy(dtype=float)).all() or (out["w_true"] <= 0).any():
        raise EvaluationError("a prediction file has a weight that is not finite (or a true weight that is not positive)")
    out["abs_err"] = abs_log_error(out["y_pred_log"].to_numpy(), out["y_true_log"].to_numpy())
    out["abs_g"] = (out["w_pred"] - out["w_true"]).abs()
    out["ape"] = out["abs_g"] / out["w_true"]
    out["nonpositive"] = (out["w_pred"] <= 0).astype(int)
    return out


def expected_seeds(cfg: dict, task: str, model: str) -> list[int]:
    """Seeds a model's final fits use: the five of the config for the neural models, 0 for the others."""
    neural = set(cfg["models"][task]["deep"]) | set(cfg["models"][task]["static_only"])
    return [int(s) for s in cfg["seeds"]["final"]] if model in neural else [0]


def check_same_samples(frame: pd.DataFrame, cfg: dict) -> None:
    """Every model of a (task, scheme, sample set) is scored on the same samples with the expected seeds (plan 4.4)."""
    for (task, scheme, sample_set), part in frame.groupby(["task", "scheme", "sample_set"], sort=True):
        reference, reference_name = None, ""
        for model, sub in part.groupby("model", sort=True):
            seeds = sorted(int(s) for s in sub["seed"].unique())
            if seeds != expected_seeds(cfg, task, model):
                raise EvaluationError(f"{task} {scheme} {model}: seeds {seeds}, expected {expected_seeds(cfg, task, model)}")
            for seed, rows in sub.groupby("seed"):
                if rows["row_id"].duplicated().any():
                    raise EvaluationError(f"{task} {scheme} {model} seed {seed}: a sample is predicted twice")
                ids = tuple(sorted(int(i) for i in rows["row_id"]))
                if reference is None:
                    reference, reference_name = ids, f"{model} seed {seed}"
                elif ids != reference:
                    raise EvaluationError(f"{task} {scheme} {sample_set}: {model} seed {seed} is scored on different "
                                          f"samples than {reference_name}")


def read_selection(layout: Layout, task: str) -> dict:
    path = layout.selection(task)
    if not path.is_file():
        raise EvaluationError(f"missing selection file {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    for field in ("primary_baseline", "primary_deep"):
        if field not in record:
            raise EvaluationError(f"{path} has no {field}")
    return record


def assert_selection_committed(layout: Layout) -> None:
    """Refuse unless both selection files are tracked and unmodified (plan 12 (d))."""
    paths = [os.path.relpath(layout.selection(task), guard.REPO_ROOT) for task in TASKS]
    guard.assert_files_committed(paths)


# ----------------------------------------------------------------------------------------
# Metrics tables
# ----------------------------------------------------------------------------------------

def horizon_parts(task: str, part: pd.DataFrame):
    """(label, rows) for each horizon stratum of a task."""
    if task == "T1":
        return [("0", part)]
    return [("1", part[part["horizon"] == 1]), ("2", part[part["horizon"] == 2]), ("all", part)]


def group_of(cfg: dict, task: str, model: str) -> str:
    for group, names in cfg["models"][task].items():
        if group != "all_sample_set" and model in names:
            return group
    return ""


def per_pond_seed_table(frame: pd.DataFrame) -> pd.DataFrame:
    """S, MAE and MAPE per pond and seed, for every stratum."""
    pieces = []
    for (task, scheme, sample_set), part in frame.groupby(["task", "scheme", "sample_set"], sort=True):
        for label, sub in horizon_parts(task, part):
            if sub.empty:
                continue
            table = sub.groupby(["model", "seed", "pond"], sort=True).agg(
                n=("abs_err", "size"), S=("abs_err", "mean"), mae_g=("abs_g", "mean"), mape_pct=("ape", "mean"),
                n_nonpositive=("nonpositive", "sum")).reset_index()
            table["mape_pct"] *= 100.0
            table.insert(0, "horizon", label)
            table.insert(0, "sample_set", sample_set)
            table.insert(0, "scheme", scheme)
            table.insert(0, "task", task)
            pieces.append(table)
    return pd.concat(pieces, ignore_index=True)


STRATUM = ["task", "scheme", "sample_set", "horizon"]


def per_pond_table(by_seed: pd.DataFrame) -> pd.DataFrame:
    """The per-pond values averaged over seeds (plan 9.0 for S)."""
    table = by_seed.groupby(STRATUM + ["model", "pond"], sort=True).agg(
        n=("n", "first"), n_seeds=("seed", "nunique"), S=("S", "mean"), mae_g=("mae_g", "mean"),
        mape_pct=("mape_pct", "mean"), n_nonpositive=("n_nonpositive", "mean")).reset_index()
    return table


def sample_sd(values: pd.Series) -> float:
    """Sample standard deviation (ddof 1); NaN for one value or if a value is infinite."""
    if len(values) < 2:
        return float("nan")
    with np.errstate(invalid="ignore"):
        return float(values.std(ddof=1))


def scores_table(cfg: dict, by_seed: pd.DataFrame, per_pond: pd.DataFrame) -> pd.DataFrame:
    """S per model and stratum with the spread over seeds and over ponds (plan 9.2)."""
    seed_s = by_seed.groupby(STRATUM + ["model", "seed"], sort=True)["S"].mean().reset_index()
    spread = seed_s.groupby(STRATUM + ["model"], sort=True)["S"].agg(S_seed_sd=sample_sd).reset_index()
    ponds = per_pond.groupby(STRATUM + ["model"], sort=True).agg(
        n_ponds=("pond", "nunique"), n_samples=("n", "sum"), S=("S", "mean"),
        sd_pond_S=("S", sample_sd), sd_pond_mae_g=("mae_g", sample_sd),
        n_nonpositive=("n_nonpositive", "sum")).reset_index()
    table = ponds.merge(spread, on=STRATUM + ["model"], validate="one_to_one")
    table["n_seeds"] = [len(expected_seeds(cfg, t, m)) for t, m in zip(table["task"], table["model"])]
    table["group"] = [group_of(cfg, t, m) for t, m in zip(table["task"], table["model"])]
    columns = STRATUM + ["model", "group", "n_ponds", "n_samples", "n_seeds", "S", "S_seed_sd", "sd_pond_S",
                         "sd_pond_mae_g", "n_nonpositive"]
    return table[columns]


def pooled_tables(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pooled MAE, RMSE, MAPE and R2 per seed, and their mean and SD over seeds."""
    rows = []
    for (task, scheme, sample_set), part in frame.groupby(["task", "scheme", "sample_set"], sort=True):
        for label, sub in horizon_parts(task, part):
            for (model, seed), rows_ in sub.groupby(["model", "seed"], sort=True):
                rows.append({"task": task, "scheme": scheme, "sample_set": sample_set, "horizon": label,
                             "model": model, "seed": seed, "n": len(rows_),
                             **pooled_metrics(rows_["w_true"], rows_["w_pred"])})
    by_seed = pd.DataFrame(rows)
    metrics = ["mae_g", "rmse_g", "mape_pct", "r2"]
    grouped = by_seed.groupby(STRATUM + ["model"], sort=True)
    summary = grouped[metrics].mean().reset_index()
    for metric in metrics:
        summary[f"{metric}_sd"] = grouped[metric].std(ddof=1).to_numpy()
    summary.insert(len(STRATUM) + 1, "n_samples", grouped["n"].first().to_numpy())
    summary.insert(len(STRATUM) + 2, "n_seeds", grouped["seed"].nunique().to_numpy())
    return by_seed, summary


def completeness_table(cfg: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Which models of the config have predictions, per task, scheme and sample set."""
    present = set(zip(frame["task"], frame["scheme"], frame["sample_set"], frame["model"]))
    rows = []
    for task in TASKS:
        for group, names in cfg["models"][task].items():
            if group == "all_sample_set":
                continue
            for model in names:
                for scheme in ("lopo", "chrono"):
                    rows.append((task, scheme, "eligible", group, model))
                if model in cfg["models"][task]["all_sample_set"]:
                    rows.append((task, "lopo", "all", group, model))
    return pd.DataFrame([{"task": t, "scheme": s, "sample_set": ss, "group": g, "model": m,
                          "has_predictions": (t, s, ss, m) in present} for t, s, ss, g, m in rows])


# ----------------------------------------------------------------------------------------
# Statistics tables (LOPO, eligible samples)
# ----------------------------------------------------------------------------------------

def sensitivity_base(model: str) -> str:
    """The model a sensitivity variant is compared with: `ridge_time_wq_w07` -> `ridge_time_wq`."""
    return SENSITIVITY_SUFFIX.sub("", model)


def comparison_pairs(cfg: dict, task: str, models: list[str], baseline: str, deep: str) -> list[tuple[str, str, str]]:
    """(model, comparator, kind) pairs reported for a task; duplicates removed, order fixed."""
    groups = cfg["models"][task]
    pairs: list[tuple[str, str, str]] = [(deep, baseline, "primary_deep_vs_primary_baseline")]
    pairs += [(m, baseline, "vs_primary_baseline") for m in models if m not in (baseline, deep)]
    pairs += [(deep, b, "primary_deep_vs_baseline") for b in groups["time_only"] if b != baseline]
    pairs += [(d, s, "deep_vs_static") for d in groups["deep"] for s in groups["static_only"]]
    pairs += [(v, sensitivity_base(v), "sensitivity_vs_base") for v in groups["sensitivity"]]
    seen, out = set(), []
    for model, comparator, kind in pairs:
        if model in models and comparator in models and model != comparator and (model, comparator) not in seen:
            seen.add((model, comparator))
            out.append((model, comparator, kind))
    return out


def stratum_matrix(per_pond: pd.DataFrame) -> tuple[list[str], list[int], np.ndarray]:
    """Models x ponds matrix of seed-averaged S for one stratum; every model must cover the same ponds."""
    wide = per_pond.pivot(index="model", columns="pond", values="S")
    if wide.isna().any().any():
        raise EvaluationError("models are scored on different ponds in one stratum")
    return list(wide.index), [int(p) for p in wide.columns], wide.to_numpy(dtype=float)


def wilcoxon_columns(result: Wilcoxon) -> dict:
    return {"n_nonzero": result.n, "n_zero_dropped": result.n_zero, "t_plus": result.t_plus,
            "t_minus": result.t_minus, "p_one_sided_lower": result.p_lower, "p_two_sided": result.p_two_sided}


def statistics(cfg: dict, per_pond: pd.DataFrame, selections: dict[str, dict]) -> dict[str, pd.DataFrame]:
    """Bootstrap, pairwise tests, influence, the fixed sequence and the merged-units sensitivity."""
    ev = cfg["evaluate"]
    n_boot, level, alpha, min_ponds = int(ev["bootstrap"]["B"]), float(ev["bootstrap"]["ci"]), float(ev["alpha"]), int(ev["min_ponds"])
    groups = [set(g) for g in ev["merged_units"]]
    rng = np.random.default_rng(int(cfg["seeds"]["bootstrap"]))
    lopo = per_pond[(per_pond["scheme"] == "lopo") & (per_pond["sample_set"] == "eligible")]
    boot_rows, pair_rows, influence_rows, merged_rows = [], [], [], []
    for task in TASKS:
        baseline, deep = selections[task]["primary_baseline"], selections[task]["primary_deep"]
        for label in HORIZONS[task]:
            part = lopo[(lopo["task"] == task) & (lopo["horizon"] == label)]
            if part.empty:
                continue
            models, ponds, matrix = stratum_matrix(part)
            if baseline not in models or deep not in models:
                raise EvaluationError(f"{task} horizon {label}: the primary models {baseline} and {deep} have no predictions")
            n_ponds = len(ponds)
            index = bootstrap_indices(rng, n_boot, n_ponds)
            boot = matrix[:, index].mean(axis=2)                           # (models, B)
            where = {m: i for i, m in enumerate(models)}
            for m in models:
                low, high = percentile_interval(boot[where[m]], level)
                boot_rows.append({"task": task, "horizon": label, "model": m,
                                  "group": group_of(cfg, task, m), "n_ponds": n_ponds,
                                  "S": float(matrix[where[m]].mean()), "ci_low": low, "ci_high": high, "B": n_boot})
            for model, comparator, kind in comparison_pairs(cfg, task, models, baseline, deep):
                a, b = matrix[where[model]], matrix[where[comparator]]
                d = pond_differences(a, b)
                test = wilcoxon_exact(d)
                with np.errstate(invalid="ignore"):
                    boot_diff = boot[where[model]] - boot[where[comparator]]
                    full_diff = float(a.mean() - b.mean())
                low, high = percentile_interval(boot_diff, level)
                pair_rows.append({
                    "task": task, "horizon": label, "model": model, "comparator": comparator, "kind": kind,
                    "n_ponds": n_ponds, "S_model": float(a.mean()), "S_comparator": float(b.mean()), "diff": full_diff,
                    "diff_ci_low": low, "diff_ci_high": high,
                    "n_lower": int(np.sum(d < 0)), "n_higher": int(np.sum(d > 0)), "n_equal": int(np.sum(d == 0)),
                    **wilcoxon_columns(test),
                    "confirmatory_pair": kind == "primary_deep_vs_primary_baseline",
                    "descriptive_only": n_ponds < min_ponds})
                for i, pond in enumerate(ponds):
                    keep = [j for j in range(n_ponds) if j != i]
                    ka, kb = a[keep], b[keep]
                    with np.errstate(invalid="ignore"):
                        influence_rows.append({
                            "task": task, "horizon": label, "model": model, "comparator": comparator,
                            "pond_removed": pond, "n_ponds_left": len(keep), "S_model": float(ka.mean()),
                            "S_comparator": float(kb.mean()), "diff": float(ka.mean() - kb.mean())})
                if kind == "primary_deep_vs_primary_baseline":
                    ua, ub = merge_units(ponds, a, groups), merge_units(ponds, b, groups)
                    merged = wilcoxon_exact(pond_differences(ua, ub))
                    merged_rows.append({"task": task, "horizon": label, "model": model, "comparator": comparator,
                                        "n_units": len(ua), "S_model": float(ua.mean()),
                                        "S_comparator": float(ub.mean()), **wilcoxon_columns(merged),
                                        "descriptive_only": True})
    pairs = pd.DataFrame(pair_rows)
    confirmatory = confirmatory_sequence(cfg, pairs, alpha, min_ponds)
    return {"bootstrap_S": pd.DataFrame(boot_rows), "pairwise": pairs, "influence": pd.DataFrame(influence_rows),
            "confirmatory": confirmatory, "merged_units": pd.DataFrame(merged_rows)}


def confirmatory_sequence(cfg: dict, pairs: pd.DataFrame, alpha: float, min_ponds: int) -> pd.DataFrame:
    """The fixed sequence of plan 9.5: each step is tested at `alpha` only if every earlier step rejected."""
    rows, open_ = [], True
    for step, (task, horizon) in enumerate(cfg["evaluate"]["confirmatory_sequence"], start=1):
        hit = pairs[(pairs["task"] == task) & (pairs["horizon"] == str(horizon)) & pairs["confirmatory_pair"]]
        if hit.empty:
            raise EvaluationError(f"no confirmatory comparison for {task} horizon {horizon}")
        row = hit.iloc[0]
        if row["n_ponds"] < min_ponds:
            status, reject = f"descriptive (fewer than {min_ponds} scored ponds)", False
        elif not open_:
            status, reject = "not tested (an earlier step did not reject)", False
        else:
            reject = bool(row["p_one_sided_lower"] <= alpha)
            status = "rejected" if reject else "not rejected"
        open_ = open_ and reject
        rows.append({"step": step, "task": task, "horizon": str(horizon), "model": row["model"],
                     "comparator": row["comparator"], "n_ponds": row["n_ponds"], "n_nonzero": row["n_nonzero"],
                     "n_zero_dropped": row["n_zero_dropped"], "t_plus": row["t_plus"], "t_minus": row["t_minus"],
                     "p_one_sided_lower": row["p_one_sided_lower"], "p_two_sided": row["p_two_sided"],
                     "n_lower": row["n_lower"], "alpha": alpha, "status": status, "reject": reject})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Permutation p-values
# ----------------------------------------------------------------------------------------

def permutation_table(cfg: dict, layout: Layout, by_seed: pd.DataFrame,
                      selections: dict[str, dict]) -> pd.DataFrame:
    """p-values of the files in results/permutation/, from the observed S of the saved predictions.

    The observed statistic is S (T1) or S at h = 1 (T2) of the LOPO predictions: seed 0 for
    the tabular models, `permutation.deep_seed` for the primary deep model (plan 8.2).
    """
    perm = cfg["permutation"]
    rows = []
    for task in TASKS:
        label = "0" if task == "T1" else "1"
        wanted = [(m, "tabular", int(perm["r_tabular"]), 0) for m in perm["tabular_models"][task]]
        wanted.append((selections[task]["primary_deep"], "deep", int(perm["r_deep"]), int(perm["deep_seed"])))
        for model, kind, n_perm, seed in wanted:
            path = layout.permutation(task, model)
            if not path.is_file():
                continue
            frame = pd.read_csv(path, float_precision="round_trip")
            if sorted(frame["r"]) != list(range(n_perm)):
                raise EvaluationError(f"{path}: permutations {len(frame)} listed, r = 0..{n_perm - 1} expected")
            observed = by_seed[(by_seed["task"] == task) & (by_seed["scheme"] == "lopo")
                               & (by_seed["sample_set"] == "eligible") & (by_seed["horizon"] == label)
                               & (by_seed["model"] == model) & (by_seed["seed"] == seed)]
            if observed.empty:
                raise EvaluationError(f"{task} {model}: no LOPO predictions for seed {seed} to compare with {path}")
            s_obs = float(observed["S"].mean())
            s_perm = frame["S_perm"].to_numpy(dtype=float)
            rows.append({"task": task, "model": model, "kind": kind, "statistic": "S" if task == "T1" else "S_h1",
                         "seed": seed, "S_obs": s_obs, "R": n_perm, "n_perm_le_obs": int(np.sum(s_perm <= s_obs)),
                         "p": permutation_p(s_obs, s_perm), "S_perm_mean": float(s_perm.mean()),
                         "S_perm_sd": float(s_perm.std(ddof=1)), "S_perm_min": float(s_perm.min()),
                         "S_perm_max": float(s_perm.max())})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------------------

def evaluate(cfg: dict, layout: Layout) -> dict[str, pd.DataFrame]:
    """Compute every table from the files under `layout.root` and write them; returns the tables."""
    selections = {task: read_selection(layout, task) for task in TASKS}
    frame = load_predictions(layout)
    check_same_samples(frame, cfg)
    by_seed = per_pond_seed_table(frame)
    per_pond = per_pond_table(by_seed)
    pooled_by_seed, pooled = pooled_tables(frame)
    tables = {
        "per_pond_seed": by_seed, "per_pond": per_pond, "scores": scores_table(cfg, by_seed, per_pond),
        "pooled_by_seed": pooled_by_seed, "pooled": pooled, "completeness": completeness_table(cfg, frame),
    }
    tables.update(statistics(cfg, per_pond, selections))
    permutation = permutation_table(cfg, layout, by_seed, selections)
    if len(permutation):
        tables["permutation"] = permutation
    for name, table in tables.items():
        folder = layout.stats if name in {"bootstrap_S", "pairwise", "influence", "confirmatory", "merged_units",
                                          "permutation"} else layout.metrics
        write_csv_atomic(table, folder(f"{name}.csv"))
    return tables


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    layout = Layout(cfg)
    try:
        assert_selection_committed(layout)
        tables = evaluate(cfg, layout)
    except (guard.GuardError, EvaluationError) as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 2
    print(f"wrote {len(tables)} tables under {layout.root / 'metrics'} and {layout.root / 'stats'}")
    print(tables["confirmatory"][["step", "task", "horizon", "model", "comparator", "n_ponds", "p_one_sided_lower",
                                  "status"]].to_string(index=False))
    missing = tables["completeness"]
    missing = missing[~missing["has_predictions"]]
    if len(missing):
        print(f"{len(missing)} models of the config have no predictions (results/metrics/completeness.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
