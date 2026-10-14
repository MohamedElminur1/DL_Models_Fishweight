"""Shared source projection for Phase 4 tables; no fitting or scoring code is imported."""

from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path
import math
import statistics
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from names import group_name, horizon_name, model_name, SCHEME_NAMES, SAMPLE_SET_NAMES, TASK_NAMES
from shared import MISSING, ROOT, Sources, literal, number, write_table

STOP = "stopped (plan 9.0)"
GROUPS = ("time_only", "water_quality", "reference", "deep", "static_only", "sensitivity")
RESULT_HEADER = ["Task / horizon", "Group", "Model", "Primary baseline", "Ponds", "Samples", "Seeds",
                 "S", "S bootstrap 95 % interval", "MAE (g)", "RMSE (g)", "MAPE (%)", "R^{2}",
                 "Ponds with lower S", "SD across ponds of S", "SD across ponds of MAE (g)"]


def setup() -> tuple[Sources, dict]:
    """Record configuration and display names before opening this family's inputs."""
    sources = Sources()
    sources.path("reporting/names.py")
    return sources, sources.config("configs/phase4.yaml")


def select(rows: list[dict[str, str]], **criteria: str) -> list[dict[str, str]]:
    return [row for row in rows if all(row[k] == str(v) for k, v in criteria.items())]


def unique(rows: list[dict[str, str]], **criteria: str) -> dict[str, str]:
    matches = select(rows, **criteria)
    if len(matches) != 1:
        raise ValueError(f"expected one source row: {criteria}, found {len(matches)}")
    return matches[0]


def horizons(task: str) -> tuple[int, ...]:
    return (0,) if task == "T1" else (1, 2)


def models(cfg: dict, task: str, groups: tuple[str, ...] = GROUPS) -> Iterator[tuple[str, str]]:
    """Iterate in frozen group/model order, never ordered by outer scores."""
    for group in groups:
        for model in cfg["models"][task][group]:
            yield group, model


def selections(sources: Sources) -> dict:
    return {t: sources.json(f"results/selection/primary_{t}.json") for t in ("T1", "T2")}


def stops(sources: Sources) -> set[tuple[str, str]]:
    """Require both completeness and the logged non-finite stop before labelling it."""
    complete = sources.csv("results/metrics/completeness.csv")
    log = sources.text("results/logs/autopilot.log")
    stopped = set()
    for row in complete:
        if row["scheme"] == "lopo" and row["sample_set"] == "eligible" and row["has_predictions"] == "False":
            if not any("stopped under plan 9.0 (non-finite prediction)" in line and
                       f"--task {row['task']} --scheme lopo --model {row['model']} " in line for line in log.splitlines()):
                raise ValueError(f"missing predictions without logged stop: {row}")
            stopped.add((row["task"], row["model"]))
    return stopped


def stop_notes(stopped: set[tuple[str, str]]) -> list[str]:
    return [f"{TASK_NAMES[t]}, {model_name(m)}: stopped (plan 9.0) on a non-finite prediction; no complete frozen metrics or chosen trial are available. Logged in results/logs/autopilot.log and listed without predictions in results/metrics/completeness.csv."
            for t, m in sorted(stopped)]


def metric(row: dict[str, str], key: str, kind: str, sd_key: str | None = None) -> str:
    """Use the mean's precision for both the mean and its saved sample SD."""
    value = number(row.get(key), kind)
    if int(row.get("n_seeds", "1")) > 1:
        sd = row.get(sd_key or key + "_sd")
        if sd in (None, ""):
            raise ValueError(f"missing SD for seeded metric {key}")
        places = len(value.split(" × ")[0].partition(".")[2])
        value += " ± " + number(sd, kind, places)
    return value


def result_table(name: str, scheme: str, sample_set: str, groups: tuple[str, ...], compact: bool = False) -> None:
    """Join saved pond-balanced and pooled metrics, retaining stopped rows."""
    sources, cfg = setup()
    scores = sources.csv("results/metrics/scores.csv")
    pooled = sources.csv("results/metrics/pooled.csv")
    ponds = sources.csv("results/metrics/per_pond.csv")
    primary = selections(sources)
    stopped = stops(sources)
    boot = sources.csv("results/stats/bootstrap_S.csv") if scheme == "lopo" and sample_set == "eligible" else []
    rows, notes = [], []
    panels = {}
    shown_stops = set()
    for task in ("T1", "T2"):
        for h in horizons(task):
            entries = models(cfg, task, groups) if sample_set == "eligible" else (("all_sample_set", m) for m in cfg['models'][task]['all_sample_set'])
            entries = list(entries)
            if entries:
                panels[task, h] = unique(scores, task=task, horizon=h, model=primary[task]['primary_baseline'], scheme=scheme, sample_set=sample_set)
            for group, model in entries:
                baseline = primary[task]["primary_baseline"]
                identity = [horizon_name(task, h), group_name(group), model_name(model), model_name(baseline)]
                if scheme == "lopo" and (task, model) in stopped:
                    rows.append(identity + [MISSING] * 3 + [STOP] + [MISSING] * (len(RESULT_HEADER) - 8))
                    shown_stops.add((task, model))
                    continue
                criteria = dict(task=task, scheme=scheme, sample_set=sample_set, horizon=h, model=model)
                score, pool = unique(scores, **criteria), unique(pooled, **criteria)
                pp = select(ponds, **criteria)
                bp = {p['pond']: float(p['S']) for p in select(ponds, **(criteria | {'model': baseline}))}
                if {p['pond'] for p in pp} != set(bp):
                    raise ValueError("model and baseline have different ponds")
                lower = sum(float(p['S']) < bp[p['pond']] for p in pp)
                interval = MISSING
                if boot:
                    b = unique(boot, task=task, horizon=h, model=model)
                    interval = f"{number(b['ci_low'], 's')} to {number(b['ci_high'], 's')}"
                rows.append(identity + [score['n_ponds'], score['n_samples'], score['n_seeds'],
                    metric(score, 'S', 's', 'S_seed_sd'), interval,
                    metric(pool, 'mae_g', 'grams'), metric(pool, 'rmse_g', 'grams'), metric(pool, 'mape_pct', 'percent'),
                    metric(pool, 'r2', 'r2'), str(lower), number(score['sd_pond_S'], 's'), number(score['sd_pond_mae_g'], 'grams')])
                if float(score['n_nonpositive']) > 0:
                    notes.append(f"{horizon_name(task,h)}, {model_name(model)}: {int(float(score['n_nonpositive']))} non-positive predictions; S is ∞; gram metrics retain these predictions.")
    header = RESULT_HEADER
    main = name in ('main_results', 'main_results_full')
    description = " Length-reference rows are descriptive." if compact else " Sensitivity and length-reference rows are descriptive."
    seed_note = "Seeded models report mean ± sample SD over five seeds; SD uses the mean's decimal precision. The last two columns describe spread across seed-averaged pond metrics."
    comparison = "Ponds with lower S"
    if compact:
        kept = [0, 1, 2, 7, 8, 9, 10, 11, 12, 13]
        header = [RESULT_HEADER[i] for i in kept]
        denominators = {score['n_ponds'] for score in panels.values()}
        denominator = next(iter(denominators)) if len(denominators) == 1 else 'N'
        comparison = f"Ponds below the primary baseline (of {denominator})"
        header[-1] = comparison
        rows = [[row[i] for i in kept] for row in rows]
        seed_note = "Deep models and the static MLP show mean ± SD over five seeds. SD uses the mean's decimal precision."
        for (task, h), score in panels.items():
            baseline = primary[task]['primary_baseline']
            notes.append(f"{horizon_name(task,h)}: primary baseline {model_name(baseline)}; {score['n_ponds']} ponds; {score['n_samples']} samples.")
            if denominator == 'N':
                notes.append(f"{horizon_name(task,h)}: N = {score['n_ponds']} scored ponds.")
    write_table(name, header, rows, sources, [
        f"Title: {name.replace('_', ' ')}; {SCHEME_NAMES[scheme]}, {SAMPLE_SET_NAMES[sample_set]}." + (description if main else " Descriptive only.") + (" Supplementary." if name == 'main_results_full' else ""),
        "S is the mean of per-pond absolute log errors. Pooled metrics use every test sample. " + seed_note,
        comparison + " counts strict comparisons against the primary baseline selected by inner validation, including zero for the baseline itself. S bootstrap intervals use saved 2,000 pond resamples; calendar and all-events intervals do not apply and are –.",
        "S and R2 use three decimals, percentages one decimal; grams use one decimal below 100 g and integers otherwise. – means not applicable or unavailable, never zero. Combined T2 horizons are not duplicated as an extra row.",
        *stop_notes(shown_stops), *notes,
    ])
