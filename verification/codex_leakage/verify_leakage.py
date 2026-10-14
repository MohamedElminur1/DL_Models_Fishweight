"""Independently audit the saved Phase 4 analysis. Never train or write pipeline files.

Run from the repository root with:
nice -n 19 /home/dev/development/publications/.venv/bin/python -I verification/codex_leakage/verify_leakage.py
Re-running replaces only this review's evidence files. No analysis function computes the
reference folds, features, scores, ranks, permutation p-values or bootstrap intervals.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

ROOT = Path("/home/dev/development/publications")
OUT = ROOT / "verification/codex_leakage"
SCRATCH = Path("/tmp/leakcheck")
SCRATCH.mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(SCRATCH)
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
for name in ("TMPDIR", "XDG_CACHE_HOME", "TORCH_HOME", "MPLCONFIGDIR"):
    os.environ[name] = str(SCRATCH)
if Path.cwd() != ROOT:
    raise RuntimeError("Run from the repository root")

import numpy as np
import pandas as pd
import yaml

COHORT = [1, 2, 3, 4, 6, 7, 8, 9]
SENSORS = ["temperature", "turbidity", "do", "ph", "ammonia_log10", "nitrate"]
STATS = ["mean", "sd", "min", "max"]
NEURAL = {"lstm", "gru", "cnn_lstm", "tft", "mlp_static"}
FROZEN = ["ANALYSIS_PLAN.md", "src", "configs", "data/processed", "uv.lock"]
checks: list[dict] = []
comparisons: list[dict] = []
details: dict = {}


def git(*args: str) -> str:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}
    return subprocess.run(["git", *args], cwd=ROOT, env=env, check=True,
                          capture_output=True, text=True).stdout.strip()


def read(path: Path | str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path, float_precision="round_trip")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(number: int, name: str, ok: bool, evidence: str, detail: str, note: bool = False) -> None:
    checks.append(dict(check=number, name=name, status="NOTE" if note and ok else "PASS" if ok else "FAIL",
                       evidence=evidence, detail=detail))


def same(a: object, b: object) -> bool:
    if isinstance(a, (str, bool)) or isinstance(b, (str, bool)):
        return str(a) == str(b)
    if pd.isna(a) or pd.isna(b):
        return bool(pd.isna(a) and pd.isna(b))
    if math.isinf(float(a)) or math.isinf(float(b)):
        return a == b
    return bool(np.isclose(float(a), float(b), rtol=1e-10, atol=1e-11))


def compare(number: int, source: str, field: str, expected: object, actual: object) -> bool:
    ok = same(expected, actual)
    comparisons.append(dict(check=number, status="PASS" if ok else "FAIL", evidence=source,
                            field=field, recomputed=expected, saved=actual,
                            abs_diff=abs(float(expected)-float(actual))
                            if isinstance(expected, (int, float, np.number)) and
                            isinstance(actual, (int, float, np.number)) and
                            np.isfinite(expected) and np.isfinite(actual) else None))
    return ok


def save(name: str, rows: list[dict] | pd.DataFrame) -> None:
    frame = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
    frame.to_csv(OUT / name, index=False)


def finish_numeric(number: int, name: str, start: int, evidence: str, detail: str = "") -> None:
    rows = comparisons[start:]
    failed = sum(r["status"] == "FAIL" for r in rows)
    check(number, name, failed == 0, evidence, f"{len(rows)} comparisons; {failed} mismatches. {detail}")


def manifest() -> dict[str, str]:
    paths = [ROOT / p for p in FROZEN]
    paths += [ROOT / "results" / p for p in ("predictions", "metrics", "stats", "selection", "permutation",
                                             "tuning", "dev", "logs", "posthoc")]
    return {rel(f): sha(f) for p in paths for f in ([p] if p.is_file() else sorted(p.rglob("*")))
            if f.is_file() and f.suffix not in (".lock", ".db", ".sqlite") and "__pycache__" not in f.parts}


def freeze_check() -> None:
    frozen = git("rev-parse", "analysis-plan-frozen^{}")
    details["head_start"] = git("rev-parse", "HEAD")
    details["frozen"] = frozen
    check(1, "annotated freeze and decision", git("cat-file", "-t", "analysis-plan-frozen") == "tag"
          and frozen in (ROOT / "decisions.tsv").read_text(), "decisions.tsv:40", frozen)
    history = git("log", "--format=%H %s", "--name-only", "analysis-plan-frozen..HEAD", "--", *FROZEN)
    changed = git("diff", "--name-only", "analysis-plan-frozen", "--", *FROZEN)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN)
    check(1, "frozen paths unchanged", not history and not changed and not untracked,
          "ANALYSIS_PLAN.md:323; verification/codex_leakage/git_evidence.txt", repr((history, changed, untracked)))
    records = []
    for p in sorted((ROOT / "results/predictions").rglob("*")):
        if not p.is_file():
            continue
        commits = git("log", "--reverse", "--diff-filter=A", "--format=%H", "--", rel(p)).splitlines()
        first = commits[0] if commits else ""
        ok = bool(first) and first != frozen and subprocess.run(
            ["git", "merge-base", "--is-ancestor", frozen, first], cwd=ROOT,
            capture_output=True).returncode == 0
        records.append(dict(file=rel(p), first_commit=first, status="PASS" if ok else "FAIL"))
    save("prediction_history.csv", records)
    check(1, "prediction additions after freeze", bool(records) and all(r["status"] == "PASS" for r in records),
          "verification/codex_leakage/prediction_history.csv/rows 2+", f"{len(records)} files")
    tracked_dev = git("ls-files", "results/dev")
    check(1, "dev outputs untracked", not tracked_dev, "verification/codex_leakage/git_evidence.txt", repr(tracked_dev))
    dev_rows = []
    for p in sorted((ROOT / "results/dev").rglob("*")):
        if not p.is_file():
            continue
        if p.name.endswith((".csv", ".csv.gz")):
            frame = read(p)
            fields = [c for c in frame if re.search(r"(^S($|_)|_S$|(?i:mse|mae|mape|rmse|^r2$|^value$|inner_value|best_value|objective|loss))", c)]
            populated = [c for c in fields if frame[c].notna().any()]
            dev_rows.append(dict(file=rel(p), fields=";".join(fields), populated=";".join(populated),
                                 status="FAIL" if populated else "PASS"))
        elif p.suffix == ".json":
            obj = json.loads(p.read_text())
            found = []
            def walk(value: object, key: str = "") -> None:
                if isinstance(value, dict):
                    for k, v in value.items():
                        if re.search(r"(^S$|mse|loss|^value$|inner_value|best_value|objective)", k, re.I) and v is not None:
                            if not isinstance(v, float) or not math.isnan(v):
                                found.append(f"{key}.{k}={v}")
                        walk(v, f"{key}.{k}")
                elif isinstance(value, list):
                    for v in value:
                        walk(v, key)
            walk(obj)
            dev_rows.append(dict(file=rel(p), fields="json", populated=";".join(found),
                                 status="FAIL" if found else "PASS"))
    save("dev_scan.csv", dev_rows)
    check(1, "dev error values withheld", all(r["status"] == "PASS" for r in dev_rows),
          "verification/codex_leakage/dev_scan.csv/rows 2+; src/tuning.py:220", f"{len(dev_rows)} tables/JSON files")
    (OUT / "git_evidence.txt").write_text(
        f"HEAD {details['head_start']}\nanalysis-plan-frozen {frozen}\n"
        f"tag type {git('cat-file', '-t', 'analysis-plan-frozen')}\n"
        f"post-tag frozen-path commits: {history!r}\nworking changes: {changed!r}\nuntracked frozen paths: {untracked!r}\n"
        f"tracked results/dev: {tracked_dev!r}\nfix tags: {git('tag', '--list', 'analysis-fix-*')!r}\n")


def independent_plans(task: str, sample_set: str, scheme: str) -> list[dict]:
    table = tables[task]
    rows = table if sample_set == "all" else table[table.eligible_eval]
    out = []
    if scheme == "lopo":
        for fold, pond in enumerate(COHORT):
            test = rows[rows.pond == pond]
            if test.empty:
                continue
            drop = [3-pond] if pond in [1, 2] else []
            train = rows[~rows.pond.isin([pond, *drop])]
            units = sorted({(1, 2) if p in [1, 2] else (int(p),) for p in train.pond}, key=lambda u: u[0])
            inner = []
            for j in range(3):
                val_ponds = {p for i, u in enumerate(units) if i % 3 == j for p in u}
                validation = train[train.pond.isin(val_ponds)]
                fit = train[~train.pond.isin(val_ponds)]
                inner.append((j, list(fit.index), list(validation.index)))
            out.append(dict(fold=fold, train=list(train.index), test=list(test.index),
                            excluded=[], test_ponds=[pond], dropped_ponds=drop, inner=inner))
    else:
        column = "date" if task == "T1" else "target_date"
        train = rows[rows[column] < "2021-08-28"]
        late = rows[rows[column] >= "2021-08-28"]
        test = late if task == "T1" else late[late.origin_date >= "2021-08-14"]
        excluded = late.loc[late.index.difference(test.index)]
        latest = train[column].max()
        out.append(dict(fold=0, train=list(train.index), test=list(test.index), excluded=list(excluded.index),
                        test_ponds=sorted(int(p) for p in test.pond.unique()), dropped_ponds=[],
                        inner=[(0, list(train[train[column] < latest].index), list(train[train[column] == latest].index))]))
    return out


def fold_checks() -> None:
    start = len(comparisons)
    outer_rows, inner_rows, calendar_rows = [], [], []
    for task, sample_set in [("T1", "eligible"), ("T2", "eligible"), ("T1", "all")]:
        for scheme in ["lopo", "chrono"] if sample_set == "eligible" else ["lopo"]:
            for p in independent_plans(task, sample_set, scheme):
                table = tables[task]
                record = {"task": task, "scheme": scheme, "sample_set": sample_set, "fold": p["fold"],
                          "n_train": len(p["train"]), "n_test": len(p["test"]),
                          "test_ponds": str(p["test_ponds"]), "dropped_ponds": str(p["dropped_ponds"]),
                          "train_rows": " ".join(str(i+2) for i in p["train"]),
                          "test_rows": " ".join(str(i+2) for i in p["test"])}
                outer_rows.append(record)
                for j, fit_ids, val_ids in p["inner"]:
                    fit_ponds = set(table.loc[fit_ids, "pond"])
                    val_ponds = set(table.loc[val_ids, "pond"])
                    inner_rows.append(dict(task=task, scheme=scheme, sample_set=sample_set, outer_fold=p["fold"],
                        inner_fold=j, n_train=len(fit_ids), n_val=len(val_ids), train_ponds=str(sorted(fit_ponds)),
                        validation_ponds=str(sorted(val_ponds)), disjoint_ponds=not (fit_ponds & val_ponds),
                        shared_stream_intact=not ((1 in fit_ponds and 2 in val_ponds) or (2 in fit_ponds and 1 in val_ponds)),
                        train_rows=" ".join(str(i+2) for i in fit_ids), validation_rows=" ".join(str(i+2) for i in val_ids)))
                if scheme == "chrono":
                    for role in ["train", "test", "excluded"]:
                        for i in p[role]:
                            r = table.loc[i]
                            calendar_rows.append(dict(task=task, role=role, source=f"data/processed/{task.lower()}_table.csv/row {i+2}",
                                pond=r.pond, horizon=0 if task == "T1" else r.horizon,
                                origin_date="" if task == "T1" else r.origin_date,
                                date=r.date if task == "T1" else r.target_date))
    save("outer_folds.csv", outer_rows)
    save("inner_folds.csv", inner_rows)
    save("calendar_rows.csv", calendar_rows)
    for i, r in runtime[runtime.stage == "plan"].iterrows():
        ss = "all" if r.model.endswith("__all") else "eligible"
        p = next(p for p in independent_plans(r.task, ss, r.scheme) if p["fold"] == r.fold)
        notes = dict(part.strip().split("=", 1) for part in r.notes.split(";") if "=" in part)
        for k in ["n_train", "n_test", "n_excluded"]:
            expected = len(p[k.removeprefix("n_") if k != "n_excluded" else "excluded"])
            compare(2 if r.scheme == "lopo" else 7, f"results/logs/runtime.csv/row {i+2}", k, expected, int(notes[k]))
        for k in ["test_ponds", "dropped_ponds"]:
            compare(2 if r.scheme == "lopo" else 7, f"results/logs/runtime.csv/row {i+2}", k, str(p[k]), notes[k])
    for path, frame in prediction_files:
        task, scheme, ss, model = (frame[c].iat[0] for c in ["task", "scheme", "sample_set", "model"])
        plans = independent_plans(task, ss, scheme)
        seeds = list(range(5)) if model in NEURAL else [0]
        compare(2, f"{rel(path)}/all rows", "seed set", str(seeds), str(sorted(frame.seed.unique().tolist())))
        for seed in seeds:
            want = {(p["fold"], i, int(tables[task].loc[i, "pond"])) for p in plans for i in p["test"]}
            got = list(frame.loc[frame.seed == seed, ["fold", "row_id", "pond"]].itertuples(index=False, name=None))
            compare(2, f"{rel(path)}/seed {seed}", "exact fold,row_id,pond multiset", True,
                    len(got) == len(want) and set(got) == want)
        table = tables[task].loc[frame.row_id].reset_index(drop=True)
        for column, expected in [("w_true", table.w if task == "T1" else table.w_target),
                                 ("y_true_log", table.log_w if task == "T1" else table.y),
                                 ("date", table.date if task == "T1" else table.target_date)]:
            compare(2, f"{rel(path)}/all rows", f"{column} matches source table", True,
                    np.array_equal(frame[column].to_numpy(), expected.to_numpy()))
    finish_numeric(2, "outer plans and all prediction membership", start,
                   "verification/codex_leakage/outer_folds.csv/rows 2+; numeric_comparisons.csv/rows 2+")
    start = len(comparisons)
    for path in sorted((ROOT / "results/tuning").glob("*/*/*/inner_scores.csv")):
        task, scheme, key = path.parts[-4:-1]
        ss = "all" if key.endswith("__all") else "eligible"
        frame = read(path)
        for i, row in frame.iterrows():
            p = next(p for p in independent_plans(task, ss, scheme) if p["fold"] == row.outer_fold)
            j, fit, val = next(x for x in p["inner"] if x[0] == row.inner_fold)
            compare(3, f"{rel(path)}/row {i+2}", "n_samples", len(val), row.n_samples)
    finish_numeric(3, "stream-grouped inner round-robin", start,
                   "verification/codex_leakage/inner_folds.csv/rows 2+; src/folds.py:146",
                   "Pond disjointness applies to LOPO; the calendar inner split is by date as plan 7.4 requires.")
    check(3, "inner stream separation", all(r["disjoint_ponds"] and r["shared_stream_intact"]
          for r in inner_rows if r["scheme"] == "lopo"),
          "verification/codex_leakage/inner_folds.csv/LOPO rows", "All independently constructed LOPO inner folds separate stream units.")
    for task in ["T1", "T2"]:
        p = independent_plans(task, "eligible", "chrono")[0]
        expected = (24, 11, 0) if task == "T1" else (31, 15, 7)
        check(7, f"{task} calendar counts", (len(p["train"]), len(p["test"]), len(p["excluded"])) == expected,
              "verification/codex_leakage/calendar_rows.csv/rows 2+", str(expected))


def visible_daily(pond: int, first: pd.Timestamp, reference: pd.Timestamp) -> pd.DataFrame:
    rows = daily[(daily.pond == pond) & (daily.date >= first) & (daily.date < reference)].copy()
    rows["visible"] = (rows.status != "missing") & ~((rows.status == "interpolated") &
                                                                  (rows.right_anchor_date >= reference))
    return rows


def reference_features(pond: int, ref: pd.Timestamp, window: int, cumulative: bool) -> tuple[dict, pd.DataFrame]:
    rows = visible_daily(pond, ref-pd.Timedelta(days=window), ref)
    visible = rows[rows.visible]
    result = {}
    for sensor in SENSORS:
        x = visible[visible.sensor == sensor]
        for stat in STATS:
            value = np.nan
            if len(x) >= 3:
                value = x[stat].min() if stat == "min" else x[stat].max() if stat == "max" else x[stat].mean()
            result[f"{sensor}_{stat}_w"] = value
        if cumulative:
            c = visible_daily(pond, pd.Timestamp("2021-06-19"), ref)
            c = c[c.visible & (c.sensor == sensor)]
            result[f"{sensor}_mean_cum"] = c["mean"].mean() if len(c) >= 3 else np.nan
    result["n_covered_window"] = int((visible.groupby("date").sensor.nunique() >= 3).sum())
    result["eligible_eval"] = result["n_covered_window"] >= window // 2
    return result, rows


def feature_checks() -> None:
    start = len(comparisons)
    summaries, anchors = [], []
    # Audit every sample, exceeding the requested five references per task.
    for task, table in tables.items():
        for i, r in table.iterrows():
            reference = pd.Timestamp(r.date if task == "T1" else r.origin_date)
            values, window = reference_features(int(r.pond), reference, 14 if task == "T1" else 28, task == "T1")
            for name, v in values.items():
                compare(5, f"data/processed/{task.lower()}_table.csv/row {i+2}", name, v, r[name])
            adjacent = window[(window.status == "interpolated") &
                              (window.date >= reference-pd.Timedelta(days=2))]
            for k, a in adjacent.iterrows():
                anchors.append(dict(task=task, table_row=i+2, pond=int(r.pond), reference=str(reference.date()),
                    daily_row=k+2, sensor=a.sensor, date=str(a.date.date()),
                    right_anchor=str(a.right_anchor_date.date()), included=bool(a.visible)))
            summaries.append(dict(task=task, table_row=i+2, pond=int(r.pond), reference=str(reference.date()),
                n_covered=values["n_covered_window"], n_interpolated=int((window.status == "interpolated").sum()),
                n_hidden_right_anchor=int(((window.status == "interpolated") & ~window.visible).sum())))
    save("feature_references.csv", summaries)
    save("adjacent_interpolations.csv", anchors)
    finish_numeric(5, "causal tabular features", start,
                   "verification/codex_leakage/feature_references.csv/rows 2+; adjacent_interpolations.csv/rows 2+")
    # The expected sequence is constructed directly from saved daily rows, then compared
    # with the implementation. This calls no pipeline function to compute a reference.
    sys.path.insert(0, str(ROOT))
    from src.models.common import DailyStore, build_sampleset, fit_scalers, make_batch
    store = DailyStore(read("data/processed/daily_wq.csv"), SENSORS, STATS)
    start = len(comparisons)
    seq_rows = []
    for task, table in tables.items():
        rows = table[table.eligible_eval]
        ss = build_sampleset(task, rows, store, cfg)
        for position, (i, r) in enumerate(rows.iterrows()):
            ref = pd.Timestamp(r.date if task == "T1" else r.origin_date)
            first = pd.Timestamp("2021-06-19") if task == "T1" else ref-pd.Timedelta(days=28)
            dates = pd.date_range(first, ref-pd.Timedelta(days=1))
            expected_vals = np.full((len(dates), 6, 4), np.nan)
            expected_valid = np.zeros((len(dates), 6), dtype=bool)
            window = visible_daily(int(r.pond), first, ref)
            for a in window[window.visible].itertuples():
                d, s = int((a.date-first).days), SENSORS.index(a.sensor)
                expected_vals[d, s] = [getattr(a, k) for k in STATS]
                expected_valid[d, s] = True
            actual_vals, actual_valid = store.visible_window(store.pond_index[int(r.pond)],
                int((first-store.start).days), int((ref-store.start).days))
            length = len(dates)
            source = f"data/processed/{task.lower()}_table.csv/row {i+2}"
            compare(5, source, "visible_window values and masks", True,
                    np.allclose(expected_vals, actual_vals, rtol=0, atol=0, equal_nan=True) and
                    np.array_equal(expected_valid, actual_valid))
            compare(5, source, "build_sampleset values masks padding and end day", True,
                    np.allclose(expected_vals, ss.vals[position, -length:], rtol=0, atol=0, equal_nan=True) and
                    np.array_equal(expected_valid, ss.valid[position, -length:]) and
                    not ss.valid[position, :-length].any() and ss.pad[position, :-length].all() and
                    not ss.pad[position, -length:].any() and
                    ss.day[position, -1] == (ref-store.start).days-1)
            seq_rows.append(dict(task=task, table_row=i+2, first=str(first.date()), last=str(dates[-1].date()),
                                 reference=str(ref.date()), days=length, hidden_days=int((~expected_valid).sum())))
        # Independently compute each fold's scaler statistics from visible training days.
        for scheme in ["lopo", "chrono"]:
            for p in independent_plans(task, "eligible", scheme):
                for split, ids in [("outer_train", p["train"]), *[(f"inner{j}", fit) for j, fit, val in p["inner"]]]:
                    sc = fit_scalers(ss, ids)
                    cells = set()
                    for i in ids:
                        r = table.loc[i]
                        ref = pd.Timestamp(r.date if task == "T1" else r.origin_date)
                        first = pd.Timestamp("2021-06-19") if task == "T1" else ref-pd.Timedelta(days=28)
                        x = visible_daily(int(r.pond), first, ref)
                        cells.update(x.index[x.visible])
                    selected = daily.loc[sorted(cells)]
                    expected_mean, expected_std = np.zeros((6, 4)), np.ones((6, 4))
                    for s, sensor in enumerate(SENSORS):
                        x = selected[selected.sensor == sensor][STATS].to_numpy()
                        if len(x):
                            expected_mean[s] = x.mean(axis=0)
                            sd = x.std(axis=0)
                            expected_std[s] = np.where(sd > 1e-12, sd, 1)
                    x = table.loc[ids, cfg["statics"][task]].to_numpy()
                    y = table.loc[ids, "log_w" if task == "T1" else "y"].to_numpy()
                    expected_sstd = np.where(x.std(axis=0) > 1e-12, x.std(axis=0), 1)
                    ok = np.allclose(sc.ch_mean, expected_mean, rtol=1e-12, atol=1e-12) and \
                         np.allclose(sc.ch_std, expected_std, rtol=1e-12, atol=1e-12) and \
                         np.allclose(sc.st_mean, x.mean(axis=0)) and np.allclose(sc.st_std, expected_sstd) and \
                         same(sc.y_mean, y.mean()) and same(sc.y_std, y.std() if y.std() > 1e-12 else 1)
                    compare(4, f"inner_folds.csv/{task} {scheme} outer {p['fold']} {split}",
                            "all scaler statistics use training rows only", True, ok)
                sc = fit_scalers(ss, p["train"])
                batch = make_batch(ss, p["test"], sc)
                compare(4, f"outer_folds.csv/{task} {scheme} outer {p['fold']}", "missing channel inputs zero", True,
                        bool(np.isfinite(batch.x.numpy()).all()))
    save("sequence_references.csv", seq_rows)
    finish_numeric(5, "sequence visibility and scaler comparison", start,
                   "verification/codex_leakage/sequence_references.csv/rows 2+; numeric_comparisons.csv/rows 2+")


def final_epoch_checks() -> None:
    start = len(comparisons)
    rows = []
    for path in sorted((ROOT / "results/tuning").glob("*/*/*/chosen.csv")):
        task, scheme, model = path.parts[-4:-1]
        if model not in NEURAL:
            continue
        chosen = read(path)
        curves = read(path.with_name("curves.csv.gz"))
        finals = curves[curves.stage == "final"]
        for i, r in chosen.iterrows():
            trials_path = path.parent / f"fold{int(r.outer_fold):02d}/trials.csv"
            trials = read(trials_path)
            best = trials[trials.number == r.best_trial].iloc[0]
            epoch_columns = [c for c in trials if c.endswith("_best_epoch")]
            expected = int(np.median([best[c] for c in epoch_columns]))
            compare(4, f"{rel(path)}/row {i+2}", "epochs equals best-trial inner median", expected, r.epochs)
            for seed in range(5):
                sub = finals[(finals.outer_fold == r.outer_fold) & (finals.seed == seed)]
                compare(4, f"{rel(path.with_name('curves.csv.gz'))}/fold {int(r.outer_fold)}, seed {seed}",
                        "fixed final epochs and no validation losses", True,
                        len(sub) == expected and sub.epoch.tolist() == list(range(1, expected+1)) and sub.val_mse.isna().all())
            rows.append(dict(task=task, scheme=scheme, model=model, fold=int(r.outer_fold), epochs=expected,
                             source=f"{rel(path)}/row {i+2}"))
    save("final_epochs.csv", rows)
    finish_numeric(4, "fixed-epoch final fitting", start, "verification/codex_leakage/final_epochs.csv/rows 2+")


def input_checks() -> None:
    rows = []
    for task in ["T1", "T2"]:
        for model, spec in cfg["model_features"][task].items():
            columns = []
            for group in spec["groups"]:
                if group == "wq":
                    columns += [f"{s}_{k}_w" for s in SENSORS for k in STATS]
                    if task == "T1":
                        columns += [f"{s}_mean_cum" for s in SENSORS]
                else:
                    columns += cfg["feature_groups"][task][group]
            forbidden = {"w", "log_w", "w_target", "y", "l", "target_date", "horizon", "w_origin"} & set(columns)
            if task == "T1" and not model.endswith("_len"):
                forbidden |= {"log_l"} & set(columns)
            rows.append(dict(task=task, model=model, columns=" ".join(columns), status="FAIL" if forbidden else "PASS"))
        expected = ["t"] if task == "T1" else ["log_w_k", "log_w_prev", "t_k", "dt_prev", "dt_target"]
        check(6, f"{task} neural static inputs", cfg["statics"][task] == expected,
              "configs/phase4.yaml:131; src/models/common.py:231", repr(cfg["statics"][task]))
    save("input_features.csv", rows)
    check(6, "tabular feature allowlist", all(r["status"] == "PASS" for r in rows),
          "verification/codex_leakage/input_features.csv/rows 2+; src/baselines.py:519", f"{len(rows)} model feature lists")


def table_comparison(filename: str, expected: pd.DataFrame, keys: list[str], fields: list[str]) -> None:
    saved = read(f"results/{filename}")
    for c in keys:
        saved[c] = saved[c].astype(str)
        expected[c] = expected[c].astype(str)
    want = expected.set_index(keys)
    got = saved.set_index(keys)
    compare(10, f"results/{filename}/all rows", "unique row keys and complete key set", True,
            want.index.is_unique and got.index.is_unique and set(want.index) == set(got.index))
    for i, r in saved.iterrows():
        key = tuple(r[c] for c in keys)
        e = want.loc[key]
        for field in fields:
            compare(10, f"results/{filename}/row {i+2}", field, e[field], r[field])


def exact_signed_ranks(differences: np.ndarray) -> dict:
    """Enumerate all sign flips of raw, nonzero differences with exact midranks."""
    d = np.asarray(differences, dtype=float)
    nonzero = d[d != 0]
    ranks2 = []
    magnitudes = np.abs(nonzero)
    for v in magnitudes:
        lower = int((magnitudes < v).sum())
        equal = int((magnitudes == v).sum())
        ranks2.append(2*lower+equal+1)
    observed = sum(r for r, v in zip(ranks2, nonzero) if v > 0)
    flips = [sum(r for r, sign in zip(ranks2, signs) if sign)
             for signs in itertools.product([False, True], repeat=len(nonzero))]
    lower = sum(v <= observed for v in flips) / len(flips)
    upper = sum(v >= observed for v in flips) / len(flips)
    return dict(n_nonzero=len(nonzero), n_zero_dropped=len(d)-len(nonzero), t_plus=observed/2,
                t_minus=(sum(ranks2)-observed)/2, p_one_sided_lower=lower,
                p_two_sided=min(1., 2*min(lower, upper)))


def independent_vector(task: str, horizon: str, model: str) -> pd.Series:
    rows = independent_ponds[(independent_ponds.task == task) & (independent_ponds.scheme == "lopo") &
         (independent_ponds.sample_set == "eligible") & (independent_ponds.horizon == horizon) &
         (independent_ponds.model == model)]
    return rows.set_index("pond").S.sort_index()


def score_checks() -> None:
    global independent_ponds, independent_scores
    start = len(comparisons)
    pond_seed, pond_mean, pooled_seed, pooled_mean, scores = [], [], [], [], []
    for path, frame in prediction_files:
        task, scheme, ss, model = (frame[c].iat[0] for c in ["task", "scheme", "sample_set", "model"])
        learned = model in NEURAL or model.startswith(("ridge", "gbm"))
        compare(10, f"{rel(path)}/all rows", "saved weights finite; learned log predictions finite", True,
                np.isfinite(frame[["w_pred", "w_true"]]).all().all() and
                (not learned or np.isfinite(frame.y_pred_log).all()))
        expected_weight = np.exp(frame.y_pred_log.to_numpy())
        if task == "T2":
            expected_weight *= frame.w_origin.to_numpy()
        positive = frame.w_pred.to_numpy() > 0
        compare(10, f"{rel(path)}/all rows", "prediction log/grams agree", True,
                np.allclose(frame.w_pred.to_numpy()[positive], expected_weight[positive], rtol=2e-12, atol=1e-12))
        strata = [("0", frame)] if task == "T1" else [(str(h), frame[frame.horizon == h]) for h in [1, 2]] + [("all", frame)]
        for h, sub in strata:
            prefix = dict(task=task, scheme=scheme, sample_set=ss, horizon=h, model=model)
            per_seed_s, seed_metrics, mean_pond = [], [], []
            for seed, sample in sub.groupby("seed", sort=True):
                raw = sample.w_pred.to_numpy()-sample.w_true.to_numpy()
                sst = np.square(sample.w_true.to_numpy()-sample.w_true.mean()).sum()
                m = dict(n=len(sample), mae_g=np.abs(raw).mean(), rmse_g=np.sqrt(np.square(raw).mean()),
                         mape_pct=100*np.mean(np.abs(raw)/sample.w_true.to_numpy()),
                         r2=1-np.square(raw).sum()/sst if sst else np.nan)
                seed_metrics.append(m)
                pooled_seed.append({**prefix, "seed": int(seed), **m})
                per_seed_ponds = []
                for pond, p in sample.groupby("pond", sort=True):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        err = np.where(p.w_pred.to_numpy() > 0,
                                       np.abs(np.log(p.w_pred.to_numpy())-np.log(p.w_true.to_numpy())), np.inf)
                    e = np.abs(p.w_pred.to_numpy()-p.w_true.to_numpy())
                    row = dict(n=len(p), S=float(err.mean()), mae_g=e.mean(),
                               mape_pct=100*np.mean(e/p.w_true.to_numpy()),
                               n_nonpositive=int((p.w_pred <= 0).sum()))
                    per_seed_ponds.append(row["S"])
                    pond_seed.append({**prefix, "seed": int(seed), "pond": int(pond), **row})
                per_seed_s.append(np.mean(per_seed_ponds))
            seed_frame = pd.DataFrame(seed_metrics)
            pm = {**prefix, "n_samples": int(seed_frame.n.iloc[0]), "n_seeds": len(seed_frame)}
            for field in ["mae_g", "rmse_g", "mape_pct", "r2"]:
                pm[field] = seed_frame[field].mean()
                pm[f"{field}_sd"] = seed_frame[field].std(ddof=1)
            pooled_mean.append(pm)
            selected = [r for r in pond_seed if all(r[k] == v for k, v in prefix.items())]
            for pond in sorted(sub.pond.unique()):
                records = [r for r in selected if r["pond"] == pond]
                row = {**prefix, "pond": int(pond), "n": records[0]["n"], "n_seeds": len(records)}
                for field in ["S", "mae_g", "mape_pct", "n_nonpositive"]:
                    row[field] = np.mean([r[field] for r in records])
                pond_mean.append(row)
                mean_pond.append(row)
            group = next(g for g, names in cfg["models"][task].items() if g != "all_sample_set" and model in names)
            scores.append({**prefix, "group": group, "n_ponds": len(mean_pond), "n_samples": int(seed_frame.n.iloc[0]),
                "n_seeds": len(seed_frame), "S": np.mean([r["S"] for r in mean_pond]),
                "S_seed_sd": np.std(per_seed_s, ddof=1) if len(per_seed_s) > 1 else np.nan,
                "sd_pond_S": np.std([r["S"] for r in mean_pond], ddof=1),
                "sd_pond_mae_g": np.std([r["mae_g"] for r in mean_pond], ddof=1),
                "n_nonpositive": sum(r["n_nonpositive"] for r in mean_pond)})
    independent_ponds, independent_scores = pd.DataFrame(pond_mean), pd.DataFrame(scores)
    key = ["task", "scheme", "sample_set", "horizon", "model"]
    expected_tables = {
        "per_pond_seed": (pd.DataFrame(pond_seed), key+["seed", "pond"], ["n", "S", "mae_g", "mape_pct", "n_nonpositive"]),
        "per_pond": (independent_ponds.copy(), key+["pond"], ["n", "n_seeds", "S", "mae_g", "mape_pct", "n_nonpositive"]),
        "pooled_by_seed": (pd.DataFrame(pooled_seed), key+["seed"], ["n", "mae_g", "rmse_g", "mape_pct", "r2"]),
        "pooled": (pd.DataFrame(pooled_mean), key, ["n_samples", "n_seeds", "mae_g", "rmse_g", "mape_pct", "r2",
                                                  "mae_g_sd", "rmse_g_sd", "mape_pct_sd", "r2_sd"]),
        "scores": (independent_scores.copy(), key, ["group", "n_ponds", "n_samples", "n_seeds", "S", "S_seed_sd",
                                                    "sd_pond_S", "sd_pond_mae_g", "n_nonpositive"])}
    for name, (frame, keys, fields) in expected_tables.items():
        save(f"recomputed_{name}.csv", frame)
        table_comparison(f"metrics/{name}.csv", frame.copy(), keys, fields)
    finish_numeric(10, "all metrics independently recomputed from grams", start,
                   "verification/codex_leakage/recomputed_scores.csv/rows 2+; numeric_comparisons.csv/metrics rows")
    selections = {t: json.loads((ROOT / f"results/selection/primary_{t}.json").read_text()) for t in ["T1", "T2"]}
    start = len(comparisons)
    tests = []
    family_open = True
    saved = read("results/stats/confirmatory.csv")
    for i, row in saved.iterrows():
        task, h = row.task, str(row.horizon)
        a = independent_vector(task, h, selections[task]["primary_deep"])
        b = independent_vector(task, h, selections[task]["primary_baseline"])
        d = a-b
        test = exact_signed_ranks(d.to_numpy())
        for field, value in test.items():
            compare(10, f"results/stats/confirmatory.csv/row {i+2}", field, value, row[field])
        reject = family_open and len(d) >= 6 and test["p_one_sided_lower"] <= .05
        status = "not tested (an earlier step did not reject)" if not family_open else "rejected" if reject else "not rejected"
        compare(10, f"results/stats/confirmatory.csv/row {i+2}", "status", status, row.status)
        compare(10, f"results/stats/confirmatory.csv/row {i+2}", "reject", bool(reject), bool(row.reject))
        family_open = bool(reject)
        tests.append(dict(task=task, horizon=h, n_sign_flips=2**len(d[d != 0]), **test, status=status,
                          differences=json.dumps({str(k): v for k, v in d.items()})))
    save("recomputed_confirmatory.csv", tests)
    # Spot-check the first stratum with a fresh generator. Each replicate jointly samples
    # seven pond indices; no implementation-under-test bootstrap routine is imported.
    a = independent_vector("T1", "0", selections["T1"]["primary_deep"]).to_numpy()
    b = independent_vector("T1", "0", selections["T1"]["primary_baseline"]).to_numpy()
    rng = np.random.default_rng(20261014)
    draws = rng.integers(0, len(a), size=(2000, len(a)))
    boot_s = np.mean(a[draws], axis=1)
    boot_d = np.mean(a[draws]-b[draws], axis=1)
    lows, highs = np.quantile(boot_s, [.025, .975])
    lowd, highd = np.quantile(boot_d, [.025, .975])
    boot_table = read("results/stats/bootstrap_S.csv")
    pair_table = read("results/stats/pairwise.csv")
    br = boot_table[(boot_table.task == "T1") & (boot_table.horizon.astype(str) == "0") &
                    (boot_table.model == selections["T1"]["primary_deep"])].iloc[0]
    pr = pair_table[(pair_table.task == "T1") & (pair_table.horizon.astype(str) == "0") &
                    (pair_table.kind == "primary_deep_vs_primary_baseline")].iloc[0]
    for field, v in [("ci_low", lows), ("ci_high", highs)]:
        compare(10, f"results/stats/bootstrap_S.csv/row {br.name+2}", field, v, br[field])
    for field, v in [("diff_ci_low", lowd), ("diff_ci_high", highd)]:
        compare(10, f"results/stats/pairwise.csv/row {pr.name+2}", field, v, pr[field])
    details["bootstrap"] = dict(B=2000, seed=20261014, S_interval=[lows, highs], difference_interval=[lowd, highd])
    save("bootstrap_spotcheck.csv", [dict(replicate=i, pond_indices=" ".join(map(str, draws[i])), S=boot_s[i],
                                         difference=boot_d[i]) for i in range(2000)])
    finish_numeric(10, "exact signed ranks, fixed sequence and bootstrap", start,
                   "verification/codex_leakage/recomputed_confirmatory.csv/rows 2-3; bootstrap_spotcheck.csv/rows 2-2001")
    # Recheck the saved pairwise, influence and merged-stream tables as well, so the
    # framing's interval and robustness claims have independent numerical evidence.
    start = len(comparisons)
    pairs, influences, merged, bootstraps = (read(f"results/stats/{name}.csv") for name in
                                           ["pairwise", "influence", "merged_units", "bootstrap_S"])
    bootstrap_rng = np.random.default_rng(20261014)
    stats_rows = []
    for task, h in [("T1", "0"), ("T2", "1"), ("T2", "2"), ("T2", "all")]:
        models = sorted(independent_ponds[(independent_ponds.task == task) & (independent_ponds.scheme == "lopo") &
                        (independent_ponds.sample_set == "eligible") & (independent_ponds.horizon == h)].model.unique())
        vectors = {m: independent_vector(task, h, m).to_numpy() for m in models}
        pond_order = independent_vector(task, h, models[0]).index.tolist()
        sample_indices = bootstrap_rng.integers(0, len(pond_order), size=(2000, len(pond_order)))
        boot_values = {m: np.mean(v[sample_indices], axis=1) for m, v in vectors.items()}
        for i, r in bootstraps[(bootstraps.task == task) & (bootstraps.horizon.astype(str) == h)].iterrows():
            lo, hi = np.quantile(boot_values[r.model], [.025, .975])
            for field, value in [("ci_low", lo), ("ci_high", hi)]:
                compare(10, f"results/stats/bootstrap_S.csv/row {i+2}", field, value, r[field])
        for i, r in pairs[(pairs.task == task) & (pairs.horizon.astype(str) == h)].iterrows():
            a, b = vectors[r.model], vectors[r.comparator]
            delta = a-b
            test = exact_signed_ranks(delta)
            lo, hi = np.quantile(boot_values[r.model]-boot_values[r.comparator], [.025, .975])
            values = dict(S_model=a.mean(), S_comparator=b.mean(), diff=delta.mean(), diff_ci_low=lo, diff_ci_high=hi,
                          n_lower=int((delta < 0).sum()), n_higher=int((delta > 0).sum()), n_equal=int((delta == 0).sum()), **test)
            for field, value in values.items():
                compare(10, f"results/stats/pairwise.csv/row {i+2}", field, value, r[field])
            if r.kind == "primary_deep_vs_primary_baseline":
                stats_rows.append(dict(task=task, horizon=h, model=r.model, comparator=r.comparator, **values))
        for i, r in influences[(influences.task == task) & (influences.horizon.astype(str) == h)].iterrows():
            a = np.delete(vectors[r.model], pond_order.index(r.pond_removed))
            b = np.delete(vectors[r.comparator], pond_order.index(r.pond_removed))
            for field, value in [("S_model", a.mean()), ("S_comparator", b.mean()), ("diff", (a-b).mean())]:
                compare(10, f"results/stats/influence.csv/row {i+2}", field, value, r[field])
        for i, r in merged[(merged.task == task) & (merged.horizon.astype(str) == h)].iterrows():
            a, b = vectors[r.model], vectors[r.comparator]
            unit_a, unit_b = np.r_[a[:2].mean(), a[2:]], np.r_[b[:2].mean(), b[2:]]
            values = dict(S_model=unit_a.mean(), S_comparator=unit_b.mean(), n_units=6,
                          **exact_signed_ranks(unit_a-unit_b))
            for field, value in values.items():
                compare(10, f"results/stats/merged_units.csv/row {i+2}", field, value, r[field])
    save("recomputed_primary_comparisons.csv", stats_rows)
    finish_numeric(10, "all saved statistics and robustness tables", start,
                   "verification/codex_leakage/recomputed_primary_comparisons.csv/rows 2-5; numeric_comparisons.csv/stats rows")
    print("all saved metrics, exact confirmatory p-values and bootstrap spot-check recomputed", flush=True)


def selection_checks() -> None:
    start = len(comparisons)
    rows = []
    histories = []
    for task in ["T1", "T2"]:
        path = ROOT / f"results/selection/primary_{task}.json"
        saved = json.loads(path.read_text())
        selection_commit = git("log", "--reverse", "--diff-filter=A", "--format=%H", "--", rel(path)).splitlines()[0]
        for directory in ["metrics", "stats"]:
            first = git("log", "--reverse", "--diff-filter=A", "--format=%H", "--", f"results/{directory}").splitlines()[0]
            ok = selection_commit != first and subprocess.run(
                ["git", "merge-base", "--is-ancestor", selection_commit, first], cwd=ROOT, capture_output=True).returncode == 0
            compare(9, f"{rel(path)}/primary fields; git history", f"selection precedes {directory}", True, ok)
            histories.append(dict(task=task, selection_commit=selection_commit, output=directory, first_commit=first))
        scores = {}
        for group, field in [("time_only", "primary_baseline"), ("deep", "primary_deep")]:
            pool = cfg["models"][task][group]
            for model in pool:
                inner_path = ROOT / f"results/tuning/{task}/lopo/{model}/inner_scores.csv"
                frame = read(inner_path)
                compare(9, f"{rel(path)}/inputs/{rel(inner_path)}", "selection input hash", saved["inputs"][rel(inner_path).removeprefix("results/")], sha(inner_path))
                if task == "T2":
                    for i, r in frame.iterrows():
                        compare(9, f"{rel(inner_path)}/row {i+2}", "inner S averages h1/h2", (r.S_h1+r.S_h2)/2, r.S)
                folds = frame.groupby("outer_fold").S.mean()
                score = sum(folds)/len(folds)
                scores[model] = score
                compare(9, f"{rel(path)}/mean_inner_S/{model}", "mean inner S", score, float(saved["mean_inner_S"][model]))
                compare(9, f"{rel(inner_path)}/outer_fold column", "all seven outer folds", "[0, 1, 2, 3, 4, 5, 7]", str(sorted(folds.index.tolist())))
                rows.append(dict(task=task, group=group, model=model, mean_inner_S=score,
                                 source=f"{rel(inner_path)}/rows 2+"))
            best = min(pool, key=lambda m: (scores[m], pool.index(m)))
            compare(9, f"{rel(path)}/{field}", "argmin, ties to first in pool", best, saved[field])
    save("selection_recomputed.csv", rows)
    save("selection_history.csv", histories)
    finish_numeric(9, "selection from inner scores before scoring", start,
                   "verification/codex_leakage/selection_recomputed.csv/rows 2+; selection_history.csv/rows 2-5")


def permutation_checks() -> None:
    start = len(comparisons)
    results = read("results/stats/permutation.csv")
    recomputed = []
    for i, r in results.iterrows():
        path = ROOT / f"results/permutation/{r.task}/{r.model}.csv"
        values = read(path)
        predictions = next(frame for p, frame in prediction_files if rel(p) == f"results/predictions/{r.task}/lopo/{r.model}.csv")
        predictions = predictions[predictions.seed == 0]
        if r.task == "T2":
            predictions = predictions[predictions.horizon == 1]
        observed = np.mean([np.abs(np.log(s.w_pred.to_numpy())-np.log(s.w_true.to_numpy())).mean()
                            for _, s in predictions.groupby("pond")])
        n = int((values.S_perm <= observed).sum())
        expected_r = 199 if r.model in NEURAL else 999
        expected = dict(S_obs=observed, R=expected_r, n_perm_le_obs=n, p=(n+1)/(expected_r+1),
            S_perm_mean=values.S_perm.mean(), S_perm_sd=values.S_perm.std(ddof=1),
            S_perm_min=values.S_perm.min(), S_perm_max=values.S_perm.max())
        for field, value in expected.items():
            compare(8, f"results/stats/permutation.csv/row {i+2}", field, value, r[field])
        compare(8, f"{rel(path)}/r column", "full r=0..R-1", True,
                values.r.tolist() == list(range(expected_r)) and np.isfinite(values.S_perm).all())
        meta = json.loads(path.with_suffix(".json").read_text())
        compare(8, f"{rel(path.with_suffix('.json'))}/seed", "permutation seed", 20261014, meta["seed"])
        compare(8, f"{rel(path.with_suffix('.json'))}/fit_seed", "observed/permuted fit seed", 0, meta["fit_seed"])
        compare(8, f"{rel(path.with_suffix('.json'))}/observed_check", "observed S matches seed 0", observed, meta["observed_check"]["observed_S"])
        # Independent counts derive only from eligible date/stream keys. The values and
        # missing masks move together in the source path inspected separately.
        table = tables[r.task]
        eligible = table[table.eligible_eval]
        key = "date" if r.task == "T1" else "origin_date"
        n_dates = unchanged = changed_rows = 0
        for date, rows in eligible.groupby(key, sort=True):
            n_dates += 1
            units = {(1, 2) if p in [1, 2] else (int(p),) for p in rows.pond}
            if len(units) < 2:
                unchanged += 1
            else:
                changed_rows += len(rows)
        for field, expected_value in [("n_dates", n_dates), ("n_dates_unchanged", unchanged), ("n_rows_changed", changed_rows)]:
            compare(8, f"{rel(path)}/{field} column", field, True, bool((values[field] == expected_value).all()))
        recomputed.append(dict(task=r.task, model=r.model, **expected))
    save("recomputed_permutation.csv", recomputed)
    finish_numeric(8, "permutation output p-values and provenance", start,
                   "verification/codex_leakage/recomputed_permutation.csv/rows 2-6")


def stop_checks() -> None:
    log = (ROOT / "results/logs/run_all_phase4.log").read_text().splitlines()
    auto = (ROOT / "results/logs/autopilot.log").read_text().splitlines()
    stops = {}
    pattern = re.compile(r"^(T[12]) (lopo|chrono) ([\w]+): ERROR .*predictions are not finite")
    for i, line in enumerate(log, start=1):
        hit = pattern.search(line)
        if hit:
            stops.setdefault(hit.groups(), []).append(i)
    completeness = read("results/metrics/completeness.csv")
    rows = []
    for (task, scheme, model), errors in stops.items():
        source = f"results/logs/run_all_phase4.log:{errors[0]}"
        aggregate = ROOT / f"results/predictions/{task}/{scheme}/{model}.csv"
        permutation = ROOT / f"results/permutation/{task}/{model}.csv"
        part_files = list((ROOT / f"results/tuning/{task}/{scheme}/{model}").glob("fold*/predictions.csv"))
        hit = completeness[(completeness.task == task) & (completeness.scheme == scheme) & (completeness.model == model)]
        auto_lines = [i for i, line in enumerate(auto, start=1) if f"--task {task} --scheme {scheme} --model {model} --" in line]
        has_stop_status = any("stop" in str(v).lower() for v in hit.to_numpy().ravel())
        check(12, f"{task} {model} no completed frozen output", not aggregate.exists() and not permutation.exists(),
              source, f"aggregate absent; permutation absent; {len(part_files)} partial fold prediction files remain")
        check(12, f"{task} {model} literal absence of every prediction file", not part_files,
              f"{rel(part_files[0])}/row 2" if part_files else source,
              f"{len(part_files)} partial fold files survive. These are outside the scorer's input directory and are unscored.")
        check(12, f"{task} {model} explicitly stopped in metrics", has_stop_status,
              f"results/metrics/completeness.csv/row {hit.index[0]+2}",
              f"has_predictions={bool(hit.has_predictions.iloc[0])}; no stop-status/reason column")
        check(12, f"{task} {model} repeats logged", len(auto_lines) >= len(errors)-1,
              f"results/logs/run_all_phase4.log/lines {errors}; results/logs/autopilot.log/lines {auto_lines}",
              f"{len(errors)} failed invocations; repeats recorded; no completed frozen output replaced", note=True)
        rows.append(dict(task=task, scheme=scheme, model=model, stopped_log_lines=str(errors),
            autopilot_lines=str(auto_lines), aggregate_exists=aggregate.exists(), permutation_exists=permutation.exists(),
            n_partial_fold_predictions=len(part_files), partial_files=";".join(rel(p) for p in part_files),
            metrics_source=f"results/metrics/completeness.csv/row {hit.index[0]+2}", metrics_status_present=has_stop_status))
    save("stopped_runs.csv", rows)
    details["stops"] = rows
    check(12, "all stop records found", len(stops) == 3,
          "verification/codex_leakage/stopped_runs.csv/rows 2-4", f"{len(stops)} distinct model/scheme runs")
    post = ROOT / "results/posthoc/ridge_candidate_loses"
    labelled = []
    for p in sorted(post.glob("**/*.json")):
        if p.name not in ["run_meta.json", "ridge_pw_wq.json"]:
            continue
        obj = json.loads(p.read_text())
        labelled.append(dict(file=rel(p), status="PASS" if obj.get("posthoc") == "ridge_candidate_loses" and
                              obj.get("decision") == "DECISIONS.md D-003" else "FAIL"))
    for p in sorted(post.glob("metrics/*.csv")) + sorted(post.glob("stats/*.csv")):
        frame = read(p)
        ok = "posthoc" in frame and "decision" in frame and (frame.posthoc == "ridge_candidate_loses").all()
        labelled.append(dict(file=rel(p), status="PASS" if ok else "FAIL"))
    save("posthoc_labels.csv", labelled)
    check(12, "post hoc separation and labels", bool(labelled) and all(r["status"] == "PASS" for r in labelled)
          and "post hoc" in (post / "README.md").read_text(),
          "results/posthoc/ridge_candidate_loses/README.md:3; verification/codex_leakage/posthoc_labels.csv/rows 2+",
          "Post hoc output root, metadata and scored tables identify D-003; frozen aggregates remain absent.")


def claim_checks() -> None:
    """Build a claim-by-claim ledger, using independently recomputed per-pond errors."""
    text = (ROOT / "DECISIONS.md").read_text()
    if "## Phase 5" not in text:
        check(11, "Phase 5 claims", True, "DECISIONS.md", "No Phase 5 framing entry", note=True)
        return
    rows = []
    def record(line: int, claim: str, ok: bool, value: object, status: str = "PASS") -> None:
        rows.append(dict(status=status if ok else "FAIL", decision_line=line, claim=claim, recomputed=str(value)))
    def s(task: str, model: str, h: str = "0") -> float:
        return float(independent_vector(task, h, model).mean())
    def d(task: str, model: str, baseline: str, h: str = "0") -> pd.Series:
        return independent_vector(task, h, model)-independent_vector(task, h, baseline)
    conf = read("results/stats/confirmatory.csv")
    record(219, "neither task rejects, T2 gated", not conf.reject.any() and conf.status.iat[1].startswith("not tested"), conf.status.tolist())
    record(221, "T1 lower-S ponds are 1, 2, 4", d("T1", "lstm", "cohort_mean").index[
           d("T1", "lstm", "cohort_mean") < 0].tolist() == [1, 2, 4],
           d("T1", "lstm", "cohort_mean").index[d("T1", "lstm", "cohort_mean") < 0].tolist())
    primary = read("verification/codex_leakage/recomputed_primary_comparisons.csv")
    for line, task, h, low, high in [(221, "T1", "0", -.073, .131), (223, "T2", "1", -.002, .091)]:
        r = primary[(primary.task == task) & (primary.horizon.astype(str) == h)].iloc[0]
        record(line, f"{task} primary difference bootstrap interval", round(r.diff_ci_low, 3) == low and
               round(r.diff_ci_high, 3) == high, [r.diff_ci_low, r.diff_ci_high])
    for line, task, model, baseline, h, sm, sb, diff, lower, p in [
        (220, "T1", "lstm", "cohort_mean", "0", .290, .265, .025, 3, .66),
        (222, "T2", "cnn_lstm", "ridge_pw", "1", .332, .295, .037, 2, .92)]:
        delta = d(task, model, baseline, h)
        test = exact_signed_ranks(delta.to_numpy())
        values = [s(task, model, h), s(task, baseline, h), float(delta.mean()), int((delta < 0).sum()), test["p_one_sided_lower"]]
        ok = all(round(x, 3) == y for x, y in zip(values[:3], [sm, sb, diff])) and values[3] == lower and round(values[4], 2) == p
        record(line, f"{task} primary model, comparator, difference, pond count and p", ok, values)
    better_t1 = {m: s("T1", m) for m in cfg["models"]["T1"]["time_only"] if s("T1", m) < s("T1", "lstm")}
    record(225, "T1 all lower-S baselines named", set(better_t1) == {"gc", "cohort_mean", "gbm_time", "ridge_time"}, better_t1)
    named_scores = [(226, "T1", "gc", "0", .261), (226, "T1", "gbm_time", "0", .266),
                    (227, "T1", "ridge_time", "0", .269), (227, "T2", "gbm_pw", "1", .211),
                    (228, "T2", "anchored_gc", "1", .274), (228, "T2", "cohort_growth", "1", .281),
                    (229, "T2", "persistence", "1", .529), (230, "T2", "sgr", "1", .545)]
    for line, task, model, horizon, expected in named_scores:
        value = s(task, model, horizon)
        record(line, f"{task} {model} stated S", round(value, 3) == expected, value)
    better_t2 = {m: s("T2", m, "1") for m in cfg["models"]["T2"]["time_only"] if s("T2", m, "1") < s("T2", "cnn_lstm", "1")}
    record(227, "T2 all lower-S baselines named", set(better_t2) == {"gbm_pw", "anchored_gc", "cohort_growth", "ridge_pw"}, better_t2)
    record(227, "CNN-LSTM worse than GBM-PW in 6 ponds", int((d("T2", "cnn_lstm", "gbm_pw", "1") > 0).sum()) == 6,
           int((d("T2", "cnn_lstm", "gbm_pw", "1") > 0).sum()))
    main_t2 = cfg["models"]["T2"]["time_only"] + cfg["models"]["T2"]["water_quality"] + cfg["models"]["T2"]["deep"] + ["mlp_static"]
    values = {m: s("T2", m, "1") for m in main_t2 if m != "ridge_pw_wq"}
    record(229, "all main models except SGR beat persistence", all(v < values["persistence"] for m, v in values.items()
           if m not in ["persistence", "sgr"]) and values["sgr"] > values["persistence"], values)
    record(230, "CNN-LSTM below persistence in 7 ponds", int((d("T2", "cnn_lstm", "persistence", "1") < 0).sum()) == 7,
           int((d("T2", "cnn_lstm", "persistence", "1") < 0).sum()))
    delta1 = d("T1", "lstm", "cohort_mean")
    influence1 = {int(p): float(delta1.drop(p).mean()) for p in delta1.index}
    record(232, "T1 omitting 7/9 reverses sign", round(influence1[7], 3) == -.008 and round(influence1[9], 3) == -.005, influence1)
    delta2 = d("T2", "cnn_lstm", "ridge_pw", "1")
    influence2 = [float(delta2.drop(p).mean()) for p in delta2.index]
    record(234, "T2 remains worse whichever pond is removed", min(influence2) > 0 and round(min(influence2), 3) == .013
           and round(max(influence2), 3) == .048, influence2)
    merged1 = np.mean([delta1.loc[[1, 2]].mean(), *delta1.drop([1, 2]).tolist()])
    record(233, "merging shared stream keeps T1 direction", merged1 > 0, merged1)
    perm = read("results/stats/permutation.csv")
    expected_p = {("T1", "lstm"): .005, ("T1", "gbm_time_wq"): .87, ("T1", "ridge_time_wq"): .95,
                  ("T2", "gbm_pw_wq"): .26, ("T2", "cnn_lstm"): .96}
    for (task, model), value in expected_p.items():
        r = perm[(perm.task == task) & (perm.model == model)].iloc[0]
        decimals = 3 if value == .005 else 2
        # The prose uses conventional half-up rounding, so use its displayed precision.
        from decimal import Decimal, ROUND_HALF_UP
        rounded = float(Decimal(str(r.p)).quantize(Decimal(10) ** -decimals, rounding=ROUND_HALF_UP))
        record(235 if task == "T1" and model == "lstm" else 238, f"{task} {model} permutation p", rounded == value, r.p)
    r = perm[(perm.task == "T1") & (perm.model == "lstm")].iloc[0]
    record(236, "T1 observed seed-0 S below all 199 permutations", r.R == 199 and r.n_perm_le_obs == 0, [r.R, r.n_perm_le_obs])
    record(237, "T1 LSTM below MLP in only 3 ponds; MLP S .306", int((d("T1", "lstm", "mlp_static") < 0).sum()) == 3
           and round(s("T1", "mlp_static"), 3) == .306, [int((d("T1", "lstm", "mlp_static") < 0).sum()), s("T1", "mlp_static")])
    for line, task, baseline, model, h, before, after in [
        (243, "T1", "gbm_time", "gbm_time_wq", "0", .266, .436),
        (243, "T1", "ridge_time", "ridge_time_wq", "0", .269, 1.582),
        (244, "T2", "gbm_pw", "gbm_pw_wq", "1", .211, .286)]:
        values = [s(task, baseline, h), s(task, model, h)]
        record(line, f"{task} adding WQ {baseline}/{model}", [round(v, 3) for v in values] == [before, after], values)
    record(244, "T2 static MLP lower than every sequence model at h1", round(s("T2", "mlp_static", "1"), 3) == .257 and
           all(s("T2", "mlp_static", "1") < s("T2", m, "1") for m in cfg["models"]["T2"]["deep"]), s("T2", "mlp_static", "1"))
    record(248, "allometric S .115, below all fish-measurement-free T1 models", round(s("T1", "allometric"), 3) == .115 and
           all(s("T1", "allometric") < s("T1", m) for m in set(independent_scores[(independent_scores.task == "T1") &
               (independent_scores.scheme == "lopo") & (independent_scores.sample_set == "eligible")].model)
               if m not in ["allometric", "ridge_time_wq_len", "gbm_time_wq_len"]), s("T1", "allometric"))
    record(250, "largest eligible scored times T1=112 T2=126", max(tables["T1"].loc[tables["T1"].eligible_eval, "t"]) == 112 and
           max((tables["T2"].t_k+tables["T2"].dt_target)[tables["T2"].eligible_eval]) == 126, [112, 126])
    record(240, "stopped T2 ridge has no main permutation result", not (perm.model == "ridge_pw_wq").any() and
           not (ROOT / "results/predictions/T2/lopo/ridge_pw_wq.csv").exists(), "main prediction and permutation absent")
    record(245, "three stops and pinned-DO diagnosis", len(details["stops"]) == 3,
           "stopped_runs.csv/rows 2-4; verification/diag_ridge_stop/step5_out.txt:2; step7_out.txt:1")
    record(224, "non-significance is not no effect; prohibited claims explicitly excluded", True,
           "DECISIONS.md:224; DECISIONS.md:250-252; frozen ANALYSIS_PLAN.md:296-311")
    record(254, "title/abstract changes and withdrawals are stated consequences", True,
           "DECISIONS.md:254-257 records intent and Gate 2; manuscript implementation is outside this review")
    save("phase5_claims.csv", rows)
    check(11, "Phase 5 claim ledger", all(r["status"] != "FAIL" for r in rows),
          "verification/codex_leakage/phase5_claims.csv/rows 2+", f"{len(rows)} claims; {sum(r['status'] == 'FAIL' for r in rows)} discrepancies")


def main() -> None:
    global cfg, tables, daily, runtime, prediction_files
    before = manifest()
    freeze_check()
    cfg = yaml.safe_load((ROOT / "configs/phase4.yaml").read_text())
    tables = {t: read(f"data/processed/{t.lower()}_table.csv") for t in ["T1", "T2"]}
    daily = read("data/processed/daily_wq.csv")
    for c in ["date", "right_anchor_date", "left_anchor_date"]:
        daily[c] = pd.to_datetime(daily[c])
    runtime = read("results/logs/runtime.csv")
    prediction_files = [(p, read(p)) for p in sorted((ROOT / "results/predictions").rglob("*.csv"))]
    fold_checks()
    print("freeze, outer/inner folds and calendar membership checked", flush=True)
    feature_checks()
    final_epoch_checks()
    input_checks()
    print("features, sequences, scalers, final epochs and feature allowlists checked", flush=True)
    # Further independent score/statistics checks are defined below.
    if "score_checks" in globals():
        score_checks()
        selection_checks()
        permutation_checks()
        stop_checks()
        claim_checks()
    after = manifest()
    changed = [p for p in set(before) | set(after) if before.get(p) != after.get(p)]
    check(1, "analysis stable throughout review", not changed,
          "verification/codex_leakage/input_manifest.csv/rows 2+", repr(changed))
    save("input_manifest.csv", [dict(file=p, sha256=before[p], unchanged=p not in changed) for p in sorted(before)])
    details["head_end"] = git("rev-parse", "HEAD")
    save("checks.csv", checks)
    save("numeric_comparisons.csv", comparisons)
    (OUT / "audit_details.json").write_text(json.dumps(details, indent=2, default=str, allow_nan=True)+"\n")
    for c in checks:
        print(f"{c['check']:02d} {c['status']} {c['name']}: {c['detail']}")


if __name__ == "__main__":
    main()
