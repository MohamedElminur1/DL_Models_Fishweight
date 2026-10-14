"""Index of the model runs stopped under plan 9.0, built from the logs: results/metrics/stopped_runs.csv.

Run from the repository root after src/evaluate.py: .venv/bin/python -I reporting/stopped_runs.py
Plan 9.0 stops a learned model's run on a non-finite prediction and logs it. The frozen scorer
lists such a run in results/metrics/completeness.csv only as has_predictions=False, as it does a
run that was never scheduled; this file, written next to it, names the stopped runs, counts the logged attempts (all
with the frozen settings) and lists the per-fold prediction files written before the stop, which
src/evaluate.py does not read. Fails if the logs, completeness.csv and the files disagree.
"""
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
STOP = re.compile(r"^(\S+)\tstopped under plan 9\.0 \(non-finite prediction\), reported, not re-run: "
                  r"--task (\S+) --scheme (\S+) --model (\S+)")
ERROR = re.compile(r"^(\S+) (\S+) (\S+): ERROR \1 \2 \3: (.*)$")
COLUMNS = ["task", "scheme", "sample_set", "model", "status", "error", "attempts_logged", "stop_recorded_utc",
           "unscored_fold_prediction_files", "posthoc_rerun"]


def main() -> None:
    stops = [m.groups() for line in (RESULTS / "logs/autopilot.log").read_text().splitlines()
             if (m := STOP.match(line))]
    errors = [m.groups() for line in (RESULTS / "logs/run_all_phase4.log").read_text().splitlines()
              if (m := ERROR.match(line))]
    complete = pd.read_csv(RESULTS / "metrics/completeness.csv")
    rows = []
    for stamp, task, scheme, model in stops:
        messages = {msg for t, s, m, msg in errors if (t, s, m) == (task, scheme, model)}
        attempts = sum((t, s, m) == (task, scheme, model) for t, s, m, _ in errors)
        if len(messages) != 1:
            sys.exit(f"{task} {scheme} {model}: expected one logged error message, found {sorted(messages)}")
        listed = complete[(complete.task == task) & (complete.scheme == scheme)
                          & (complete.sample_set == "eligible") & (complete.model == model)]
        if listed.has_predictions.tolist() != [False]:
            sys.exit(f"{task} {scheme} {model}: completeness.csv does not list it once without predictions")
        if (RESULTS / f"predictions/{task}/{scheme}/{model}.csv").exists():
            sys.exit(f"{task} {scheme} {model}: an aggregate prediction file exists for a stopped run")
        folds = sorted(p.parent.name for p in (RESULTS / f"tuning/{task}/{scheme}/{model}").glob("fold*/predictions.csv"))
        posthoc = RESULTS / f"posthoc/ridge_candidate_loses/predictions/{task}/{scheme}/{model}.csv"
        rows.append({"task": task, "scheme": scheme, "sample_set": "eligible", "model": model,
                     "status": "stopped (plan 9.0)", "error": messages.pop(), "attempts_logged": attempts,
                     "stop_recorded_utc": stamp, "unscored_fold_prediction_files": ";".join(folds),
                     "posthoc_rerun": posthoc.relative_to(ROOT).as_posix() if posthoc.exists() else ""})
    out = RESULTS / "metrics/stopped_runs.csv"
    pd.DataFrame(rows, columns=COLUMNS).to_csv(out, index=False)
    print(f"{len(rows)} stopped runs -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
