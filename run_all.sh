#!/usr/bin/env bash
# Rebuilds the results of the aquacj-4561230 revision from the public raw files.
#
#   ./run_all.sh                 every stage, in order
#   ./run_all.sh phase4 figures  only the named stages
#   WORKERS=2 ./run_all.sh phase4
#
# Stages: check, audit, phase2, data, tests, phase4, select, permute, evaluate, figures.
# Each stage can be re-run: model runs that finished with the same config and inputs are
# skipped, unfinished folds and permutations resume, and the data stages rewrite their files
# with the same bytes. Outer test predictions need the frozen analysis (git tag
# analysis-plan-frozen, see ANALYSIS_PLAN.md section 12); `select` stops when the selection
# files differ from the last commit, because src/evaluate.py scores only committed selections.
set -euo pipefail
cd "$(dirname "$0")"

PY=(.venv/bin/python -I)
WORKERS="${WORKERS:-4}"

models() {  # every model of a task in the given config groups
  "${PY[@]}" -c '
import sys, yaml
cfg = yaml.safe_load(open("configs/phase4.yaml"))
task, groups = sys.argv[1], sys.argv[2:]
print(" ".join(m for g in groups for m in cfg["models"][task][g]))' "$@"
}

primary_deep() {
  "${PY[@]}" -c 'import json, sys; print(json.load(open(f"results/selection/primary_{sys.argv[1]}.json"))["primary_deep"])' "$1"
}

stage_check() {  # the 11 pond files (data/IoTPond_Dataset/) must be the published ones
  grep 'data/IoTPond_Dataset/' data/raw/SHA256SUMS | sha256sum -c -
}

stage_audit() { "${PY[@]}" src/audit.py; }

stage_phase2() {  # about 2 h on 4 CPU cores
  "${PY[@]}" src/phase2_original.py --config configs/phase2.yaml
  "${PY[@]}" src/phase2_posthoc.py
}

stage_data() {
  "${PY[@]}" src/events.py
  "${PY[@]}" src/qc.py
  "${PY[@]}" src/features.py
}

stage_tests() { "${PY[@]}" -m pytest -q tests/; }

stage_phase4() {
  local task model
  for task in T1 T2; do
    for model in $(models "$task" time_only water_quality reference deep static_only sensitivity); do
      "${PY[@]}" src/train.py --task "$task" --scheme lopo --model "$model" --workers "$WORKERS"
    done
  done
  for model in $(models T1 all_sample_set); do
    "${PY[@]}" src/train.py --task T1 --scheme lopo --model "$model" --sample-set all --workers "$WORKERS"
  done
  for task in T1 T2; do  # calendar hold-out, descriptive (one fold per model)
    for model in $(models "$task" time_only water_quality reference deep static_only); do
      "${PY[@]}" src/train.py --task "$task" --scheme chrono --model "$model"
    done
  done
}

stage_select() {
  "${PY[@]}" src/select_primary.py
  if ! git diff --quiet HEAD -- results/selection || [ -n "$(git ls-files --others --exclude-standard results/selection)" ]; then
    echo "results/selection/ differs from the last commit: review it, commit it, then run the remaining stages." >&2
    exit 3
  fi
}

perm_models() {  # the tabular models of a task's permutation test (configs/phase4.yaml)
  "${PY[@]}" -c '
import sys, yaml
print(" ".join(yaml.safe_load(open("configs/phase4.yaml"))["permutation"]["tabular_models"][sys.argv[1]]))' "$1"
}

stage_permute() {  # tabular models, then the primary deep model of each task
  local task model
  for task in T1 T2; do
    for model in $(perm_models "$task") "$(primary_deep "$task")"; do
      "${PY[@]}" src/permute.py --task "$task" --model "$model" --workers "$WORKERS"
    done
  done
}

stage_evaluate() { "${PY[@]}" src/evaluate.py; }

stage_figures() {
  local script
  for script in reporting/figs/fig_*.py; do
    "${PY[@]}" "$script"
  done
}

ALL=(check audit phase2 data tests phase4 select permute evaluate figures)
if [ $# -eq 0 ]; then set -- "${ALL[@]}"; fi
for stage in "$@"; do
  declare -F "stage_$stage" >/dev/null || { echo "unknown stage: $stage (stages: ${ALL[*]})" >&2; exit 2; }
done
for stage in "$@"; do
  echo "== $stage"
  "stage_$stage"
done
