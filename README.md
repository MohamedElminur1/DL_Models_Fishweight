# Revision of aquacj-4561230: fish weight from IoT water-quality data

Code, data tables, results and decision trail for the major revision of "Optimizing
Aquaponics through Deep Learning: A Multi-Model Approach for Accurate Fish Weight Prediction
and Forecasting" (Aquaculture Journal, MDPI). The revision re-analyses the public dataset of
Udanor et al. (2022) from scratch: the original analysis is reproduced and diagnosed
(`DECISIONS.md`, Gate 2), and a new design, frozen before any test score was seen
(`ANALYSIS_PLAN.md`, git tag `analysis-plan-frozen`), decides what the paper claims.

## What is where

| Path | Content |
|---|---|
| `ANALYSIS_PLAN.md` | The pre-registered analysis; frozen by the annotated tag `analysis-plan-frozen` together with `src/`, `configs/`, `data/processed/` and `uv.lock`. |
| `DECISIONS.md` | Gate verdicts: data audit (Gate 1), original analysis (Gate 2), later gates. |
| `LOG.md`, `decisions.tsv` | Narrative log and one row per decision, with evidence pointers. |
| `src/` | Data audit, Phase 2 diagnostics, data pipeline, models, validation, statistics. |
| `configs/` | Every setting of the pipeline and the models (seeds, budgets, search spaces). |
| `data/raw/` | Symbolic links to the public pond files and their SHA-256 sums. The files themselves are not redistributed. |
| `data/processed/` | Weighing events, daily water quality and the T1 and T2 tables. |
| `results/` | Written only by scripts: audit, QC counts, Phase 2 diagnostics, predictions, tuning records, metrics, statistics, logs. |
| `reporting/`, `figures/` | Figure scripts and the figures (600 dpi PNG, PDF, SVG for diagrams); captions with their source files in `figures/CAPTIONS.md`. |
| `verification/` | Blind re-implementations by a second agent (Codex) from written specifications, and their comparison with the pipeline. |
| `tests/` | Unit tests of the pipeline, models, folds, guard and statistics. |
| `manuscript/` | Revised manuscript (clean and changes marked) and the response to the reviewers. |
| `tasks/` | The task specifications given to each role during the revision. |

## Environment

CPU only. Python 3.12 managed by [uv](https://docs.astral.sh/uv/); the exact versions are in
`uv.lock`.

```bash
uv sync --frozen --group dev
```

The runs reported in the paper used 4 CPU cores (AMD EPYC 7B12); runtimes and hardware are
logged per training in `results/logs/runtime.csv`.

## Data

Download the "Sensor Based Aquaponics Fish Pond Datasets" (Udanor et al. 2022,
doi:10.34740/KAGGLE/DSV/3748790) and put the 11 pond files in `data/IoTPond_Dataset/` under
their original names (`IoTpond1.csv`, `IoTPond2.csv`, ...). `./run_all.sh check` verifies
them against `data/raw/SHA256SUMS`. The raw files are read-only inputs: cleaning happens in
code and is logged.

## Running

```bash
./run_all.sh                 # every stage in order
./run_all.sh data tests      # selected stages
```

| Stage | What it does |
|---|---|
| `check` | Verifies the raw files. |
| `audit` | Phase 1 audit of the raw files (`results/audit/`, `figures/audit/`). |
| `phase2` | Reproduction and diagnosis of the original pond-2 analysis (`results/phase2/`, about 2 h). |
| `data` | Weighing events, sensor QC, daily aggregates, T1 and T2 tables. |
| `tests` | Unit tests. |
| `phase4` | All models of the plan under leave-one-pond-out and the calendar hold-out. |
| `select` | Chooses the primary baseline and deep model from inner-validation scores only. Stops if the choice differs from the committed one. |
| `permute` | Within-date permutation tests. |
| `evaluate` | Metrics and statistics from the saved predictions (`results/metrics/`, `results/stats/`). |
| `figures` | All figures. |

Stages can be re-run: finished model runs are skipped, unfinished ones resume. Outer test
predictions are refused unless the working tree matches the frozen tag
(`src/guard.py`), so changing the analysis after the freeze is visible.

## Research integrity

Only real weighings from the public dataset support claims about fish growth: models are
scored only on real measurement dates, fish weight and length are never imputed between
them, and no synthetic data enter any table or figure. Every number in the manuscript and the
response letter traces to a file in `results/`. The original analysis is described as it was
done (`DECISIONS.md`, Gate 2). AI assistance was used for code, analysis and text editing;
the authors verified all code, results and text.

## Citation

Udanor, C.N. et al. An Internet of Things labelled dataset for aquaponics fish pond water
quality monitoring system. Data Brief 2022, 43, 108400.
https://doi.org/10.1016/j.dib.2022.108400
