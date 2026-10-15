# Fish weight from aquaponics pond sensor data: a leakage-aware re-analysis

Code, processed data, results and figures of the revised manuscript "Deep Learning Versus
Simple Baselines for Fish Weight Prediction from Aquaponics Fish Pond Sensor Data: A
Leakage-Aware Evaluation" (Aquaculture Journal, MDPI, under review). The analysis uses the
public dataset of Udanor et al. (2022). It reproduces and diagnoses the analysis of the
submitted version (Section 3.7 of the paper), then evaluates a new design that was frozen
before any test score was seen (`ANALYSIS_PLAN.md`, annotated git tag `analysis-plan-frozen`).

This repository is a filtered copy of the history of the authors' working repository. It holds
the analysis material only; manuscript drafts and correspondence with the journal are not
included. Commits that touched only those files are absent, and the remaining commits keep
their dates and messages. `decisions.tsv` records the freeze for `src/guard.py`, and
`DECISIONS.md` holds the analysis decisions D-001 to D-003 that the code refers to.

## What is where

| Path | Content |
|---|---|
| `ANALYSIS_PLAN.md` | The pre-registered analysis; frozen by the annotated tag `analysis-plan-frozen` together with `src/`, `configs/`, `data/processed/` and `uv.lock`. |
| `decisions.tsv` | The freeze record that the guard checks. |
| `DECISIONS.md` | The analysis decisions D-001 to D-003 (pond 2, the protocol of the diagnostics, the post hoc ridge variant), copied from the working repository. |
| `src/` | Data audit, diagnostics of the original analysis, data pipeline, models, validation, statistics. |
| `configs/` | Every setting of the pipeline and the models (seeds, budgets, search spaces). |
| `data/raw/` | Symbolic links to the public pond files and their SHA-256 sums. The files themselves are not redistributed. |
| `data/processed/` | Weighing events, daily water quality and the T1 and T2 tables. |
| `results/` | Written only by scripts: audit, QC counts, diagnostics of the original analysis, predictions, tuning records, metrics, statistics, logs. |
| `posthoc/` | The post hoc ridge variant for the three ridge runs stopped under plan section 9.0; outside the frozen analysis and reported in the Supplementary Materials only. |
| `reporting/`, `figures/` | Scripts that turn `results/` into the figures, the table CSVs and the diagnosis of the stopped runs; the figures (600 dpi PNG, PDF, SVG for diagrams) with captions and source files in `figures/CAPTIONS.md`. |
| `verification/` | Independent re-implementations of the data tables, folds, baselines, scores and leakage checks from written specifications, and their comparison with the pipeline. |
| `tests/` | Unit tests of the pipeline, models, folds, guard and statistics. |
| `original_code/` | The only code of the submitted study that was available: a notebook that loads two saved models (Section 3.7 of the paper). |

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
doi:10.34740/KAGGLE/DSV/3748790, CC BY 4.0) and put the 11 pond files in
`data/IoTPond_Dataset/` under their original names (`IoTpond1.csv`, `IoTPond2.csv`, ...).
`./run_all.sh check` verifies them against `data/raw/SHA256SUMS`. The raw files are read-only
inputs: cleaning happens in code and is logged.

## Running

```bash
./run_all.sh                 # every stage in order
./run_all.sh data tests      # selected stages
```

| Stage | What it does |
|---|---|
| `check` | Verifies the raw files. |
| `audit` | Audit of the raw files (`results/audit/`, `figures/audit/`). |
| `phase2` | Reproduction and diagnosis of the original pond-2 analysis (`results/phase2/`, about 2 h). |
| `data` | Weighing events, sensor QC, daily aggregates, T1 and T2 tables. |
| `tests` | Unit tests. |
| `phase4` | All models of the plan under leave-one-pond-out and the calendar hold-out. |
| `select` | Chooses the primary baseline and deep model from inner-validation scores only. Stops if the choice differs from the committed one. |
| `permute` | Within-date permutation tests. |
| `evaluate` | Metrics and statistics from the saved predictions (`results/metrics/`, `results/stats/`). |
| `posthoc` | The post hoc ridge variant and its scores (`results/posthoc/`). |
| `diagnostics` | Where the three stopped ridge runs overflowed, from inner splits only (`results/diagnostics/`). |
| `figures` | All figures. |
| `tables` | The table CSVs of the paper from `results/` (written to `manuscript/src/tables/`), then the independent checks. |

Stages can be re-run: finished model runs are skipped, unfinished ones resume. Outer test
predictions are refused unless the working tree matches the frozen tag (`src/guard.py`), so
changing the analysis after the freeze is visible. Files in `results/` that record the frozen
commit give its id in the working repository, 91b38b99494313ed4b1e6eb570bcdf0e41ab0639; here
the tag points to a8706c331365cf5c9661d41e840738cbab3afac1, whose plan, `src/`, `configs/`,
`data/processed/` and `uv.lock` are identical.

## Research integrity

Only real weighings from the public dataset support claims about fish growth: models are
scored only on real measurement dates, fish weight and length are never imputed between
them, and no synthetic data enter any table or figure. Every number in the paper traces to a
file in `results/`. Generative AI tools assisted with writing and checking the code, running
the analysis and editing the text; the authors reviewed all code, results and text.

## Citation

Udanor, C.N. et al. An Internet of Things labelled dataset for aquaponics fish pond water
quality monitoring system. Data Brief 2022, 43, 108400.
https://doi.org/10.1016/j.dib.2022.108400
