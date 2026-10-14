Created eight 600 dpi PNG/PDF pairs, with matching new scripts under `reporting/figs/`. Figure 2 remains incomplete because its requested pond differences are not saved.

- [fig_pond_scores](/home/dev/development/publications/figures/main/fig_pond_scores.png): per-pond scores, model means and primary labels; sources `results/metrics/{per_pond,scores,completeness}.csv`.
- [fig_confirmatory](/home/dev/development/publications/figures/main/fig_confirmatory.png): saved mean differences, bootstrap intervals and exact p-values, with the blocked T2 test labelled; sources `results/stats/{pairwise,confirmatory}.csv`.
- [fig_permutation](/home/dev/development/publications/figures/main/fig_permutation.png): permutation histograms, observed scores and saved p-values; sources `results/permutation/{T1,T2}/*.csv` and `results/stats/permutation.csv`.
- [fig_trajectories](/home/dev/development/publications/figures/main/fig_trajectories.png): T1 LOPO predictions and real weighings, retaining later observations; sources `results/predictions/T1/lopo/*.csv` and `data/processed/events.csv`.
- [fig_t2_scatter](/home/dev/development/publications/figures/main/fig_t2_scatter.png): predicted versus observed target weights at both horizons; sources `results/predictions/T2/lopo/*.csv`.
- [fig_calendar](/home/dev/development/publications/figures/supp/fig_calendar.png): descriptive T1 calendar predictions with the cut-off marked; sources `results/predictions/T1/chrono/*.csv` and `data/processed/events.csv`.
- [fig_learning_curves](/home/dev/development/publications/figures/supp/fig_learning_curves.png): chosen-trial validation and final-fit training curves for every neural model; sources `results/tuning/{T1,T2}/lopo/*/{curves.csv.gz,chosen.csv}`.
- [fig_ablation](/home/dev/development/publications/figures/supp/fig_ablation.png): descriptive ridge and GBM feature-set scores; sources `results/metrics/{per_pond,scores,completeness}.csv`.

Each linked PNG has a PDF alongside it. [CAPTIONS.md](/home/dev/development/publications/figures/CAPTIONS.md) contains eight appended captions and exact input lists, including selection files. [run_part2.py](/home/dev/development/publications/reporting/figs/run_part2.py) rebuilds all figures; [verify_part2.py](/home/dev/development/publications/reporting/figs/verify_part2.py) passed the saved-data and export checks. Every PNG was opened and visually checked. Part 1 scripts are unchanged. No commit was made.

Could not draw Figure 2's pond dots: `pairwise.csv` holds aggregate differences only, and the task prohibits calculating unsaved statistics. T2 Ridge (past weights + WQ) also has no LOPO scores or permutation distribution after its plan 9.0 stop; those absences are marked. No post hoc results were drawn.

This report is saved as [DONE_part2.md](/home/dev/development/publications/reporting/figs/DONE_part2.md).

## Revision 1

Applied the six items in the lead's review. The earlier Figure 2 source limitation is
resolved by the review's instruction to subtract the saved per-pond scores. All seven
changed PNGs were opened and visually checked, including the final versions after fixing
overlapping outlier labels. Each changed figure also has its regenerated PDF.

1. `fig_confirmatory` now shows seven pond-coloured dots in each panel, with small vertical
   offsets. Each dot subtracts the primary baseline's S from the primary deep model's
   seed-averaged S in `results/metrics/per_pond.csv`, restricted to LOPO, eligible samples
   and the task-horizon. Panels are shorter, p-values have three decimals, the omission text
   is removed, and the legend reads "Mean difference over ponds, pond-bootstrap 95 %
   interval". The x-range includes every pond difference. The new verifier check independently
   enumerates the signed-rank distribution and reproduces the saved counts and exact p-values.
2. `fig_trajectories` shares a 1 to 1000 g range across its seven panels; `fig_calendar`
   shares a 10 to 1000 g range. Both include every displayed real weighing, end on
   2021-10-31 and label ticks as plain numbers. Predictions outside the range have open
   triangles pointing outward at the edge, with labelled gram values and connector lines.
   There are seven such predictions in LOPO and two in the calendar hold-out. Captions
   explain that later, unscored weighings are not shown.
3. `fig_pond_scores` and `fig_ablation` retain their existing scales. Their T1 panels now
   mark the transition at 0.5 with a light vertical line and an axis note explaining the
   linear and logarithmic portions; captions match.
4. `fig_permutation` displays p-values to three decimals, including 0.870 and 0.260.
   Panel (d) reads "No permutation test: run stopped under plan 9.0" across two lines.
5. `fig_t2_scatter` has separate model-marker and pond-colour legend blocks.
6. Every Python invocation used the requested repository-root command with
   `PYTHONDONTWRITEBYTECODE=1` and `MPLCONFIGDIR=/tmp/mplconfig_figs`. The effective
   Matplotlib configuration directory was checked. Part 2 rebuild instructions now include
   these environment settings.

Changed scripts and notes:

- `reporting/figs/_part2.py`
- `reporting/figs/_part2_trajectories.py`
- `reporting/figs/fig_confirmatory.py`
- `reporting/figs/fig_trajectories.py`
- `reporting/figs/fig_calendar.py`
- `reporting/figs/fig_pond_scores.py`
- `reporting/figs/fig_ablation.py`
- `reporting/figs/fig_permutation.py`
- `reporting/figs/fig_t2_scatter.py`
- `reporting/figs/verify_part2.py`
- `reporting/figs/run_part2.py`
- `figures/CAPTIONS.md`
- `reporting/figs/DONE_part2.md`, this appended section only.

Changed exports, PNG and PDF for each stem:

- `figures/main/fig_confirmatory`: pond differences, saved bootstrap interval and test status.
- `figures/main/fig_trajectories`: shared weighing scale with labelled ridge outliers.
- `figures/supp/fig_calendar`: shared weighing scale and descriptive calendar predictions.
- `figures/main/fig_pond_scores`: marked scale transition.
- `figures/supp/fig_ablation`: marked scale transition.
- `figures/main/fig_permutation`: three-decimal p-values and stopped-test wording.
- `figures/main/fig_t2_scatter`: separate model and pond legends.

The confirmatory verification check failed before the dot reader was implemented, with
`ImportError: cannot import name 'pond_differences'`, and passed after implementation.
The final command completed with exit code 0:

```text
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs .venv/bin/python -I reporting/figs/verify_part2.py
```

Its output:

```text
PASS T1: confirmatory pond dots reproduce n_lower = 3 of 7, exact one-sided p = 0.65625
PASS T2, 14 days ahead: confirmatory pond dots reproduce n_lower = 2 of 7, exact one-sided p = 0.921875
PASS T1: all required saved model rows, seven ponds each
PASS T2, 14 days ahead: all required saved model rows, seven ponds each
PASS T2, 28 days ahead: all required saved model rows, seven ponds each
PASS T1: shown models share sample IDs and expected seeds; all five neural models have complete selected curves
PASS T2: shown models share sample IDs and expected seeds; all five neural models have complete selected curves
PASS fig_pond_scores: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_confirmatory: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_permutation: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_trajectories: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_t2_scatter: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_calendar: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_learning_curves: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_ablation: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
LIMITATION T2 Ridge (past weights + WQ): LOPO and permutation outputs absent after plan 9.0 stop
```

Additional checks run from `/tmp/figs-revision1/check_revision.py` confirmed the rendered
trajectory limits, real weighing coverage, date endpoint, tick formatting and outlier markers.
Every part 2 caption body matches its script; re-appending the captions changes nothing.
SHA-256 comparisons against the starting snapshot confirmed 4205 protected files unchanged,
including `results/`, `src/`, `configs/`, `ANALYSIS_PLAN.md` and both learning-curve exports.
Part 1 scripts were not edited. No packages were added and no commit was made. Temporary
inspection scripts, snapshots and verification output remain under `/tmp/figs-revision1`;
Matplotlib uses `/tmp/mplconfig_figs`.

No revision item remains incomplete. The saved T2 Ridge (past weights + WQ) LOPO scores and
permutation distribution remain unavailable because the model stopped under plan 9.0. The
figures retain the required stopped-model row and panel; no analysis was rerun to fill them.
