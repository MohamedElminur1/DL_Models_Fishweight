# Pond score figures completed

Completed 2026-10-15. No commit was made. Edits stayed within the phase 7c-F
ownership list; manuscript, results, task files and other figure scripts were left alone.

## Changes

1. `fig_pond_scores` now contains T1 only, without a panel letter or title. It retains
   every configured model, group heading, primary label, pond marker, black Model S
   diamond, the symmetric-logarithmic scale and the line and note at 0.5.
   See `reporting/figs/fig_pond_scores.py:116`.
2. `fig_pond_scores_t2` places 14 days ahead and 28 days ahead side by side, labeled
   `(a)` and `(b)`. The panels share vertical positions; only the left panel has model
   labels and group headings. Both keep their original independent linear ranges and
   the stopped Ridge note. One pond and Model S legend sits below both panels.
   See `reporting/figs/fig_pond_scores.py:124`.
3. Both figures come from the original script and saved metrics, with primary labels
   still read from the saved selections. `draw_rows` has a `show_labels=True` default
   that preserves its existing output for the ablation caller.
   See `reporting/figs/fig_pond_scores.py:42` and `:108`.
4. The four requested main PNG/PDF files were written. The ablation PNG/PDF were
   re-rendered in place; neither has a Git diff. No file was deleted.
5. The runner registers nine figures and calls the shared score script once, collecting
   both caption records. The verifier resolves both outputs to that script and checks
   their dimensions, PDF labels and notes, and the ablation pixels against `HEAD`.
   See `reporting/figs/run_part2.py:19`, `:67` and
   `reporting/figs/verify_part2.py:75`. Caption entries are split at
   `figures/CAPTIONS.md:111` and `:134`, using "labeled" and "colored".

## Sizes and fonts

| PNG under `figures/main/` | Pixels, width × height | Print size, cm | Smallest visible font |
| --- | --- | --- | --- |
| `fig_pond_scores.png` | 3779 × 3070 | 16 × 13 | 8 pt |
| `fig_pond_scores_t2.png` | 3779 × 2598 | 16 × 11 | 8 pt |

Both PNGs are 600 dpi, stored as 599.9988 dpi by PNG resolution metadata. Integer
pixel dimensions imply 15.997767 × 12.996333 cm and 15.997767 × 10.998200 cm at
600 dpi. The vector PDF pages retain the requested 16 × 13 cm and 16 × 11 cm sizes.
`pdfimages -list` reports no raster images in either PDF.

The minimum font size was measured from every visible, nonempty Matplotlib Text artist
immediately before saving. It is 8 pt in both figures at their full 16 cm print width;
T2 panel headings use the shared 9 pt panel style. Both PNGs were opened and visually
checked for row readability, clipping, legends and the scale-change note.

## Verification

All Python commands ran from `/home/dev/development/publications`, using
`env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs nice -n 19 .venv/bin/python -I`.
No packages were installed and no web access was used.

The existing verifier passed before editing. After adding the new size requirement,
it failed against the original 33 cm score figure at the height assertion. After
rendering the split figures, the updated verifier passed with exit code 0.

The runner was exercised through `run_part2.main()` with `--captions`. Its save hook
allowed writes only for the two score figures and the ablation figure; other figures
were built in memory and closed. All nine generated caption records matched the saved
captions without writing them. The shared score script ran once and produced both outputs.
Live checks confirmed shared T2 vertical axes, labels only on the left, separate linear
ranges, and T1's symmetric-logarithmic scale.

A separate comparison called the committed and updated `draw_rows` on the same saved
data for all three task-horizons. Plotted x/y coordinates, marker styles, labels, group
headings, notes, scales and ranges matched `HEAD` in every case.

The re-rendered `figures/supp/fig_ablation.png` was compared with
`git show HEAD:figures/supp/fig_ablation.png` after decoding both with Pillow.
Dimensions, image mode and every pixel matched. This check also runs in the verifier.
`git diff --check` passed.

Command:

```sh
cd /home/dev/development/publications
env PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/mplconfig_figs nice -n 19 .venv/bin/python -I reporting/figs/verify_part2.py
```

Output:

```text
PASS T1: confirmatory pond dots reproduce n_lower = 3 of 7, exact one-sided p = 0.65625
PASS T2, 14 days ahead: confirmatory pond dots reproduce n_lower = 2 of 7, exact one-sided p = 0.921875
PASS T1: all required saved model rows, seven ponds each
PASS T2, 14 days ahead: all required saved model rows, seven ponds each
PASS T2, 28 days ahead: all required saved model rows, seven ponds each
PASS T1: shown models share sample IDs and expected seeds; all five neural models have complete selected curves
PASS T2: shown models share sample IDs and expected seeds; all five neural models have complete selected curves
PASS fig_pond_scores: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_pond_scores_t2: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_confirmatory: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_permutation: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_trajectories: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_t2_scatter: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_calendar: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_learning_curves: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_ablation: parseable script, 16 cm / 600 dpi PNG, one-page PDF with text, caption
PASS fig_ablation: decoded PNG pixels identical to HEAD
LIMITATION T2 Ridge (past weights + WQ): LOPO and permutation outputs absent after plan 9.0 stop
```

The reported Ridge limitation is the existing saved-data condition. Both T2 panels
retain its row and stopped-run note. Nothing in this task remains unfinished.
