# Why the ridge models with water quality stopped (diagnosis, inner splits only)

Read-only reproduction with the repository's own learners (`Context.learner`, `RidgeModel`)
and `predict_checked`, on the inner splits of the T1 and T2 ridge models with water quality.
No outer-test row was used. The scripts were written for `/tmp/diag_w28/` and run from the
repository root with `.venv/bin/python -I`; their outputs are the `step*_out.txt` files and
the two `*_zmax_by_column.csv` tables.

T1 (steps 1 to 5): `t1w28_do_min_w` (28-day minimum of dissolved oxygen) takes only the values
0.007 (4 rows) and 0.008 (16 rows) in the inner-training ponds 1, 2, 6, 7 of outer folds 2 and 3
(SD 4.0e-4); pond 9 on 2021-09-25 has 1.201, a z-score of 2983. Its term contributes 1744 of
the row's log prediction of 1750 at alpha 1e-3. From alpha 1e-3 to 1 that prediction is 815 to
1750, above the largest value whose exp is finite in double precision (about 709.8), so the
weight in grams overflows and `predict_checked` stops the model under plan 9.0; from alpha 3.16
up it stays finite (543 and lower). This is not floating-point noise: the SD is the real spread
of near-constant readings. The other T1 ridge water-quality models stay finite, with inner
log predictions at small alphas up to 200 (`ridge_time_wq_noind`), 166 (`_len`), 163
(`ridge_time_wq`) and 84 (`_w07`).

T2 (steps 6 and 7): the same row stops `ridge_pw_wq`. `do_min_w` is 0.007 or 0.008 in the
inner-training ponds 1, 2, 6, 7 of outer folds 2 and 3, pond 9 on 2021-09-25 (h1) has 1.201
(z = 2854), and its term contributes 1061.6 of the log-ratio prediction of 1063.6 at alpha
1e-3, the only alpha that overflows. Step 6 predicts the same stop for `ridge_pw_wq_noind`
(974 at alpha 1e-3) and none for `ridge_pw_wq_w14` and `_w42` (largest finite 244 and 615, on
pond 2 at 2021-10-09, where `do_max_w` is 4.181 against 19.989 to 19.998 in the training ponds
3, 4, 7, z = 8199).

Why the Codex rebuild (`verification/codex_p4models/`) stayed finite: its spec scores tuning
candidates on the log scale (`verify.py`, `choose_alpha` and `score`), where these rows are
finite, very large errors, so the alpha loses its trial; the exp overflow showed only as a
warning (`DONE.md`). The repository code converts every candidate's prediction to grams and
applies plan 9.0 to it (`src/tuning.py`, `predict_checked`).
