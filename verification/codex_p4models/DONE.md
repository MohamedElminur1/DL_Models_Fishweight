# Verification complete

Files:

- verify.py
- test_verify.py
- audit_outputs.py
- pytest.ini
- codex_folds.csv
- codex_inner.csv
- codex_predictions.csv
- codex_curves.csv
- codex_scores.csv
- DONE.md

T1: 7 outer folds, 35 test rows per model.
T2: 7 outer folds, 53 test rows per model.

Linear outer-test predictions at or below 0 g: 0.

Linear inner-validation predictions at or below 0 g: 0.

Fallbacks (count means predicted rows; outer is test, inner0–2 are validation):

| Task | Model | Held-out pond | Stage | Fallback | Count |
| --- | --- | --- | --- | --- | --- |
| T1 | cohort_mean | 1 | inner0 | cohort_mean -> gc | 5 |
| T1 | cohort_mean | 1 | inner1 | cohort_mean -> gc | 4 |
| T1 | cohort_mean | 1 | inner2 | cohort_mean -> gc | 1 |
| T1 | cohort_mean | 2 | inner0 | cohort_mean -> gc | 5 |
| T1 | cohort_mean | 2 | inner1 | cohort_mean -> gc | 4 |
| T1 | cohort_mean | 2 | inner2 | cohort_mean -> gc | 1 |
| T1 | cohort_mean | 2 | outer | cohort_mean -> gc | 1 |
| T1 | cohort_mean | 3 | inner0 | cohort_mean -> gc | 9 |
| T1 | cohort_mean | 3 | inner1 | cohort_mean -> gc | 1 |
| T1 | cohort_mean | 4 | inner0 | cohort_mean -> gc | 9 |
| T1 | cohort_mean | 4 | inner1 | cohort_mean -> gc | 1 |
| T1 | cohort_mean | 6 | inner0 | cohort_mean -> gc | 9 |
| T1 | cohort_mean | 6 | inner1 | cohort_mean -> gc | 2 |
| T1 | cohort_mean | 7 | inner0 | cohort_mean -> gc | 7 |
| T1 | cohort_mean | 7 | inner1 | cohort_mean -> gc | 2 |
| T1 | cohort_mean | 9 | inner0 | cohort_mean -> gc | 3 |
| T1 | cohort_mean | 9 | inner1 | cohort_mean -> gc | 1 |
| T1 | cohort_mean | 9 | outer | cohort_mean -> gc | 1 |
| T2 | cohort_growth | 1 | inner0 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 1 | inner1 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 1 | inner2 | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 1 | outer | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 2 | inner0 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 2 | inner1 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 2 | inner2 | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 2 | outer | cohort_growth -> anchored_gc | 4 |
| T2 | cohort_growth | 3 | inner0 | cohort_growth -> anchored_gc | 10 |
| T2 | cohort_growth | 3 | inner1 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 3 | inner2 | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 4 | inner0 | cohort_growth -> anchored_gc | 10 |
| T2 | cohort_growth | 4 | inner1 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 4 | inner2 | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 6 | inner0 | cohort_growth -> anchored_gc | 6 |
| T2 | cohort_growth | 6 | inner1 | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 6 | outer | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 7 | inner0 | cohort_growth -> anchored_gc | 6 |
| T2 | cohort_growth | 7 | inner1 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 7 | outer | cohort_growth -> anchored_gc | 1 |
| T2 | cohort_growth | 9 | inner0 | cohort_growth -> anchored_gc | 6 |
| T2 | cohort_growth | 9 | inner1 | cohort_growth -> anchored_gc | 5 |
| T2 | cohort_growth | 9 | outer | cohort_growth -> anchored_gc | 1 |

Assumptions and conventions:

- dropped_ponds lists the shared-stream exclusion only; the held-out pond is already named. Pond 8 has no eligible rows and is absent from all fits and tests.
- codex_curves.csv records final outer-training curve fits only, one per gc or anchored_gc model per fold. Parameters follow the selected formula’s table order. A failed curve means all three starts failed.
- Curve and start ties retain the first in the specified order. Ridge ties use exact computed equality and select the larger alpha.
- For deduplicated anchored events, origins precede targets and the first occurrence is retained after checking consistent times and log weights (absolute tolerance 1e-12 for floating-point roundoff).
- If no T1 curve fits and cohort_mean without fallback lacks three ponds for a requested day, execution raises rather than inventing an unspecified prediction. This case did not occur.
- T1 has blank origin_date, S_h1 and S_h2; non-ridge alpha and absent pond scores are blank. Nonpositive linear weights remain in w_pred, have blank y_pred_log, and score as infinity.
- The logistic formula uses logaddexp for numerical stability. The growth cache reuses only fits with identical ordered training times and log weights.

Verification:

Run /home/dev/development/publications/.venv/bin/python -I -m pytest -c pytest.ini --rootdir=. --confcutdir=. test_verify.py -q.
Run /home/dev/development/publications/.venv/bin/python -I verify.py to regenerate every CSV and this report.
CSV floats use 17 significant digits.

Executed verification: 13 hand-checkable pytest tests passed. The independent saved-output audit passed for 14 folds, 252 inner scores, 546 predictions, 14 curve records, and 19 aggregate scores. It checked eligible-row coverage, fold membership, inner validation sizes, log/weight consistency, alpha grid membership, horizon averaging, and recomputed every overall and per-pond score from the saved predictions.

Run /home/dev/development/publications/.venv/bin/python -I audit_outputs.py to repeat that audit.

All saved predictions and scores are finite, and no outer growth curve failed. Exponentiation overflow warnings occurred during candidate-alpha inner evaluation; the log-scale tuning scores and the selected models’ saved predictions remained finite.
