# Ridge-stop diagnosis completed

`ridge_stop.py` writes `results/diagnostics/ridge_stop.csv` with 1,365 candidate rows and the 26 requested columns. Each model has 273 rows across outer folds 0, 1, 2, 3, 4, 5 and 7, three inner splits and 13 alphas. No commit was made.

## Execution

Both runs used this command from `/home/dev/development/publications` on 2026-10-15. `/usr/bin/time` measured the complete process, including imports; the script also printed its calculation runtime.

```bash
cd /home/dev/development/publications
nice -n 19 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 .venv/bin/python -I -B reporting/diagnostics/ridge_stop.py
```

Hardware was an AMD EPYC 7B12 VM with four logical CPUs, Linux x86_64, glibc 2.39 and Python 3.12.3. The command limited the numerical libraries to one thread and used nice level 19.

| Run | Calculation seconds | Whole-process seconds | Peak RSS KiB |
|---|---:|---:|---:|
| 1 | 10.602423701435328 | 14.64 | 355648 |
| 2 | 10.809783258009702 | 14.88 | 356076 |

The script calls the same `load_config`, `load_tables`, `plan_folds`, `Context`, ridge learner, `predict_checked` and `ridge_alphas` as `verification/diag_ridge_stop/common.py`. It selects frames only with inner-training and inner-validation labels, checks their membership against outer-training labels, and never accesses outer-test labels or calculates scores. Loading and planning use the repository's required setup; all subsequent fits, predictions and diagnostic statistics use inner frames only. The learner's own `_design`, `mean_`, `scale_` and coefficients supply the contribution diagnosis. Missing raw values are blank, and distinct training values are counted after the learner's imputation.

Only `reporting/diagnostics/ridge_stop.py`, `results/diagnostics/ridge_stop.csv` and this report were written as task deliverables. Other repository files were left alone. No packages were installed and no web access was used. CSV floats use `repr(float_value)`; no scientific result is rounded in the file. Runtime and hardware are recorded here rather than in the deterministic CSV.

A rerun recomputes the full table before atomically replacing the CSV. If calculation fails, the previous CSV remains. Interruption during CSV writing can leave `ridge_stop.csv.tmp`; the next successful run overwrites that temporary file and replaces the CSV.

## Comparison with the one-off outputs

The following keys name exact CSV rows. Line numbers include the header. A reference such as A2/A3 means both separately identified rows have the displayed value. The tuple order is task, model, outer fold, inner split, alpha.

| Key | CSV row tuple | CSV line |
|---|---|---:|
| A2 | `T1, ridge_time_wq_w28, 2, 1, 0.001` | 93 |
| A3 | `T1, ridge_time_wq_w28, 3, 1, 0.001` | 132 |
| B2 | `T2, ridge_pw_wq, 2, 1, 0.001` | 366 |
| B3 | `T2, ridge_pw_wq, 3, 1, 0.001` | 405 |
| C2 | `T2, ridge_pw_wq_noind, 2, 1, 0.001` | 639 |
| C3 | `T2, ridge_pw_wq_noind, 3, 1, 0.001` | 678 |
| F2 | `T1, ridge_time_wq_w28, 2, 1, 3.1622776601683795` | 100 |
| F3 | `T1, ridge_time_wq_w28, 3, 1, 3.1622776601683795` | 139 |
| D | `T2, ridge_pw_wq_w14, 7, 0, 0.001` | 1055 |
| E | `T2, ridge_pw_wq_w42, 7, 0, 0.001` | 1328 |
| M | `T2, ridge_pw_wq, 7, 0, 0.001` | 509 |
| N | `T2, ridge_pw_wq_noind, 7, 0, 0.001` | 782 |

Each line below identifies the CSV row and column holding one requested value. Values agree with the one-off outputs at their printed precision, except for the source wording issues explained below.

| Value checked | Row key | CSV column | CSV value | One-off source |
|---|---|---|---|---|
| T1 stopping pond | A2/A3 | `row_pond` | `9` | step2_out.txt; step5_out.txt |
| T1 stopping date | A2/A3 | `row_date` | `2021-09-25` | step2_out.txt; step5_out.txt |
| T1 dissolved-oxygen window minimum | A2/A3 | `top_raw_value` | `1.201` | step2_out.txt; step5_out.txt |
| T1 training ponds | A2/A3 | `train_ponds` | `1;2;6;7` | step2_out.txt; step5_out.txt |
| T1 smallest training value | A2/A3 | `top_train_min` | `0.007` | step2_out.txt; step5_out.txt |
| T1 largest training value | A2/A3 | `top_train_max` | `0.008` | step2_out.txt; step5_out.txt |
| T1 training values and counts | A2/A3 | `top_train_values` | `0.007x4;0.008x16` | step2_out.txt; step5_out.txt |
| T1 z-score | A2/A3 | `top_z` | `2983.0` | step2_out.txt; step5_out.txt |
| T1 log prediction | A2/A3 | `row_log_pred` | `1750.353568923219` | step2_out.txt; step5_out.txt |
| T1 dominant design column | A2/A3 | `top_column` | `t1w28_do_min_w` | step2_out.txt; step5_out.txt |
| T1 dominant contribution | A2/A3 | `top_contribution` | `1743.9058327120622` | step2_out.txt; step5_out.txt |
| T2 ridge stopping pond | B2/B3 | `row_pond` | `9` | step6_out.txt; step7_out.txt |
| T2 ridge stopping date | B2/B3 | `row_date` | `2021-09-25` | step6_out.txt; step7_out.txt |
| T2 ridge stopping horizon | B2/B3 | `row_horizon` | `1` | step6_out.txt; step7_out.txt |
| T2 ridge dissolved-oxygen window minimum | B2/B3 | `top_raw_value` | `1.201` | step6_out.txt; step7_out.txt |
| T2 ridge training ponds | B2/B3 | `train_ponds` | `1;2;6;7` | step6_out.txt; step7_out.txt |
| T2 ridge smallest training value | B2/B3 | `top_train_min` | `0.007` | step6_out.txt; step7_out.txt |
| T2 ridge largest training value | B2/B3 | `top_train_max` | `0.008` | step6_out.txt; step7_out.txt |
| T2 ridge training values and counts | B2/B3 | `top_train_values` | `0.007x7;0.008x24` | step6_out.txt; step7_out.txt; counts newly recorded |
| T2 ridge z-score | B2/B3 | `top_z` | `2853.84045754889` | step6_out.txt; step7_out.txt |
| T2 ridge log prediction | B2/B3 | `row_log_pred` | `1063.6162677565512` | step6_out.txt; step7_out.txt |
| T2 ridge dominant design column | B2/B3 | `top_column` | `do_min_w` | step6_out.txt; step7_out.txt |
| T2 ridge dominant contribution | B2/B3 | `top_contribution` | `1061.5774072158367` | step6_out.txt; step7_out.txt |
| T2 noind stopping pond | C2/C3 | `row_pond` | `9` | step6_out.txt |
| T2 noind stopping date | C2/C3 | `row_date` | `2021-09-25` | step6_out.txt |
| T2 noind stopping horizon | C2/C3 | `row_horizon` | `1` | step6_out.txt |
| T2 noind dissolved-oxygen window minimum | C2/C3 | `top_raw_value` | `1.201` | step6_out.txt |
| T2 noind training ponds | C2/C3 | `train_ponds` | `1;2;6;7` | step6_out.txt |
| T2 noind smallest training value | C2/C3 | `top_train_min` | `0.007` | step6_out.txt |
| T2 noind largest training value | C2/C3 | `top_train_max` | `0.008` | step6_out.txt |
| T2 noind training values and counts | C2/C3 | `top_train_values` | `0.007x7;0.008x24` | step6_out.txt; counts newly recorded |
| T2 noind z-score | C2/C3 | `top_z` | `2853.8404575488903` | step6_out.txt |
| T2 noind log prediction | C2/C3 | `row_log_pred` | `973.6802181763805` | step6_out.txt |
| T2 noind dominant design column | C2/C3 | `top_column` | `do_min_w` | step6_out.txt |
| T2 noind dominant contribution | C2/C3 | `top_contribution` | `972.4368026548393` | new diagnostic detail; not printed in step6_out.txt |
| T1 first finite alpha in the stopping splits | F2/F3 | `alpha` | `3.1622776601683795` | step2_out.txt: 3.16228 |
| T1 log prediction at first finite alpha | F2/F3 | `row_log_pred` | `542.8791487459366` | step2_out.txt: 542.879 |
| T1 first finite candidate does not stop | F2/F3 | `stops_run` | `False` | step2_out.txt |
| 14-day window largest finite log prediction | D | `max_abs_log_pred` | `244.1820481792326` | step6_out.txt |
| 14-day window pond | D | `row_pond` | `2` | step6_out.txt |
| 14-day window origin date | D | `row_date` | `2021-10-09` | step6_out.txt |
| 14-day window horizon | D | `row_horizon` | `1` | new diagnostic detail; step6_out.txt omits the maximum row horizon |
| 14-day window training ponds | D | `train_ponds` | `3;4;7` | step6_out.txt |
| 14-day window dominant design column | D | `top_column` | `t2w14_do_max_w` | step6_out.txt |
| 14-day window raw dissolved oxygen | D | `top_raw_value` | `3.2` | step6_out.txt |
| 14-day window training minimum | D | `top_train_min` | `19.972` | step6_out.txt |
| 14-day window training maximum | D | `top_train_max` | `19.998` | step6_out.txt |
| 14-day window training values and counts | D | `top_train_values` | `19.972x2;19.989x2;19.996x4;19.997x2;19.998x12` | step6_out.txt; counts newly recorded |
| 14-day window signed z-score | D | `top_z` | `-2236.0066745905706` | step6_out.txt |
| 42-day window largest finite log prediction | E | `max_abs_log_pred` | `615.2296371627393` | step6_out.txt |
| 42-day window pond | E | `row_pond` | `2` | step6_out.txt |
| 42-day window origin date | E | `row_date` | `2021-10-09` | step6_out.txt |
| 42-day window horizon | E | `row_horizon` | `1` | new diagnostic detail; step6_out.txt omits the maximum row horizon |
| 42-day window training ponds | E | `train_ponds` | `3;4;7` | step6_out.txt |
| 42-day window dominant design column | E | `top_column` | `t2w42_do_max_w` | step6_out.txt |
| 42-day window raw dissolved oxygen | E | `top_raw_value` | `9.203` | step6_out.txt |
| 42-day window training minimum | E | `top_train_min` | `19.996` | step6_out.txt |
| 42-day window training maximum | E | `top_train_max` | `19.998` | step6_out.txt |
| 42-day window training values and counts | E | `top_train_values` | `19.996x2;19.997x2;19.998x18` | step6_out.txt; counts newly recorded |
| 42-day window signed z-score | E | `top_z` | `-17507.552757723402` | step6_out.txt |
| Main T2 window raw value | M | `top_raw_value` | `4.181` | step6_out.txt: main ridge, not the supplementary windows |
| Main T2 window training minimum | M | `top_train_min` | `19.989` | step6_out.txt: main ridge, not the supplementary windows |
| Main T2 window training maximum | M | `top_train_max` | `19.998` | step6_out.txt: main ridge, not the supplementary windows |
| Main T2 window training ponds | M | `train_ponds` | `3;4;7` | step6_out.txt: main ridge, not the supplementary windows |
| Main T2 window signed z-score | M | `top_z` | `-8199.147917192215` | step6_out.txt: main ridge, not the supplementary windows |
| Exponential overflow threshold in every CSV row | A2 | `exp_overflow_threshold` | `709.782712893384` | step4_out.txt: 709.782712893384 |

The three stopped-run identities agree with `results/metrics/stopped_runs.csv`. This diagnostic exhausts the inner candidates after a stop for inspection; it does not restart or score the stopped runs.

| Model | Exact stopping alpha set | Outer folds | Inner split | Stop candidates | CSV rows and column |
|---|---|---|---:|---:|---|
| `T1 ridge_time_wq_w28` | `0.001;0.0031622776601683794;0.01;0.03162277660168379;0.1;0.31622776601683794;1.0` | 2, 3 | 1 | 14 | Lines 93, 94, 95, 96, 97, 98, 99, 132, 133, 134, 135, 136, 137, 138, `stops_run=True` |
| `T2 ridge_pw_wq` | `0.001` | 2, 3 | 1 | 2 | Lines 366, 405, `stops_run=True` |
| `T2 ridge_pw_wq_noind` | `0.001` | 2, 3 | 1 | 2 | Lines 639, 678, `stops_run=True` |

T1 stops at every grid alpha from 0.001 through 1, matching `step1_out.txt`; the next grid alpha, 3.1622776601683795, is finite in both affected splits. Each T2 model stops only at 0.001, matching `step6_out.txt`. Each supplementary T2 model has zero stop candidates, also matching `step6_out.txt`.

For T1 `ridge_time_wq_w28`, every `stops_run=True` CSV row identifies the same validation sample, pond 9 at weighing date 2021-09-25. This holds for all 14 stopping candidates.

For T2 `ridge_pw_wq`, every `stops_run=True` CSV row identifies the same validation sample, pond 9 at origin 2021-09-25, horizon 1. This holds for both stopping candidates.

For T2 `ridge_pw_wq_noind`, every `stops_run=True` CSV row identifies that same T2 validation sample. This holds for both stopping candidates. I separately refitted every stopping candidate with the frozen learner and checked that its one actual nonfinite gram prediction belongs to the sample named in the CSV, rather than assuming the maximum absolute log prediction was the overflowing sample.

## Differences and source wording

No recomputed requested quantity differs numerically from its corresponding one-off output at the output's printed precision. The CSV retains the full precision of the learner.

The task text and the diagnosis README associate the main-window value 4.181, training range 19.989 to 19.998 and absolute z-score about 8199 with the supplementary windows. `step6_out.txt` instead assigns those quantities to the main `do_max_w` column of `ridge_pw_wq` and `ridge_pw_wq_noind`. They are reproduced in CSV rows M and N. The supplementary maxima do occur on the requested pond and origin date and round to 244 and 615. Their actual dominant columns are `t2w14_do_max_w` and `t2w42_do_max_w`, with raw values 3.2 and 9.203, training ranges 19.972 to 19.998 and 19.996 to 19.998, and absolute z-scores 2236.0066745905706 and 17507.552757723402. The CSV records signed z-scores, which are negative for these low validation readings; the one-off summary prints absolute values.

The noind prediction is 973.6802181763805, agreeing with `step6_out.txt`'s 973.68 and rounding to the task's 974. Its z-score is 2853.8404575488903. The tiny difference from the indicator model's 2853.84045754889 follows the learner's stored means and scales; neither value was adjusted.

`step4_out.txt` prints a z-score of approximately -2 in part a because `step4_verify.py` selects the first pond-9 validation row, dated 2021-07-03. That is a different sample from the stopping row. The stopping row's z-score 2983 agrees with `step2_out.txt` and the corrected selection in `step5_out.txt`. This CSV stores only the maximum absolute log-prediction row for each candidate, as requested, and therefore does not add a record for that unrelated first validation row.

The one-off phrase "largest finite log prediction" means the log-scale prediction is finite. It does not establish that conversion to grams is finite. The CSV reports both scales' relevant diagnostics through `max_abs_log_pred`, `n_nonfinite_grams` and `stops_run`.

## Verification and repeatability

The check in `/tmp/check_phase7c_ridge.py` failed before generation with `FileNotFoundError` for the absent CSV. After generation it passed with the following output. The initial failure only establishes the missing artifact; the numerical and coverage checks below establish its contents.

```text
T1 ridge_time_wq_w28 all stop candidates identify the same actual overflowing validation row: 14
T2 ridge_pw_wq all stop candidates identify the same actual overflowing validation row: 2
T2 ridge_pw_wq_noind all stop candidates identify the same actual overflowing validation row: 2
PASS: 1365 unique candidates, full grid coverage, CSV invariants, step2 and step6 values, all 18 actual overflow rows, unknown model and empty split failures.
```

The check reconstructs the expected candidate keys from the frozen fold plans and alpha grid, checks exact column order and uniqueness, compares every per-alpha prediction and gram-nonfinite count in `step2_out.txt`, and compares the four T2 extrema and stop counts in `step6_out.txt`. It also checks contribution products, standardization, training-value ordering and truncation, the common overflow threshold and T1's blank horizon. An unknown configured model raises `KeyError`; an empty inner split raises `ValueError`.

After copying the first CSV to `/tmp/phase7c_ridge_stop_first.csv`, I ran the diagnostic a second time with the same command and then ran:

```bash
cmp /tmp/phase7c_ridge_stop_first.csv results/diagnostics/ridge_stop.csv
```

`cmp` exited 0 with no output. The two CSV files are byte-identical. The CSV SHA-256 is `c2d1030860d6b154330a3127c46a2a079079ad71cc1aadbf408240fc68c0ae87`.

Verification reached execution of the frozen learner on all inner candidates, numerical comparison against saved outputs, and byte comparison across two complete script runs. No outer-test predictions or scores were computed. No requested deliverable remains unfinished.
