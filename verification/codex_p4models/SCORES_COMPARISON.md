# Codex part 1 outer scores against results/metrics

Compared 17 task-model-horizon scores; largest |S difference| 5.71e-09, largest per-pond difference 4.42e-08.

```
task         model horizon  S_codex  S_pipeline  max_pond_diff                                       note      S_diff
  T1            gc       0 0.261001    0.261001    4.42108e-08                                            5.70722e-09
  T1   cohort_mean       0 0.265477    0.265477    1.59385e-09                                             2.0737e-10
  T1    ridge_time       0 0.268928    0.268928    9.99201e-16                                            1.11022e-16
  T1 ridge_time_wq       0  1.58239     1.58239     1.9984e-14                                            3.77476e-15
  T1    allometric       0 0.115125    0.115125    3.20577e-15                                            2.08167e-16
  T2   persistence       1    0.529       0.529    2.22045e-16                                            1.11022e-16
  T2   persistence       2 0.983359    0.983359    1.11022e-16                                            2.22045e-16
  T2        linear       1 0.344188    0.344188    3.33067e-16                                            1.11022e-16
  T2        linear       2 0.399441    0.399441    2.22045e-16                                                      0
  T2           sgr       1 0.544703    0.544703    6.66134e-16                                                      0
  T2           sgr       2 0.900289    0.900289    6.66134e-16                                                      0
  T2   anchored_gc       1 0.274265    0.274265     9.0701e-09                                            1.25472e-09
  T2   anchored_gc       2 0.263344    0.263344    3.38705e-09                                            7.06827e-10
  T2 cohort_growth       1 0.281121    0.281121     5.2546e-10                                             3.9716e-11
  T2 cohort_growth       2 0.253955    0.253955    2.93516e-09                                            6.01858e-10
  T2      ridge_pw       1 0.294721    0.294721      4.996e-16                                            1.11022e-16
  T2      ridge_pw       2 0.242875    0.242875    3.33067e-16                                            1.11022e-16
  T2   ridge_pw_wq       1 0.887273         NaN            NaN no pipeline score (stopped under plan 9.0)         NaN
  T2   ridge_pw_wq       2  0.42744         NaN            NaN no pipeline score (stopped under plan 9.0)         NaN
```
