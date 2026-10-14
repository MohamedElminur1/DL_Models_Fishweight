# Gate 4 part 2: blind rebuild of the pond 2 fold against the saved predictions

Every saved prediction is matched (0 unmatched); largest |difference| 8.88e-16 in log weight and 5.68e-14 g; largest pond 2 S difference 1.39e-16.

S_pond2 is the seed-averaged S of pond 2 (T2: both horizons, the `all` stratum of per_pond.csv).

```
task    model saved_fold  rows_saved  rows_codex  unmatched  max_abs_d_log_pred  max_abs_d_w_pred_g  S_pond2_codex  S_pond2_saved      S_diff
  T1     lstm        [1]          35          35          0         8.88178e-16         5.68434e-14        0.19525        0.19525 1.38778e-16
  T2 cnn_lstm        [1]          55          55          0         1.11022e-16         5.68434e-14       0.258077       0.258077 5.55112e-17
```
