# Completed independent verification

Implemented SPEC.md in `g4p2.py` and ran all ten final fits: seeds 0–4 for
T1 LSTM and T2 CNN-LSTM. Settings were used as specified, without tuning.
Training excluded ponds 1 and 2 and retained eligible rows in file order.
T1 used 25 training rows, 7 test rows and L = 112; T2 used 36 training rows,
11 test rows and 28-day windows with both horizons pooled.

The implementation uses each sample's strict interpolation-anchor cutoff,
training-only scalers over distinct valid cells, float64 scaling and float32
network tensors. Networks use the prescribed initialization order and final
epoch weights. Training follows the specified seeded minibatches and records
sample-weighted training MSE.

Created:

- `g4p2.py`: independent preprocessing, networks and training program.
- `test_g4p2.py`: six hand-checkable preprocessing tests.
- `codex_g4p2_predictions.csv`: 90 prediction rows, saved with `%.17g`.
- `codex_g4p2_curves.csv`: 285 epoch records, saved with `%.17g`.
- `codex_g4p2_scalers.json`: both models' channel, static and target scalers;
  T1 also records L.

## Verification

Ran the program successfully using:

```bash
nice -n 19 /home/dev/development/publications/.venv/bin/python -I g4p2.py
```

The tests passed: **6 passed in 1.91s**. They cover the strict anchor cutoff,
T1 padding and window cut, T2 origin-based window cut, channel and mask layout,
distinct-cell scaler population, missing-channel defaults, constant SD handling
and T2 static ordering. The initial test run failed collection because
`g4p2.py` did not yet exist; that initial failure was not a behavioral test.

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 nice -n 19 /home/dev/development/publications/.venv/bin/python -I -m pytest -p no:cacheprovider -c /dev/null --rootdir=/home/dev/development/publications/verification/codex_g4p2 test_g4p2.py
```

The explicit pytest configuration and disabled plugin discovery prevent
ancestor configuration and unrelated plugins from being loaded.

Verified the saved artifacts against the permitted inputs: all test rows remain
in source order for every seed, epoch sequences are complete, predictions and
losses are finite, and gram conversions match exactly after round-trip CSV
parsing. A separate pandas calculation recomputed the training-cell union and
confirmed every saved channel, static and target scaler. Scalers also matched
across all five fresh fits of each model.

Runtime versions: PyTorch 2.14.1+cpu, NumPy 2.5.4, pandas 3.0.6. Both torch
thread counts were set to one and deterministic algorithms were enabled with
`warn_only=True`. Python processes ran sequentially with `nice -n 19` and `-I`.

## Interpretation and access

No substantive ambiguity was found. The unspecified JSON nesting is keyed by
model name (`lstm`, `cnn_lstm`), with 24 channel entries in sensor/statistic
order. T1 `origin_date` is blank. A daily cell absent from the table is treated
as invalid, consistently with the invalid-cell rule.

Source reads were limited to SPEC.md and the three named processed tables.
No prohibited project files or directories were opened or listed. All created
files are in this verification directory. No commit was made.
