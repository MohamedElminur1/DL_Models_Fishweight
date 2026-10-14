# Independent verification task (Codex): two deep models, final fits on one outer fold

You are verifying another team's deep-learning pipeline blind. Work only inside this directory
(`verification/codex_g4p2/`). Read the three input tables below and this file. Do NOT open,
list or read anything else under `/home/dev/development/publications/` (in particular not
`src`, `results`, `configs`, `tasks`, `tests`, `reporting`, `posthoc`, other `verification/`
folders or any Markdown file outside this directory). The point is an independent
implementation from this description alone.

Python: `/home/dev/development/publications/.venv/bin/python -I` (numpy, pandas, torch
2.14.1+cpu). Run from this directory with `nice -n 19`; one process at a time. Write your code
in `g4p2.py`, outputs in this directory, and finish with `DONE.md`. Do not tune anything and
do not change any setting below to make results look better: implement this description.

## 1. Inputs (read-only)

`/home/dev/development/publications/data/processed/daily_wq.csv`: one row per pond, calendar
date and sensor. Columns used: `pond`, `date`, `sensor`, `status` (`observed`, `interpolated`,
`missing`), `right_anchor_date` (set only on interpolated rows), `mean`, `sd`, `min`, `max`.

`.../data/processed/t1_table.csv` (task T1, one row per weighing event): `pond`, `date`, `t`
(days since 2021-06-19), `log_w` (natural log of the weight in g), `eligible_eval`.

`.../data/processed/t2_table.csv` (task T2, one row per origin event and horizon): `pond`,
`origin_date`, `target_date`, `horizon` (1 or 2), `t_k`, `dt_prev`, `dt_target`, `log_w_k`,
`log_w_prev`, `w_origin`, `y` = ln(w_target / w_origin), `eligible_eval`.

Use only rows with `eligible_eval` true. Keep every table in its file row order.

## 2. The fold

Outer fold: the held-out (test) pond is 2. Its sensor-stream partner, pond 1, is removed from
training. Training rows: the eligible rows of ponds 3, 4, 6, 7, 8 and 9, in file row order
(pond 8 has none). Test rows: the eligible rows of pond 2. Expected counts: T1 25 training,
7 test; T2 36 training, 11 test (both horizons together).

## 3. Sequences (one per sample)

Daily store: index days from the first date in `daily_wq.csv` (2021-05-22). For pond p, day d,
sensor s, the four statistics in the order mean, sd, min, max. Sensor order: temperature,
turbidity, do, ph, ammonia_log10, nitrate.

Reference day r of a sample: T1 its `date`; T2 its `origin_date`. A cell (p, d, s) is valid
for that sample if its status is `observed`, or if its status is `interpolated` and its
`right_anchor_date` is before r. Every other cell is invalid for that sample.

Window: T1 the days from 2021-06-19 to r - 1 (that is `t` days), left-padded with padding
steps to L = the largest `t` over all eligible T1 rows of all ponds (L = 112), so every T1
sequence has L steps and its last step is r - 1. T2 the 28 days r - 28 to r - 1, no padding.

Each step has 30 channels: channel 4*s + k (s = sensor index 0..5, k = statistic index 0..3)
is the standardized statistic (section 4) if the cell is valid, else 0.0; channel 24 + s is
1.0 if the cell of sensor s is valid, else 0.0. Padding steps are all 0.0.

Statics: T1 one column, `t`. T2 five columns in this order: `log_w_k`, `log_w_prev`, `t_k`,
`dt_prev`, `dt_target`. Target: T1 `log_w`; T2 `y` (both horizons pooled in one model).

## 4. Scaling (fitted on the training rows only, inside each fit)

- Channel (s, k): mean and population SD (ddof 0) of statistic k over the set of distinct
  cells (pond, day, sensor s) that are valid in at least one training sample's window under
  that sample's own validity rule. A cell counts once however many windows contain it. If a
  channel has no such cell, mean 0 and SD 1. An SD of at most 1e-12 is replaced by 1.
- Valid cells: (value - mean) / SD. Invalid cells and padding: 0.0. Masks: not scaled.
- Statics: per column, mean and population SD over the training samples (same SD rule).
- Target: mean and population SD over the training targets (same SD rule).
- Do the arithmetic in float64; give the network float32 tensors.

## 5. Networks (PyTorch, default initialisation, no custom init)

T1 LSTM, hidden H, layers N, dropout p:
`lstm = nn.LSTM(30, H, num_layers=N, batch_first=True)` (unidirectional, bias, zero initial
state, no dropout between layers), then `head = nn.Sequential(nn.Linear(H + 1, H), nn.ReLU(),
nn.Dropout(p), nn.Linear(H, 1))`. Forward: run the LSTM over all L steps (padding included,
no packing), take the top-layer output at the last step (index L - 1), concatenate the static
vector, apply the head, squeeze to shape [batch]. Create the modules in exactly this order:
LSTM, then the head.

T2 CNN-LSTM, filters F, kernel K, pool P, hidden H, layers N, dropout p:
`conv = nn.Conv1d(30, F, K, padding="same")`, ReLU, then `nn.MaxPool1d(P, ceil_mode=True)` if
P > 1 (nothing if P = 1), then `lstm = nn.LSTM(F, H, num_layers=N, batch_first=True)` over the
pooled steps, the top-layer output at the last pooled step, concatenated with the 5 statics,
then the same head as T1 with H + 5 inputs. The input [batch, 28, 30] is transposed to
[batch, 30, 28] for the convolution and back for the LSTM. Create the modules in this order:
Conv1d, LSTM, then the head.

The network outputs the standardized target.

## 6. Training one fit (seed s, E epochs)

1. Fit the scalers of section 4 on the training rows and build the training tensors.
2. Seed: `random.seed(s)`, `numpy.random.seed(s)`, `torch.manual_seed(s)`; then create the
   network (section 5 order).
3. `torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=WD)`, other arguments at their
   defaults. Loss: `torch.nn.functional.mse_loss` on the standardized target (batch mean).
4. `gen = torch.Generator().manual_seed(s)`. For each epoch 1..E: `net.train()`,
   `perm = torch.randperm(n_train, generator=gen)`, consecutive slices of `perm` of length
   batch size B (the last, shorter slice is used), and per slice:
   `opt.zero_grad(set_to_none=True)`, forward, loss, `backward()`, `opt.step()`. Record the
   epoch's training MSE as the sample-weighted mean of the batch losses.
5. No validation, no early stopping, no clipping, no learning-rate schedule. Keep the final
   weights.
6. Predict the test rows: `net.eval()`, `torch.no_grad()`, one forward pass over all test rows,
   prediction on the log scale = output (as float64) * target SD + target mean. Grams: T1
   exp(prediction); T2 `w_origin` * exp(prediction).

At the start of the program: `torch.set_num_threads(1)`, `torch.set_num_interop_threads(1)`,
`torch.use_deterministic_algorithms(True, warn_only=True)`.

Run seeds s = 0, 1, 2, 3, 4 for each model, each a fresh fit as above.

## 7. Settings of this fold

| | T1 LSTM | T2 CNN-LSTM |
|---|---|---|
| epochs E | 35 | 22 |
| batch size B | 16 | 8 |
| hidden H | 16 | 32 |
| layers N | 2 | 2 |
| dropout p | 0.21945567008211916 | 0.24947606909722403 |
| learning rate LR | 0.009430527524826749 | 0.009292720124427695 |
| weight decay WD | 3.339621571998313e-06 | 0.007865935076179597 |
| filters F | | 8 |
| kernel K | | 7 |
| pool P | | 2 |

## 8. Outputs

- `codex_g4p2_predictions.csv`: one row per model, seed and test row: `task`, `model`
  (`lstm` or `cnn_lstm`), `seed`, `pond`, `date` (T1 the event date; T2 the target date),
  `origin_date` (T2 only), `horizon` (T1: 0), `y_pred_log`, `w_pred`, with full precision
  (`float_format="%.17g"`).
- `codex_g4p2_curves.csv`: `task`, `model`, `seed`, `epoch`, `train_mse`.
- `codex_g4p2_scalers.json`: for each model, the 24 channel means and SDs, the static means
  and SDs, the target mean and SD, and (T1) L.
- `test_g4p2.py` with small hand-checkable tests of the validity rule, the window cut, the
  channel layout and the scaler population, run with `-m pytest -p no:cacheprovider`.
- `DONE.md`: what you built, the test result, anything in this description you found
  ambiguous and how you resolved it.
