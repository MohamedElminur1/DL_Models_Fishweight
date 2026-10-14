"""Independent implementation of the pond-2 final fits described in SPEC.md."""

from dataclasses import dataclass
import json
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F


OUTPUT_DIR = Path(__file__).resolve().parent
INPUT_DIR = Path("/home/dev/development/publications/data/processed")
SENSORS = ("temperature", "turbidity", "do", "ph", "ammonia_log10", "nitrate")
STATISTICS = ("mean", "sd", "min", "max")
STATIC_COLUMNS = {
    "T1": ("t",),
    "T2": ("log_w_k", "log_w_prev", "t_k", "dt_prev", "dt_target"),
}
TARGET_COLUMN = {"T1": "log_w", "T2": "y"}
TRAIN_PONDS = (3, 4, 6, 7, 8, 9)
SETTINGS = {
    "T1": dict(model="lstm", epochs=35, batch_size=16, hidden=16, layers=2,
               dropout=0.21945567008211916, lr=0.009430527524826749,
               weight_decay=3.339621571998313e-06),
    "T2": dict(model="cnn_lstm", epochs=22, batch_size=8, hidden=32, layers=2,
               dropout=0.24947606909722403, lr=0.009292720124427695,
               weight_decay=0.007865935076179597, filters=8, kernel=7, pool=2),
}


def cell_valid(status: str, right_anchor: pd.Timestamp, reference: pd.Timestamp) -> bool:
    """Check a sensor cell using the reference date of its own sample."""
    return status == "observed" or (
        status == "interpolated" and pd.notna(right_anchor) and right_anchor < reference
    )


class DailyStore:
    """Retain daily cells in file order and look them up by pond, indexed day and sensor."""

    def __init__(self, frame: pd.DataFrame):
        dates = pd.to_datetime(frame["date"])
        self.first_date = dates.iloc[0]
        self.values = frame.loc[:, list(STATISTICS)].to_numpy(dtype=np.float64)
        self.sensor_indices = frame["sensor"].map(dict(zip(SENSORS, range(6)))).to_numpy()
        if pd.isna(self.sensor_indices).any():
            raise ValueError("Unknown sensor in daily table")
        self.statuses = frame["status"].to_numpy()
        self.anchors = list(pd.to_datetime(frame["right_anchor_date"]))
        self.lookup = {}
        for index, (pond, date, sensor) in enumerate(zip(frame["pond"], dates, self.sensor_indices)):
            key = (int(pond), self.day_index(date), int(sensor))
            if key in self.lookup:
                raise ValueError(f"Duplicate daily cell: {key}")
            self.lookup[key] = index

    def day_index(self, date: pd.Timestamp) -> int:
        return int((pd.Timestamp(date) - self.first_date).days)


@dataclass
class Window:
    values: np.ndarray
    valid: np.ndarray
    indices: np.ndarray


def make_window(store: DailyStore, row, task: str, length: int | None) -> Window:
    """Construct the unscaled, reference-exclusive window including T1 left padding."""
    reference = pd.Timestamp(row["date"] if task == "T1" else row["origin_date"])
    stop = store.day_index(reference)
    if task == "T1":
        start = store.day_index(pd.Timestamp("2021-06-19"))
        count = stop - start
        if count != int(row["t"]) or count < 0 or length is None or count > length:
            raise ValueError("T1 date/t/window length disagreement")
        padding = length - count
        steps = length
    else:
        start = stop - 28
        padding = 0
        steps = 28
    values = np.zeros((steps, 6, 4), dtype=np.float64)
    valid = np.zeros((steps, 6), dtype=bool)
    indices = np.full((steps, 6), -1, dtype=np.int64)
    for step, day in enumerate(range(start, stop), start=padding):
        for sensor in range(6):
            index = store.lookup.get((int(row["pond"]), day, sensor))
            if index is None:
                continue
            indices[step, sensor] = index
            if cell_valid(store.statuses[index], store.anchors[index], reference):
                if not np.isfinite(store.values[index]).all():
                    raise ValueError("Valid sensor cell has nonfinite statistics")
                valid[step, sensor] = True
                values[step, sensor] = store.values[index]
    return Window(values, valid, indices)


def population_scale(values: np.ndarray, axis=None):
    """Compute float64 population moments and replace only negligible SDs."""
    mean = np.mean(values, axis=axis, dtype=np.float64)
    sd = np.std(values, axis=axis, ddof=0, dtype=np.float64)
    return mean, np.where(sd <= 1e-12, 1.0, sd)


def fit_scalers(store: DailyStore, windows: list[Window], rows: pd.DataFrame,
                task: str, length: int | None) -> dict:
    """Fit channel moments on the union of distinct valid training cells, in file order."""
    covered = np.zeros(len(store.values), dtype=bool)
    for window in windows:
        covered[window.indices[window.valid]] = True
    channel_mean = np.zeros((6, 4), dtype=np.float64)
    channel_sd = np.ones((6, 4), dtype=np.float64)
    for sensor in range(6):
        values = store.values[covered & (store.sensor_indices == sensor)]
        if len(values):
            channel_mean[sensor], channel_sd[sensor] = population_scale(values, axis=0)
    statics = rows.loc[:, list(STATIC_COLUMNS[task])].to_numpy(dtype=np.float64)
    target = rows[TARGET_COLUMN[task]].to_numpy(dtype=np.float64)
    if not np.isfinite(statics).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite training statics or target")
    static_mean, static_sd = population_scale(statics, axis=0)
    target_mean, target_sd = population_scale(target)
    result = dict(channel_mean=channel_mean.reshape(24), channel_sd=channel_sd.reshape(24),
                  static_mean=static_mean, static_sd=static_sd,
                  target_mean=float(target_mean), target_sd=float(target_sd))
    if task == "T1":
        result["L"] = length
    return result


def scale_window(window: Window, scalers: dict) -> np.ndarray:
    """Place sensor-major statistics first, then the six unscaled validity masks."""
    means = scalers["channel_mean"].reshape(6, 4)
    sds = scalers["channel_sd"].reshape(6, 4)
    scaled = np.zeros_like(window.values)
    standardized = (window.values - means) / sds
    scaled[window.valid] = standardized[window.valid]
    sequence = np.concatenate((scaled.reshape(len(scaled), 24),
                               window.valid.astype(np.float64)), axis=1)
    return sequence.astype(np.float32)


def tensors_for_rows(windows: list[Window], rows: pd.DataFrame, task: str, scalers: dict):
    sequence = torch.from_numpy(np.stack([scale_window(window, scalers) for window in windows]))
    statics = rows.loc[:, list(STATIC_COLUMNS[task])].to_numpy(dtype=np.float64)
    statics = (statics - scalers["static_mean"]) / scalers["static_sd"]
    target = rows[TARGET_COLUMN[task]].to_numpy(dtype=np.float64)
    target = (target - scalers["target_mean"]) / scalers["target_sd"]
    return sequence, torch.from_numpy(statics.astype(np.float32)), torch.from_numpy(target.astype(np.float32))


class T1LSTM(nn.Module):
    def __init__(self, settings: dict):
        super().__init__()
        hidden = settings["hidden"]
        self.lstm = nn.LSTM(30, hidden, num_layers=settings["layers"], batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 1, hidden), nn.ReLU(),
                                  nn.Dropout(settings["dropout"]), nn.Linear(hidden, 1))

    def forward(self, sequence: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        outputs, _ = self.lstm(sequence)
        return self.head(torch.cat((outputs[:, -1, :], statics), dim=1)).squeeze(-1)


class T2CNNLSTM(nn.Module):
    def __init__(self, settings: dict):
        super().__init__()
        hidden = settings["hidden"]
        self.conv = nn.Conv1d(30, settings["filters"], settings["kernel"], padding="same")
        self.lstm = nn.LSTM(settings["filters"], hidden, num_layers=settings["layers"], batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.ReLU(),
                                  nn.Dropout(settings["dropout"]), nn.Linear(hidden, 1))
        self.pool = nn.MaxPool1d(settings["pool"], ceil_mode=True) if settings["pool"] > 1 else None

    def forward(self, sequence: torch.Tensor, statics: torch.Tensor) -> torch.Tensor:
        convolved = F.relu(self.conv(sequence.transpose(1, 2)))
        if self.pool is not None:
            convolved = self.pool(convolved)
        outputs, _ = self.lstm(convolved.transpose(1, 2))
        return self.head(torch.cat((outputs[:, -1, :], statics), dim=1)).squeeze(-1)


def fit_one(store: DailyStore, train_rows: pd.DataFrame, test_rows: pd.DataFrame,
            task: str, length: int | None, seed: int):
    """Run one fresh fit with its own training-only scalers and final-epoch predictions."""
    settings = SETTINGS[task]
    train_windows = [make_window(store, row, task, length) for row in train_rows.to_dict("records")]
    scalers = fit_scalers(store, train_windows, train_rows, task, length)
    train_x, train_static, train_y = tensors_for_rows(train_windows, train_rows, task, scalers)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    net = T1LSTM(settings) if task == "T1" else T2CNNLSTM(settings)
    optimizer = torch.optim.AdamW(net.parameters(), lr=settings["lr"], weight_decay=settings["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    curves = []
    for epoch in range(1, settings["epochs"] + 1):
        net.train()
        permutation = torch.randperm(len(train_rows), generator=generator)
        loss_sum = 0.0
        for start in range(0, len(train_rows), settings["batch_size"]):
            batch = permutation[start:start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = net(train_x[batch], train_static[batch])
            loss = F.mse_loss(prediction, train_y[batch])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch)
        curves.append(dict(task=task, model=settings["model"], seed=seed, epoch=epoch,
                           train_mse=loss_sum / len(train_rows)))
    test_windows = [make_window(store, row, task, length) for row in test_rows.to_dict("records")]
    test_x, test_static, _ = tensors_for_rows(test_windows, test_rows, task, scalers)
    net.eval()
    with torch.no_grad():
        standardized = net(test_x, test_static).numpy().astype(np.float64)
    log_prediction = standardized * scalers["target_sd"] + scalers["target_mean"]
    grams = np.exp(log_prediction)
    if task == "T2":
        grams *= test_rows["w_origin"].to_numpy(dtype=np.float64)
    predictions = []
    for row, log_value, weight in zip(test_rows.to_dict("records"), log_prediction, grams):
        predictions.append(dict(task=task, model=settings["model"], seed=seed, pond=int(row["pond"]),
                                date=row["date"] if task == "T1" else row["target_date"],
                                origin_date="" if task == "T1" else row["origin_date"],
                                horizon=0 if task == "T1" else int(row["horizon"]),
                                y_pred_log=float(log_value), w_pred=float(weight)))
    return predictions, curves, scalers


def serializable_scalers(scalers: dict) -> dict:
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in scalers.items()}


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)
    daily = pd.read_csv(INPUT_DIR / "daily_wq.csv", usecols=[
        "pond", "date", "sensor", "status", "right_anchor_date", *STATISTICS])
    tables = {
        "T1": pd.read_csv(INPUT_DIR / "t1_table.csv", usecols=[
            "pond", "date", "t", "log_w", "eligible_eval"]),
        "T2": pd.read_csv(INPUT_DIR / "t2_table.csv", usecols=[
            "pond", "origin_date", "target_date", "horizon", "t_k", "dt_prev", "dt_target",
            "log_w_k", "log_w_prev", "w_origin", "y", "eligible_eval"]),
    }
    for task, table in tables.items():
        if table["eligible_eval"].dtype != bool:
            raise ValueError("eligible_eval must parse as booleans")
        tables[task] = table.loc[table["eligible_eval"]]
    store = DailyStore(daily)
    assert store.first_date == pd.Timestamp("2021-05-22")
    assert pd.to_datetime(daily["date"]).min() == store.first_date
    length = int(tables["T1"]["t"].max())
    assert length == 112
    print(f"torch={torch.__version__}; numpy={np.__version__}; pandas={pd.__version__}", flush=True)
    all_predictions, all_curves, all_scalers = [], [], {}
    expected_counts = {"T1": (25, 7), "T2": (36, 11)}
    for task, table in tables.items():
        train_rows = table.loc[table["pond"].isin(TRAIN_PONDS)]
        test_rows = table.loc[table["pond"] == 2]
        assert (len(train_rows), len(test_rows)) == expected_counts[task]
        print(f"{task}: train={len(train_rows)}, test={len(test_rows)}, L={length if task == 'T1' else 28}", flush=True)
        for seed in range(5):
            predictions, curves, scalers = fit_one(store, train_rows, test_rows, task,
                                                  length if task == "T1" else None, seed)
            serialized = serializable_scalers(scalers)
            model = SETTINGS[task]["model"]
            if model in all_scalers:
                assert serialized == all_scalers[model]
            else:
                all_scalers[model] = serialized
            all_predictions.extend(predictions)
            all_curves.extend(curves)
            print(f"{task} seed={seed}: epochs={len(curves)}, final_train_mse={curves[-1]['train_mse']:.17g}", flush=True)
    predictions_frame = pd.DataFrame(all_predictions)
    curves_frame = pd.DataFrame(all_curves)
    assert len(predictions_frame) == 90
    assert len(curves_frame) == 285
    assert np.isfinite(predictions_frame[["y_pred_log", "w_pred"]].to_numpy()).all()
    assert np.isfinite(curves_frame["train_mse"]).all()
    predictions_frame.to_csv(OUTPUT_DIR / "codex_g4p2_predictions.csv", index=False, float_format="%.17g")
    curves_frame.to_csv(OUTPUT_DIR / "codex_g4p2_curves.csv", index=False, float_format="%.17g")
    with (OUTPUT_DIR / "codex_g4p2_scalers.json").open("w") as handle:
        json.dump(all_scalers, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("Wrote 90 predictions, 285 curve rows, and both models' scalers.", flush=True)


if __name__ == "__main__":
    main()
