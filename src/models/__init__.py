"""Neural models of Phase 4: LSTM, GRU, CNN-LSTM, an encoder-only TFT and `mlp_static`.

`build_network(arch, hp, ...)` is the one place that turns an architecture name and its
hyperparameters into a torch module with the signature `net(x, static, pad) -> [B]`.
`DEEP_MODELS` are the sequence models (the pool of the primary deep model, plan 9.4);
`mlp_static` is their static-only twin. Shared data and training code is in
`src.models.common`.
"""

from __future__ import annotations

from torch import nn

from src.models.cnn_lstm import CNNLSTMRegressor
from src.models.mlp import StaticMLP
from src.models.rnn import RNNRegressor
from src.models.tft import TFTRegressor

DEEP_MODELS = ("lstm", "gru", "cnn_lstm", "tft")
STATIC_MODEL = "mlp_static"
NETWORK_MODELS = DEEP_MODELS + (STATIC_MODEL,)


def build_network(arch: str, hp: dict, n_static: int, n_sensors: int, n_stats: int) -> nn.Module:
    """The network called `arch` with hyperparameters `hp` (keys as in configs/phase4.yaml)."""
    n_in = n_sensors * n_stats + n_sensors  # aggregates plus one validity mask per sensor
    if arch in ("lstm", "gru"):
        return RNNRegressor(arch, n_in, n_static, int(hp["hidden"]), int(hp["layers"]), float(hp["dropout"]))
    if arch == "cnn_lstm":
        return CNNLSTMRegressor(
            n_in, n_static, int(hp["hidden"]), int(hp["layers"]), float(hp["dropout"]),
            int(hp["filters"]), int(hp["kernel"]), int(hp["pool"]),
        )
    if arch == "tft":
        return TFTRegressor(n_sensors, n_stats, n_static, int(hp["hidden"]), int(hp["heads"]), float(hp["dropout"]))
    if arch == STATIC_MODEL:
        return StaticMLP(n_static, int(hp["hidden"]), float(hp["dropout"]))
    raise KeyError(f"unknown architecture {arch!r}; choose from {NETWORK_MODELS}")
