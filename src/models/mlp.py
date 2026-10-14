"""`mlp_static`: the static-inputs-only twin of the sequence models (plan 6.3, 13).

The same two-layer head that the LSTM, GRU and CNN-LSTM put on top of their last hidden
state (linear, ReLU, dropout, linear), applied to the static inputs alone: T1 uses t; T2 uses
log W_k, log W_{k-1}, t_k, t_k - t_{k-1} and the target gap. The daily sequence is ignored.
It is tuned, trained and seeded exactly like the sequence models, so a difference between a
sequence model and `mlp_static` measures what the water-quality sequence adds, not what the
training procedure adds.
"""

from __future__ import annotations

import torch
from torch import nn


class StaticMLP(nn.Module):
    """`net(x, static, pad)` like the sequence models; only `static` is used."""

    def __init__(self, n_static: int, hidden: int, dropout: float):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(n_static, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        return self.head(static).squeeze(-1)
