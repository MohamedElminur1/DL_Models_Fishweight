"""LSTM and GRU regressors (plan 6.3, 13).

Input: daily sequence [B, L, 30] (24 standardized aggregates and 6 sensor masks, left-padded
with zeros), statics [B, S], padding mask [B, L]. The hidden state of the last real step is
concatenated with the statics and passed through a two-layer head (hidden units, ReLU,
dropout, one output). Dropout acts in the head only.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.common import gather_last


class RNNRegressor(nn.Module):
    """`cell` is "lstm" or "gru"; `layers` recurrent layers of `hidden` units."""

    def __init__(self, cell: str, n_in: int, n_static: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        recurrent = {"lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = recurrent(n_in, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.head(torch.cat([gather_last(out, pad), static], dim=1)).squeeze(-1)
