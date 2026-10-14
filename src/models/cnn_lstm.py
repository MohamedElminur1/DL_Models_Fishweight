"""CNN-LSTM regressor (plan 6.3, 13).

Conv1d over the days (`filters` filters, `kernel` wide, same padding) with ReLU, then max
pooling of width `pool` (1 means no pooling), then an LSTM, then the same two-layer head as
the plain recurrent models, fed with the last LSTM output and the statics. Pooling uses
ceil mode, so the most recent day is never dropped when the sequence length is odd; the
sequences are left-padded, so the last pooled step is always a real step.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CNNLSTMRegressor(nn.Module):
    def __init__(self, n_in: int, n_static: int, hidden: int, layers: int, dropout: float,
                 filters: int, kernel: int, pool: int):
        super().__init__()
        self.conv = nn.Conv1d(n_in, filters, kernel, padding="same")
        self.pool = nn.MaxPool1d(pool, ceil_mode=True) if pool > 1 else nn.Identity()
        self.rnn = nn.LSTM(filters, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1)
        )

    def forward(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        features = self.pool(F.relu(self.conv(x.transpose(1, 2)))).transpose(1, 2)
        out, _ = self.rnn(features)
        return self.head(torch.cat([out[:, -1], static], dim=1)).squeeze(-1)
