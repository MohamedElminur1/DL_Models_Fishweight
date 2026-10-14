"""Temporal Fusion Transformer, encoder only, point output (Lim et al. 2021; plan 6.3, 13).

Blocks, in the order of the forward pass:
- Static variables (T1: t; T2: log W_k, log W_{k-1}, t_k, t_k - t_{k-1}, t_{k+h} - t_k): each
  scalar is embedded by its own linear layer, a variable selection network (VSN) weights and
  sums them into one vector, and four static covariate encoders (GRNs) turn it into the
  contexts c_select (variable selection), c_enrich (static enrichment) and c_h, c_c (the
  initial hidden and cell state of the LSTM encoder).
- Temporal variables: the six sensors. Sensor s at day d is the vector (its four standardized
  daily aggregates, its validity mask), embedded by a linear layer of its own. A VSN with
  context c_select weights the six embeddings at every day.
- LSTM encoder started from (c_h, c_c), gated skip connection with add and layer norm to the
  selected inputs, static enrichment GRN with context c_enrich.
- Interpretable multi-head attention with the last real day as the single query over all
  days (padded days masked, values shared across heads, heads averaged), gated skip, a
  position-wise GRN, and a gated skip to the output of the LSTM stage.
- A linear layer maps the final vector of the last day to one number.

Deviations from the paper, all by design of the plan: mean squared error instead of the
quantile loss (point output), no known-future inputs and no decoder (there is one prediction
per sequence), and a single query instead of a query per time step. Dropout acts inside the
gated linear units and on the attention weights.
"""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from src.models.common import last_real_index


class GatedLinearUnit(nn.Module):
    """Dropout, then a linear map to 2 * d_out and the gate sigmoid(a) * b."""

    def __init__(self, d_in: int, d_out: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(d_in, 2 * d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.glu(self.proj(self.dropout(x)), dim=-1)


class GateAddNorm(nn.Module):
    """LayerNorm(skip + GLU(x)): the gated skip connection of the paper."""

    def __init__(self, d: int, dropout: float):
        super().__init__()
        self.glu = GatedLinearUnit(d, d, dropout)
        self.norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return self.norm(skip + self.glu(x))


class GatedResidualNetwork(nn.Module):
    """GRN(a, c) = LayerNorm(a + GLU(W1 ELU(W2 a + W3 c))), with an optional context c."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float, d_context: int = 0):
        super().__init__()
        self.fc_in = nn.Linear(d_in, d_hidden)
        self.fc_context = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc_hidden = nn.Linear(d_hidden, d_hidden)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.glu = GatedLinearUnit(d_hidden, d_out, dropout)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.fc_in(x)
        if context is not None:
            projected = self.fc_context(context)
            if hidden.dim() == 3 and projected.dim() == 2:
                projected = projected.unsqueeze(1)  # one context per sequence, shared by all days
            hidden = hidden + projected
        hidden = self.fc_hidden(F.elu(hidden))
        return self.norm(self.skip(x) + self.glu(hidden))


class VariableSelection(nn.Module):
    """Softmax weights over `n_vars` embedded variables, each also transformed by its own GRN.

    Input `[..., n_vars, d]`; returns the weighted sum `[..., d]` and the weights `[..., n_vars]`.
    """

    def __init__(self, n_vars: int, d: int, dropout: float, d_context: int = 0):
        super().__init__()
        self.weights = GatedResidualNetwork(n_vars * d, d, n_vars, dropout, d_context)
        self.variable_grns = nn.ModuleList(GatedResidualNetwork(d, d, d, dropout) for _ in range(n_vars))

    def forward(self, embedded: torch.Tensor, context: torch.Tensor | None = None):
        weights = torch.softmax(self.weights(embedded.flatten(start_dim=-2), context), dim=-1)
        processed = torch.stack(
            [grn(embedded[..., i, :]) for i, grn in enumerate(self.variable_grns)], dim=-2
        )
        return (weights.unsqueeze(-1) * processed).sum(dim=-2), weights


class InterpretableMultiHeadAttention(nn.Module):
    """Per-head queries and keys, one value projection shared by all heads, heads averaged."""

    def __init__(self, d: int, heads: int, dropout: float):
        super().__init__()
        if d % heads:
            raise ValueError(f"hidden size {d} is not divisible by {heads} heads")
        self.d_head = d // heads
        self.queries = nn.ModuleList(nn.Linear(d, self.d_head) for _ in range(heads))
        self.keys = nn.ModuleList(nn.Linear(d, self.d_head) for _ in range(heads))
        self.value = nn.Linear(d, self.d_head)
        self.out = nn.Linear(self.d_head, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding: torch.Tensor):
        """`query` [B, Q, d]; `key`, `value` [B, L, d]; `key_padding` [B, L], True = ignore.

        Returns the attended vectors [B, Q, d] and the head-averaged attention [B, Q, L].
        """
        values = self.value(value)
        attended, weights = [], []
        for project_q, project_k in zip(self.queries, self.keys):
            scores = project_q(query) @ project_k(key).transpose(1, 2) / math.sqrt(self.d_head)
            scores = scores.masked_fill(key_padding.unsqueeze(1), float("-inf"))
            attention = self.dropout(torch.softmax(scores, dim=-1))
            attended.append(attention @ values)
            weights.append(attention)
        mixed = torch.stack(attended).mean(dim=0)
        return self.dropout(self.out(mixed)), torch.stack(weights).mean(dim=0)


class TFTRegressor(nn.Module):
    """Encoder-only TFT. `x` is [B, L, sensors * stats + sensors] as built by `make_batch`."""

    def __init__(self, n_sensors: int, n_stats: int, n_static: int, hidden: int, heads: int,
                 dropout: float):
        super().__init__()
        d = hidden
        self.n_sensors, self.n_stats = n_sensors, n_stats
        self.static_embed = nn.ModuleList(nn.Linear(1, d) for _ in range(n_static))
        self.static_select = VariableSelection(n_static, d, dropout)
        self.context_select = GatedResidualNetwork(d, d, d, dropout)
        self.context_enrich = GatedResidualNetwork(d, d, d, dropout)
        self.context_h = GatedResidualNetwork(d, d, d, dropout)
        self.context_c = GatedResidualNetwork(d, d, d, dropout)
        self.sensor_embed = nn.ModuleList(nn.Linear(n_stats + 1, d) for _ in range(n_sensors))
        self.temporal_select = VariableSelection(n_sensors, d, dropout, d_context=d)
        self.encoder = nn.LSTM(d, d, batch_first=True)
        self.post_encoder = GateAddNorm(d, dropout)
        self.enrichment = GatedResidualNetwork(d, d, d, dropout, d_context=d)
        self.attention = InterpretableMultiHeadAttention(d, heads, dropout)
        self.post_attention = GateAddNorm(d, dropout)
        self.position_wise = GatedResidualNetwork(d, d, d, dropout)
        self.post_block = GateAddNorm(d, dropout)
        self.output = nn.Linear(d, 1)

    def forward(self, x: torch.Tensor, static: torch.Tensor, pad: torch.Tensor,
                return_details: bool = False):
        batch, steps, _ = x.shape
        n_agg = self.n_sensors * self.n_stats
        aggregates = x[..., :n_agg].reshape(batch, steps, self.n_sensors, self.n_stats)
        per_sensor = torch.cat([aggregates, x[..., n_agg:].unsqueeze(-1)], dim=-1)
        embedded = torch.stack(
            [embed(per_sensor[:, :, s]) for s, embed in enumerate(self.sensor_embed)], dim=2
        )
        static_embedded = torch.stack(
            [embed(static[:, j:j + 1]) for j, embed in enumerate(self.static_embed)], dim=1
        )
        zeta, static_weights = self.static_select(static_embedded)
        c_select, c_enrich = self.context_select(zeta), self.context_enrich(zeta)
        h0, c0 = self.context_h(zeta).unsqueeze(0), self.context_c(zeta).unsqueeze(0)

        selected, temporal_weights = self.temporal_select(embedded, c_select)
        encoded, _ = self.encoder(selected, (h0.contiguous(), c0.contiguous()))
        phi = self.post_encoder(encoded, selected)
        theta = self.enrichment(phi, c_enrich)

        last = last_real_index(pad)
        rows = torch.arange(batch, device=x.device)
        query, phi_last = theta[rows, last].unsqueeze(1), phi[rows, last].unsqueeze(1)
        attended, attention = self.attention(query, theta, theta, pad)
        delta = self.post_attention(attended, query)
        final = self.post_block(self.position_wise(delta), phi_last)
        prediction = self.output(final).reshape(batch)
        if return_details:
            return prediction, {"attention": attention.squeeze(1), "static_weights": static_weights,
                                "temporal_weights": temporal_weights}
        return prediction
