"""Short-term traffic forecasting models: GRU / LSTM baselines and PatchTST."""
from __future__ import annotations

import torch
from torch import nn


class RNNForecaster(nn.Module):
    """GRU or LSTM over the multivariate history; predicts all horizons for all channels."""

    def __init__(self, n_channels: int, n_horizons: int, hidden: int = 64, layers: int = 2,
                 cell: str = "gru", dropout: float = 0.1):
        super().__init__()
        rnn = nn.GRU if cell == "gru" else nn.LSTM
        self.rnn = rnn(n_channels, hidden, layers, batch_first=True, dropout=dropout)
        self.head = nn.Linear(hidden, n_channels * n_horizons)
        self.n_channels, self.n_horizons = n_channels, n_horizons

    def forward(self, x):  # x: (B, T, C)
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).view(-1, self.n_horizons, self.n_channels)


class PatchTST(nn.Module):
    """Channel-independent patch transformer (Nie et al., ICLR 2023), compact version.

    Each channel's history is split into patches, embedded, passed through a shared
    Transformer encoder, and flattened into a linear head that outputs every horizon.
    """

    def __init__(self, n_channels: int, n_horizons: int, history: int, patch_len: int = 6,
                 stride: int | None = None, d_model: int = 64, n_heads: int = 4, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        stride = stride or patch_len // 2
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = (history - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, dropout, batch_first=True,
                                           norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(self.n_patches * d_model, n_horizons)
        # Cross-channel mixing on top of the channel-independent forecast
        self.mix = nn.Linear(n_channels, n_channels)
        nn.init.eye_(self.mix.weight)
        nn.init.zeros_(self.mix.bias)
        self.n_channels, self.n_horizons = n_channels, n_horizons

    def forward(self, x):  # x: (B, T, C)
        B, T, C = x.shape
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True) + 1e-5
        z = ((x - mean) / std).permute(0, 2, 1).reshape(B * C, T)  # RevIN-style normalisation
        patches = z.unfold(1, self.patch_len, self.stride)  # (B*C, P, L)
        h = self.encoder(self.embed(patches) + self.pos)
        y = self.head(h.flatten(1)).view(B, C, self.n_horizons).permute(0, 2, 1)  # (B, H, C)
        y = y * std + mean
        return self.mix(y)


def build_model(name: str, n_channels: int, n_horizons: int, history: int, patch_len: int = 6) -> nn.Module:
    name = name.lower()
    if name in ("gru", "lstm"):
        return RNNForecaster(n_channels, n_horizons, cell=name)
    if name == "patchtst":
        return PatchTST(n_channels, n_horizons, history, patch_len=patch_len)
    raise ValueError(f"unknown model {name}")
