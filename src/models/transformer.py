from __future__ import annotations

import math

import torch


class PositionalEncoding(torch.nn.Module):
    """Sinusoidal positional encoding for sequence inputs."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, T, D], got {x.shape}")
        return x + self.pe[:, : x.shape[1], :]


class TransformerForecast(torch.nn.Module):
    """Transformer encoder trajectory regressor."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int = 30,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.pred_len = pred_len
        self.input_proj = torch.nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = torch.nn.Sequential(
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, pred_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, T, F], got {x.shape}")
        encoded = self.input_proj(x)
        encoded = self.positional_encoding(encoded)
        encoded = self.encoder(encoded)
        pooled = encoded[:, -1, :]
        return self.output(pooled).view(x.shape[0], self.pred_len, 2)
