from __future__ import annotations

import torch


class LSTMForecast(torch.nn.Module):
    """Encoder-decoder LSTM for future trajectory regression."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int = 30,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.pred_len = pred_len
        self.num_layers = num_layers
        self.encoder = torch.nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = torch.nn.LSTM(
            input_size=2,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.output = torch.nn.Linear(hidden_dim, 2)

    def forward(
        self,
        x: torch.Tensor,
        teacher_y: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, T, F], got {x.shape}")
        if teacher_y is not None and teacher_y.shape[:2] != (x.shape[0], self.pred_len):
            raise ValueError(
                f"teacher_y must have shape [B, {self.pred_len}, 2], got {teacher_y.shape}"
            )

        _, state = self.encoder(x)
        decoder_input = x[:, -1, 0:2]
        outputs: list[torch.Tensor] = []

        for step in range(self.pred_len):
            decoder_out, state = self.decoder(decoder_input[:, None, :], state)
            pred_step = self.output(decoder_out[:, -1, :])
            outputs.append(pred_step)

            use_teacher = (
                self.training
                and teacher_y is not None
                and teacher_forcing_ratio > 0.0
                and torch.rand((), device=x.device).item() < teacher_forcing_ratio
            )
            decoder_input = teacher_y[:, step, :] if use_teacher else pred_step

        return torch.stack(outputs, dim=1)
