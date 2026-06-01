from __future__ import annotations

import torch


class LinearExtrapolation:
    """Constant-velocity trajectory extrapolation baseline."""

    def __init__(self, pred_len: int = 30, dt: float = 0.1):
        if pred_len <= 0:
            raise ValueError("pred_len must be positive")
        if dt <= 0:
            raise ValueError("dt must be positive")
        self.pred_len = pred_len
        self.dt = dt

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if X.ndim != 3 or X.shape[-1] < 4:
            raise ValueError(f"Expected X shape [B, T, F>=4], got {X.shape}")

        p_last = X[:, -1, 0:2]
        v_last = X[:, -1, 2:4]
        steps = torch.arange(
            1,
            self.pred_len + 1,
            device=X.device,
            dtype=X.dtype,
        ).view(1, self.pred_len, 1)
        return p_last[:, None, :] + v_last[:, None, :] * (steps * self.dt)
