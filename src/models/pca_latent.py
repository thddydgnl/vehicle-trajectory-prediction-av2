from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from src.models.diffusion import DiffusionDenoiser, TrajectoryConditionEncoder


class PCATrajectoryCodec:
    """PCA codec for flattening future trajectories into a low-dimensional latent space."""

    def __init__(self, n_components: int = 12):
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.pred_len: int | None = None

    def fit(self, Y_train: np.ndarray) -> None:
        if Y_train.ndim != 3 or Y_train.shape[-1] != 2:
            raise ValueError(f"Expected Y_train shape [N, T, 2], got {Y_train.shape}")
        self.pred_len = int(Y_train.shape[1])
        self.pca.fit(Y_train.reshape(Y_train.shape[0], -1))

    def transform(self, Y: np.ndarray) -> np.ndarray:
        if self.pred_len is None:
            raise RuntimeError("PCATrajectoryCodec must be fit before transform")
        if Y.ndim != 3 or Y.shape[-1] != 2 or Y.shape[1] != self.pred_len:
            raise ValueError(f"Expected Y shape [N, {self.pred_len}, 2], got {Y.shape}")
        return self.pca.transform(Y.reshape(Y.shape[0], -1)).astype(np.float32)

    def inverse_transform(self, z: np.ndarray) -> np.ndarray:
        if self.pred_len is None:
            raise RuntimeError("PCATrajectoryCodec must be fit before inverse_transform")
        Y_flat = self.pca.inverse_transform(z)
        return Y_flat.reshape(z.shape[0], self.pred_len, 2).astype(np.float32)

    def save(self, path: str | Path) -> None:
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "PCATrajectoryCodec":
        codec = joblib.load(path)
        if not isinstance(codec, cls):
            raise TypeError(f"Expected PCATrajectoryCodec at {path}, got {type(codec)}")
        return codec


class PCALatentDiffusionTrajectory(torch.nn.Module):
    """Conditional diffusion model that denoises PCA latents and decodes trajectories."""

    def __init__(
        self,
        input_dim: int,
        codec_path: str | Path,
        pred_len: int = 30,
        latent_dim: int = 12,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        diffusion_steps: int = 100,
        sampling_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_samples: int = 6,
    ):
        super().__init__()
        codec = PCATrajectoryCodec.load(codec_path)
        if codec.pred_len != pred_len:
            raise ValueError(f"Codec pred_len {codec.pred_len} does not match model pred_len {pred_len}")
        if codec.n_components != latent_dim:
            raise ValueError(f"Codec components {codec.n_components} does not match latent_dim {latent_dim}")

        self.pred_len = pred_len
        self.latent_dim = latent_dim
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        self.num_samples = num_samples
        self.condition_encoder = TrajectoryConditionEncoder(input_dim, cond_dim=cond_dim, hidden_dim=cond_dim)
        self.denoiser = DiffusionDenoiser(latent_dim, cond_dim=cond_dim, hidden_dim=hidden_dim, time_dim=cond_dim)

        self.register_buffer("pca_mean", torch.as_tensor(codec.pca.mean_, dtype=torch.float32))
        self.register_buffer("pca_components", torch.as_tensor(codec.pca.components_, dtype=torch.float32))
        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def encode_y(self, Y: torch.Tensor) -> torch.Tensor:
        y_flat = Y.reshape(Y.shape[0], -1)
        return (y_flat - self.pca_mean) @ self.pca_components.T

    def decode_z(self, z: torch.Tensor) -> torch.Tensor:
        y_flat = z @ self.pca_components + self.pca_mean
        return y_flat.view(z.shape[0], self.pred_len, 2)

    def q_sample(self, z0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self.sqrt_alpha_bars[t].view(-1, 1) * z0 + self.sqrt_one_minus_alpha_bars[t].view(-1, 1) * noise

    def training_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        z0 = self.encode_y(Y)
        noise = torch.randn_like(z0)
        t = torch.randint(0, self.diffusion_steps, (Y.shape[0],), device=Y.device)
        zt = self.q_sample(z0, t, noise)
        cond = self.condition_encoder(X)
        noise_hat = self.denoiser(zt, t, cond)
        return F.mse_loss(noise_hat, noise)

    @torch.no_grad()
    def sample(self, X: torch.Tensor, num_samples: int | None = None, num_steps: int | None = None) -> torch.Tensor:
        batch_size = X.shape[0]
        sample_count = int(num_samples or self.num_samples)
        step_count = int(num_steps or self.sampling_steps)
        step_count = max(1, min(step_count, self.diffusion_steps))
        cond = self.condition_encoder(X)
        cond = cond[:, None, :].expand(batch_size, sample_count, -1).reshape(batch_size * sample_count, -1)
        zt = torch.randn(batch_size * sample_count, self.latent_dim, device=X.device)
        timesteps = torch.linspace(self.diffusion_steps - 1, 0, step_count, device=X.device).long()

        for t_value in timesteps:
            t = torch.full((zt.shape[0],), int(t_value.item()), device=X.device, dtype=torch.long)
            noise_hat = self.denoiser(zt, t, cond)
            alpha_bar = self.alpha_bars[t].view(-1, 1)
            z0 = (zt - self.sqrt_one_minus_alpha_bars[t].view(-1, 1) * noise_hat) / torch.sqrt(alpha_bar).clamp_min(1e-6)
            if int(t_value.item()) > 0:
                prev_alpha_bar = self.alpha_bars[t - 1].view(-1, 1)
                zt = torch.sqrt(prev_alpha_bar) * z0 + torch.sqrt(1.0 - prev_alpha_bar) * noise_hat
            else:
                zt = z0

        decoded = self.decode_z(zt)
        return decoded.view(batch_size, sample_count, self.pred_len, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sample(X, num_samples=1)[:, 0]
