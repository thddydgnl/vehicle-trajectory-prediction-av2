from __future__ import annotations

import math

import torch
import torch.nn.functional as F


class SinusoidalTimeEmbedding(torch.nn.Module):
    """Sinusoidal timestep embedding."""

    def __init__(self, dim: int):
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=t.device, dtype=torch.float32) * -scale)
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class TrajectoryConditionEncoder(torch.nn.Module):
    """Encode observed trajectory features into a conditioning vector."""

    def __init__(self, input_dim: int, cond_dim: int = 128, hidden_dim: int = 128):
        super().__init__()
        self.encoder = torch.nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, cond_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B, T, F], got {x.shape}")
        _, hidden = self.encoder(x)
        return self.proj(hidden[-1])


class DiffusionDenoiser(torch.nn.Module):
    """MLP denoiser for flattened future trajectories."""

    def __init__(self, trajectory_dim: int, cond_dim: int = 128, hidden_dim: int = 256, time_dim: int = 128):
        super().__init__()
        self.time_embedding = SinusoidalTimeEmbedding(time_dim)
        self.net = torch.nn.Sequential(
            torch.nn.Linear(trajectory_dim + cond_dim + time_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, trajectory_dim),
        )

    def forward(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embedding(t)
        return self.net(torch.cat([xt, cond, time_emb], dim=-1))


class GaussianDiffusionTrajectory(torch.nn.Module):
    """Conditional Gaussian diffusion model for direct trajectory generation."""

    def __init__(
        self,
        input_dim: int,
        pred_len: int = 30,
        trajectory_dim: int = 60,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        diffusion_steps: int = 100,
        sampling_steps: int = 50,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        num_samples: int = 6,
    ):
        super().__init__()
        if trajectory_dim != pred_len * 2:
            raise ValueError("trajectory_dim must equal pred_len * 2")
        self.pred_len = pred_len
        self.trajectory_dim = trajectory_dim
        self.diffusion_steps = diffusion_steps
        self.sampling_steps = sampling_steps
        self.num_samples = num_samples
        self.condition_encoder = TrajectoryConditionEncoder(input_dim, cond_dim=cond_dim, hidden_dim=cond_dim)
        self.denoiser = DiffusionDenoiser(trajectory_dim, cond_dim=cond_dim, hidden_dim=hidden_dim, time_dim=cond_dim)

        betas = torch.linspace(beta_start, beta_end, diffusion_steps, dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar = self.sqrt_alpha_bars[t].view(-1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bars[t].view(-1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def predict_noise(self, xt: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.denoiser(xt, t, cond)

    def training_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        x0 = Y.reshape(Y.shape[0], self.trajectory_dim)
        noise = torch.randn_like(x0)
        t = torch.randint(0, self.diffusion_steps, (Y.shape[0],), device=Y.device)
        xt = self.q_sample(x0, t, noise)
        cond = self.condition_encoder(X)
        noise_hat = self.predict_noise(xt, t, cond)
        return F.mse_loss(noise_hat, noise)

    @torch.no_grad()
    def sample(self, X: torch.Tensor, num_samples: int | None = None, num_steps: int | None = None) -> torch.Tensor:
        batch_size = X.shape[0]
        sample_count = int(num_samples or self.num_samples)
        step_count = int(num_steps or self.sampling_steps)
        step_count = max(1, min(step_count, self.diffusion_steps))
        cond = self.condition_encoder(X)
        cond = cond[:, None, :].expand(batch_size, sample_count, -1).reshape(batch_size * sample_count, -1)
        xt = torch.randn(batch_size * sample_count, self.trajectory_dim, device=X.device)

        timesteps = torch.linspace(self.diffusion_steps - 1, 0, step_count, device=X.device).long()
        for t_value in timesteps:
            t = torch.full((xt.shape[0],), int(t_value.item()), device=X.device, dtype=torch.long)
            noise_hat = self.predict_noise(xt, t, cond)
            alpha_bar = self.alpha_bars[t].view(-1, 1)
            xt = (xt - self.sqrt_one_minus_alpha_bars[t].view(-1, 1) * noise_hat) / torch.sqrt(alpha_bar).clamp_min(1e-6)
            if int(t_value.item()) > 0:
                prev_alpha_bar = self.alpha_bars[t - 1].view(-1, 1)
                xt = torch.sqrt(prev_alpha_bar) * xt + torch.sqrt(1.0 - prev_alpha_bar) * noise_hat

        return xt.view(batch_size, sample_count, self.pred_len, 2)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.sample(X, num_samples=1)[:, 0]
