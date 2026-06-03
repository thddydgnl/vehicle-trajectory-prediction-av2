import torch

from src.models.diffusion import (
    DiffusionDenoiser,
    GaussianDiffusionTrajectory,
    SinusoidalTimeEmbedding,
    TrajectoryConditionEncoder,
    reverse_diffusion_timesteps,
)


def test_sinusoidal_time_embedding_shape() -> None:
    module = SinusoidalTimeEmbedding(dim=15)
    t = torch.tensor([0, 1, 2])

    assert module(t).shape == (3, 15)


def test_condition_encoder_shape() -> None:
    encoder = TrajectoryConditionEncoder(input_dim=6, cond_dim=16, hidden_dim=16)
    x = torch.randn(4, 50, 6)

    assert encoder(x).shape == (4, 16)


def test_denoiser_shape() -> None:
    denoiser = DiffusionDenoiser(trajectory_dim=60, cond_dim=16, hidden_dim=32, time_dim=16)
    xt = torch.randn(4, 60)
    t = torch.randint(0, 10, (4,))
    cond = torch.randn(4, 16)

    assert denoiser(xt, t, cond).shape == (4, 60)


def test_gaussian_diffusion_training_loss_is_scalar() -> None:
    model = GaussianDiffusionTrajectory(
        input_dim=6,
        pred_len=30,
        trajectory_dim=60,
        cond_dim=16,
        hidden_dim=32,
        diffusion_steps=8,
        sampling_steps=4,
        num_samples=3,
    )
    X = torch.randn(4, 50, 6)
    Y = torch.randn(4, 30, 2)

    loss = model.training_loss(X, Y)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_gaussian_diffusion_sample_shape() -> None:
    model = GaussianDiffusionTrajectory(
        input_dim=6,
        pred_len=30,
        trajectory_dim=60,
        cond_dim=16,
        hidden_dim=32,
        diffusion_steps=8,
        sampling_steps=4,
        num_samples=3,
    )
    X = torch.randn(2, 50, 6)

    samples = model.sample(X)

    assert samples.shape == (2, 3, 30, 2)


def test_reverse_diffusion_timesteps_use_actual_skipped_steps() -> None:
    timesteps = reverse_diffusion_timesteps(diffusion_steps=10, sampling_steps=4)

    assert timesteps.tolist() == [9, 6, 3, 0]
