from pathlib import Path

import numpy as np
import torch

from src.models.pca_latent import PCALatentDiffusionTrajectory, PCATrajectoryCodec


def test_pca_trajectory_codec_round_trip_shape(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    Y = rng.normal(size=(32, 30, 2)).astype(np.float32)
    codec = PCATrajectoryCodec(n_components=6)
    codec.fit(Y)

    z = codec.transform(Y)
    recovered = codec.inverse_transform(z)

    assert z.shape == (32, 6)
    assert recovered.shape == (32, 30, 2)


def test_pca_trajectory_codec_save_load(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    Y = rng.normal(size=(32, 30, 2)).astype(np.float32)
    codec = PCATrajectoryCodec(n_components=6)
    codec.fit(Y)
    path = tmp_path / "codec.pkl"
    codec.save(path)

    loaded = PCATrajectoryCodec.load(path)

    assert loaded.n_components == 6
    assert loaded.pred_len == 30


def test_pca_latent_diffusion_training_loss_and_sample_shape(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    Y_np = rng.normal(size=(64, 30, 2)).astype(np.float32)
    codec = PCATrajectoryCodec(n_components=8)
    codec.fit(Y_np)
    codec_path = tmp_path / "codec.pkl"
    codec.save(codec_path)

    model = PCALatentDiffusionTrajectory(
        input_dim=6,
        codec_path=codec_path,
        pred_len=30,
        latent_dim=8,
        cond_dim=16,
        hidden_dim=32,
        diffusion_steps=8,
        sampling_steps=4,
        num_samples=3,
    )
    X = torch.randn(4, 50, 6)
    Y = torch.from_numpy(Y_np[:4])

    loss = model.training_loss(X, Y)
    samples = model.sample(X)

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert samples.shape == (4, 3, 30, 2)
