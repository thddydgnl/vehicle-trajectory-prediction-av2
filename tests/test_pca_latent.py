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


def test_pca_trajectory_codec_normalizes_latents_from_train_split() -> None:
    rng = np.random.default_rng(42)
    Y = rng.normal(size=(128, 30, 2)).astype(np.float32)
    codec = PCATrajectoryCodec(n_components=8)
    codec.fit(Y)

    z = codec.transform(Y)
    recovered = codec.inverse_transform(z)

    assert np.allclose(z.mean(axis=0), 0.0, atol=1e-5)
    assert np.allclose(z.std(axis=0), 1.0, atol=1e-5)
    assert recovered.shape == Y.shape


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


def test_pca_latent_diffusion_encode_decode_uses_codec_latent_scale(tmp_path: Path) -> None:
    rng = np.random.default_rng(123)
    Y_np = rng.normal(size=(96, 30, 2)).astype(np.float32)
    codec = PCATrajectoryCodec(n_components=6)
    codec.fit(Y_np)
    codec_path = tmp_path / "codec.pkl"
    codec.save(codec_path)
    model = PCALatentDiffusionTrajectory(
        input_dim=6,
        codec_path=codec_path,
        pred_len=30,
        latent_dim=6,
        cond_dim=16,
        hidden_dim=32,
        diffusion_steps=8,
        sampling_steps=4,
        num_samples=3,
    )

    z = model.encode_y(torch.from_numpy(Y_np))

    assert torch.allclose(z.mean(dim=0), torch.zeros(6), atol=1e-5)
    assert torch.allclose(z.std(dim=0, unbiased=False), torch.ones(6), atol=1e-5)
