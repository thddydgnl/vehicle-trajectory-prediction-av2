from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.datasets.av2_dataset import TrajectoryDataset, create_dataloader
from src.models.diffusion import GaussianDiffusionTrajectory
from src.models.lstm import LSTMForecast
from src.models.transformer import TransformerForecast
from src.training.trainer import Trainer, TrainerConfig
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.seed import set_seed


class TinyTrajectoryRegressor(torch.nn.Module):
    """Small trainable smoke model for validating the common training loop."""

    def __init__(self, obs_len: int, input_dim: int, pred_len: int, hidden_dim: int = 64):
        super().__init__()
        self.pred_len = pred_len
        self.net = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(obs_len * input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, pred_len * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(x.shape[0], self.pred_len, 2)


def _resolve_device(config: dict[str, Any]) -> torch.device:
    device_name = str(config.get("device", "auto"))
    if device_name == "auto":
        return get_device()
    return torch.device(device_name)


def _build_model(config: dict[str, Any], data_path: Path) -> torch.nn.Module:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("config.model must be a mapping")

    dataset = TrajectoryDataset(data_path)
    sample = dataset[0]["X"]
    obs_len = int(sample.shape[0])
    input_dim = int(sample.shape[1])
    pred_len = int(model_config.get("pred_len", 30))
    hidden_dim = int(model_config.get("hidden_dim", 64))
    architecture = str(model_config.get("architecture", "tiny_regressor"))
    if architecture == "tiny_regressor":
        return TinyTrajectoryRegressor(
            obs_len=obs_len,
            input_dim=input_dim,
            pred_len=pred_len,
            hidden_dim=hidden_dim,
        )
    if architecture == "lstm":
        return LSTMForecast(
            input_dim=int(model_config.get("input_dim", input_dim)),
            pred_len=pred_len,
            hidden_dim=hidden_dim,
            num_layers=int(model_config.get("num_layers", 2)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if architecture == "transformer":
        return TransformerForecast(
            input_dim=int(model_config.get("input_dim", input_dim)),
            pred_len=pred_len,
            d_model=int(model_config.get("d_model", 128)),
            nhead=int(model_config.get("nhead", 4)),
            num_layers=int(model_config.get("num_layers", 3)),
            dim_feedforward=int(model_config.get("dim_feedforward", 256)),
            dropout=float(model_config.get("dropout", 0.1)),
        )
    if architecture == "diffusion_direct":
        return GaussianDiffusionTrajectory(
            input_dim=int(model_config.get("input_dim", input_dim)),
            pred_len=pred_len,
            trajectory_dim=int(model_config.get("trajectory_dim", pred_len * 2)),
            cond_dim=int(model_config.get("cond_dim", 128)),
            hidden_dim=int(model_config.get("hidden_dim", 256)),
            diffusion_steps=int(model_config.get("diffusion_steps", 100)),
            sampling_steps=int(model_config.get("sampling_steps", 50)),
            beta_start=float(model_config.get("beta_start", 0.0001)),
            beta_end=float(model_config.get("beta_end", 0.02)),
            num_samples=int(model_config.get("num_samples", 6)),
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def train_from_config(config_path: Path, data_path: Path, val_data_path: Path, max_epochs: int | None = None) -> dict[str, float | int | str]:
    config = load_yaml_config(config_path)
    training_config = config.get("training", {})
    model_config = config.get("model", {})
    if not isinstance(training_config, dict) or not isinstance(model_config, dict):
        raise ValueError("config must contain model and training mappings")

    seed = int(training_config.get("seed", 42))
    set_seed(seed)
    device = _resolve_device(training_config)
    model = _build_model(config, data_path)
    batch_size = int(training_config.get("batch_size", 64))
    num_workers = int(training_config.get("num_workers", 0))
    train_loader = create_dataloader(data_path, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = create_dataloader(val_data_path, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    epochs = int(max_epochs if max_epochs is not None else training_config.get("epochs", 1))
    trainer_config = TrainerConfig(
        model_name=str(model_config.get("name", "tiny_regressor")),
        epochs=epochs,
        learning_rate=float(training_config.get("learning_rate", 1e-3)),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
        gradient_clip=float(training_config["gradient_clip"]) if "gradient_clip" in training_config else None,
        early_stopping_patience=int(training_config.get("early_stopping_patience", 5)),
        loss=str(training_config.get("loss", "smooth_l1")),
        endpoint_weight=float(training_config.get("endpoint_weight", 0.0)),
        out_dir=Path(training_config.get("out_dir", "outputs")),
        metadata={
            "config_path": str(config_path),
            "train_data": str(data_path),
            "val_data": str(val_data_path),
            "seed": seed,
            "device": str(device),
            "model": model_config,
            "training": training_config,
        },
    )
    return Trainer(model, train_loader, val_loader, trainer_config, device).fit()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a trajectory forecasting model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val_data", type=Path, required=True)
    parser.add_argument("--max_epochs", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = train_from_config(args.config, args.data, args.val_data, args.max_epochs)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
