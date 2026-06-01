from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Any

import torch

from src.datasets.av2_dataset import create_dataloader
from src.evaluation.metrics import ade, count_parameters, fde, min_ade, min_fde, miss_rate
from src.models.diffusion import GaussianDiffusionTrajectory
from src.models.linear import LinearExtrapolation
from src.models.lstm import LSTMForecast
from src.models.transformer import TransformerForecast
from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import save_json
from src.utils.paths import ensure_dir


def _build_linear_model(config: dict[str, Any]) -> LinearExtrapolation:
    model_config = config.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("config.model must be a mapping")
    return LinearExtrapolation(
        pred_len=int(model_config.get("pred_len", 30)),
        dt=float(model_config.get("dt", 0.1)),
    )


def _save_metrics_csv(metrics: dict[str, float | int | str], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def evaluate_linear(data_path: Path, config_path: Path, out_dir: Path) -> dict[str, float | int | str]:
    config = load_yaml_config(config_path)
    eval_config = config.get("evaluation", {})
    if not isinstance(eval_config, dict):
        raise ValueError("config.evaluation must be a mapping")

    model = _build_linear_model(config)
    dataloader = create_dataloader(
        data_path,
        batch_size=int(eval_config.get("batch_size", 128)),
        shuffle=False,
        num_workers=int(eval_config.get("num_workers", 0)),
    )

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    scenario_ids: list[str] = []
    track_ids: list[str] = []

    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            pred = model.predict(batch["X"])
            predictions.append(pred.cpu())
            targets.append(batch["Y"].cpu())
            masks.append(batch["mask_y"].cpu())
            scenario_ids.extend(batch["scenario_id"])
            track_ids.extend(batch["track_id"])
    elapsed = time.perf_counter() - start

    pred_tensor = torch.cat(predictions, dim=0)
    gt_tensor = torch.cat(targets, dim=0)
    mask_tensor = torch.cat(masks, dim=0)
    metrics: dict[str, float | int | str] = {
        "model": "linear",
        "data": str(data_path),
        "num_samples": int(pred_tensor.shape[0]),
        "ADE": float(ade(pred_tensor, gt_tensor, mask_tensor).item()),
        "FDE": float(fde(pred_tensor, gt_tensor, mask_tensor).item()),
        "Miss Rate": float(miss_rate(pred_tensor, gt_tensor, threshold=float(eval_config.get("miss_threshold", 2.0)), mask=mask_tensor).item()),
        "Latency": float(elapsed / max(int(pred_tensor.shape[0]), 1)),
        "Parameters": 0,
    }

    predictions_dir = ensure_dir(out_dir / "predictions")
    metrics_dir = ensure_dir(out_dir / "metrics")
    tables_dir = ensure_dir(out_dir / "tables")

    with (predictions_dir / "linear_val.pkl").open("wb") as f:
        pickle.dump(
            {
                "pred": pred_tensor.numpy(),
                "gt": gt_tensor.numpy(),
                "mask_y": mask_tensor.numpy(),
                "scenario_id": scenario_ids,
                "track_id": track_ids,
            },
            f,
        )
    save_json(metrics, metrics_dir / "linear_val_metrics.json")
    _save_metrics_csv(metrics, tables_dir / "linear_val_metrics.csv")
    return metrics


def _build_lstm_from_checkpoint(checkpoint: dict[str, Any]) -> LSTMForecast:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    model_config = metadata.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint metadata.model must be a mapping")
    if str(model_config.get("architecture")) != "lstm":
        raise ValueError("checkpoint does not describe an LSTM model")
    return LSTMForecast(
        input_dim=int(model_config.get("input_dim", 6)),
        pred_len=int(model_config.get("pred_len", 30)),
        hidden_dim=int(model_config.get("hidden_dim", 128)),
        num_layers=int(model_config.get("num_layers", 2)),
        dropout=float(model_config.get("dropout", 0.1)),
    )


def _build_transformer_from_checkpoint(checkpoint: dict[str, Any]) -> TransformerForecast:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    model_config = metadata.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint metadata.model must be a mapping")
    if str(model_config.get("architecture")) != "transformer":
        raise ValueError("checkpoint does not describe a Transformer model")
    return TransformerForecast(
        input_dim=int(model_config.get("input_dim", 6)),
        pred_len=int(model_config.get("pred_len", 30)),
        d_model=int(model_config.get("d_model", 128)),
        nhead=int(model_config.get("nhead", 4)),
        num_layers=int(model_config.get("num_layers", 3)),
        dim_feedforward=int(model_config.get("dim_feedforward", 256)),
        dropout=float(model_config.get("dropout", 0.1)),
    )


def _build_diffusion_from_checkpoint(checkpoint: dict[str, Any]) -> GaussianDiffusionTrajectory:
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata must be a mapping")
    model_config = metadata.get("model", {})
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint metadata.model must be a mapping")
    if str(model_config.get("architecture")) != "diffusion_direct":
        raise ValueError("checkpoint does not describe a direct diffusion model")
    return GaussianDiffusionTrajectory(
        input_dim=int(model_config.get("input_dim", 6)),
        pred_len=int(model_config.get("pred_len", 30)),
        trajectory_dim=int(model_config.get("trajectory_dim", 60)),
        cond_dim=int(model_config.get("cond_dim", 128)),
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        diffusion_steps=int(model_config.get("diffusion_steps", 100)),
        sampling_steps=int(model_config.get("sampling_steps", 50)),
        beta_start=float(model_config.get("beta_start", 0.0001)),
        beta_end=float(model_config.get("beta_end", 0.02)),
        num_samples=int(model_config.get("num_samples", 6)),
    )


def evaluate_lstm(
    data_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_lstm_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloader = create_dataloader(data_path, batch_size=batch_size, shuffle=False)

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(device))
            predictions.append(pred.cpu())
            targets.append(batch["Y"].cpu())
            masks.append(batch["mask_y"].cpu())
    elapsed = time.perf_counter() - start

    pred_tensor = torch.cat(predictions, dim=0)
    gt_tensor = torch.cat(targets, dim=0)
    mask_tensor = torch.cat(masks, dim=0)
    metrics: dict[str, float | int | str] = {
        "model": "lstm",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "num_samples": int(pred_tensor.shape[0]),
        "ADE": float(ade(pred_tensor, gt_tensor, mask_tensor).item()),
        "FDE": float(fde(pred_tensor, gt_tensor, mask_tensor).item()),
        "Miss Rate": float(miss_rate(pred_tensor, gt_tensor, mask=mask_tensor).item()),
        "Latency": float(elapsed / max(int(pred_tensor.shape[0]), 1)),
        "Parameters": count_parameters(model),
    }

    metrics_dir = ensure_dir(out_dir / "metrics")
    tables_dir = ensure_dir(out_dir / "tables")
    save_json(metrics, metrics_dir / "lstm_eval_metrics.json")
    _save_metrics_csv(metrics, tables_dir / "lstm_eval_metrics.csv")
    return metrics


def evaluate_transformer(
    data_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_transformer_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloader = create_dataloader(data_path, batch_size=batch_size, shuffle=False)

    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            pred = model(batch["X"].to(device))
            predictions.append(pred.cpu())
            targets.append(batch["Y"].cpu())
            masks.append(batch["mask_y"].cpu())
    elapsed = time.perf_counter() - start

    pred_tensor = torch.cat(predictions, dim=0)
    gt_tensor = torch.cat(targets, dim=0)
    mask_tensor = torch.cat(masks, dim=0)
    metrics: dict[str, float | int | str] = {
        "model": "transformer",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "num_samples": int(pred_tensor.shape[0]),
        "ADE": float(ade(pred_tensor, gt_tensor, mask_tensor).item()),
        "FDE": float(fde(pred_tensor, gt_tensor, mask_tensor).item()),
        "Miss Rate": float(miss_rate(pred_tensor, gt_tensor, mask=mask_tensor).item()),
        "Latency": float(elapsed / max(int(pred_tensor.shape[0]), 1)),
        "Parameters": count_parameters(model),
    }

    metrics_dir = ensure_dir(out_dir / "metrics")
    tables_dir = ensure_dir(out_dir / "tables")
    save_json(metrics, metrics_dir / "transformer_eval_metrics.json")
    _save_metrics_csv(metrics, tables_dir / "transformer_eval_metrics.csv")
    return metrics


def evaluate_diffusion_direct(
    data_path: Path,
    checkpoint_path: Path,
    out_dir: Path,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, float | int | str]:
    device = device or get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = _build_diffusion_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    dataloader = create_dataloader(data_path, batch_size=batch_size, shuffle=False)

    sample_batches: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    start = time.perf_counter()
    with torch.no_grad():
        for batch in dataloader:
            samples = model.sample(batch["X"].to(device))
            sample_batches.append(samples.cpu())
            targets.append(batch["Y"].cpu())
            masks.append(batch["mask_y"].cpu())
    elapsed = time.perf_counter() - start

    samples_tensor = torch.cat(sample_batches, dim=0)
    pred_tensor = samples_tensor[:, 0]
    gt_tensor = torch.cat(targets, dim=0)
    mask_tensor = torch.cat(masks, dim=0)
    metrics: dict[str, float | int | str] = {
        "model": "diffusion_direct",
        "checkpoint": str(checkpoint_path),
        "data": str(data_path),
        "num_samples": int(pred_tensor.shape[0]),
        "num_prediction_samples": int(samples_tensor.shape[1]),
        "ADE": float(ade(pred_tensor, gt_tensor, mask_tensor).item()),
        "FDE": float(fde(pred_tensor, gt_tensor, mask_tensor).item()),
        "minADE": float(min_ade(samples_tensor, gt_tensor, mask_tensor).item()),
        "minFDE": float(min_fde(samples_tensor, gt_tensor).item()),
        "Miss Rate": float(miss_rate(pred_tensor, gt_tensor, mask=mask_tensor).item()),
        "Latency": float(elapsed / max(int(pred_tensor.shape[0]), 1)),
        "Parameters": count_parameters(model),
    }

    metrics_dir = ensure_dir(out_dir / "metrics")
    tables_dir = ensure_dir(out_dir / "tables")
    save_json(metrics, metrics_dir / "diffusion_direct_eval_metrics.json")
    _save_metrics_csv(metrics, tables_dir / "diffusion_direct_eval_metrics.csv")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trajectory forecasting models.")
    parser.add_argument("--model", choices=["linear", "lstm", "transformer", "diffusion_direct"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "linear":
        if args.config is None:
            raise ValueError("--config is required for linear evaluation")
        metrics = evaluate_linear(args.data, args.config, args.out_dir)
    elif args.model == "lstm":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for lstm evaluation")
        metrics = evaluate_lstm(args.data, args.checkpoint, args.out_dir, batch_size=args.batch_size)
    elif args.model == "transformer":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for transformer evaluation")
        metrics = evaluate_transformer(args.data, args.checkpoint, args.out_dir, batch_size=args.batch_size)
    elif args.model == "diffusion_direct":
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required for diffusion_direct evaluation")
        metrics = evaluate_diffusion_direct(args.data, args.checkpoint, args.out_dir, batch_size=args.batch_size)
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
