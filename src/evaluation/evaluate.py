from __future__ import annotations

import argparse
import csv
import pickle
import time
from pathlib import Path
from typing import Any

import torch

from src.datasets.av2_dataset import create_dataloader
from src.evaluation.metrics import ade, fde, miss_rate
from src.models.linear import LinearExtrapolation
from src.utils.config import load_yaml_config
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
        "FDE": float(fde(pred_tensor, gt_tensor).item()),
        "Miss Rate": float(miss_rate(pred_tensor, gt_tensor, threshold=float(eval_config.get("miss_threshold", 2.0))).item()),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trajectory forecasting models.")
    parser.add_argument("--model", choices=["linear"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model != "linear":
        raise ValueError(f"Unsupported model: {args.model}")

    metrics = evaluate_linear(args.data, args.config, args.out_dir)
    for key, value in metrics.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
