from __future__ import annotations

import csv
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader

from src.evaluation.metrics import ade, fde
from src.training.losses import combined_trajectory_loss, trajectory_mse_loss, trajectory_smooth_l1_loss
from src.utils.io import save_json
from src.utils.paths import ensure_dir


LossFn = Callable[[torch.Tensor, torch.Tensor, torch.Tensor | None], torch.Tensor]


@dataclass(frozen=True)
class TrainerConfig:
    model_name: str
    epochs: int = 1
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip: float | None = 1.0
    early_stopping_patience: int = 5
    loss: str = "smooth_l1"
    endpoint_weight: float = 0.0
    out_dir: Path = Path("outputs")
    metadata: dict[str, Any] = field(default_factory=dict)


def _select_loss(config: TrainerConfig) -> LossFn:
    if config.loss == "mse":
        return trajectory_mse_loss
    if config.loss == "smooth_l1":
        return trajectory_smooth_l1_loss
    if config.loss == "combined":
        return lambda pred, gt, mask: combined_trajectory_loss(
            pred,
            gt,
            mask,
            endpoint_weight=config.endpoint_weight,
        )
    raise ValueError(f"Unsupported loss: {config.loss}")


class Trainer:
    """Common supervised training loop for trajectory regressors."""

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.loss_fn = _select_loss(config)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.checkpoint_dir = ensure_dir(config.out_dir / "checkpoints")
        self.logs_dir = ensure_dir(config.out_dir / "logs")
        self.metrics_dir = ensure_dir(config.out_dir / "metrics")

    def fit(self) -> dict[str, float | int | str]:
        best_val_ade = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        rows: list[dict[str, float | int]] = []

        for epoch in range(1, self.config.epochs + 1):
            train_loss = self._train_one_epoch()
            val_metrics = self.validate()
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "val_ADE": val_metrics["ADE"],
                "val_FDE": val_metrics["FDE"],
            }
            rows.append(row)
            self._save_checkpoint("last", epoch, val_metrics)

            if val_metrics["ADE"] < best_val_ade:
                best_val_ade = val_metrics["ADE"]
                best_epoch = epoch
                epochs_without_improvement = 0
                self._save_checkpoint("best", epoch, val_metrics)
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.early_stopping_patience:
                break

        self._write_log(rows)
        final_metrics: dict[str, float | int | str] = {
            "model": self.config.model_name,
            "best_epoch": best_epoch,
            "ADE": best_val_ade,
            "FDE": rows[best_epoch - 1]["val_FDE"] if best_epoch else float("nan"),
            "epochs_ran": len(rows),
            "metadata": self.config.metadata,
        }
        save_json(final_metrics, self.metrics_dir / f"{self.config.model_name}_val_metrics.json")
        return final_metrics

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        total_items = 0
        for batch in self.train_loader:
            X = batch["X"].to(self.device)
            Y = batch["Y"].to(self.device)
            mask_y = batch["mask_y"].to(self.device)

            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(X)
            loss = self.loss_fn(pred, Y, mask_y)
            loss.backward()
            if self.config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()

            valid_steps = int(mask_y.sum().detach().cpu().item())
            total_loss += float(loss.detach().cpu().item()) * valid_steps
            total_items += valid_steps
        return total_loss / max(total_items, 1)

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        total_items = 0
        preds: list[torch.Tensor] = []
        gts: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        for batch in self.val_loader:
            X = batch["X"].to(self.device)
            Y = batch["Y"].to(self.device)
            mask_y = batch["mask_y"].to(self.device)
            pred = self.model(X)
            loss = self.loss_fn(pred, Y, mask_y)

            valid_steps = int(mask_y.sum().detach().cpu().item())
            total_loss += float(loss.detach().cpu().item()) * valid_steps
            total_items += valid_steps
            preds.append(pred.cpu())
            gts.append(Y.cpu())
            masks.append(mask_y.cpu())

        pred_tensor = torch.cat(preds, dim=0)
        gt_tensor = torch.cat(gts, dim=0)
        mask_tensor = torch.cat(masks, dim=0)
        return {
            "val_loss": total_loss / max(total_items, 1),
            "ADE": float(ade(pred_tensor, gt_tensor, mask_tensor).item()),
            "FDE": float(fde(pred_tensor, gt_tensor, mask_tensor).item()),
        }

    def _save_checkpoint(self, kind: str, epoch: int, metrics: dict[str, float]) -> None:
        path = self.checkpoint_dir / f"{kind}_{self.config.model_name}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "model_name": self.config.model_name,
                "trainer_config": {
                    "model_name": self.config.model_name,
                    "epochs": self.config.epochs,
                    "learning_rate": self.config.learning_rate,
                    "weight_decay": self.config.weight_decay,
                    "gradient_clip": self.config.gradient_clip,
                    "early_stopping_patience": self.config.early_stopping_patience,
                    "loss": self.config.loss,
                    "endpoint_weight": self.config.endpoint_weight,
                    "out_dir": str(self.config.out_dir),
                },
                "metadata": self.config.metadata,
            },
            path,
        )

    def _write_log(self, rows: list[dict[str, float | int]]) -> None:
        path = self.logs_dir / f"{self.config.model_name}_train_log.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_ADE", "val_FDE"])
            writer.writeheader()
            writer.writerows(rows)
