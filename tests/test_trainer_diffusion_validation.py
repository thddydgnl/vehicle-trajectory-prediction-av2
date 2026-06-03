import torch
from torch.utils.data import DataLoader, Dataset

from src.training.trainer import Trainer, TrainerConfig


class _TinyTrajectoryDataset(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "X": torch.zeros(50, 6),
            "Y": torch.zeros(30, 2),
            "mask_y": torch.ones(30, dtype=torch.bool),
        }


class _ToyDiffusion(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))

    def training_loss(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.weight.square() + 0.1

    def sample(
        self,
        X: torch.Tensor,
        num_samples: int | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        sample_count = int(num_samples or 1)
        samples = torch.zeros(X.shape[0], sample_count, 30, 2, device=X.device)
        samples[:, 0, :, 0] = 5.0
        return samples


def test_trainer_validation_records_diffusion_min_metrics(tmp_path) -> None:
    loader = DataLoader(_TinyTrajectoryDataset(), batch_size=2)
    config = TrainerConfig(
        model_name="toy_diffusion",
        selection_metric="minADE",
        validation_num_samples=3,
        validation_seed=7,
        out_dir=tmp_path,
    )
    trainer = Trainer(_ToyDiffusion(), loader, loader, config, torch.device("cpu"))

    metrics = trainer.validate()

    assert metrics["ADE"] == 5.0
    assert metrics["minADE"] == 0.0
    assert metrics["minFDE"] == 0.0
    assert metrics["Sample_Diversity"] > 0.0
