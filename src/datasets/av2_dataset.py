from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


REQUIRED_KEYS = (
    "X",
    "Y",
    "mask_x",
    "mask_y",
    "object_type",
    "scenario_id",
    "track_id",
    "origin",
    "theta",
)


class TrajectoryDataset(Dataset):
    """PyTorch Dataset for processed trajectory `.npz` files."""

    def __init__(self, npz_path: str | Path):
        self.npz_path = Path(npz_path)
        if not self.npz_path.exists():
            raise FileNotFoundError(f"Processed trajectory file not found: {self.npz_path}")

        with np.load(self.npz_path, allow_pickle=True) as data:
            missing = [key for key in REQUIRED_KEYS if key not in data.files]
            if missing:
                raise KeyError(f"Missing required keys in {self.npz_path}: {missing}")
            self.data = {key: data[key] for key in data.files}

        self._length = int(self.data["X"].shape[0])
        self._validate_lengths()

    def _validate_lengths(self) -> None:
        for key in REQUIRED_KEYS:
            if self.data[key].shape[0] != self._length:
                raise ValueError(
                    f"Key {key} has length {self.data[key].shape[0]}, expected {self._length}"
                )

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "X": torch.as_tensor(self.data["X"][idx], dtype=torch.float32),
            "Y": torch.as_tensor(self.data["Y"][idx], dtype=torch.float32),
            "mask_x": torch.as_tensor(self.data["mask_x"][idx], dtype=torch.bool),
            "mask_y": torch.as_tensor(self.data["mask_y"][idx], dtype=torch.bool),
            "object_type": torch.as_tensor(self.data["object_type"][idx], dtype=torch.long),
            "scenario_id": str(self.data["scenario_id"][idx]),
            "track_id": str(self.data["track_id"][idx]),
            "origin": torch.as_tensor(self.data["origin"][idx], dtype=torch.float32),
            "theta": torch.as_tensor(self.data["theta"][idx], dtype=torch.float32),
        }


def create_dataloader(
    npz_path: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for a processed trajectory `.npz` file."""
    dataset = TrajectoryDataset(npz_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
