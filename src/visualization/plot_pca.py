from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from src.visualization.common import load_processed_npz, save_figure


OBJECT_LABELS = {
    0: "vehicle",
    1: "pedestrian",
}


def plot_pca_trajectory_space(
    data_path: str | Path,
    out_dir: str | Path,
    max_points: int = 2000,
) -> Path:
    data = load_processed_npz(data_path)
    full_horizon = data["mask_y"].all(axis=1)
    Y = data["Y"][full_horizon]
    object_type = data["object_type"][full_horizon]
    if Y.shape[0] < 2:
        raise ValueError("At least two full-horizon trajectories are required for PCA plotting")

    if Y.shape[0] > max_points:
        indices = np.linspace(0, Y.shape[0] - 1, max_points).astype(int)
        Y = Y[indices]
        object_type = object_type[indices]

    flat = Y.reshape(Y.shape[0], -1)
    coords = PCA(n_components=2, random_state=42).fit_transform(flat)

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    for value in sorted(set(object_type.tolist())):
        mask = object_type == value
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=18,
            alpha=0.7,
            label=OBJECT_LABELS.get(int(value), f"type {int(value)}"),
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Future Trajectory PCA Space")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    return save_figure(fig, Path(out_dir) / "pca_trajectory_space.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PCA view of future trajectories.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--max_points", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(plot_pca_trajectory_space(args.data, args.out_dir, args.max_points))


if __name__ == "__main__":
    main()
