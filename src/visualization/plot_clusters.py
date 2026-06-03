from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.visualization.common import load_processed_npz, save_figure


def plot_kmeans_clusters(
    data_path: str | Path,
    out_dir: str | Path,
    n_clusters: int = 5,
    max_points: int = 2000,
) -> Path:
    data = load_processed_npz(data_path)
    full_horizon = data["mask_y"].all(axis=1)
    Y = data["Y"][full_horizon]
    if Y.shape[0] < 2:
        raise ValueError("At least two full-horizon trajectories are required for cluster plotting")

    if Y.shape[0] > max_points:
        indices = np.linspace(0, Y.shape[0] - 1, max_points).astype(int)
        Y = Y[indices]

    flat = Y.reshape(Y.shape[0], -1)
    pca_dims = min(12, flat.shape[0], flat.shape[1])
    latent = PCA(n_components=pca_dims, random_state=42).fit_transform(flat)
    k = min(max(n_clusters, 1), latent.shape[0])
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(latent)
    coords = latent[:, :2] if latent.shape[1] >= 2 else np.column_stack([latent[:, 0], np.zeros(latent.shape[0])])

    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-means Trajectory Clusters")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(*scatter.legend_elements(), title="cluster", frameon=False)
    ax.add_artist(legend)
    return save_figure(fig, Path(out_dir) / "kmeans_clusters.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot K-means clusters over trajectory PCA space.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--max_points", type=int, default=2000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(plot_kmeans_clusters(args.data, args.out_dir, args.n_clusters, args.max_points))


if __name__ == "__main__":
    main()
