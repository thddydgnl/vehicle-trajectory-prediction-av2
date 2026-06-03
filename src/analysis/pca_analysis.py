from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.common import full_horizon_mask, load_processed_npz
from src.models.pca_latent import PCATrajectoryCodec
from src.utils.paths import ensure_dir


def _full_horizon_y(data_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = load_processed_npz(data_path)
    mask = full_horizon_mask(data)
    Y = data["Y"][mask].astype(np.float32)
    if Y.shape[0] == 0:
        raise ValueError(f"No full-horizon trajectories found in {data_path}")
    return Y, mask


def _save_latent_csv(latent: np.ndarray, source_indices: np.ndarray, path: Path, max_rows: int) -> None:
    ensure_dir(path.parent)
    limit = min(latent.shape[0], max_rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["source_index", *[f"pc{i + 1}" for i in range(latent.shape[1])]]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row_idx in range(limit):
            row = {"source_index": int(source_indices[row_idx])}
            row.update({f"pc{i + 1}": float(latent[row_idx, i]) for i in range(latent.shape[1])})
            writer.writerow(row)


def _plot_pca_space(latent: np.ndarray, path: Path) -> None:
    if latent.shape[1] < 2:
        raise ValueError("At least two PCA dimensions are required for a PCA space plot")
    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], s=16, alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Future Trajectory PCA Space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def run_pca_analysis(
    train_data: Path,
    out_dir: Path,
    n_components: int = 12,
    data: Path | None = None,
    max_export_rows: int = 5000,
) -> dict[str, Path]:
    Y_train, _ = _full_horizon_y(train_data)
    fit_components = min(n_components, Y_train.shape[0], Y_train.reshape(Y_train.shape[0], -1).shape[1])
    if fit_components < 2:
        raise ValueError("At least two PCA components are required for Phase 13 analysis")

    checkpoints_dir = ensure_dir(out_dir / "checkpoints")
    figures_dir = ensure_dir(out_dir / "figures")
    tables_dir = ensure_dir(out_dir / "tables")
    codec = PCATrajectoryCodec(n_components=fit_components)
    codec.fit(Y_train)
    codec_path = checkpoints_dir / "pca_codec.pkl"
    codec.save(codec_path)

    figure_path = figures_dir / "pca_explained_variance.png"
    plt.figure(figsize=(6, 4))
    cumulative = np.cumsum(codec.pca.explained_variance_ratio_)
    plt.plot(np.arange(1, fit_components + 1), cumulative, marker="o")
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()

    target_data = data or train_data
    target, target_mask = _full_horizon_y(target_data)
    latent = codec.transform(target)
    pca_space_path = figures_dir / "pca_trajectory_space.png"
    _plot_pca_space(latent, pca_space_path)
    latent_path = tables_dir / "pca_latent.csv"
    _save_latent_csv(latent, np.flatnonzero(target_mask), latent_path, max_rows=max_export_rows)
    return {"codec": codec_path, "explained_variance": figure_path, "pca_space": pca_space_path, "latent": latent_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCA trajectory codec on processed train data.")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=None, help="Optional data split to transform after fitting on train_data.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n_components", type=int, default=12)
    parser.add_argument("--max_export_rows", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_pca_analysis(args.train_data, args.out_dir, args.n_components, data=args.data, max_export_rows=args.max_export_rows)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
