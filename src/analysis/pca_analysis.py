from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.models.pca_latent import PCATrajectoryCodec
from src.utils.paths import ensure_dir


def run_pca_analysis(train_data: Path, out_dir: Path, n_components: int = 12) -> dict[str, Path]:
    with np.load(train_data, allow_pickle=True) as data:
        Y_train = data["Y"].astype(np.float32)

    checkpoints_dir = ensure_dir(out_dir / "checkpoints")
    figures_dir = ensure_dir(out_dir / "figures")
    codec = PCATrajectoryCodec(n_components=n_components)
    codec.fit(Y_train)
    codec_path = checkpoints_dir / "pca_codec.pkl"
    codec.save(codec_path)

    figure_path = figures_dir / "pca_explained_variance.png"
    plt.figure(figsize=(6, 4))
    cumulative = np.cumsum(codec.pca.explained_variance_ratio_)
    plt.plot(np.arange(1, n_components + 1), cumulative, marker="o")
    plt.xlabel("PCA components")
    plt.ylabel("Cumulative explained variance")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(figure_path, dpi=160)
    plt.close()
    return {"codec": codec_path, "figure": figure_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit PCA trajectory codec on processed train data.")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n_components", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_pca_analysis(args.train_data, args.out_dir, args.n_components)
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
