from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from src.analysis.common import (
    collect_model_outputs,
    full_horizon_mask,
    future_flat,
    load_processed_npz,
    per_sample_metrics,
    summarize_metrics,
)
from src.utils.paths import ensure_dir


def _fit_pca_kmeans(train_data: dict[str, np.ndarray], n_components: int, n_clusters: int) -> tuple[PCA, KMeans]:
    train_flat, _ = future_flat(train_data, full_only=True)
    if train_flat.shape[0] < 2:
        raise ValueError("At least two full-horizon train trajectories are required for K-means analysis")
    pca_dims = min(n_components, train_flat.shape[0], train_flat.shape[1])
    pca = PCA(n_components=pca_dims, random_state=42)
    train_latent = pca.fit_transform(train_flat)
    k = min(max(n_clusters, 1), train_latent.shape[0])
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(train_latent)
    return pca, kmeans


def _plot_clusters(latent: np.ndarray, labels: np.ndarray, path: Path) -> None:
    coords = latent[:, :2] if latent.shape[1] >= 2 else np.column_stack([latent[:, 0], np.zeros(latent.shape[0])])
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=labels, cmap="tab10", s=18, alpha=0.75)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("K-means Trajectory Clusters")
    ax.grid(True, alpha=0.25)
    legend = ax.legend(*scatter.legend_elements(), title="cluster", frameon=False)
    ax.add_artist(legend)
    ensure_dir(path.parent)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _cluster_summary(data: dict[str, np.ndarray], labels: np.ndarray, full_mask: np.ndarray) -> pd.DataFrame:
    Y = data["Y"][full_mask]
    object_type = data["object_type"][full_mask]
    rows = []
    for cluster in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == cluster)
        final_points = Y[indices, -1]
        rows.append(
            {
                "cluster": int(cluster),
                "count": int(indices.size),
                "mean_final_x": float(final_points[:, 0].mean()),
                "mean_final_y": float(final_points[:, 1].mean()),
                "vehicle_count": int((object_type[indices] == 0).sum()),
                "pedestrian_count": int((object_type[indices] == 1).sum()),
            }
        )
    return pd.DataFrame(rows)


def _cluster_metrics(
    data: dict[str, np.ndarray],
    labels: np.ndarray,
    full_mask: np.ndarray,
    predictions_dir: Path | None,
    miss_threshold: float,
    required_models: tuple[str, ...] | None,
) -> pd.DataFrame:
    outputs = collect_model_outputs(data, predictions_dir, required_models=required_models)
    full_indices = np.flatnonzero(full_mask)
    rows = []
    for model_name, model_output in outputs.items():
        metrics = per_sample_metrics(
            model_output["pred"],
            data["Y"],
            data["mask_y"],
            samples=model_output.get("samples"),
            miss_threshold=miss_threshold,
        )
        for cluster in sorted(set(labels.tolist())):
            cluster_full_positions = np.flatnonzero(labels == cluster)
            source_indices = full_indices[cluster_full_positions]
            summary = summarize_metrics(metrics, source_indices)
            rows.append(
                {
                    "cluster": int(cluster),
                    "model": model_name,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def run_kmeans_analysis(
    train_data: str | Path,
    data: str | Path,
    predictions_dir: str | Path | None,
    out_dir: str | Path,
    n_components: int = 12,
    n_clusters: int = 5,
    miss_threshold: float = 2.0,
    required_models: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    train = load_processed_npz(train_data)
    target = load_processed_npz(data)
    pca, kmeans = _fit_pca_kmeans(train, n_components=n_components, n_clusters=n_clusters)
    target_flat, target_full_mask = future_flat(target, full_only=True)
    latent = pca.transform(target_flat)
    labels = kmeans.predict(latent)

    output_root = Path(out_dir)
    tables_dir = ensure_dir(output_root / "tables")
    figures_dir = ensure_dir(output_root / "figures")
    cluster_summary = _cluster_summary(target, labels, target_full_mask)
    cluster_metrics = _cluster_metrics(
        target,
        labels,
        target_full_mask,
        Path(predictions_dir) if predictions_dir else None,
        miss_threshold,
        required_models,
    )

    summary_path = tables_dir / "cluster_summary.csv"
    metrics_path = tables_dir / "cluster_metrics.csv"
    figure_path = figures_dir / "kmeans_clusters.png"
    cluster_summary.to_csv(summary_path, index=False)
    cluster_metrics.to_csv(metrics_path, index=False)
    _plot_clusters(latent, labels, figure_path)
    return {"cluster_summary": summary_path, "cluster_metrics": metrics_path, "figure": figure_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PCA + K-means trajectory pattern analysis.")
    parser.add_argument("--train_data", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs"))
    parser.add_argument("--n_components", type=int, default=12)
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument("--miss_threshold", type=float, default=2.0)
    parser.add_argument("--required_models", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = run_kmeans_analysis(
        train_data=args.train_data,
        data=args.data,
        predictions_dir=args.predictions,
        out_dir=args.out_dir,
        n_components=args.n_components,
        n_clusters=args.n_clusters,
        miss_threshold=args.miss_threshold,
        required_models=tuple(args.required_models) if args.required_models else None,
    )
    for key, path in paths.items():
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
