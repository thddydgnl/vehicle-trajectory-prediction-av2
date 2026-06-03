from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils.paths import ensure_dir
from src.visualization.common import (
    available_predictions,
    load_prediction_payloads,
    load_processed_npz,
    per_sample_ade_fde,
    prediction_samples,
    save_figure,
)


MODEL_COLORS = {
    "linear": "#2563eb",
    "lstm": "#16a34a",
    "transformer": "#dc2626",
    "diffusion_direct": "#9333ea",
    "diffusion_pca": "#ea580c",
}


def _set_square_bounds(ax: plt.Axes, arrays: list[np.ndarray]) -> None:
    points = [arr.reshape(-1, 2) for arr in arrays if arr.size > 0]
    if not points:
        return
    stacked = np.concatenate(points, axis=0)
    x_min, y_min = stacked.min(axis=0)
    x_max, y_max = stacked.max(axis=0)
    span = max(float(x_max - x_min), float(y_max - y_min), 1.0)
    pad = span * 0.08
    x_center = float((x_min + x_max) / 2.0)
    y_center = float((y_min + y_max) / 2.0)
    half = span / 2.0 + pad
    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)


def _plot_case(
    ax: plt.Axes,
    X: np.ndarray,
    Y: np.ndarray,
    mask_x: np.ndarray,
    mask_y: np.ndarray,
    predictions: dict[str, np.ndarray],
    index: int,
) -> None:
    past = X[index, :, 0:2]
    future = Y[index]
    ax.plot(past[mask_x[index], 0], past[mask_x[index], 1], color="#111827", linewidth=1.8, label="past")
    ax.plot(future[mask_y[index], 0], future[mask_y[index], 1], color="#f59e0b", linewidth=1.8, label="future")
    for model_name, pred in predictions.items():
        if index >= pred.shape[0]:
            continue
        color = MODEL_COLORS.get(model_name, "#64748b")
        ax.plot(
            pred[index, :, 0],
            pred[index, :, 1],
            linestyle="--",
            linewidth=1.2,
            color=color,
            label=model_name,
        )
    plotted_arrays = [past[mask_x[index]], future[mask_y[index]]]
    plotted_arrays.extend(pred[index] for pred in predictions.values() if index < pred.shape[0])
    _set_square_bounds(ax, plotted_arrays)
    ax.scatter([0.0], [0.0], color="#111827", s=12, zorder=3)
    ax.set_title(f"case {index}", fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def plot_model_overlay(
    data_path: str | Path,
    predictions_dir: str | Path,
    out_dir: str | Path,
    num_cases: int = 20,
) -> Path:
    data = load_processed_npz(data_path)
    predictions = available_predictions(data, predictions_dir)
    preferred = ["linear", "lstm", "transformer"]
    selected_predictions = {name: predictions[name] for name in preferred if name in predictions}

    total_cases = int(data["X"].shape[0])
    num_panels = min(max(num_cases, 1), total_cases, 12)
    cols = min(4, num_panels)
    rows = int(np.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0), squeeze=False)
    for panel_idx in range(rows * cols):
        ax = axes[panel_idx // cols][panel_idx % cols]
        if panel_idx >= num_panels:
            ax.axis("off")
            continue
        _plot_case(
            ax,
            data["X"],
            data["Y"],
            data["mask_x"],
            data["mask_y"],
            selected_predictions,
            panel_idx,
        )
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Trajectory Overlay", y=0.94, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return save_figure(fig, Path(out_dir) / "trajectory_overlay_linear_lstm_transformer.png")


def plot_diffusion_samples(
    data_path: str | Path,
    predictions_dir: str | Path,
    out_dir: str | Path,
    num_cases: int = 6,
) -> Path:
    data = load_processed_npz(data_path)
    payloads = load_prediction_payloads(predictions_dir)
    samples = None
    for name in ("diffusion_direct", "diffusion_pca", "diffusion"):
        if name in payloads:
            samples = prediction_samples(payloads[name])
            if samples is not None:
                break

    total_cases = int(data["X"].shape[0])
    num_panels = min(max(num_cases, 1), total_cases, 6)
    cols = min(3, num_panels)
    rows = int(np.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2), squeeze=False)
    for panel_idx in range(rows * cols):
        ax = axes[panel_idx // cols][panel_idx % cols]
        if panel_idx >= num_panels:
            ax.axis("off")
            continue
        past = data["X"][panel_idx, :, 0:2]
        future = data["Y"][panel_idx]
        ax.plot(past[data["mask_x"][panel_idx], 0], past[data["mask_x"][panel_idx], 1], color="#111827", linewidth=1.8, label="past")
        ax.plot(future[data["mask_y"][panel_idx], 0], future[data["mask_y"][panel_idx], 1], color="#f59e0b", linewidth=1.8, label="future")
        if samples is not None and panel_idx < samples.shape[0]:
            for sample_idx in range(min(samples.shape[1], 8)):
                ax.plot(
                    samples[panel_idx, sample_idx, :, 0],
                    samples[panel_idx, sample_idx, :, 1],
                    color="#9333ea",
                    linewidth=0.9,
                    alpha=0.45,
                    label="diffusion sample" if sample_idx == 0 else None,
                )
        else:
            ax.text(0.5, 0.08, "diffusion samples unavailable", transform=ax.transAxes, ha="center", fontsize=8)
        plotted_arrays = [past[data["mask_x"][panel_idx]], future[data["mask_y"][panel_idx]]]
        if samples is not None and panel_idx < samples.shape[0]:
            plotted_arrays.extend(samples[panel_idx, sample_idx] for sample_idx in range(min(samples.shape[1], 8)))
        _set_square_bounds(ax, plotted_arrays)
        ax.set_title(f"case {panel_idx}", fontsize=9)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Diffusion Sample Overlay", y=0.94, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    return save_figure(fig, Path(out_dir) / "trajectory_overlay_diffusion_samples.png")


def plot_top_error_cases(
    data_path: str | Path,
    predictions_dir: str | Path,
    out_dir: str | Path,
    num_cases: int = 5,
) -> list[Path]:
    data = load_processed_npz(data_path)
    predictions = available_predictions(data, predictions_dir)
    pred = predictions.get("linear")
    if pred is None:
        raise ValueError("Linear prediction is required for top-error case selection")
    _, fde = per_sample_ade_fde(pred, data["Y"], data["mask_y"])
    indices = np.argsort(fde)[::-1][: max(num_cases, 1)]
    case_dir = ensure_dir(Path(out_dir) / "top_error_cases")
    saved: list[Path] = []
    for rank, index in enumerate(indices, start=1):
        fig, ax = plt.subplots(figsize=(4.0, 3.6))
        _plot_case(ax, data["X"], data["Y"], data["mask_x"], data["mask_y"], {"linear": pred}, int(index))
        ax.set_title(f"top FDE case {rank}: sample {int(index)}", fontsize=10)
        saved.append(save_figure(fig, case_dir / f"top_error_case_{rank:02d}.png"))
        plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot trajectory overlays.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, default=Path("outputs/predictions"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--num_cases", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        plot_model_overlay(args.data, args.predictions, args.out_dir, args.num_cases),
        plot_diffusion_samples(args.data, args.predictions, args.out_dir, min(args.num_cases, 6)),
        *plot_top_error_cases(args.data, args.predictions, args.out_dir, min(args.num_cases, 5)),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
