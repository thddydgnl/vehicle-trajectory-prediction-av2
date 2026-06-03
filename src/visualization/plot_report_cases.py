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
from src.visualization.plot_trajectories import MODEL_COLORS


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


def _choose_reference_model(
    predictions: dict[str, np.ndarray],
    gt: np.ndarray,
    mask_y: np.ndarray,
    requested_model: str | None,
) -> str:
    if requested_model:
        if requested_model not in predictions:
            raise ValueError(f"Requested model {requested_model!r} is unavailable; available: {sorted(predictions)}")
        return requested_model
    scores = []
    for model_name, pred in predictions.items():
        ade_values, _ = per_sample_ade_fde(pred, gt, mask_y)
        scores.append((float(np.nanmean(ade_values)), model_name))
    if not scores:
        raise ValueError("No predictions are available for report-case plotting")
    return min(scores)[1]


def _plot_overlay_case(
    ax: plt.Axes,
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    index: int,
    title: str,
) -> None:
    past = data["X"][index, :, 0:2]
    future = data["Y"][index]
    mask_x = data["mask_x"][index].astype(bool)
    mask_y = data["mask_y"][index].astype(bool)
    ax.plot(past[mask_x, 0], past[mask_x, 1], color="#111827", linewidth=1.8, label="past")
    ax.plot(future[mask_y, 0], future[mask_y, 1], color="#f59e0b", linewidth=1.8, label="future")
    plotted_arrays = [past[mask_x], future[mask_y]]
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
        plotted_arrays.append(pred[index])
    _set_square_bounds(ax, plotted_arrays)
    ax.scatter([0.0], [0.0], color="#111827", s=12, zorder=3)
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)


def _save_case_grid(
    data: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    indices: np.ndarray,
    title: str,
    path: Path,
) -> Path:
    num_panels = int(indices.size)
    cols = min(3, num_panels)
    rows = int(np.ceil(num_panels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.2), squeeze=False)
    for panel_idx in range(rows * cols):
        ax = axes[panel_idx // cols][panel_idx % cols]
        if panel_idx >= num_panels:
            ax.axis("off")
            continue
        index = int(indices[panel_idx])
        _plot_overlay_case(ax, data, predictions, index, f"sample {index}")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 5), frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle(title, y=0.94, fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    saved = save_figure(fig, path)
    plt.close(fig)
    return saved


def plot_best_worst_cases(
    data_path: str | Path,
    predictions_dir: str | Path,
    out_dir: str | Path,
    model_name: str | None = None,
    num_cases: int = 6,
) -> tuple[Path, Path]:
    data = load_processed_npz(data_path)
    predictions = available_predictions(data, predictions_dir)
    reference_model = _choose_reference_model(predictions, data["Y"], data["mask_y"], model_name)
    ade_values, fde_values = per_sample_ade_fde(predictions[reference_model], data["Y"], data["mask_y"])
    del ade_values
    count = min(max(num_cases, 1), int(fde_values.shape[0]))
    best_indices = np.argsort(fde_values)[:count]
    worst_indices = np.argsort(fde_values)[::-1][:count]
    output_root = ensure_dir(out_dir)
    best_path = _save_case_grid(
        data,
        predictions,
        best_indices,
        f"Best Cases by {reference_model} FDE",
        output_root / "trajectory_overlay_best_cases.png",
    )
    worst_path = _save_case_grid(
        data,
        predictions,
        worst_indices,
        f"Worst Cases by {reference_model} FDE",
        output_root / "trajectory_overlay_worst_cases.png",
    )
    return best_path, worst_path


def _sample_fde(samples: np.ndarray, gt: np.ndarray, mask_y: np.ndarray) -> np.ndarray:
    valid = mask_y.astype(bool)
    final_indices = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    return np.linalg.norm(samples[np.arange(samples.shape[0]), :, final_indices] - gt[np.arange(gt.shape[0]), None, final_indices], axis=-1)


def plot_diffusion_interesting_case(
    data_path: str | Path,
    predictions_dir: str | Path,
    out_dir: str | Path,
    model_name: str | None = None,
    max_samples: int = 12,
) -> Path:
    data = load_processed_npz(data_path)
    payloads = load_prediction_payloads(predictions_dir)
    candidates = [model_name] if model_name else ["diffusion_direct", "diffusion_pca", "diffusion"]
    samples = None
    selected_model = None
    for candidate in candidates:
        if candidate is None or candidate not in payloads:
            continue
        samples = prediction_samples(payloads[candidate])
        if samples is not None:
            selected_model = candidate
            break
    if samples is None or selected_model is None:
        raise ValueError(f"No diffusion sample payload found in {predictions_dir}")

    fde_values = _sample_fde(samples, data["Y"], data["mask_y"])
    best_fde = fde_values.min(axis=1)
    first_fde = fde_values[:, 0]
    improvement = first_fde - best_fde
    index = int(np.nanargmax(improvement))
    best_sample_index = int(np.nanargmin(fde_values[index]))

    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    past = data["X"][index, :, 0:2]
    future = data["Y"][index]
    mask_x = data["mask_x"][index].astype(bool)
    mask_y = data["mask_y"][index].astype(bool)
    ax.plot(past[mask_x, 0], past[mask_x, 1], color="#111827", linewidth=2.0, label="past")
    ax.plot(future[mask_y, 0], future[mask_y, 1], color="#f59e0b", linewidth=2.0, label="future")
    plotted_arrays = [past[mask_x], future[mask_y]]
    for sample_idx in range(min(samples.shape[1], max_samples)):
        is_best = sample_idx == best_sample_index
        ax.plot(
            samples[index, sample_idx, :, 0],
            samples[index, sample_idx, :, 1],
            color="#9333ea" if not is_best else "#16a34a",
            linewidth=0.9 if not is_best else 2.0,
            alpha=0.35 if not is_best else 0.95,
            label="diffusion samples" if sample_idx == 0 else ("best sample" if is_best else None),
        )
        plotted_arrays.append(samples[index, sample_idx])
    _set_square_bounds(ax, plotted_arrays)
    ax.set_title(f"{selected_model} interesting case {index}", fontsize=11)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, loc="best")
    saved = save_figure(fig, Path(out_dir) / "diffusion_samples_interesting_case.png")
    plt.close(fig)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Phase 15 report case figures.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/full_av2/figures"))
    parser.add_argument("--reference_model", default=None)
    parser.add_argument("--diffusion_model", default=None)
    parser.add_argument("--num_cases", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [
        *plot_best_worst_cases(
            args.data,
            args.predictions,
            args.out_dir,
            model_name=args.reference_model,
            num_cases=args.num_cases,
        ),
        plot_diffusion_interesting_case(
            args.data,
            args.predictions,
            args.out_dir,
            model_name=args.diffusion_model,
        ),
    ]
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
