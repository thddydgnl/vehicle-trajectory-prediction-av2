from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

from src.utils.paths import ensure_dir


matplotlib.use("Agg")


def load_processed_npz(data_path: str | Path) -> dict[str, np.ndarray]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def compute_linear_prediction(X: np.ndarray, pred_len: int, dt: float = 0.1) -> np.ndarray:
    if X.ndim != 3 or X.shape[-1] < 4:
        raise ValueError(f"Expected X shape [N, T, F>=4], got {X.shape}")
    steps = np.arange(1, pred_len + 1, dtype=np.float32).reshape(1, pred_len, 1)
    return (X[:, -1:, 0:2] + X[:, -1:, 2:4] * (steps * np.float32(dt))).astype(np.float32)


def _model_name_from_path(path: Path) -> str:
    stem = path.stem
    for suffix in ("_val", "_eval", "_predictions", "_prediction"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def load_prediction_payloads(predictions_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    if predictions_dir is None:
        return {}
    directory = Path(predictions_dir)
    if not directory.exists():
        return {}
    payloads: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.glob("*.pkl")):
        with path.open("rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            continue
        payloads[_model_name_from_path(path)] = payload
    return payloads


def prediction_array(payload: dict[str, Any]) -> np.ndarray | None:
    for key in ("pred", "prediction", "predictions"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 4:
                arr = arr[:, 0]
            return arr.astype(np.float32)
    for key in ("samples", "pred_samples"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 4:
                return arr[:, 0].astype(np.float32)
    return None


def prediction_samples(payload: dict[str, Any]) -> np.ndarray | None:
    for key in ("samples", "pred_samples"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 4:
                return arr.astype(np.float32)
    for key in ("pred", "prediction", "predictions"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 4:
                return arr.astype(np.float32)
    return None


def available_predictions(
    data: dict[str, np.ndarray],
    predictions_dir: str | Path | None,
    include_linear: bool = True,
) -> dict[str, np.ndarray]:
    pred_len = int(data["Y"].shape[1])
    predictions: dict[str, np.ndarray] = {}
    if include_linear:
        predictions["linear"] = compute_linear_prediction(data["X"], pred_len)
    for name, payload in load_prediction_payloads(predictions_dir).items():
        arr = prediction_array(payload)
        if arr is not None:
            predictions[name] = arr
    return predictions


def masked_displacement_errors(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> np.ndarray:
    n = min(pred.shape[0], gt.shape[0], mask.shape[0])
    errors = np.linalg.norm(pred[:n] - gt[:n], axis=-1)
    return np.where(mask[:n], errors, np.nan)


def per_sample_ade_fde(pred: np.ndarray, gt: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    errors = masked_displacement_errors(pred, gt, mask)
    valid = mask[: errors.shape[0]]
    counts = valid.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every sample must have at least one valid future step")
    ade = np.nansum(errors, axis=1) / counts
    final_indices = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    fde = errors[np.arange(errors.shape[0]), final_indices]
    return ade.astype(np.float32), fde.astype(np.float32)


def save_figure(fig: Any, path: str | Path, dpi: int = 180) -> Path:
    output_path = Path(path)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    return output_path
