from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_processed_npz(data_path: str | Path) -> dict[str, np.ndarray]:
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed data not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def full_horizon_mask(data: dict[str, np.ndarray]) -> np.ndarray:
    return data["mask_y"].astype(bool).all(axis=1)


def future_flat(data: dict[str, np.ndarray], full_only: bool = True) -> tuple[np.ndarray, np.ndarray]:
    mask = full_horizon_mask(data) if full_only else np.ones(data["Y"].shape[0], dtype=bool)
    Y = data["Y"][mask].astype(np.float32)
    return Y.reshape(Y.shape[0], -1), mask


def compute_linear_prediction(X: np.ndarray, pred_len: int, dt: float = 0.1) -> np.ndarray:
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
        if isinstance(payload, dict):
            payloads[_model_name_from_path(path)] = payload
    return payloads


def _single_prediction(payload: dict[str, Any]) -> np.ndarray | None:
    for key in ("pred", "prediction", "predictions"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 4:
                arr = arr[:, 0]
            if arr.ndim == 3:
                return arr.astype(np.float32)
    for key in ("samples", "pred_samples"):
        if key in payload:
            arr = np.asarray(payload[key])
            if arr.ndim == 4:
                return arr[:, 0].astype(np.float32)
    return None


def _prediction_samples(payload: dict[str, Any]) -> np.ndarray | None:
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


def _metadata_list(payload: dict[str, Any], key: str) -> list[str] | None:
    if key not in payload:
        return None
    return [str(value) for value in payload[key]]


def _validate_payload_alignment(model_name: str, payload: dict[str, Any], data: dict[str, np.ndarray], pred: np.ndarray) -> None:
    expected_shape = data["Y"].shape
    if pred.shape != expected_shape:
        raise ValueError(f"{model_name} prediction shape {pred.shape} does not match data Y shape {expected_shape}")
    if "gt" in payload and not np.allclose(np.asarray(payload["gt"], dtype=np.float32), data["Y"]):
        raise ValueError(f"{model_name} payload gt does not match the requested data split")
    if "mask_y" in payload and not np.array_equal(np.asarray(payload["mask_y"], dtype=bool), data["mask_y"]):
        raise ValueError(f"{model_name} payload mask_y does not match the requested data split")

    for key in ("scenario_id", "track_id"):
        payload_values = _metadata_list(payload, key)
        if payload_values is None:
            continue
        data_values = [str(value) for value in data[key]]
        if payload_values != data_values:
            raise ValueError(f"{model_name} payload {key} order does not match the requested data split")


def _validate_required_models(outputs: dict[str, dict[str, np.ndarray]], required_models: tuple[str, ...] | None) -> None:
    if not required_models:
        return
    missing = sorted(set(required_models).difference(outputs))
    if missing:
        available = sorted(outputs)
        raise ValueError(f"Missing required model predictions: {missing}; available models: {available}")


def collect_model_outputs(
    data: dict[str, np.ndarray],
    predictions_dir: str | Path | None,
    include_linear: bool = True,
    required_models: tuple[str, ...] | None = None,
) -> dict[str, dict[str, np.ndarray]]:
    outputs: dict[str, dict[str, np.ndarray]] = {}
    if include_linear:
        outputs["linear"] = {"pred": compute_linear_prediction(data["X"], int(data["Y"].shape[1]))}
    for model_name, payload in load_prediction_payloads(predictions_dir).items():
        if include_linear and model_name == "linear":
            continue
        pred = _single_prediction(payload)
        if pred is None:
            continue
        _validate_payload_alignment(model_name, payload, data, pred)
        model_outputs = {"pred": pred}
        samples = _prediction_samples(payload)
        if samples is not None:
            if samples.shape[0] != data["Y"].shape[0] or samples.shape[2:] != data["Y"].shape[1:]:
                raise ValueError(f"{model_name} sample shape {samples.shape} does not match data Y shape {data['Y'].shape}")
            model_outputs["samples"] = samples
        outputs[model_name] = model_outputs
    _validate_required_models(outputs, required_models)
    return outputs


def per_sample_metrics(
    pred: np.ndarray,
    gt: np.ndarray,
    mask: np.ndarray,
    samples: np.ndarray | None = None,
    miss_threshold: float = 2.0,
) -> dict[str, np.ndarray]:
    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match ground truth shape {gt.shape}")
    if mask.shape != gt.shape[:2]:
        raise ValueError(f"Mask shape {mask.shape} does not match ground truth time shape {gt.shape[:2]}")
    valid = mask.astype(bool)
    n = int(pred.shape[0])
    errors = np.linalg.norm(pred - gt, axis=-1)
    counts = valid.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every sample must have at least one valid future step")
    ade_numerator = (errors * valid).sum(axis=1)
    ade = ade_numerator / counts
    final_indices = valid.shape[1] - 1 - np.argmax(valid[:, ::-1], axis=1)
    fde = errors[np.arange(n), final_indices]
    miss = (fde > miss_threshold).astype(np.float32)

    if samples is None:
        min_ade = ade.copy()
        min_fde = fde.copy()
        min_miss = miss.copy()
    else:
        if samples.ndim != 4 or samples.shape[0] != n or samples.shape[2:] != gt.shape[1:]:
            raise ValueError(f"Sample prediction shape {samples.shape} is incompatible with ground truth shape {gt.shape}")
        sample_errors = np.linalg.norm(samples - gt[:, None, :, :], axis=-1)
        sample_valid = valid[:, None, :]
        sample_counts = sample_valid.sum(axis=2)
        sample_ade = (sample_errors * sample_valid).sum(axis=2) / sample_counts
        min_ade = sample_ade.min(axis=1)
        sample_fde = sample_errors[np.arange(n), :, final_indices]
        min_fde = sample_fde.min(axis=1)
        min_miss = (min_fde > miss_threshold).astype(np.float32)

    return {
        "ADE": ade.astype(np.float32),
        "ADE Numerator": ade_numerator.astype(np.float32),
        "Valid Steps": counts.astype(np.float32),
        "FDE": fde.astype(np.float32),
        "minADE": min_ade.astype(np.float32),
        "minFDE": min_fde.astype(np.float32),
        "Miss Rate": miss.astype(np.float32),
        "minMiss Rate": min_miss.astype(np.float32),
    }


def summarize_metrics(metrics: dict[str, np.ndarray], indices: np.ndarray) -> dict[str, float | int]:
    selected = np.asarray(indices, dtype=int)
    if selected.size == 0:
        return {"count": 0, "ADE": np.nan, "FDE": np.nan, "minADE": np.nan, "minFDE": np.nan, "Miss Rate": np.nan, "minMiss Rate": np.nan}
    valid_steps = metrics["Valid Steps"][selected]
    ade_denominator = float(valid_steps.sum())
    ade_value = float(metrics["ADE Numerator"][selected].sum() / ade_denominator) if ade_denominator > 0 else np.nan
    return {
        "count": int(selected.size),
        "ADE": ade_value,
        "FDE": float(np.nanmean(metrics["FDE"][selected])),
        "minADE": float(np.nanmean(metrics["minADE"][selected])),
        "minFDE": float(np.nanmean(metrics["minFDE"][selected])),
        "Miss Rate": float(np.nanmean(metrics["Miss Rate"][selected])),
        "minMiss Rate": float(np.nanmean(metrics["minMiss Rate"][selected])),
    }
