from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED_KEYS = {
    "X",
    "Y",
    "mask_x",
    "mask_y",
    "object_type",
    "scenario_id",
    "track_id",
    "origin",
    "theta",
}


def validate_npz(npz_path: str | Path, obs_len: int = 50, pred_len: int = 30) -> dict[str, int | str]:
    path = Path(npz_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed npz not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        missing = REQUIRED_KEYS.difference(data.files)
        if missing:
            raise KeyError(f"{path} missing required keys: {sorted(missing)}")
        arrays = {key: data[key] for key in data.files}

    X = arrays["X"]
    Y = arrays["Y"]
    mask_x = arrays["mask_x"]
    mask_y = arrays["mask_y"]
    object_type = arrays["object_type"]
    scenario_id = arrays["scenario_id"]
    track_id = arrays["track_id"]
    origin = arrays["origin"]
    theta = arrays["theta"]
    n = X.shape[0]

    expected_shapes = {
        "X": (n, obs_len, 6),
        "Y": (n, pred_len, 2),
        "mask_x": (n, obs_len),
        "mask_y": (n, pred_len),
        "object_type": (n,),
        "scenario_id": (n,),
        "track_id": (n,),
        "origin": (n, 2),
        "theta": (n,),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} shape {arrays[key].shape} != expected {shape}")
    if n == 0:
        raise ValueError(f"{path} contains no samples")

    dtype_checks = {
        "X": np.float32,
        "Y": np.float32,
        "mask_x": np.bool_,
        "mask_y": np.bool_,
        "object_type": np.int64,
        "origin": np.float32,
        "theta": np.float32,
    }
    for key, dtype in dtype_checks.items():
        if arrays[key].dtype != dtype:
            raise TypeError(f"{key} dtype {arrays[key].dtype} != expected {dtype}")

    for key in ("X", "Y", "origin", "theta"):
        if np.isnan(arrays[key]).any():
            raise ValueError(f"{key} contains NaN")
        if np.isinf(arrays[key]).any():
            raise ValueError(f"{key} contains Inf")
    if not np.allclose(X[:, -1, 0:2], 0.0, atol=1e-4):
        raise ValueError("final observed relative position is not near zero")
    if not np.isin(object_type, [0, 1]).all():
        raise ValueError("object_type contains unsupported values")
    if not mask_x[:, -1].all():
        raise ValueError("final observed timestep must be valid for every sample")
    if not mask_y.all(axis=1).all():
        raise ValueError("every sample must have a full valid future horizon")
    if not np.allclose(X[~mask_x], 0.0, atol=1e-6):
        raise ValueError("masked X payload must be zero")
    if not np.allclose(Y[~mask_y], 0.0, atol=1e-6):
        raise ValueError("masked Y payload must be zero")

    return {"path": str(path), "num_samples": int(n), "num_scenarios": int(len(set(scenario_id.tolist())))}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed trajectory npz schema.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--obs_len", type=int, default=50)
    parser.add_argument("--pred_len", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_npz(args.npz, obs_len=args.obs_len, pred_len=args.pred_len)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
