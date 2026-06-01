from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


OBJECT_TYPE_VEHICLE = 0
OBJECT_TYPE_PEDESTRIAN = 1
FEATURE_DIM = 6
PATTERNS = (
    "straight",
    "slowdown",
    "left_turn",
    "right_turn",
    "stop_and_go",
    "pedestrian_random",
)


@dataclass(frozen=True)
class SyntheticConfig:
    out_dir: Path
    num_samples: int = 1000
    obs_len: int = 50
    pred_len: int = 30
    seed: int = 42
    dt: float = 0.1
    train_ratio: float = 0.7
    val_ratio: float = 0.15


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _split_counts(num_samples: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    if num_samples < 3:
        raise ValueError("num_samples must be at least 3")
    if not 0.0 < train_ratio < 1.0 or not 0.0 <= val_ratio < 1.0:
        raise ValueError("split ratios must be in [0, 1)")
    train_count = int(num_samples * train_ratio)
    val_count = int(num_samples * val_ratio)
    test_count = num_samples - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError("train, val, and test splits must all be non-empty")
    return train_count, val_count, test_count


def _rotation(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def _local_velocity(pattern: str, step: int, rng: np.random.Generator, dt: float) -> np.ndarray:
    if pattern == "straight":
        speed = rng.uniform(7.0, 15.0)
        lateral = rng.normal(0.0, 0.05)
        return np.array([speed, lateral], dtype=np.float32)
    if pattern == "slowdown":
        speed = max(1.0, rng.uniform(13.0, 18.0) - 0.16 * step)
        return np.array([speed, rng.normal(0.0, 0.03)], dtype=np.float32)
    if pattern == "left_turn":
        speed = rng.uniform(5.0, 10.0)
        turn_rate = 0.035
        return np.array([speed * np.cos(turn_rate * step), speed * np.sin(turn_rate * step)], dtype=np.float32)
    if pattern == "right_turn":
        speed = rng.uniform(5.0, 10.0)
        turn_rate = -0.035
        return np.array([speed * np.cos(turn_rate * step), speed * np.sin(turn_rate * step)], dtype=np.float32)
    if pattern == "stop_and_go":
        cycle = step % 40
        speed = 9.0 if cycle < 15 else 2.0 if cycle < 26 else 6.0
        return np.array([speed + rng.normal(0.0, 0.2), rng.normal(0.0, 0.04)], dtype=np.float32)
    if pattern == "pedestrian_random":
        angle = 0.18 * np.sin(step / 5.0) + rng.normal(0.0, 0.08)
        speed = rng.uniform(0.8, 1.8)
        return np.array([speed * np.cos(angle), speed * np.sin(angle)], dtype=np.float32)
    raise ValueError(f"Unknown synthetic pattern: {pattern}")


def _make_single_trajectory(
    sample_idx: int,
    obs_len: int,
    pred_len: int,
    dt: float,
    rng: np.random.Generator,
) -> dict[str, np.ndarray | int | str | float]:
    pattern = PATTERNS[sample_idx % len(PATTERNS)]
    object_type = OBJECT_TYPE_PEDESTRIAN if pattern == "pedestrian_random" else OBJECT_TYPE_VEHICLE
    total_len = obs_len + pred_len
    global_heading = rng.uniform(-np.pi, np.pi)
    rot = _rotation(global_heading)
    position = rng.normal(0.0, 1.0, size=2).astype(np.float32)
    positions = np.zeros((total_len, 2), dtype=np.float32)

    for step in range(total_len):
        velocity = _local_velocity(pattern, step, rng, dt)
        if step > 0:
            position = position + (rot @ velocity) * dt
        positions[step] = position + rng.normal(0.0, 0.02, size=2).astype(np.float32)

    origin = positions[obs_len - 1].copy()
    delta = positions[obs_len - 1] - positions[max(obs_len - 2, 0)]
    theta = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-6 else global_heading
    rel_rot = _rotation(-theta)
    rel_positions = ((positions - origin) @ rel_rot.T).astype(np.float32)

    velocities = np.zeros_like(rel_positions)
    velocities[1:] = (rel_positions[1:] - rel_positions[:-1]) / dt
    velocities[0] = velocities[1]
    headings = np.zeros(total_len, dtype=np.float32)
    headings[1:] = np.arctan2(velocities[1:, 1], velocities[1:, 0])
    headings[0] = headings[1]
    rel_headings = _wrap_angle(headings)

    X = np.zeros((obs_len, FEATURE_DIM), dtype=np.float32)
    X[:, 0:2] = rel_positions[:obs_len]
    X[:, 2:4] = velocities[:obs_len]
    X[:, 4] = np.sin(rel_headings[:obs_len])
    X[:, 5] = np.cos(rel_headings[:obs_len])

    return {
        "X": X,
        "Y": rel_positions[obs_len:].astype(np.float32),
        "mask_x": np.ones(obs_len, dtype=bool),
        "mask_y": np.ones(pred_len, dtype=bool),
        "object_type": object_type,
        "scenario_id": f"synthetic_scenario_{sample_idx:06d}",
        "track_id": f"synthetic_track_{sample_idx:06d}",
        "origin": origin.astype(np.float32),
        "theta": np.float32(theta),
        "pattern": pattern,
    }


def generate_synthetic_dataset(
    num_samples: int = 1000,
    obs_len: int = 50,
    pred_len: int = 30,
    seed: int = 42,
    dt: float = 0.1,
) -> dict[str, np.ndarray]:
    """Generate synthetic trajectories in the project's processed schema."""
    if obs_len < 2:
        raise ValueError("obs_len must be at least 2")
    if pred_len < 1:
        raise ValueError("pred_len must be positive")

    rng = np.random.default_rng(seed)
    samples = [_make_single_trajectory(i, obs_len, pred_len, dt, rng) for i in range(num_samples)]

    return {
        "X": np.stack([sample["X"] for sample in samples]).astype(np.float32),
        "Y": np.stack([sample["Y"] for sample in samples]).astype(np.float32),
        "mask_x": np.stack([sample["mask_x"] for sample in samples]).astype(bool),
        "mask_y": np.stack([sample["mask_y"] for sample in samples]).astype(bool),
        "object_type": np.array([sample["object_type"] for sample in samples], dtype=np.int64),
        "scenario_id": np.array([sample["scenario_id"] for sample in samples], dtype=object),
        "track_id": np.array([sample["track_id"] for sample in samples], dtype=object),
        "origin": np.stack([sample["origin"] for sample in samples]).astype(np.float32),
        "theta": np.array([sample["theta"] for sample in samples], dtype=np.float32),
        "pattern": np.array([sample["pattern"] for sample in samples], dtype=object),
    }


def _slice_dataset(dataset: dict[str, np.ndarray], start: int, end: int) -> dict[str, np.ndarray]:
    return {key: value[start:end] for key, value in dataset.items()}


def save_synthetic_splits(config: SyntheticConfig) -> dict[str, Path]:
    """Generate and save train/val/test smoke splits."""
    set_seed(config.seed)
    out_dir = ensure_dir(config.out_dir)
    dataset = generate_synthetic_dataset(
        num_samples=config.num_samples,
        obs_len=config.obs_len,
        pred_len=config.pred_len,
        seed=config.seed,
        dt=config.dt,
    )
    train_count, val_count, test_count = _split_counts(
        config.num_samples,
        config.train_ratio,
        config.val_ratio,
    )
    split_ranges = {
        "train": (0, train_count),
        "val": (train_count, train_count + val_count),
        "test": (train_count + val_count, train_count + val_count + test_count),
    }
    paths: dict[str, Path] = {}
    for split_name, (start, end) in split_ranges.items():
        path = out_dir / f"{split_name}_smoke.npz"
        np.savez_compressed(path, **_slice_dataset(dataset, start, end))
        paths[split_name] = path
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic trajectory smoke data.")
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--num_samples", type=int, default=1000)
    parser.add_argument("--obs_len", type=int, default=50)
    parser.add_argument("--pred_len", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dt", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = save_synthetic_splits(
        SyntheticConfig(
            out_dir=args.out_dir,
            num_samples=args.num_samples,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            seed=args.seed,
            dt=args.dt,
        )
    )
    for split_name, path in paths.items():
        print(f"{split_name}: {path}")


if __name__ == "__main__":
    main()
