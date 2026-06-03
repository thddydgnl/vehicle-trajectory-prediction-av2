from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.config import load_yaml_config
from src.utils.geometry import rotation_matrix, to_relative_coords, wrap_angle
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


OBJECT_TYPE_TO_ID = {
    "vehicle": 0,
    "pedestrian": 1,
}
SUPPORTED_TARGET_MODES = {"focal", "scored", "all_supported"}
REQUIRED_COLUMNS = {
    "track_id",
    "object_type",
    "timestep",
    "position_x",
    "position_y",
    "heading",
    "velocity_x",
    "velocity_y",
    "scenario_id",
    "focal_track_id",
    "observed",
}


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    out_dir: Path
    num_scenarios: int | None = 100
    target_types: tuple[str, ...] = ("VEHICLE", "PEDESTRIAN")
    obs_len: int = 50
    pred_len: int = 30
    target_mode: str = "focal"
    seed: int = 42
    splits: tuple[str, ...] = ("train", "val")


def _scenario_files(raw_dir: Path, split: str, limit: int | None) -> list[Path]:
    split_dir = raw_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"AV2 split directory not found: {split_dir}")
    files = sorted(split_dir.glob("*/scenario_*.parquet"))
    if limit is not None:
        files = files[:limit]
    return files


def _normalize_target_types(target_types: tuple[str, ...]) -> set[str]:
    return {target_type.lower() for target_type in target_types}


def _candidate_track_ids(df: pd.DataFrame, target_mode: str, target_types: set[str]) -> list[str]:
    if target_mode not in SUPPORTED_TARGET_MODES:
        raise ValueError(f"Unsupported target_mode: {target_mode}")
    if target_mode == "focal":
        return [str(df["focal_track_id"].iloc[0])]

    candidates = df[df["object_type"].str.lower().isin(target_types)].copy()
    if target_mode == "scored" and "object_category" in candidates.columns:
        candidates = candidates[candidates["object_category"] >= 2]
    return sorted(str(track_id) for track_id in candidates["track_id"].unique())


def _build_track_sample(
    df: pd.DataFrame,
    track_id: str,
    target_types: set[str],
    obs_len: int,
    pred_len: int,
) -> dict[str, np.ndarray | str | int | float] | None:
    track = df[df["track_id"].astype(str) == str(track_id)].sort_values("timestep")
    if track.empty:
        return None
    object_type_name = str(track["object_type"].iloc[0]).lower()
    if object_type_name not in target_types or object_type_name not in OBJECT_TYPE_TO_ID:
        return None

    total_len = obs_len + pred_len
    by_timestep = track.set_index("timestep")
    positions = np.zeros((total_len, 2), dtype=np.float32)
    velocities = np.zeros((total_len, 2), dtype=np.float32)
    headings = np.zeros(total_len, dtype=np.float32)
    mask = np.zeros(total_len, dtype=bool)

    for t in range(total_len):
        if t not in by_timestep.index:
            continue
        row = by_timestep.loc[t]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if bool(row["observed"]) != (t < obs_len):
            return None
        positions[t] = np.array([row["position_x"], row["position_y"]], dtype=np.float32)
        velocities[t] = np.array([row["velocity_x"], row["velocity_y"]], dtype=np.float32)
        headings[t] = np.float32(row["heading"])
        mask[t] = True

    mask_x = mask[:obs_len]
    mask_y = mask[obs_len:]
    if not mask_x[-1] or not mask_x.any() or not mask_y.all():
        return None

    origin = positions[obs_len - 1].copy()
    theta = float(headings[obs_len - 1])
    rel_positions = to_relative_coords(positions, origin, theta)
    rel_velocities = (velocities @ rotation_matrix(-theta).T).astype(np.float32)
    rel_heading = wrap_angle(headings - theta).astype(np.float32)

    X = np.zeros((obs_len, 6), dtype=np.float32)
    X[:, 0:2] = rel_positions[:obs_len]
    X[:, 2:4] = rel_velocities[:obs_len]
    X[:, 4] = np.sin(rel_heading[:obs_len])
    X[:, 5] = np.cos(rel_heading[:obs_len])
    X[~mask_x] = 0.0
    Y = rel_positions[obs_len:].astype(np.float32)
    Y[~mask_y] = 0.0

    return {
        "X": X,
        "Y": Y,
        "mask_x": mask_x.astype(bool),
        "mask_y": mask_y.astype(bool),
        "object_type": OBJECT_TYPE_TO_ID[object_type_name],
        "scenario_id": str(track["scenario_id"].iloc[0]),
        "track_id": str(track_id),
        "origin": origin.astype(np.float32),
        "theta": np.float32(theta),
    }


def _empty_processed(obs_len: int, pred_len: int) -> dict[str, np.ndarray]:
    return {
        "X": np.zeros((0, obs_len, 6), dtype=np.float32),
        "Y": np.zeros((0, pred_len, 2), dtype=np.float32),
        "mask_x": np.zeros((0, obs_len), dtype=bool),
        "mask_y": np.zeros((0, pred_len), dtype=bool),
        "object_type": np.zeros((0,), dtype=np.int64),
        "scenario_id": np.array([], dtype=object),
        "track_id": np.array([], dtype=object),
        "origin": np.zeros((0, 2), dtype=np.float32),
        "theta": np.zeros((0,), dtype=np.float32),
    }


def _samples_to_arrays(samples: list[dict[str, np.ndarray | str | int | float]], obs_len: int, pred_len: int) -> dict[str, np.ndarray]:
    if not samples:
        return _empty_processed(obs_len, pred_len)
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
    }


def _save_scaler(train_arrays: dict[str, np.ndarray], out_dir: Path) -> None:
    X = train_arrays["X"]
    mask = train_arrays["mask_x"]
    if X.shape[0] == 0 or not mask.any():
        return
    valid_x = X[mask]
    scaler = {
        "mean": valid_x.mean(axis=0).astype(np.float32),
        "std": np.maximum(valid_x.std(axis=0), 1e-6).astype(np.float32),
        "feature_order": ["rel_x", "rel_y", "velocity_x", "velocity_y", "sin_heading", "cos_heading"],
    }
    joblib.dump(scaler, out_dir / "scaler.pkl")


def preprocess_split(config: PreprocessConfig, split: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    target_types = _normalize_target_types(config.target_types)
    samples: list[dict[str, np.ndarray | str | int | float]] = []
    records: list[dict[str, str | int]] = []
    files = _scenario_files(config.raw_dir, split, config.num_scenarios)
    for path in tqdm(files, desc=f"preprocess {split}"):
        df = pd.read_parquet(path)
        missing = REQUIRED_COLUMNS.difference(df.columns)
        if missing:
            raise KeyError(f"{path} is missing required columns: {sorted(missing)}")
        for track_id in _candidate_track_ids(df, config.target_mode, target_types):
            sample = _build_track_sample(df, track_id, target_types, config.obs_len, config.pred_len)
            if sample is None:
                continue
            samples.append(sample)
            records.append(
                {
                    "split": split,
                    "scenario_id": str(sample["scenario_id"]),
                    "track_id": str(sample["track_id"]),
                    "object_type": int(sample["object_type"]),
                }
            )
    return _samples_to_arrays(samples, config.obs_len, config.pred_len), pd.DataFrame.from_records(records)


def preprocess_av2(config: PreprocessConfig) -> dict[str, Path]:
    set_seed(config.seed)
    ensure_dir(config.out_dir)
    metadata_dir = ensure_dir(config.out_dir / "metadata")
    suffix = "small" if config.num_scenarios is not None else "full"
    output_paths: dict[str, Path] = {}
    metadata_frames: list[pd.DataFrame] = []

    for split in config.splits:
        arrays, metadata = preprocess_split(config, split)
        if arrays["X"].shape[0] == 0:
            raise ValueError(f"No valid samples were produced for split {split}")
        path = config.out_dir / f"{split}_{suffix}.npz"
        np.savez_compressed(path, **arrays)
        output_paths[split] = path
        metadata_frames.append(metadata)
        metadata.to_csv(metadata_dir / f"{split}_{suffix}_metadata.csv", index=False)
        if split == "train":
            _save_scaler(arrays, config.out_dir)

    if metadata_frames:
        pd.concat(metadata_frames, ignore_index=True).to_csv(metadata_dir / f"{suffix}_metadata.csv", index=False)
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Argoverse 2 Motion Forecasting parquet files.")
    parser.add_argument("--config", type=Path, help="Optional YAML config with preprocessing arguments.")
    parser.add_argument("--raw_dir", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--num_scenarios", type=int)
    parser.add_argument("--full", action="store_true", help="Process all scenarios and write *_full.npz files.")
    parser.add_argument("--target_types", nargs="+")
    parser.add_argument("--obs_len", type=int)
    parser.add_argument("--pred_len", type=int)
    parser.add_argument("--target_mode", choices=sorted(SUPPORTED_TARGET_MODES))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--splits", nargs="+")
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    values = load_yaml_config(args.config) if args.config else {}
    raw_dir = args.raw_dir if args.raw_dir is not None else values.get("raw_dir")
    out_dir = args.out_dir if args.out_dir is not None else values.get("out_dir")
    if raw_dir is None or out_dir is None:
        raise ValueError("--raw_dir and --out_dir are required unless provided by --config")

    num_scenarios = values.get("num_scenarios", 100)
    if args.num_scenarios is not None:
        num_scenarios = args.num_scenarios
    if args.full:
        num_scenarios = None

    target_types = args.target_types if args.target_types is not None else values.get("target_types", ["VEHICLE", "PEDESTRIAN"])
    splits = args.splits if args.splits is not None else values.get("splits", ["train", "val"])
    return PreprocessConfig(
        raw_dir=Path(raw_dir),
        out_dir=Path(out_dir),
        num_scenarios=num_scenarios,
        target_types=tuple(target_types),
        obs_len=int(args.obs_len if args.obs_len is not None else values.get("obs_len", 50)),
        pred_len=int(args.pred_len if args.pred_len is not None else values.get("pred_len", 30)),
        target_mode=str(args.target_mode if args.target_mode is not None else values.get("target_mode", "focal")),
        seed=int(args.seed if args.seed is not None else values.get("seed", 42)),
        splits=tuple(splits),
    )


def main() -> None:
    args = parse_args()
    paths = preprocess_av2(config_from_args(args))
    for split, path in paths.items():
        print(f"{split}: {path}")


if __name__ == "__main__":
    main()
