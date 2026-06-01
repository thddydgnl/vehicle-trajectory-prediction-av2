from pathlib import Path

import numpy as np

from src.datasets.synthetic import FEATURE_DIM, PATTERNS, SyntheticConfig, save_synthetic_splits


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
    "pattern",
}


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def test_save_synthetic_splits_creates_expected_files(tmp_path: Path) -> None:
    paths = save_synthetic_splits(SyntheticConfig(out_dir=tmp_path, num_samples=60, seed=7))

    assert set(paths) == {"train", "val", "test"}
    assert all(path.exists() for path in paths.values())


def test_synthetic_npz_schema_and_shapes(tmp_path: Path) -> None:
    paths = save_synthetic_splits(SyntheticConfig(out_dir=tmp_path, num_samples=60, obs_len=50, pred_len=30))
    train = _load_npz(paths["train"])

    assert REQUIRED_KEYS.issubset(train)
    assert train["X"].shape == (42, 50, FEATURE_DIM)
    assert train["Y"].shape == (42, 30, 2)
    assert train["mask_x"].shape == (42, 50)
    assert train["mask_y"].shape == (42, 30)
    assert train["object_type"].shape == (42,)
    assert train["origin"].shape == (42, 2)
    assert train["theta"].shape == (42,)

    assert train["X"].dtype == np.float32
    assert train["Y"].dtype == np.float32
    assert train["mask_x"].dtype == np.bool_
    assert train["mask_y"].dtype == np.bool_
    assert train["object_type"].dtype == np.int64
    assert train["origin"].dtype == np.float32
    assert train["theta"].dtype == np.float32


def test_synthetic_data_is_reproducible(tmp_path: Path) -> None:
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"

    first_paths = save_synthetic_splits(SyntheticConfig(out_dir=first_dir, num_samples=60, seed=123))
    second_paths = save_synthetic_splits(SyntheticConfig(out_dir=second_dir, num_samples=60, seed=123))

    first = _load_npz(first_paths["train"])
    second = _load_npz(second_paths["train"])

    assert np.allclose(first["X"], second["X"])
    assert np.allclose(first["Y"], second["Y"])
    assert np.array_equal(first["scenario_id"], second["scenario_id"])


def test_synthetic_last_observation_is_relative_origin(tmp_path: Path) -> None:
    paths = save_synthetic_splits(SyntheticConfig(out_dir=tmp_path, num_samples=60))
    train = _load_npz(paths["train"])

    assert np.allclose(train["X"][:, -1, 0:2], 0.0, atol=1e-5)


def test_synthetic_includes_all_motion_patterns(tmp_path: Path) -> None:
    paths = save_synthetic_splits(SyntheticConfig(out_dir=tmp_path, num_samples=60))
    all_patterns: set[str] = set()
    for path in paths.values():
        split = _load_npz(path)
        all_patterns.update(str(pattern) for pattern in split["pattern"].tolist())

    assert set(PATTERNS).issubset(all_patterns)
