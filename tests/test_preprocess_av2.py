from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.preprocess_av2 import PreprocessConfig, config_from_args, parse_args, preprocess_av2
from src.datasets.validate_processed import validate_npz


def _write_mock_scenario(
    split_dir: Path,
    scenario_id: str,
    focal_track_id: str = "focal",
    observed_mismatch: bool = False,
    drop_future_tail: bool = False,
) -> None:
    scenario_dir = split_dir / scenario_id
    scenario_dir.mkdir(parents=True)
    rows = []
    end_timestep = 75 if drop_future_tail else 80
    for timestep in range(end_timestep):
        rows.append(
            {
                "observed": timestep < 50 or (observed_mismatch and timestep == 50),
                "track_id": focal_track_id,
                "object_type": "vehicle",
                "object_category": 2,
                "timestep": timestep,
                "position_x": float(timestep),
                "position_y": 2.0,
                "heading": 0.0,
                "velocity_x": 10.0,
                "velocity_y": 0.0,
                "scenario_id": scenario_id,
                "start_timestamp": 0.0,
                "end_timestamp": 0.0,
                "num_timestamps": 110,
                "focal_track_id": focal_track_id,
                "city": "mock",
            }
        )
    pd.DataFrame(rows).to_parquet(scenario_dir / f"scenario_{scenario_id}.parquet")
    (scenario_dir / f"log_map_archive_{scenario_id}.json").write_text("{}", encoding="utf-8")


def test_preprocess_av2_mock_focal_schema(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    for split in ("train", "val"):
        _write_mock_scenario(raw_dir / split, f"{split}_scenario")
    out_dir = tmp_path / "processed"

    paths = preprocess_av2(
        PreprocessConfig(
            raw_dir=raw_dir,
            out_dir=out_dir,
            num_scenarios=1,
            obs_len=50,
            pred_len=30,
            splits=("train", "val"),
        )
    )

    assert set(paths) == {"train", "val"}
    train_result = validate_npz(paths["train"])
    val_result = validate_npz(paths["val"])
    assert train_result["num_samples"] == 1
    assert val_result["num_samples"] == 1
    assert (out_dir / "scaler.pkl").exists()
    assert (out_dir / "metadata" / "small_metadata.csv").exists()

    with np.load(paths["train"], allow_pickle=True) as data:
        assert np.allclose(data["X"][0, -1, 0:2], 0.0)
        assert data["object_type"][0] == 0
        assert data["scenario_id"][0] == "train_scenario"


def test_preprocess_av2_reads_yaml_config(tmp_path: Path, monkeypatch) -> None:
    raw_dir = tmp_path / "raw"
    for split in ("train", "val"):
        _write_mock_scenario(raw_dir / split, f"{split}_scenario")
    out_dir = tmp_path / "processed"
    config_path = tmp_path / "preprocess.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"raw_dir: {raw_dir.as_posix()}",
                f"out_dir: {out_dir.as_posix()}",
                "num_scenarios: 1",
                "target_types:",
                "  - VEHICLE",
                "obs_len: 50",
                "pred_len: 30",
                "target_mode: focal",
                "seed: 42",
                "splits:",
                "  - train",
                "  - val",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr("sys.argv", ["preprocess_av2", "--config", str(config_path)])
    config = config_from_args(parse_args())
    paths = preprocess_av2(config)

    assert paths["train"] == out_dir / "train_small.npz"
    assert validate_npz(paths["val"])["num_samples"] == 1


def test_preprocess_av2_rejects_observed_mismatch_and_truncated_future(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    _write_mock_scenario(raw_dir / "train", "bad_observed", observed_mismatch=True)
    _write_mock_scenario(raw_dir / "val", "short_future", drop_future_tail=True)
    out_dir = tmp_path / "processed"

    try:
        preprocess_av2(
            PreprocessConfig(
                raw_dir=raw_dir,
                out_dir=out_dir,
                num_scenarios=1,
                obs_len=50,
                pred_len=30,
                splits=("train", "val"),
            )
        )
    except ValueError as exc:
        assert "No valid samples" in str(exc)
    else:
        raise AssertionError("preprocess_av2 should reject invalid observed/future masks")


def test_validate_processed_rejects_bad_final_origin(tmp_path: Path) -> None:
    npz_path = tmp_path / "bad.npz"
    np.savez_compressed(
        npz_path,
        X=np.ones((1, 50, 6), dtype=np.float32),
        Y=np.zeros((1, 30, 2), dtype=np.float32),
        mask_x=np.ones((1, 50), dtype=bool),
        mask_y=np.ones((1, 30), dtype=bool),
        object_type=np.array([0], dtype=np.int64),
        scenario_id=np.array(["s"], dtype=object),
        track_id=np.array(["t"], dtype=object),
        origin=np.zeros((1, 2), dtype=np.float32),
        theta=np.zeros((1,), dtype=np.float32),
    )

    try:
        validate_npz(npz_path)
    except ValueError as exc:
        assert "final observed" in str(exc)
    else:
        raise AssertionError("validate_npz should reject non-zero final observed position")


def test_validate_processed_rejects_truncated_future_and_masked_payload(tmp_path: Path) -> None:
    npz_path = tmp_path / "bad_mask.npz"
    mask_y = np.ones((1, 30), dtype=bool)
    mask_y[0, -1] = False
    Y = np.zeros((1, 30, 2), dtype=np.float32)
    Y[0, -1] = 1.0
    np.savez_compressed(
        npz_path,
        X=np.zeros((1, 50, 6), dtype=np.float32),
        Y=Y,
        mask_x=np.ones((1, 50), dtype=bool),
        mask_y=mask_y,
        object_type=np.array([0], dtype=np.int64),
        scenario_id=np.array(["s"], dtype=object),
        track_id=np.array(["t"], dtype=object),
        origin=np.zeros((1, 2), dtype=np.float32),
        theta=np.zeros((1,), dtype=np.float32),
    )

    try:
        validate_npz(npz_path)
    except ValueError as exc:
        assert "full valid future" in str(exc)
    else:
        raise AssertionError("validate_npz should reject truncated future horizons")
