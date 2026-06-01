from pathlib import Path

import numpy as np
import torch

from src.utils.config import load_yaml_config
from src.utils.device import get_device
from src.utils.io import load_json, save_json
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


def test_load_yaml_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("model:\n  name: smoke\n", encoding="utf-8")

    config = load_yaml_config(config_path)

    assert config == {"model": {"name": "smoke"}}


def test_json_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "nested" / "metrics.json"
    save_json({"ADE": 1.25, "model": "linear"}, output_path)

    assert load_json(output_path) == {"ADE": 1.25, "model": "linear"}


def test_ensure_dir(tmp_path: Path) -> None:
    path = ensure_dir(tmp_path / "a" / "b")

    assert path.exists()
    assert path.is_dir()


def test_get_device_returns_torch_device() -> None:
    assert isinstance(get_device(), torch.device)


def test_set_seed_is_reproducible() -> None:
    set_seed(42)
    np_values = np.random.rand(3)
    torch_values = torch.rand(3)

    set_seed(42)

    assert np.allclose(np.random.rand(3), np_values)
    assert torch.allclose(torch.rand(3), torch_values)
