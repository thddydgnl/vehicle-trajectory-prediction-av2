import json
from pathlib import Path
from typing import Any

from src.utils.paths import ensure_dir


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    """Save a dictionary as pretty-printed JSON."""
    output_path = Path(path)
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object: {input_path}")
    return data
