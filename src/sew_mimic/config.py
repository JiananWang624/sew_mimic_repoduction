"""Load the single project-level YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict[str, Any]:
    """Load a YAML mapping and fail clearly for a missing or invalid file."""
    config_path = Path(path)
    try:
        with config_path.open("r", encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"SEW-Mimic config not found: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"SEW-Mimic config must contain a YAML mapping: {config_path}")
    return config


def project_path(value: str | Path) -> Path:
    """Resolve a configured path relative to the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


CONFIG = load_config()
