"""Configuration loader and path management."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration and resolve relative paths to absolute."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path) as f:
        config = yaml.safe_load(f)

    for key, rel_path in config.get("paths", {}).items():
        config["paths"][key] = str((PROJECT_ROOT / rel_path).resolve())

    return config


def ensure_dirs(config: dict[str, Any]) -> None:
    """Create all data/output directories if missing."""
    for path in config["paths"].values():
        Path(path).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["plots"], "candidates").mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["plots"], "summary").mkdir(parents=True, exist_ok=True)
