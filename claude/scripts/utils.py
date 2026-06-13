"""
Shared utilities for the car logo CNN pipeline.

Provides configuration loading, device selection,
path resolution, and logging helpers used across all scripts.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logger(name: str = "car-logo-cnn") -> logging.Logger:
    """Configure and return a project-scoped logger with console output."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s  %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)

    return logger


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_json(path: Path) -> Dict[str, Any]:
    """Load and return a JSON file. Raises on parse failure."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def load_model_config(config_dir: Path | None = None) -> Dict[str, Any]:
    """Load the model architecture configuration from JSON."""
    config_dir = config_dir or PROJECT_ROOT / "configs"
    return load_json(config_dir / "model_config.json")


def load_training_config(config_dir: Path | None = None) -> Dict[str, Any]:
    """Load the training hyperparameter configuration from JSON."""
    config_dir = config_dir or PROJECT_ROOT / "configs"
    return load_json(config_dir / "training_config.json")


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def resolve_device(device_pref: str = "auto") -> torch.device:
    """Resolve a torch device string to a torch.device.

    ``"auto"`` picks CUDA if available, otherwise CPU.
    """
    if device_pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_pref)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def build_class_index(data_dir: Path) -> Dict[str, int]:
    """Scan a directory of class-named subdirectories and build label map.

    Returns ``{class_name: integer_index}`` sorted alphabetically.
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    classes = sorted(
        p.name for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )

    if not classes:
        raise ValueError(
            f"No class subdirectories found in {data_dir}. "
            "Each brand should be a separate folder."
        )

    return {name: idx for idx, name in enumerate(classes)}


def index_to_class(class_index: Dict[str, int]) -> Dict[int, str]:
    """Reverse the class index: integer -> class name."""
    return {idx: name for name, idx in class_index.items()}
