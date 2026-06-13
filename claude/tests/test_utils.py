"""
Unit tests for shared utilities.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

# Allow imports from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import utils


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestConfigLoading:
    def test_load_json_valid(self) -> None:
        data = {"a": 1, "b": "hello"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            result = utils.load_json(Path(f.name))
        Path(f.name).unlink()
        assert result == data

    def test_load_json_invalid(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json",
                                         delete=False, encoding="utf-8") as f:
            f.write("not json")
            f.flush()
            with pytest.raises(ValueError, match="Invalid JSON"):
                utils.load_json(Path(f.name))
        Path(f.name).unlink()

    def test_load_model_config_returns_dict(self) -> None:
        config = utils.load_model_config()
        assert "input_channels" in config
        assert "conv_blocks" in config

    def test_load_training_config_returns_dict(self) -> None:
        config = utils.load_training_config()
        assert "batch_size" in config
        assert "learning_rate" in config


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------

class TestDeviceResolution:
    def test_auto_returns_device(self) -> None:
        device = utils.resolve_device("auto")
        assert isinstance(device, type(device))

    def test_cpu_forced(self) -> None:
        device = utils.resolve_device("cpu")
        assert str(device) == "cpu"


# ---------------------------------------------------------------------------
# Class index helpers
# ---------------------------------------------------------------------------

class TestClassIndex:
    def test_build_class_index(self, tmp_path: Path) -> None:
        for brand in ["audi", "bmw", "mercedes"]:
            (tmp_path / brand).mkdir()
        index = utils.build_class_index(tmp_path)
        assert index == {"audi": 0, "bmw": 1, "mercedes": 2}

    def test_build_class_index_empty(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No class subdirectories"):
            utils.build_class_index(tmp_path)

    def test_index_to_class(self) -> None:
        index = {"audi": 0, "bmw": 1}
        reverse = utils.index_to_class(index)
        assert reverse == {0: "audi", 1: "bmw"}

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").mkdir()
        (tmp_path / "visible").mkdir()
        index = utils.build_class_index(tmp_path)
        assert ".hidden" not in index
        assert "visible" in index
