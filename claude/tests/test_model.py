"""
Unit tests for the CNN model architecture.

Verifies forward-pass shapes, config-driven construction,
and the tiny-model smoke-test variant.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# Allow imports from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from model import CarLogoCNN, ConvBlock


# ---------------------------------------------------------------------------
# ConvBlock
# ---------------------------------------------------------------------------

class TestConvBlock:
    def test_output_shape(self) -> None:
        block = ConvBlock(in_channels=3, out_channels=16, kernel_size=3,
                          pool_size=2, dropout=0.1)
        x = torch.randn(4, 3, 64, 64)
        out = block(x)
        # pool_size=2 halves spatial dims
        assert out.shape == (4, 16, 32, 32), f"Got {out.shape}"

    def test_no_pool(self) -> None:
        block = ConvBlock(in_channels=3, out_channels=8, kernel_size=3,
                          pool_size=1, dropout=0.0)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        assert out.shape == (2, 8, 32, 32), f"Got {out.shape}"


# ---------------------------------------------------------------------------
# CarLogoCNN
# ---------------------------------------------------------------------------

class TestCarLogoCNN:
    def test_tiny_forward(self) -> None:
        model = CarLogoCNN.tiny(num_classes=10)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 10), f"Got {logits.shape}"

    def test_from_config_defaults(self) -> None:
        config = {
            "input_channels": 3,
            "input_size": [128, 128],
            "conv_blocks": [
                {"out_channels": 8, "kernel_size": 3, "pool_size": 2, "dropout": 0},
                {"out_channels": 16, "kernel_size": 3, "pool_size": 2, "dropout": 0},
            ],
            "fc_layers": [32],
            "fc_dropout": 0.0,
            "num_classes": 5,
        }
        model = CarLogoCNN.from_config(config)
        x = torch.randn(1, 3, 128, 128)
        out = model(x)
        assert out.shape == (1, 5), f"Got {out.shape}"

    def test_empty_conv_blocks(self) -> None:
        """Model with only FC layers should still produce correct logits."""
        model = CarLogoCNN(
            conv_blocks_config=[],
            fc_layers=[64],
            num_classes=5,
            input_size=(32, 32),
        )
        x = torch.randn(1, 3, 32, 32)
        out = model(x)
        assert out.shape == (1, 5), f"Got {out.shape}"

    def test_multiple_batches(self) -> None:
        model = CarLogoCNN.tiny(num_classes=3)
        for batch_size in [1, 4, 16]:
            x = torch.randn(batch_size, 3, 224, 224)
            out = model(x)
            assert out.shape[0] == batch_size
            assert out.shape[1] == 3

    def test_logits_not_normalized(self) -> None:
        """Raw logits should not be probabilities (no softmax in forward)."""
        model = CarLogoCNN.tiny(num_classes=5)
        x = torch.randn(1, 3, 224, 224)
        logits = model(x)
        # If output contains values outside [0,1], it's raw logits
        assert (logits > 1.0).any() or (logits < 0).any(), (
            "Output appears to be probabilities, not logits"
        )
