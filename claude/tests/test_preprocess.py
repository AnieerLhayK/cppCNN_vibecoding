"""
Unit tests for image preprocessing transforms.

Verifies that transform pipelines produce tensors with correct shapes
and expected value ranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Allow imports from scripts/
SCRIPT_DIR = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from preprocess import eval_transforms, train_transforms


def _dummy_image(size: int = 256) -> Image.Image:
    return Image.fromarray(
        np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)
    )


class TestTrainTransforms:
    def test_output_shape(self) -> None:
        pipeline = train_transforms((224, 224))
        img = _dummy_image(300)
        tensor = pipeline(img)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 224, 224)

    def test_output_is_normalized(self) -> None:
        """After normalization, values should roughly be in [-3, 3]."""
        pipeline = train_transforms((64, 64))
        img = _dummy_image(100)
        tensor = pipeline(img)
        assert tensor.min() >= -3.5
        assert tensor.max() <= 3.5

    def test_multiple_calls_different(self) -> None:
        """Augmentation should produce different crops/flips."""
        pipeline = train_transforms((64, 64))
        img = _dummy_image(200)
        tensors = [pipeline(img) for _ in range(5)]
        # At least two should differ (virtually guaranteed with RandomResizedCrop)
        identical = all(
            torch.equal(tensors[0], t) for t in tensors[1:]
        )
        assert not identical, "Augmentation should produce variation"


class TestEvalTransforms:
    def test_output_shape(self) -> None:
        pipeline = eval_transforms((224, 224))
        img = _dummy_image(200)
        tensor = pipeline(img)
        assert tensor.shape == (3, 224, 224)

    def test_deterministic(self) -> None:
        """Evaluation transforms should be deterministic."""
        pipeline = eval_transforms((64, 64))
        img = _dummy_image(128)
        t1 = pipeline(img)
        t2 = pipeline(img)
        assert torch.equal(t1, t2)

    def test_small_image(self) -> None:
        """Should handle images smaller than target size."""
        pipeline = eval_transforms((224, 224))
        img = _dummy_image(100)
        tensor = pipeline(img)
        assert tensor.shape == (3, 224, 224)
