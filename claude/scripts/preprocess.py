"""
Image preprocessing — data augmentation, normalization, and DataLoader creation.

Provides transform pipelines for:
- **Training**: random resized crop, horizontal flip, color jitter, normalization.
- **Validation / Test**: resize, center crop, normalization.

Also provides a factory function to build PyTorch DataLoaders from
a directory of class-named subdirectories.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

import utils


# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

def train_transforms(input_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Augmentation-heavy pipeline for training data."""
    return transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def eval_transforms(input_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """Deterministic pipeline for validation / test evaluation."""
    return transforms.Compose([
        transforms.Resize(int(max(input_size)) + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    data_root: Path,
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train / val / test DataLoaders from a structured data directory.

    Expects::

        data_root/
            train/
                class_a/...
                class_b/...
            val/
                class_a/...
                class_b/...
            test/
                class_a/...
                class_b/...

    Returns ``(train_loader, val_loader, test_loader, class_names)``.
    """
    input_size = tuple(model_config.get("input_size", [224, 224]))
    batch_size = training_config.get("batch_size", 32)
    num_workers = training_config.get("num_workers", 2)

    train_path = data_root / "train"
    val_path = data_root / "val"
    test_path = data_root / "test"

    if not train_path.is_dir():
        raise NotADirectoryError(
            f"Training directory not found: {train_path}. "
            "Run prepare_data.py first."
        )

    # Build datasets
    train_dataset = datasets.ImageFolder(
        root=str(train_path),
        transform=train_transforms(input_size),
    )
    val_dataset = datasets.ImageFolder(
        root=str(val_path) if val_path.is_dir() else str(train_path),
        transform=eval_transforms(input_size),
    )
    test_dataset = datasets.ImageFolder(
        root=str(test_path) if test_path.is_dir() else str(val_path) if val_path.is_dir() else str(train_path),
        transform=eval_transforms(input_size),
    )

    class_names = train_dataset.classes

    # Build loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names
