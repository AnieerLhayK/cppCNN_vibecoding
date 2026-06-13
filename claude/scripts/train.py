"""
Training engine — epoch loop with validation, checkpointing,
and console-based progress reporting.

Usage::

    python scripts/train.py
    python scripts/train.py --epochs 100 --lr 0.0005
    python scripts/train.py --resume models/checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

import utils
from model import CarLogoCNN
from preprocess import build_dataloaders


# ---------------------------------------------------------------------------
# Core training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> Tuple[float, float]:
    """Run one training epoch. Returns ``(avg_loss, accuracy)``."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"Train E{epoch:2d}/{total_epochs}",
                leave=False, ncols=80)
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split_name: str = "Val",
) -> Tuple[float, float]:
    """Run evaluation on a validation/test loader. Returns ``(avg_loss, accuracy)``."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc=split_name, leave=False, ncols=80):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    acc = 100.0 * correct / total
    return avg_loss, acc


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    model_config: Dict[str, Any],
    class_names: List[str],
) -> None:
    """Persist model state, optimizer state, and training metadata."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_acc": val_acc,
        "model_config": model_config,
        "class_names": class_names,
    }, path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the car logo CNN model."
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override training_config num_epochs.")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override training_config learning_rate.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override training_config batch_size.")
    parser.add_argument("--resume", type=Path, default=None,
                        help="Path to a checkpoint .pth to resume from.")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Override data directory (default: PROJECT_ROOT/data).")
    parser.add_argument("--tiny", action="store_true",
                        help="Use tiny model for quick smoke test.")
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = utils.setup_logger("train")

    # -- Config -----------------------------------------------------------
    model_cfg = utils.load_model_config()
    train_cfg = utils.load_training_config()

    # CLI overrides
    if args.epochs is not None:
        train_cfg["num_epochs"] = args.epochs
    if args.lr is not None:
        train_cfg["learning_rate"] = args.lr
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size

    device = utils.resolve_device(train_cfg.get("device", "auto"))
    logger.info("Device: %s", device)
    logger.info("Epochs: %d | LR: %s | Batch: %d",
                train_cfg["num_epochs"], train_cfg["learning_rate"],
                train_cfg["batch_size"])

    # -- Data -------------------------------------------------------------
    data_dir = args.data_dir or utils.PROJECT_ROOT / "data"
    train_loader, val_loader, _, class_names = build_dataloaders(
        data_root=data_dir,
        model_config=model_cfg,
        training_config=train_cfg,
    )
    if train_cfg.get("num_classes", None) is None:
        model_cfg["num_classes"] = len(class_names)
    else:
        model_cfg["num_classes"] = len(class_names)

    logger.info("Classes (%d): %s", len(class_names),
                ", ".join(class_names))

    # -- Model ------------------------------------------------------------
    if args.tiny:
        model = CarLogoCNN.tiny(num_classes=len(class_names))
        logger.info("Using TINY model (smoke-test mode)")
    else:
        model = CarLogoCNN.from_config(model_cfg)
        logger.info("Model created with %d conv blocks, %d params",
                    len(model_cfg.get("conv_blocks", [])),
                    sum(p.numel() for p in model.parameters()))

    model = model.to(device)

    # -- Loss, optimizer, scheduler ---------------------------------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 0.0001),
    )

    scheduler = StepLR(
        optimizer,
        step_size=train_cfg.get("lr_scheduler_step", 15),
        gamma=train_cfg.get("lr_scheduler_gamma", 0.1),
    )

    # -- Resume -----------------------------------------------------------
    start_epoch = 1
    best_val_acc = 0.0

    if args.resume:
        if not args.resume.exists():
            logger.error("Checkpoint not found: %s", args.resume)
            return 1
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_acc = ckpt.get("val_acc", 0.0)
        logger.info("Resumed from epoch %d (best val_acc %.2f%%)",
                    ckpt.get("epoch", 0), best_val_acc)

    # -- Training loop ----------------------------------------------------
    num_epochs = train_cfg["num_epochs"]
    patience = train_cfg.get("early_stop_patience", 10)
    epochs_without_improvement = 0
    best_model_dir = Path(train_cfg.get("checkpoint_dir", "models/checkpoints"))
    best_model_path = best_model_dir / "best_model.pth"

    logger.info("=" * 60)
    logger.info("Training started")
    logger.info("=" * 60)

    for epoch in range(start_epoch, num_epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer,
            device, epoch, num_epochs,
        )
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, "Val"
        )
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Progress summary
        logger.info(
            "Epoch %3d/%d | train loss %.4f (%.2f%%) | val loss %.4f (%.2f%%) | lr %.6f",
            epoch, num_epochs,
            train_loss, train_acc,
            val_loss, val_acc,
            current_lr,
        )

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            save_checkpoint(
                best_model_path, model, optimizer,
                epoch, val_acc, model_cfg, class_names,
            )
            logger.info("  >> New best model saved (val_acc=%.2f%%)", val_acc)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if patience > 0 and epochs_without_improvement >= patience:
            logger.info(
                "Early stopping triggered — no improvement for %d epochs.",
                patience,
            )
            break

    logger.info("=" * 60)
    logger.info("Training complete. Best val accuracy: %.2f%%", best_val_acc)
    logger.info("Best model saved to: %s", best_model_path)
    logger.info("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
