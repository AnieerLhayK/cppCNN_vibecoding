"""
Comprehensive evaluation on the test set — overall accuracy, per-class metrics,
confusion matrix, and classification report.

Usage::

    python scripts/evaluate.py
    python scripts/evaluate.py --checkpoint models/checkpoints/best_model.pth
    python scripts/evaluate.py --top-k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

import utils
from model import CarLogoCNN
from preprocess import build_dataloaders


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    top_k: int = 1,
) -> Tuple[float, float, List[int], List[int], torch.Tensor]:
    """Run full evaluation and return metrics + predictions.

    Returns:
        - top-1 accuracy (percentage)
        - top-k accuracy (percentage)
        - list of true class indices
        - list of predicted class indices
        - softmax probabilities tensor (num_samples, num_classes)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct_1 = 0
    correct_k = 0
    total = 0

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[torch.Tensor] = []

    for inputs, targets in tqdm(loader, desc="Evaluating", ncols=80):
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * inputs.size(0)

        # Top-1 accuracy
        _, pred_1 = outputs.max(1)
        correct_1 += pred_1.eq(targets).sum().item()

        # Top-k accuracy
        if top_k > 1:
            _, pred_topk = outputs.topk(min(top_k, outputs.size(1)), dim=1)
            correct_k += pred_topk.eq(targets.view(-1, 1)).sum().item()

        probs = torch.softmax(outputs, dim=1)
        all_probs.append(probs.cpu())
        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(pred_1.cpu().tolist())
        total += targets.size(0)

    avg_loss = running_loss / total
    acc_1 = 100.0 * correct_1 / total
    acc_k = 100.0 * correct_k / total if top_k > 1 else acc_1
    probs_tensor = torch.cat(all_probs, dim=0)

    return avg_loss, acc_1, acc_k, all_targets, all_preds, probs_tensor


def print_confusion_matrix(cm: List[List[int]], class_names: List[str]) -> None:
    """Print a compact confusion matrix to the console."""
    n = len(class_names)
    label_width = max(len(n) for n in class_names) + 2

    # Header
    print(f"\n{'':>{label_width}}", end="")
    for short in class_names:
        print(f"{short[:4]:>5}", end="")
    print()

    for i in range(n):
        print(f"{class_names[i]:>{label_width}}", end="")
        for j in range(n):
            val = cm[i][j]
            if val > 0:
                print(f"{val:5d}", end="")
            else:
                print(f"    .", end="")
        print()
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained model on test set."
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=utils.PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help="Data directory (default: PROJECT_ROOT/data).",
    )
    parser.add_argument(
        "--top-k", type=int, default=1,
        help="Compute top-k accuracy (default: 1).",
    )
    parser.add_argument(
        "--full-matrix", action="store_true",
        help="Print full confusion matrix (may be large for many classes).",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = utils.setup_logger("evaluate")

    # -- Validate checkpoint ----------------------------------------------
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        logger.info("Train a model first: python scripts/train.py")
        return 1

    # -- Load checkpoint --------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("model_config", utils.load_model_config())
    class_names: List[str] = ckpt.get("class_names", [])

    logger.info("Loaded checkpoint from epoch %d (val_acc=%.2f%%)",
                ckpt.get("epoch", 0), ckpt.get("val_acc", 0.0))

    # -- Model ------------------------------------------------------------
    model_cfg["num_classes"] = len(class_names) if class_names else model_cfg.get("num_classes", 50)
    model = CarLogoCNN.from_config(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Model loaded with %d classes", model_cfg["num_classes"])

    if not class_names:
        logger.warning("No class names in checkpoint; using numeric indices.")
        class_names = [str(i) for i in range(model_cfg["num_classes"])]

    # -- Data -------------------------------------------------------------
    train_cfg = utils.load_training_config()
    data_dir = args.data_dir or utils.PROJECT_ROOT / "data"
    _, _, test_loader, _ = build_dataloaders(
        data_root=data_dir,
        model_config=model_cfg,
        training_config=train_cfg,
    )

    # -- Evaluate ---------------------------------------------------------
    logger.info("Evaluating on test set ...")
    test_loss, acc_1, acc_k, targets, preds, probs = evaluate_model(
        model, test_loader, device, top_k=args.top_k,
    )

    # -- Results ----------------------------------------------------------
    n = len(targets)
    print("\n" + "=" * 60)
    print("TEST SET RESULTS")
    print("=" * 60)
    print(f"  Test loss:          {test_loss:.4f}")
    print(f"  Top-1 accuracy:     {acc_1:.2f}%")
    if args.top_k > 1:
        print(f"  Top-{args.top_k} accuracy:     {acc_k:.2f}%")
    print(f"  Total samples:      {n}")
    print(f"  Number of classes:  {len(class_names)}")
    print("=" * 60)

    # -- Per-class report -------------------------------------------------
    print("\nClassification Report:")
    print("-" * 60)
    print(classification_report(
        targets, preds,
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    # -- Confusion matrix -------------------------------------------------
    if args.full_matrix:
        cm = confusion_matrix(targets, preds).tolist()
        print("Confusion Matrix (rows=true, cols=predicted):")
        print_confusion_matrix(cm, class_names)

    # -- Worst-performing classes -----------------------------------------
    from sklearn.metrics import recall_score
    per_class_recall = recall_score(targets, preds, average=None, zero_division=0)
    worst_idx = sorted(
        range(len(per_class_recall)),
        key=lambda i: per_class_recall[i],
    )[:max(5, len(class_names) // 10)]

    print("\nWorst-performing classes (lowest recall):")
    print("-" * 60)
    for idx in worst_idx:
        if per_class_recall[idx] < 1.0:
            print(f"  {class_names[idx]:25s}  recall={per_class_recall[idx]:.3f}")
    print()

    logger.info("Evaluation complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
