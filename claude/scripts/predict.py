"""
Single-image inference — load a trained checkpoint and classify a car logo image.

Shows top-K predictions with confidence scores.

Usage::

    python scripts/predict.py --image path/to/logo.jpg
    python scripts/predict.py --image path/to/logo.jpg --checkpoint models/checkpoints/best_model.pth
    python scripts/predict.py --image path/to/logo.jpg --top-k 5
    python scripts/predict.py --image path/to/logo.jpg --show-probs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

import utils
from model import CarLogoCNN


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_and_preprocess(
    image_path: Path,
    input_size: Tuple[int, int] = (224, 224),
) -> torch.Tensor:
    """Load an image file and return a normalized ``(1, C, H, W)`` tensor."""
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    pil_image = Image.open(image_path).convert("RGB")

    pipeline = transforms.Compose([
        transforms.Resize(int(max(input_size)) + 32),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    tensor = pipeline(pil_image)
    return tensor.unsqueeze(0)  # add batch dimension


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> List[Tuple[int, str, float]]:
    """Run inference and return ``[(class_id, class_name, confidence), ...]``."""
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor.to(device))
        probs = F.softmax(logits, dim=1).squeeze(0)

    top_probs, top_indices = probs.topk(min(top_k, len(probs)))

    results: List[Tuple[int, str, float]] = []
    for prob, cls_idx in zip(top_probs, top_indices):
        results.append((
            int(cls_idx),
            str(cls_idx.item()),  # placeholder — replaced by class_names
            float(prob),
        ))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify a car logo image using a trained CNN."
    )
    parser.add_argument(
        "--image", "-i", type=Path, required=True,
        help="Path to the input image file.",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=Path,
        default=utils.PROJECT_ROOT / "models" / "checkpoints" / "best_model.pth",
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Show top-K predictions (default: 5).",
    )
    parser.add_argument(
        "--show-probs", action="store_true",
        help="Display numerical probabilities for all classes.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = utils.setup_logger("predict")

    # -- Load checkpoint --------------------------------------------------
    if not args.checkpoint.exists():
        logger.error("Checkpoint not found: %s", args.checkpoint)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_cfg = ckpt.get("model_config", utils.load_model_config())
    class_names: List[str] = ckpt.get("class_names", [])

    model_cfg["num_classes"] = len(class_names) if class_names else model_cfg.get("num_classes", 50)
    model = CarLogoCNN.from_config(model_cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    logger.info("Model loaded from %s", args.checkpoint)

    if not class_names:
        logger.warning("No class names found in checkpoint; using indices.")
        class_names = [str(i) for i in range(model_cfg["num_classes"])]

    # -- Load image -------------------------------------------------------
    try:
        input_size = tuple(model_cfg.get("input_size", [224, 224]))
        tensor = load_and_preprocess(args.image, input_size)
    except Exception as exc:
        logger.error("Failed to load image: %s", exc)
        return 1

    logger.info("Image: %s (%s)", args.image, "x".join(str(s) for s in input_size))

    # -- Predict ----------------------------------------------------------
    top_results = predict(model, tensor, device, top_k=args.top_k)

    print("\n" + "=" * 60)
    print(f"  PREDICTION:  {args.image.name}")
    print("=" * 60)
    for rank, (cls_idx, _, confidence) in enumerate(top_results, 1):
        cls_name = class_names[cls_idx] if cls_idx < len(class_names) else str(cls_idx)
        bar_len = int(confidence * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)
        print(f"  #{rank}  {cls_name:25s}  {confidence:.4f}  {bar}")
    print("=" * 60)

    # -- Full probability distribution ------------------------------------
    if args.show_probs:
        print("\nFull probability distribution:")
        print("-" * 60)
        model.eval()
        with torch.no_grad():
            logits = model(tensor.to(device))
            probs = torch.softmax(logits, dim=1).squeeze(0)

        for cls_idx in range(len(class_names)):
            p = float(probs[cls_idx])
            if p > 0.01:
                print(f"  {class_names[cls_idx]:25s}  {p:.4f}")
        print()

    logger.info("Prediction complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
