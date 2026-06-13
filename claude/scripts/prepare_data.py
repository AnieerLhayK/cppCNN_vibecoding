"""
Dataset preparation — validate structure, split into train/val/test,
and report class distribution.

Expected input layout::

    data/raw/
        audi/
            img001.jpg
            img002.jpg
            ...
        bmw/
            ...
        ...

Output layout (after split)::

    data/
        train/
            audi/...
            bmw/...
        val/
            audi/...
            bmw/...
        test/
            audi/...
            bmw/...
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import utils

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_classes(data_dir: Path) -> Dict[str, List[Path]]:
    """Scan a directory of class subdirectories and return {class: [image paths]}."""
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory not found: {data_dir}")

    collection: Dict[str, List[Path]] = {}

    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir() or class_dir.name.startswith("."):
            continue

        images = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if images:
            collection[class_dir.name] = images

    if not collection:
        raise ValueError(
            f"No brand subdirectories with supported images found in {data_dir}.\n"
            f"Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    return collection


def print_class_distribution(collection: Dict[str, List[Path]], header: str) -> None:
    """Print a formatted per-class image count summary."""
    total = sum(len(paths) for paths in collection.values())
    print(f"\n{header}  ({total} images, {len(collection)} classes)")
    print("-" * 50)
    for cls, paths in sorted(collection.items()):
        print(f"  {cls:20s}  {len(paths):4d}")
    print()


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def train_val_test_split(
    collection: Dict[str, List[Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """Partition each class into train / val / test by relative filenames.

    Returns three dictionaries keyed by class name with lists of
    relative path strings (relative to the raw data root).
    """
    random.seed(seed)
    train_set: Dict[str, List[str]] = {}
    val_set: Dict[str, List[str]] = {}
    test_set: Dict[str, List[str]] = {}

    for cls, paths in collection.items():
        random.shuffle(paths)
        n = len(paths)
        n_train = max(1, round(n * train_ratio))
        n_val = max(0, round(n * val_ratio))

        train_set[cls] = [str(p) for p in paths[:n_train]]
        val_set[cls] = [str(p) for p in paths[n_train:n_train + n_val]]
        test_set[cls] = [str(p) for p in paths[n_train + n_val:]]

    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def _copy_split(
    split_data: Dict[str, List[str]],
    source_root: Path,
    dest_root: Path,
) -> int:
    """Copy files from source to destination, preserving class folder structure.

    Returns the number of files copied.
    """
    count = 0
    for cls, paths in split_data.items():
        target_dir = dest_root / cls
        target_dir.mkdir(parents=True, exist_ok=True)
        for src_path_str in paths:
            src = Path(src_path_str)
            dst = target_dir / src.name
            if not dst.exists():
                shutil.copy2(src, dst)
            count += 1
    return count


def write_split_manifest(
    split_data: Dict[str, List[str]],
    output_path: Path,
    split_name: str,
) -> None:
    """Write a text manifest listing all files in a split."""
    lines = [f"# {split_name} split manifest", f"# Classes: {len(split_data)}"]
    total = sum(len(paths) for paths in split_data.values())
    lines.append(f"# Total images: {total}\n")
    for cls in sorted(split_data):
        for p in split_data[cls]:
            lines.append(f"{cls}/{Path(p).name}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate raw car logo dataset and split into train/val/test."
    )
    parser.add_argument(
        "--raw-dir",
        default=utils.PROJECT_ROOT / "data" / "raw",
        type=Path,
        help="Directory containing class-named subfolders with images.",
    )
    parser.add_argument(
        "--output-dir",
        default=utils.PROJECT_ROOT / "data",
        type=Path,
        help="Output root for train/val/test directories.",
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7,
        help="Proportion of data for training (default: 0.7).",
    )
    parser.add_argument(
        "--val-ratio", type=float, default=0.15,
        help="Proportion of data for validation (default: 0.15).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and report without copying anything.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    logger = utils.setup_logger("prepare_data")

    # Scan
    logger.info("Scanning %s ...", args.raw_dir)
    collection = scan_classes(args.raw_dir)
    print_class_distribution(collection, "Raw dataset")

    # Split
    train_set, val_set, test_set = train_val_test_split(
        collection,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"Train: {sum(len(v) for v in train_set.values())} images")
    print(f"Val:   {sum(len(v) for v in val_set.values())} images")
    print(f"Test:  {sum(len(v) for v in test_set.values())} images")

    if args.dry_run:
        logger.info("Dry-run mode — no files copied.")
        return 0

    # Copy splits
    dest_train = args.output_dir / "train"
    dest_val = args.output_dir / "val"
    dest_test = args.output_dir / "test"

    n_train = _copy_split(train_set, args.raw_dir, dest_train)
    n_val = _copy_split(val_set, args.raw_dir, dest_val)
    n_test = _copy_split(test_set, args.raw_dir, dest_test)

    # Write manifests
    manifest_dir = args.output_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    write_split_manifest(train_set, manifest_dir / "train_manifest.txt", "train")
    write_split_manifest(val_set, manifest_dir / "val_manifest.txt", "validation")
    write_split_manifest(test_set, manifest_dir / "test_manifest.txt", "test")

    logger.info(
        "Done — %d train / %d val / %d test images copied.",
        n_train, n_val, n_test,
    )
    logger.info("Manifests written to %s", manifest_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
