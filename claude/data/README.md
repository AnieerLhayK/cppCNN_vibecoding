# Dataset Preparation

## Supported Datasets

This project expects car logo images organized in the following structure:

```
data/
  raw/
    train/
      brand_a/
        img1.jpg
        img2.jpg
        ...
      brand_b/
        ...
    test/
      brand_a/
        ...
```

Or a single flat directory that gets split automatically:

```
data/
  raw/
    brand_a/
      img1.jpg
      ...
    brand_b/
      ...
```

## Recommended Datasets

1. **XMU Car Logo Dataset** — ~75 car brands, ~10,000 images
2. **Vehicle Logo Dataset (VLD)** — ~40 car brands, ~7,000 images
3. **Chinese Car Logo Dataset (CCLD)** — ~50 Chinese/global car brands

## Quick Start

Place your dataset images under `data/raw/` organized by brand folders,
then run:

```bash
python scripts/prepare_data.py
```

This will verify the structure, split into train/val/test sets,
and report class distribution.
