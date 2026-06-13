# Car Logo Recognition — CNN

A convolutional neural network project for recognizing vehicle brand logos from images.
Console-based pipeline with modular scripts, configuration-driven architecture,
and full training/evaluation/inference workflow.

## Project Structure

```
claude/
├── configs/              # JSON configuration files
│   ├── model_config.json       # CNN architecture parameters
│   └── training_config.json    # Training hyperparameters
├── data/                 # Dataset directory (see data/README.md)
├── scripts/              # Modular pipeline scripts
│   ├── utils.py               # Shared utilities
│   ├── prepare_data.py        # Dataset validation and splitting
│   ├── preprocess.py          # Image transforms and augmentation
│   ├── model.py               # CNN model definition
│   ├── train.py               # Training engine
│   ├── evaluate.py            # Evaluation on test set
│   └── predict.py             # Single-image inference
├── models/               # Trained model checkpoints
├── tests/                # Unit tests
└── requirements.txt      # Python dependencies
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Prepare dataset (place images in data/raw/ first)
python scripts/prepare_data.py

# Train the model
python scripts/train.py

# Evaluate on test set
python scripts/evaluate.py

# Predict a single image
python scripts/predict.py --image path/to/image.jpg
```

## Pipeline Phases

| Phase | Script | Description |
|-------|--------|-------------|
| Data | `prepare_data.py` | Validate structure, split train/val/test |
| Transform | `preprocess.py` | Augmentation, normalization, tensor conversion |
| Model | `model.py` | Configurable CNN architecture |
| Train | `train.py` | Training loop, checkpointing, logging |
| Evaluate | `evaluate.py` | Accuracy, per-class metrics, confusion matrix |
| Predict | `predict.py` | Load checkpoint, classify single image |

## Design Principles

- **Configuration-driven**: Model architecture and training hyperparams in JSON, not hardcoded
- **Console-native**: No GUI dependencies, clean terminal output
- **Modular**: Each script has a single responsibility
- **Portable**: Pure PyTorch, no exotic dependencies
