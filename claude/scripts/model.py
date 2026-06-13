"""
Configurable CNN model for car logo recognition.

The architecture is defined by ``model_config.json`` and consists of:

1. **Convolutional blocks** — each block: Conv2d → BatchNorm → ReLU → MaxPool → Dropout
2. **Fully-connected head** — Linear → ReLU → Dropout → Linear (logits)

The number and size of conv/fc layers are read from config, making
the model adaptable to different dataset scales.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Convolutional block factory
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d → BatchNorm → ReLU → MaxPool → Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2  # same-padding

        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,  # bias is redundant when BatchNorm follows
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size) if pool_size > 1 else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.dropout(x)
        return x


# ---------------------------------------------------------------------------
# Main CNN
# ---------------------------------------------------------------------------

class CarLogoCNN(nn.Module):
    """Configurable CNN for vehicle logo classification.

    Usage::

        config = load_model_config()
        model  = CarLogoCNN.from_config(config)
    """

    def __init__(
        self,
        input_channels: int = 3,
        conv_blocks_config: List[Dict[str, Any]] | None = None,
        fc_layers: List[int] | None = None,
        fc_dropout: float = 0.5,
        num_classes: int = 50,
        input_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        conv_blocks_config = conv_blocks_config or []
        fc_layers = fc_layers or [512, 256]

        # -- Convolutional feature extractor --------------------------------
        blocks: List[nn.Module] = []
        in_ch = input_channels

        for block_cfg in conv_blocks_config:
            block = ConvBlock(
                in_channels=in_ch,
                out_channels=block_cfg["out_channels"],
                kernel_size=block_cfg.get("kernel_size", 3),
                pool_size=block_cfg.get("pool_size", 2),
                dropout=block_cfg.get("dropout", 0.0),
            )
            blocks.append(block)
            in_ch = block_cfg["out_channels"]

        self.features = nn.Sequential(*blocks)

        # -- Classifier head ------------------------------------------------
        # Compute flattened feature size after convolutions
        dummy = torch.zeros(1, input_channels, *input_size)
        with torch.no_grad():
            flat_size = self.features(dummy).view(1, -1).size(1)

        classifier: List[nn.Module] = []
        prev_dim = flat_size

        for hidden_dim in fc_layers:
            classifier.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(fc_dropout),
            ])
            prev_dim = hidden_dim

        classifier.append(nn.Linear(prev_dim, num_classes))
        self.classifier = nn.Sequential(*classifier)

        # -- Weight initialization ------------------------------------------
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Kaiming normal initialization to conv and linear layers."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits of shape ``(batch, num_classes)``."""
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.classifier(x)
        return x

    # -------------------------------------------------------------------
    # Config-driven factory
    # -------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> CarLogoCNN:
        """Construct a model from the ``model_config.json`` dictionary."""
        return cls(
            input_channels=config.get("input_channels", 3),
            conv_blocks_config=config.get("conv_blocks", []),
            fc_layers=config.get("fc_layers", [512, 256]),
            fc_dropout=config.get("fc_dropout", 0.5),
            num_classes=config.get("num_classes", 50),
            input_size=tuple(config.get("input_size", [224, 224])),
        )

    @classmethod
    def tiny(cls, num_classes: int = 50) -> CarLogoCNN:
        """Minimal variant for quick smoke-testing."""
        return cls(
            conv_blocks_config=[
                {"out_channels": 16, "kernel_size": 3, "pool_size": 2, "dropout": 0},
                {"out_channels": 32, "kernel_size": 3, "pool_size": 2, "dropout": 0},
            ],
            fc_layers=[64],
            fc_dropout=0.0,
            num_classes=num_classes,
        )
