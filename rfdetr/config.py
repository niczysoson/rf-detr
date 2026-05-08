# Copyright 2024 Roboflow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration classes and defaults for RF-DETR models."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# Default model registry mapping model names to their download URLs
MODEL_REGISTRY: Dict[str, str] = {
    "rfdetr-base": "https://storage.googleapis.com/rfdetr/rfdetr-base.pth",
    "rfdetr-large": "https://storage.googleapis.com/rfdetr/rfdetr-large.pth",
}

# Supported input resolutions (must be multiples of 56 for RF-DETR)
SUPPORTED_RESOLUTIONS: List[int] = [560, 616, 672, 728, 784, 840, 896, 952, 1008]

DEFAULT_RESOLUTION: int = 560
# Lowered from 0.5 to 0.4 — I found 0.5 misses too many valid detections
# in my use case (small objects at distance). Adjust per dataset as needed.
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.4
DEFAULT_NUM_CLASSES: int = 91  # COCO classes


@dataclass
class RFDETRConfig:
    """Configuration for RF-DETR model initialization and inference.

    Attributes:
        model_name: Name of the model variant to use.
        resolution: Input image resolution (height and width). Must be a
            multiple of 56.
        num_classes: Number of object detection classes.
        confidence_threshold: Minimum confidence score for detections.
        device: Device string for inference (e.g., 'cpu', 'cuda', 'cuda:0').
        pretrained: Whether to load pretrained weights.
        checkpoint_url: Optional explicit URL to download weights from.
            Overrides the registry URL for the given model_name.
        class_names: Optional list of class name strings. If provided, its
            length must match num_classes.
    """

    model_name: str = "rfdetr-base"
    resolution: int = DEFAULT_RESOLUTION
    num_classes: int = DEFAULT_NUM_CLASSES
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD
    device: Optional[str] = None  # None means auto-detect
    pretrained: bool = True
    checkpoint_url: Optional[str] = None
    class_names: Optional[List[str]] = None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        """Validate configuration values and raise informative errors."""
        if self.model_name not in MODEL_REGISTRY and self.checkpoint_url is None:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Available models: {list(MODEL_REGISTRY.keys())}. "
                "Provide a 'checkpoint_url' to use a custom checkpoint."
            )

        if self.resolution % 56 != 0:
            raise ValueError(
                f"Resolution must be a multiple of 56, got {self.resolution}. "
                f"Supported resolutions: {SUPPORTED_RESOLUTIONS}"
            )

        if self.resolution not in SUPPORTED_RESOLUTIONS:
            import warnings
            warnings.warn(
                f"Resolution {self.resolution} is not in the recommended list "
                f"{SUPPORTED_RESOLUTIONS}. Performance may be sub"
            )
