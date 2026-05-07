# Copyright 2024 Roboflow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Core RF-DETR model module.

Provides the main RFDETRModel class for loading, running inference,
and managing RF-DETR object detection models.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# Registry of known pretrained model URLs
MODEL_REGISTRY: Dict[str, str] = {
    "rfdetr-base": "https://storage.googleapis.com/rfdetr/rfdetr-base.pth",
    "rfdetr-large": "https://storage.googleapis.com/rfdetr/rfdetr-large.pth",
}

# Lowered from 0.5 to reduce missed detections in my use case (crowded scenes).
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.35
DEFAULT_IMAGE_SIZE: int = 640


@dataclass
class DetectionResult:
    """Container for object detection predictions.

    Attributes:
        boxes: Bounding boxes in [x1, y1, x2, y2] format, shape (N, 4).
        scores: Confidence scores for each detection, shape (N,).
        labels: Class label indices for each detection, shape (N,).
        class_names: Optional list of human-readable class names.
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    class_names: Optional[List[str]] = field(default=None)

    def __len__(self) -> int:
        return len(self.scores)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Return a new DetectionResult with only high-confidence detections."""
        mask = self.scores >= threshold
        return DetectionResult(
            boxes=self.boxes[mask],
            scores=self.scores[mask],
            labels=self.labels[mask],
            class_names=self.class_names,
        )


class RFDETRModel:
    """RF-DETR object detection model wrapper.

    Supports loading from a pretrained model name or a local checkpoint path.

    Example::

        model = RFDETRModel("rfdetr-base")
        results = model.predict(image)
    """

    def __init__(
        self,
        model_name_or_path: str = "rfdetr-base",
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        image_size: int = DEFAULT_IMAGE_SIZE,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            model_name_or_path: Pretrained model name (see MODEL_REGISTRY) or
                path to a local ``.pth`` checkpoint.
            confidence_threshold: Minimum score to keep a detection.
                Defaults to 0.35 (lowered from upstream 0.5).
            image_size: Input resolution used during inference.
            device: Target device string (e.g. ``"cpu"``, ``"cuda:0"``).
                Defaults to CUDA if available, otherwise CPU.
        """
        self.model_name_or_path = model_name_or_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device or self._default_device()
        self._model = None  # Lazy-loaded on first predict call
