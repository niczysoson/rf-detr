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

DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5
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
            image_size: Input resolution used during inference.
            device: Target device string (e.g. ``"cpu"``, ``"cuda:0"``).
                Defaults to CUDA if available, otherwise CPU.
        """
        self.model_name_or_path = model_name_or_path
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size
        self.device = device or self._default_device()
        self._model = None  # Lazy-loaded on first predict call

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        image: Union[np.ndarray, str, Path],
        confidence_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """Run inference on a single image.

        Args:
            image: RGB image as a NumPy array (H, W, 3), or a file path.
            confidence_threshold: Override the instance-level threshold for
                this call only.

        Returns:
            :class:`DetectionResult` containing boxes, scores, and labels.
        """
        if self._model is None:
            self._load_model()

        threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
        raw = self._run_inference(image)
        return raw.filter_by_confidence(threshold)

    def predict_batch(
        self,
        images: List[Union[np.ndarray, str, Path]],
        confidence_threshold: Optional[float] = None,
    ) -> List[DetectionResult]:
        """Run inference on a batch of images.

        Args:
            images: List of images (NumPy arrays or file paths).
            confidence_threshold: Override the instance-level threshold.

        Returns:
            List of :class:`DetectionResult`, one per input image.
        """
        return [self.predict(img, confidence_threshold) for img in images]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Resolve and load the model weights (lazy initialisation)."""
        checkpoint_path = self._resolve_checkpoint()
        # Actual weight loading will be wired up once the backbone module
        # is implemented.  For now we store the resolved path.
        self._checkpoint_path = checkpoint_path

    def _resolve_checkpoint(self) -> str:
        """Return a local path to the checkpoint, downloading if needed."""
        candidate = self.model_name_or_path

        # Local file — use directly
        if os.path.isfile(candidate):
            return candidate

        # Known pretrained name — download from registry
        if candidate in MODEL_REGISTRY:
            return self._download_checkpoint(candidate, MODEL_REGISTRY[candidate])

        raise ValueError(
            f"Unknown model '{candidate}'. "
            f"Available pretrained models: {list(MODEL_REGISTRY.keys())}"
        )

    @staticmethod
    def _download_checkpoint(name: str, url: str) -> str:
        """Download a checkpoint and cache it locally.

        Returns the path to the cached file.
        """
        cache_dir = Path.home() / ".cache" / "rfdetr"
        cache_dir.mkdir(parents=True, exist_ok=True)
        dest = cache_dir / f"{name}.pth"

        if dest.exists():
            return str(dest)

        import urllib.request

        print(f"Downloading RF-DETR checkpoint '{name}' from {url} …")
        urllib.request.urlretrieve(url, dest)
        print(f"Saved to {dest}")
        return str(dest)

    def _run_inference(self, image: Union[np.ndarray, str, Path]) -> DetectionResult:
        """Execute a forward pass and return raw (unfiltered) detections.

        This is a stub that returns empty detections until the backbone and
        transformer head modules are integrated.
        """
        return DetectionResult(
            boxes=np.empty((0, 4), dtype=np.float32),
            scores=np.empty((0,), dtype=np.float32),
            labels=np.empty((0,), dtype=np.int64),
        )

    @staticmethod
    def _default_device() -> str:
        """Select CUDA if available, otherwise fall back to CPU."""
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model='{self.model_name_or_path}', "
            f"device='{self.device}', "
            f"confidence_threshold={self.confidence_threshold})"
        )
