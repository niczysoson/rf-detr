# Copyright 2024 Roboflow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Inference utilities for RF-DETR object detection models."""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from PIL import Image

from rfdetr.config import RFDETRConfig
from rfdetr.model import DetectionResult


def load_image(source: Union[str, Path, np.ndarray, Image.Image]) -> Image.Image:
    """Load an image from various source types into a PIL Image.

    Args:
        source: Image source — can be a file path (str or Path),
                a NumPy array (HxWxC, uint8), or an existing PIL Image.

    Returns:
        A PIL Image in RGB mode.

    Raises:
        TypeError: If the source type is not supported.
        FileNotFoundError: If a path is given but the file does not exist.
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
        return Image.open(path).convert("RGB")

    if isinstance(source, np.ndarray):
        return Image.fromarray(source).convert("RGB")

    if isinstance(source, Image.Image):
        return source.convert("RGB")

    raise TypeError(
        f"Unsupported image source type: {type(source)}. "
        "Expected str, Path, np.ndarray, or PIL.Image.Image."
    )


def preprocess_images(
    images: List[Image.Image],
    input_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Resize, normalise and batch a list of PIL images into a model-ready tensor.

    Images are resized to ``(input_size, input_size)``, converted to float32
    tensors in [0, 1], and normalised with ImageNet statistics.

    Args:
        images: List of PIL Images (RGB).
        input_size: Target spatial resolution (square).
        device: Torch device to place the output tensor on.

    Returns:
        Float32 tensor of shape ``(N, 3, input_size, input_size)``.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    tensors = []
    for img in images:
        img_resized = img.resize((input_size, input_size), Image.BILINEAR)
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0  # HxWxC
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # 1xCxHxW
        tensors.append(t)

    batch = torch.cat(tensors, dim=0).to(device)  # NxCxHxW
    batch = (batch - mean) / std
    return batch


def postprocess_detections(
    raw_logits: torch.Tensor,
    raw_boxes: torch.Tensor,
    orig_sizes: List[tuple],
    confidence_threshold: float = 0.5,
) -> List[DetectionResult]:
    """Convert raw model outputs into :class:`DetectionResult` objects.

    Args:
        raw_logits: Class logits tensor of shape ``(N, num_queries, num_classes)``.
        raw_boxes: Bounding-box tensor of shape ``(N, num_queries, 4)`` in
                   normalised ``[cx, cy, w, h]`` format.
        orig_sizes: List of ``(width, height)`` tuples for each image in the batch,
                    used to scale boxes back to pixel coordinates.
        confidence_threshold: Minimum score to keep a detection.

    Returns:
        A list of :class:`DetectionResult` instances, one per image.
    """
    scores = torch.sigmoid(raw_logits)  # (N, Q, C)
    results: List[DetectionResult] = []

    for i, (orig_w, orig_h) in enumerate(orig_sizes):
        img_scores, img_labels = scores[i].max(dim=-1)  # (Q,)
        img_boxes_norm = raw_boxes[i]  # (Q, 4)  cx,cy,w,h in [0,1]

        # Convert normalised cx,cy,w,h -> x1,y1,x2,y2 in pixel space
        cx, cy, bw, bh = img_boxes_norm.unbind(-1)
        x1 = (cx - bw / 2) * orig_w
        y1 = (cy - bh / 2) * orig_h
        x2 = (cx + bw / 2) * orig_w
        y2 = (cy + bh / 2) * orig_h
        boxes_px = torch.stack([x1, y1, x2, y2], dim=-1)  # (Q, 4)

        result = DetectionResult(
            boxes=boxes_px.cpu().numpy(),
            labels=img_labels.cpu().numpy(),
            scores=img_scores.cpu().numpy(),
        )
        results.append(result.filter_by_confidence(confidence_threshold))

    return results
