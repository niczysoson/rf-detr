# Copyright 2024 Roboflow. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RF-DETR: Real-time object detection transformer model.

This package provides a high-performance, real-time object detection model
based on the DETR (Detection Transformer) architecture, optimized for
production use cases.

Example usage::

    from rfdetr import RFDETRBase

    model = RFDETRBase()
    detections = model.predict("image.jpg")

    # Run inference with a custom confidence threshold
    detections = model.predict("image.jpg", threshold=0.4)
"""

from rfdetr.models.rf_detr import RFDETRBase, RFDETRLarge

__version__ = "1.0.0"
__author__ = "Roboflow"
__all__ = ["RFDETRBase", "RFDETRLarge"]
