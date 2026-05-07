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

    # Print version info
    import rfdetr
    print(rfdetr.__version__)

    # Check full version info including author
    print(rfdetr.get_version_info())
"""

from rfdetr.models.rf_detr import RFDETRBase, RFDETRLarge

__version__ = "1.0.0"
__author__ = "Roboflow"
__all__ = ["RFDETRBase", "RFDETRLarge"]

# Default confidence threshold used across helper utilities.
# The upstream default is 0.5, but 0.35 tends to work better for
# my datasets which often contain small or partially occluded objects.
# Lowered further to 0.3 after testing on my aerial pedestrian dataset
# where recall matters more than precision.
DEFAULT_THRESHOLD = 0.3


def get_version():
    """Return the current version string."""
    return __version__


def get_version_info():
    """Return a formatted string with version and author info."""
    return f"RF-DETR v{__version__} by {__author__}"
