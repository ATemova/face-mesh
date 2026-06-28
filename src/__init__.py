"""Face Mesh — real-time facial landmark detection and mesh overlay.

A small, clean wrapper around MediaPipe's Tasks ``FaceLandmarker`` API
(478 3D landmarks) with OpenCV rendering for webcam, image and video input.
Includes head-pose estimation, blink detection and per-frame export.
"""

from .detector import FaceMeshDetector
from .drawing import (
    FEATURE_SETS,
    draw_blendshapes_overlay,
    draw_face_landmarks,
    draw_head_pose_axes,
    draw_metrics_panel,
)
from .export import MetricsExporter
from .metrics import (
    BlinkCounter,
    average_ear,
    eye_aspect_ratio,
    head_pose_axes_2d,
    head_pose_from_matrix,
)
from .model import DEFAULT_MODEL_URL, ensure_model
from .utils import FPSMeter, resolve_source

__all__ = [
    "FaceMeshDetector",
    "FEATURE_SETS",
    "draw_face_landmarks",
    "draw_blendshapes_overlay",
    "draw_head_pose_axes",
    "draw_metrics_panel",
    "MetricsExporter",
    "BlinkCounter",
    "average_ear",
    "eye_aspect_ratio",
    "head_pose_from_matrix",
    "head_pose_axes_2d",
    "ensure_model",
    "DEFAULT_MODEL_URL",
    "FPSMeter",
    "resolve_source",
]

__version__ = "0.2.0"
