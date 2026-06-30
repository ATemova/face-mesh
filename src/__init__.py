"""Face Mesh — real-time facial landmark detection and mesh overlay.

A clean wrapper around MediaPipe's Tasks ``FaceLandmarker`` API (478 3D
landmarks) with OpenCV rendering for webcam, image and video input. Includes
head-pose estimation (with optional 1-Euro smoothing), blink detection with
per-eye counts and blink-rate, gaze estimation, a low-latency live-stream
mode, and per-frame export to JSONL or CSV.
"""

from .detector import FaceMeshDetector
from .drawing import (
    FEATURE_SETS,
    draw_blendshapes_overlay,
    draw_face_landmarks,
    draw_gaze,
    draw_head_pose_axes,
    draw_metrics_panel,
)
from .export import (
    CsvExporter,
    JsonlExporter,
    MetricsExporter,
    create_exporter,
    flatten_face,
    landmarks_to_list,
)
from .filters import OneEuroFilter, PoseSmoother
from .metrics import (
    BlinkCounter,
    BlinkTracker,
    average_ear,
    eye_aspect_ratio,
    gaze_direction,
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
    "draw_gaze",
    "draw_head_pose_axes",
    "draw_metrics_panel",
    "create_exporter",
    "CsvExporter",
    "JsonlExporter",
    "MetricsExporter",
    "flatten_face",
    "landmarks_to_list",
    "OneEuroFilter",
    "PoseSmoother",
    "BlinkCounter",
    "BlinkTracker",
    "average_ear",
    "eye_aspect_ratio",
    "gaze_direction",
    "head_pose_from_matrix",
    "head_pose_axes_2d",
    "ensure_model",
    "DEFAULT_MODEL_URL",
    "FPSMeter",
    "resolve_source",
]

__version__ = "0.3.0"
