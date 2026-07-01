"""Face Mesh — real-time facial landmark detection and mesh overlay.

A clean wrapper around MediaPipe's Tasks ``FaceLandmarker`` API (478 3D
landmarks) with OpenCV rendering for webcam, image and video input. Includes
head-pose (with 1-Euro smoothing), per-eye blink + rate, gaze estimation with
screen calibration, drowsiness monitoring (PERCLOS / microsleep / yawns),
multi-face panels, a live-stream mode, an optional GPU delegate, JSONL/CSV
export, and an offline replay viewer.
"""

from .calibration import GazeCalibrator, grid_targets
from .detector import FaceMeshDetector
from .recognition import FaceTracker
from .drawing import (
    FEATURE_SETS,
    draw_blendshapes_overlay,
    draw_calibration_target,
    draw_drowsiness_banner,
    draw_face_landmarks,
    draw_face_panels,
    draw_gaze,
    draw_gaze_cursor,
    draw_head_pose_axes,
    draw_metrics_panel,
)
from .drowsiness import DrowsinessMonitor
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
    mouth_aspect_ratio,
)
from .model import DEFAULT_MODEL_URL, ensure_model
from .utils import FPSMeter, resolve_source

__all__ = [
    "FaceMeshDetector",
    "FaceTracker",
    "GazeCalibrator",
    "grid_targets",
    "DrowsinessMonitor",
    "FEATURE_SETS",
    "draw_face_landmarks",
    "draw_blendshapes_overlay",
    "draw_gaze",
    "draw_gaze_cursor",
    "draw_head_pose_axes",
    "draw_metrics_panel",
    "draw_face_panels",
    "draw_drowsiness_banner",
    "draw_calibration_target",
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
    "mouth_aspect_ratio",
    "gaze_direction",
    "head_pose_from_matrix",
    "head_pose_axes_2d",
    "ensure_model",
    "DEFAULT_MODEL_URL",
    "FPSMeter",
    "resolve_source",
]

__version__ = "1.0.0"
