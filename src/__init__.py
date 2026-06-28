"""Face Mesh — real-time facial landmark detection and mesh overlay.

A small, clean wrapper around MediaPipe's Tasks ``FaceLandmarker`` API
(478 3D landmarks) with OpenCV rendering for webcam, image and video input.
"""

from .detector import FaceMeshDetector
from .drawing import FEATURE_SETS, draw_face_landmarks, draw_blendshapes_overlay
from .model import ensure_model, DEFAULT_MODEL_URL
from .utils import FPSMeter, resolve_source

__all__ = [
    "FaceMeshDetector",
    "FEATURE_SETS",
    "draw_face_landmarks",
    "draw_blendshapes_overlay",
    "ensure_model",
    "DEFAULT_MODEL_URL",
    "FPSMeter",
    "resolve_source",
]

__version__ = "0.1.0"
