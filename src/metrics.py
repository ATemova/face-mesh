"""Derived metrics — eye-aspect-ratio / blink detection and head pose.

These build on the raw 478 landmarks (and the optional facial transformation
matrix) returned by the detector. Nothing here touches MediaPipe directly, so
the functions are easy to unit-test with plain numbers.
"""

from __future__ import annotations

import math

import numpy as np

# --- Eye landmarks (MediaPipe 478-point topology) -------------------------
# Each list is the classic 6-point EAR set, ordered:
#   [corner, top-a, top-b, corner, bottom-b, bottom-a]
# "left"/"right" are the subject's eyes (left eye appears on the right of a
# non-mirrored image).
EYE_LANDMARKS_LEFT = (362, 385, 387, 263, 373, 380)
EYE_LANDMARKS_RIGHT = (33, 160, 158, 133, 153, 144)

# Nose tip — a stable anchor for the head-pose gizmo.
NOSE_TIP = 1


def _px(landmark, width: int, height: int) -> np.ndarray:
    return np.array([landmark.x * width, landmark.y * height], dtype=float)


def eye_aspect_ratio(landmarks, indices, width: int, height: int) -> float:
    """Eye Aspect Ratio for one eye, computed in pixel space.

    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|). Open eyes sit around
    0.25-0.35; a closed eye drops toward ~0.1.
    """
    p = [_px(landmarks[i], width, height) for i in indices]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = np.linalg.norm(p[0] - p[3])
    return float(vertical / (2.0 * horizontal)) if horizontal > 0 else 0.0


def average_ear(landmarks, width: int, height: int) -> float:
    """Mean EAR across both eyes."""
    left = eye_aspect_ratio(landmarks, EYE_LANDMARKS_LEFT, width, height)
    right = eye_aspect_ratio(landmarks, EYE_LANDMARKS_RIGHT, width, height)
    return (left + right) / 2.0


class BlinkCounter:
    """Count blinks via EAR thresholding with a short debounce.

    A blink is registered when the eye reopens after having been below
    ``ear_threshold`` for at least ``consec_frames`` consecutive frames.
    """

    def __init__(self, ear_threshold: float = 0.21, consec_frames: int = 2) -> None:
        self.ear_threshold = ear_threshold
        self.consec_frames = consec_frames
        self.count = 0
        self._below = 0
        self._closed = False

    def update(self, ear: float) -> int:
        if ear < self.ear_threshold:
            self._below += 1
            if self._below >= self.consec_frames:
                self._closed = True
        else:
            if self._closed:
                self.count += 1  # rising edge: eye just reopened
            self._below = 0
            self._closed = False
        return self.count

    @property
    def is_closed(self) -> bool:
        return self._closed


# --- Head pose ------------------------------------------------------------
def head_pose_from_matrix(matrix) -> tuple[float, float, float]:
    """Decompose a facial transformation matrix into Euler angles (degrees).

    Args:
        matrix: 4x4 (or 3x3) facial transformation matrix from the detector
            (``output_transformation_matrixes=True``).

    Returns:
        ``(pitch, yaw, roll)`` in degrees — nod, shake, tilt respectively.
    """
    R = np.asarray(matrix, dtype=float)[:3, :3]
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(R[2, 1], R[2, 2])
        yaw = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:  # gimbal-lock fallback
        pitch = math.atan2(-R[1, 2], R[1, 1])
        yaw = math.atan2(-R[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def head_pose_axes_2d(matrix, length: float = 70.0) -> list[tuple[int, int]]:
    """Project the rotated unit axes to 2D screen offsets.

    Returns three ``(dx, dy)`` pixel offsets for the X, Y and Z axes,
    relative to a chosen origin (image y grows downward, so it is negated).
    """
    R = np.asarray(matrix, dtype=float)[:3, :3]
    axes = np.eye(3) * length
    offsets = []
    for axis in axes:
        v = R @ axis
        offsets.append((int(round(v[0])), int(round(-v[1]))))
    return offsets
