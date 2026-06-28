"""Detector wrapper around MediaPipe's Tasks ``FaceLandmarker``.

The legacy ``mp.solutions.face_mesh`` API was removed in recent MediaPipe
releases (0.10.30+), so this project targets the current Tasks API. The
wrapper hides the running-mode plumbing and guarantees monotonically
increasing timestamps in VIDEO mode (a hard requirement of the API).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    FaceLandmarkerResult,
    RunningMode,
)

_RUNNING_MODES = {
    "image": RunningMode.IMAGE,
    "video": RunningMode.VIDEO,
}


class FaceMeshDetector:
    """Detect 478 facial landmarks per face.

    Args:
        model_path: Path to a ``face_landmarker.task`` bundle.
        running_mode: ``"image"`` for single shots, ``"video"`` for streams
            (webcam / video files). VIDEO mode enables temporal tracking.
        num_faces: Maximum number of faces to track.
        min_face_detection_confidence: Detection threshold in ``[0, 1]``.
        min_face_presence_confidence: Presence threshold in ``[0, 1]``.
        min_tracking_confidence: Tracking threshold in ``[0, 1]``.
        output_blendshapes: Also return 52 expression blendshape scores.
        output_transformation_matrixes: Also return 4x4 head-pose matrices.
    """

    def __init__(
        self,
        model_path: str | Path,
        running_mode: str = "video",
        num_faces: int = 1,
        min_face_detection_confidence: float = 0.5,
        min_face_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        output_blendshapes: bool = False,
        output_transformation_matrixes: bool = False,
    ) -> None:
        if running_mode not in _RUNNING_MODES:
            raise ValueError(
                f"running_mode must be one of {list(_RUNNING_MODES)}, got {running_mode!r}"
            )
        self.running_mode = running_mode
        self._last_ts_ms = -1

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=_RUNNING_MODES[running_mode],
            num_faces=num_faces,
            min_face_detection_confidence=min_face_detection_confidence,
            min_face_presence_confidence=min_face_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=output_blendshapes,
            output_facial_transformation_matrixes=output_transformation_matrixes,
        )
        self._landmarker = FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr, timestamp_ms: int | None = None) -> FaceLandmarkerResult:
        """Run detection on a single BGR frame (OpenCV's native format).

        Args:
            frame_bgr: ``H x W x 3`` uint8 array in BGR channel order.
            timestamp_ms: Frame timestamp for VIDEO mode. If omitted, a
                monotonic wall-clock value is used. Ignored in IMAGE mode.

        Returns:
            A :class:`FaceLandmarkerResult` with ``face_landmarks`` and,
            if enabled, ``face_blendshapes`` / transformation matrices.
        """
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        if self.running_mode == "image":
            return self._landmarker.detect(mp_image)

        # VIDEO mode requires strictly increasing timestamps.
        if timestamp_ms is None:
            timestamp_ms = int(time.monotonic() * 1000)
        if timestamp_ms <= self._last_ts_ms:
            timestamp_ms = self._last_ts_ms + 1
        self._last_ts_ms = timestamp_ms
        return self._landmarker.detect_for_video(mp_image, timestamp_ms)

    def close(self) -> None:
        """Release the underlying MediaPipe graph."""
        self._landmarker.close()

    def __enter__(self) -> "FaceMeshDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
