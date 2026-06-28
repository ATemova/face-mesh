"""Export per-frame detection results to JSON Lines (.jsonl).

One JSON object per processed frame, e.g.::

    {"frame": 0, "faces": [{"ear": 0.29, "blink_count": 0,
                            "head_pose": {"pitch": 3.1, "yaw": -12.4, "roll": 1.0}}]}

With ``include_landmarks=True`` each face also carries a ``landmarks`` list of
``[x, y, z]`` normalized coordinates. JSONL streams cleanly and handles a
variable number of faces per frame without a fixed schema.
"""

from __future__ import annotations

import json
from pathlib import Path


class MetricsExporter:
    """Append-friendly JSONL writer for per-frame face metrics."""

    def __init__(self, path: str | Path, include_landmarks: bool = False) -> None:
        self.path = Path(path)
        self.include_landmarks = include_landmarks
        self._fh = self.path.open("w", encoding="utf-8")
        self._frames = 0

    def write_frame(self, frame_idx: int, faces: list[dict]) -> None:
        """Write one frame's list of per-face metric dicts."""
        record = {"frame": frame_idx, "faces": faces}
        self._fh.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._frames += 1

    @staticmethod
    def landmarks_to_list(face_landmarks, ndigits: int = 5) -> list[list[float]]:
        """Convert a face's landmarks to a compact ``[[x, y, z], ...]`` list."""
        return [
            [round(lm.x, ndigits), round(lm.y, ndigits), round(lm.z, ndigits)]
            for lm in face_landmarks
        ]

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "MetricsExporter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
