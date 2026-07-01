"""Gaze calibration — map raw iris ratios to screen coordinates.

Raw gaze from :func:`metrics.gaze_direction` is a normalized ``(h, v)`` pair
that only loosely corresponds to where a person is looking. Calibration fits
an affine map from ``(h, v)`` to actual screen ``(x, y)`` using a handful of
known fixation points (typically a 3x3 grid), turning tendency into a usable
on-screen gaze cursor.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class GazeCalibrator:
    """Collect (raw_gaze -> screen) samples and fit an affine mapping.

    The mapping is ``[x, y]^T = A @ [h, v, 1]^T`` where ``A`` is 2x3, solved
    by least squares. Needs at least 3 non-collinear points; a 5- or 9-point
    grid is more robust.
    """

    def __init__(self) -> None:
        self._h: list[float] = []
        self._v: list[float] = []
        self._x: list[float] = []
        self._y: list[float] = []
        self.matrix: np.ndarray | None = None  # 2x3

    def add_sample(self, raw_h: float, raw_v: float, screen_x: float, screen_y: float) -> None:
        self._h.append(raw_h)
        self._v.append(raw_v)
        self._x.append(screen_x)
        self._y.append(screen_y)

    @property
    def n_samples(self) -> int:
        return len(self._h)

    @property
    def is_fitted(self) -> bool:
        return self.matrix is not None

    def fit(self) -> np.ndarray:
        """Solve for the 2x3 affine matrix; returns it."""
        if self.n_samples < 3:
            raise ValueError("need at least 3 calibration points to fit")
        A = np.column_stack([self._h, self._v, np.ones(self.n_samples)])  # N x 3
        targets = np.column_stack([self._x, self._y])                     # N x 2
        sol, *_ = np.linalg.lstsq(A, targets, rcond=None)                 # 3 x 2
        self.matrix = sol.T                                               # 2 x 3
        return self.matrix

    def apply(self, raw_h: float, raw_v: float) -> tuple[float, float]:
        """Map a raw gaze ratio to screen coordinates (requires a fit)."""
        if self.matrix is None:
            raise RuntimeError("calibrator is not fitted; call fit() first")
        vec = np.array([raw_h, raw_v, 1.0])
        x, y = self.matrix @ vec
        return float(x), float(y)

    def save(self, path: str | Path) -> None:
        if self.matrix is None:
            raise RuntimeError("nothing to save; calibrator is not fitted")
        Path(path).write_text(json.dumps({"matrix": self.matrix.tolist()}))

    @classmethod
    def load(cls, path: str | Path) -> "GazeCalibrator":
        data = json.loads(Path(path).read_text())
        obj = cls()
        obj.matrix = np.asarray(data["matrix"], dtype=float)
        return obj


def grid_targets(width: int, height: int, margin: float = 0.12) -> list[tuple[int, int]]:
    """A 3x3 grid of screen fixation points inset from the edges."""
    xs = [margin, 0.5, 1 - margin]
    ys = [margin, 0.5, 1 - margin]
    return [(int(x * width), int(y * height)) for y in ys for x in xs]
