"""Render self-test — verify drawing works without a camera or model.

This synthesizes a structured 478-point landmark cloud and renders every
mesh feature onto a blank canvas. It exercises the full drawing pipeline
(normalized->pixel mapping, all connection sets, blendshape overlay) so you
can confirm your install renders correctly before plugging in a webcam.

    python scripts/render_demo.py            # writes render_demo.png

Note: the output is an abstract mesh, not a real face — real landmark
positions come from the model at runtime via main.py.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.drawing import FEATURE_GROUPS, draw_blendshapes_overlay, draw_face_landmarks


@dataclass
class _LM:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0
    name: str = ""


@dataclass
class _Cat:
    score: float
    category_name: str
    index: int = 0
    display_name: str = ""


def synth_landmarks(n: int = 478, seed: int = 7) -> list[_LM]:
    """478 deterministic points inside a face-shaped oval."""
    rng = np.random.default_rng(seed)
    pts: list[_LM] = []
    for _ in range(n):
        # Sample within a centered ellipse so the mesh looks face-ish.
        r = math.sqrt(rng.random())
        theta = rng.random() * 2 * math.pi
        x = 0.5 + 0.30 * r * math.cos(theta)
        y = 0.5 + 0.38 * r * math.sin(theta)
        pts.append(_LM(x=x, y=y, z=rng.random() * 0.1))
    return pts


def main() -> int:
    canvas = np.full((600, 600, 3), 18, dtype=np.uint8)
    landmarks = synth_landmarks()

    draw_face_landmarks(canvas, landmarks, FEATURE_GROUPS["all"], draw_points=True)

    fake_blendshapes = [
        _Cat(0.82, "mouthSmileLeft"), _Cat(0.79, "mouthSmileRight"),
        _Cat(0.41, "eyeBlinkLeft"), _Cat(0.12, "browInnerUp"),
        _Cat(0.05, "jawOpen"),
    ]
    draw_blendshapes_overlay(canvas, fake_blendshapes)

    out = Path(__file__).resolve().parent.parent / "render_demo.png"
    cv2.imwrite(str(out), canvas)
    print(f"Rendered {len(landmarks)} landmarks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
