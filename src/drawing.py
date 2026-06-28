"""Rendering helpers — turn landmarks into a mesh overlay with OpenCV.

Colors are BGR tuples (OpenCV convention). The public entry point is
:func:`draw_face_landmarks`, which renders one or more "feature sets"
(tesselation, contours, irises, ...) onto a frame.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2

from mediapipe.tasks.python.vision import FaceLandmarksConnections as _C

# Each feature maps to (connection_set, line_color_BGR, thickness).
# Connection sets are iterables of objects with ``.start`` / ``.end`` indices.
FEATURE_SETS = {
    "tesselation": (_C.FACE_LANDMARKS_TESSELATION, (90, 110, 90), 1),
    "face_oval": (_C.FACE_LANDMARKS_FACE_OVAL, (224, 224, 224), 1),
    "lips": (_C.FACE_LANDMARKS_LIPS, (120, 120, 255), 1),
    "left_eye": (_C.FACE_LANDMARKS_LEFT_EYE, (80, 230, 80), 1),
    "right_eye": (_C.FACE_LANDMARKS_RIGHT_EYE, (80, 230, 80), 1),
    "left_eyebrow": (_C.FACE_LANDMARKS_LEFT_EYEBROW, (80, 230, 80), 1),
    "right_eyebrow": (_C.FACE_LANDMARKS_RIGHT_EYEBROW, (80, 230, 80), 1),
    "nose": (_C.FACE_LANDMARKS_NOSE, (224, 224, 224), 1),
    "left_iris": (_C.FACE_LANDMARKS_LEFT_IRIS, (0, 200, 255), 2),
    "right_iris": (_C.FACE_LANDMARKS_RIGHT_IRIS, (0, 200, 255), 2),
}

# Convenient named groups the CLI exposes via --features.
FEATURE_GROUPS = {
    "tesselation": ["tesselation"],
    "contours": [
        "face_oval", "lips", "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow", "nose",
    ],
    "irises": ["left_iris", "right_iris"],
    "all": [
        "tesselation", "face_oval", "lips", "left_eye", "right_eye",
        "left_eyebrow", "right_eyebrow", "nose", "left_iris", "right_iris",
    ],
}


def _to_pixels(landmarks, width: int, height: int) -> list[tuple[int, int]]:
    """Map normalized landmark coords to integer pixel coordinates."""
    pts = []
    for lm in landmarks:
        x = min(max(lm.x, 0.0), 1.0)
        y = min(max(lm.y, 0.0), 1.0)
        pts.append((int(x * width), int(y * height)))
    return pts


def _draw_feature(image, pts: Sequence[tuple[int, int]], connections, color, thickness) -> None:
    n = len(pts)
    for conn in connections:
        s, e = conn.start, conn.end
        if s < n and e < n:
            cv2.line(image, pts[s], pts[e], color, thickness, cv2.LINE_AA)


def draw_face_landmarks(
    image,
    face_landmarks,
    features: Iterable[str] = ("tesselation", "contours", "irises"),
    *,
    draw_points: bool = False,
    point_color: tuple[int, int, int] = (0, 255, 0),
) -> None:
    """Draw the mesh for one face, in place.

    Args:
        image: BGR frame to draw on (modified in place).
        face_landmarks: A single face's list of normalized landmarks.
        features: Feature names or groups (see ``FEATURE_GROUPS`` /
            ``FEATURE_SETS``). Unknown names are ignored.
        draw_points: Also plot a dot at every landmark.
        point_color: BGR color for landmark dots.
    """
    height, width = image.shape[:2]
    pts = _to_pixels(face_landmarks, width, height)

    # Expand groups (e.g. "contours") into concrete feature names, de-duped.
    resolved: list[str] = []
    for f in features:
        for name in FEATURE_GROUPS.get(f, [f]):
            if name not in resolved:
                resolved.append(name)

    for name in resolved:
        spec = FEATURE_SETS.get(name)
        if spec is None:
            continue
        connections, color, thickness = spec
        _draw_feature(image, pts, connections, color, thickness)

    if draw_points:
        for p in pts:
            cv2.circle(image, p, 1, point_color, -1, cv2.LINE_AA)


def draw_blendshapes_overlay(image, blendshapes, top_n: int = 5) -> None:
    """Render the strongest expression blendshapes as a small HUD panel.

    Args:
        image: BGR frame (modified in place).
        blendshapes: One face's list of blendshape ``Category`` objects.
        top_n: How many of the highest-scoring expressions to show.
    """
    if not blendshapes:
        return
    ranked = sorted(blendshapes, key=lambda c: c.score, reverse=True)[:top_n]

    pad, line_h, bar_w = 8, 22, 120
    panel_w = 230
    panel_h = pad * 2 + line_h * len(ranked)
    x0, y0 = 10, 10

    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for i, cat in enumerate(ranked):
        y = y0 + pad + i * line_h + 14
        name = (cat.category_name or "")[:14]
        cv2.putText(image, name, (x0 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (235, 235, 235), 1, cv2.LINE_AA)
        bx = x0 + panel_w - bar_w - pad
        cv2.rectangle(image, (bx, y - 9), (bx + bar_w, y - 1), (60, 60, 60), -1)
        fill = int(bar_w * min(max(cat.score, 0.0), 1.0))
        cv2.rectangle(image, (bx, y - 9), (bx + fill, y - 1), (0, 200, 255), -1)


def draw_head_pose_axes(image, origin_xy, axis_offsets, *, thickness: int = 2) -> None:
    """Draw a 3D orientation gizmo (X red, Y green, Z blue) at ``origin_xy``.

    Args:
        image: BGR frame (modified in place).
        origin_xy: ``(x, y)`` pixel anchor — typically the nose tip.
        axis_offsets: three ``(dx, dy)`` screen offsets from
            :func:`metrics.head_pose_axes_2d`, in X, Y, Z order.
    """
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    colors = ((0, 0, 255), (0, 255, 0), (255, 60, 60))  # X, Y, Z in BGR
    for (dx, dy), color in zip(axis_offsets, colors):
        cv2.line(image, (ox, oy), (ox + dx, oy + dy), color, thickness, cv2.LINE_AA)
    cv2.circle(image, (ox, oy), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_metrics_panel(image, lines, *, anchor: str = "br") -> None:
    """Render a small translucent text panel of metric lines.

    Args:
        image: BGR frame (modified in place).
        lines: list of short strings to display, one per row.
        anchor: ``"br"`` (bottom-right) or ``"tr"`` (top-right).
    """
    if not lines:
        return
    h, w = image.shape[:2]
    pad, line_h = 8, 20
    panel_w = 200
    panel_h = pad * 2 + line_h * len(lines)
    x1 = w - panel_w - 10
    y1 = 10 if anchor == "tr" else h - panel_h - 10

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x1 + panel_w, y1 + panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for i, text in enumerate(lines):
        y = y1 + pad + i * line_h + 13
        cv2.putText(image, text, (x1 + pad, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (60, 255, 120), 1, cv2.LINE_AA)
