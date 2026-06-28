"""Face Mesh — command line entry point.

Examples
--------
    # Webcam (default device 0), live window
    python main.py

    # A single image, written next to the input as *_mesh.jpg
    python main.py --source photo.jpg

    # A video file, rendered to an output file
    python main.py --source clip.mp4 --output clip_mesh.mp4

    # Show expression blendshapes, track up to 2 faces
    python main.py --blendshapes --num-faces 2

Interactive keys (windowed mode): q/ESC quit · s snapshot · m toggle mesh
· f cycle features · p toggle points · b toggle blendshapes.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src import (
    FaceMeshDetector,
    FPSMeter,
    draw_blendshapes_overlay,
    draw_face_landmarks,
    ensure_model,
    resolve_source,
)
from src.drawing import FEATURE_GROUPS

# Order in which the 'f' key cycles through feature presets.
_FEATURE_CYCLE = ["all", "tesselation", "contours", "irises"]


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time face mesh detection (MediaPipe + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="Webcam index (e.g. 0), or path to an image/video.")
    p.add_argument("--model", default=None,
                   help="Path to face_landmarker.task (auto-downloaded if omitted).")
    p.add_argument("--features", default="all", choices=list(FEATURE_GROUPS),
                   help="Which mesh features to draw.")
    p.add_argument("--num-faces", type=int, default=1,
                   help="Maximum number of faces to track.")
    p.add_argument("--points", action="store_true",
                   help="Also plot a dot at every landmark.")
    p.add_argument("--blendshapes", action="store_true",
                   help="Show top expression blendshapes as an overlay.")
    p.add_argument("--det-conf", type=float, default=0.5,
                   help="Minimum face detection confidence.")
    p.add_argument("--track-conf", type=float, default=0.5,
                   help="Minimum tracking confidence (video).")
    p.add_argument("--output", default=None,
                   help="Write annotated result to this path (image or video).")
    p.add_argument("--no-display", action="store_true",
                   help="Process without opening a window (implies --output for streams).")
    return p.parse_args(argv)


def _annotate(frame, result, features, *, points, show_blendshapes) -> None:
    """Draw all detected faces (and optional blendshapes) onto a frame."""
    for face in result.face_landmarks:
        draw_face_landmarks(frame, face, features, draw_points=points)
    if show_blendshapes and result.face_blendshapes:
        draw_blendshapes_overlay(frame, result.face_blendshapes[0])


def _hud(frame, text: str) -> None:
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 120), 1, cv2.LINE_AA)


def run_image(args, model_path: Path) -> int:
    img = cv2.imread(args.source)
    if img is None:
        print(f"error: could not read image {args.source!r}", file=sys.stderr)
        return 2

    with FaceMeshDetector(
        model_path, running_mode="image", num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        output_blendshapes=args.blendshapes,
    ) as det:
        result = det.detect(img)

    features = FEATURE_GROUPS[args.features]
    _annotate(img, result, features, points=args.points, show_blendshapes=args.blendshapes)
    n = len(result.face_landmarks)
    print(f"Detected {n} face(s).")

    out = args.output or str(Path(args.source).with_name(Path(args.source).stem + "_mesh.jpg"))
    cv2.imwrite(out, img)
    print(f"Saved -> {out}")

    if not args.no_display:
        try:
            cv2.imshow("Face Mesh (press any key to close)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass  # headless build, output already saved
    return 0


def run_stream(args, model_path: Path, kind: str, value) -> int:
    cap = cv2.VideoCapture(value)
    if not cap.isOpened():
        print(f"error: could not open source {value!r}", file=sys.stderr)
        return 2

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps or 25.0, (w, h))

    display = not args.no_display
    features = list(FEATURE_GROUPS[args.features])
    feature_idx = _FEATURE_CYCLE.index(args.features) if args.features in _FEATURE_CYCLE else 0
    mesh_on, points_on, blend_on = True, args.points, args.blendshapes
    fps = FPSMeter()
    frame_idx = 0
    snapshots = 0
    start = time.monotonic()

    with FaceMeshDetector(
        model_path, running_mode="video", num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        output_blendshapes=args.blendshapes,
    ) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # Webcams feel more natural mirrored; files are left as-is.
            if kind == "webcam":
                frame = cv2.flip(frame, 1)

            # Timestamp: frame-derived for files, wall-clock for webcam.
            if kind == "video" and src_fps > 0:
                ts_ms = int(frame_idx / src_fps * 1000)
            else:
                ts_ms = int((time.monotonic() - start) * 1000)

            result = det.detect(frame, timestamp_ms=ts_ms)
            if mesh_on:
                _annotate(frame, result, features, points=points_on, show_blendshapes=blend_on)

            current_fps = fps.tick()
            _hud(frame, f"{current_fps:4.1f} FPS | faces:{len(result.face_landmarks)} "
                        f"| {_FEATURE_CYCLE[feature_idx]} | q quit")

            if writer is not None:
                writer.write(frame)

            if display:
                try:
                    cv2.imshow("Face Mesh", frame)
                except cv2.error:
                    display = False  # headless; keep processing/writing
                else:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                    elif key == ord("m"):
                        mesh_on = not mesh_on
                    elif key == ord("p"):
                        points_on = not points_on
                    elif key == ord("b"):
                        blend_on = not blend_on
                    elif key == ord("f"):
                        feature_idx = (feature_idx + 1) % len(_FEATURE_CYCLE)
                        features = list(FEATURE_GROUPS[_FEATURE_CYCLE[feature_idx]])
                    elif key == ord("s"):
                        snap = f"snapshot_{snapshots:03d}.jpg"
                        cv2.imwrite(snap, frame)
                        print(f"Saved {snap}")
                        snapshots += 1

            frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved -> {args.output} ({frame_idx} frames)")
    cv2.destroyAllWindows()
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    model_path = ensure_model(args.model) if args.model else ensure_model()

    kind, value = resolve_source(args.source)
    if kind == "image":
        return run_image(args, model_path)
    return run_stream(args, model_path, kind, value)


if __name__ == "__main__":
    raise SystemExit(main())
