"""Face Mesh — command line entry point.

Examples
--------
    # Webcam, low-latency live-stream mode, with everything on
    python main.py --live --head-pose --blink --gaze --smooth

    # A single image -> writes *_mesh.jpg next to it
    python main.py --source photo.jpg --head-pose --gaze

    # A video, rendered out + metrics to CSV (or .jsonl)
    python main.py --source clip.mp4 --output clip_mesh.mp4 \
        --head-pose --blink --gaze --export metrics.csv

Interactive keys (windowed mode): q/ESC quit · s snapshot · m mesh
· f cycle features · p points · b blendshapes · h head pose · e blink · g gaze.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from src import (
    BlinkTracker,
    FaceMeshDetector,
    FPSMeter,
    PoseSmoother,
    create_exporter,
    draw_blendshapes_overlay,
    draw_face_landmarks,
    draw_gaze,
    draw_head_pose_axes,
    draw_metrics_panel,
    ensure_model,
    gaze_direction,
    head_pose_axes_2d,
    head_pose_from_matrix,
    landmarks_to_list,
    resolve_source,
)
from src.drawing import FEATURE_GROUPS
from src.metrics import NOSE_TIP

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
    p.add_argument("--live", action="store_true",
                   help="Use low-latency LIVE_STREAM mode (webcam only).")
    p.add_argument("--points", action="store_true",
                   help="Also plot a dot at every landmark.")
    p.add_argument("--blendshapes", action="store_true",
                   help="Show top expression blendshapes as an overlay.")
    p.add_argument("--head-pose", action="store_true",
                   help="Estimate head pose (pitch/yaw/roll) and draw axes.")
    p.add_argument("--smooth", action="store_true",
                   help="Apply a 1-Euro filter to head-pose angles.")
    p.add_argument("--blink", action="store_true",
                   help="Per-eye Eye Aspect Ratio, blink counts and rate.")
    p.add_argument("--gaze", action="store_true",
                   help="Estimate gaze direction from iris position.")
    p.add_argument("--ear-threshold", type=float, default=0.21,
                   help="EAR below this counts as a closed eye (blink).")
    p.add_argument("--det-conf", type=float, default=0.5,
                   help="Minimum face detection confidence.")
    p.add_argument("--track-conf", type=float, default=0.5,
                   help="Minimum tracking confidence (video/live).")
    p.add_argument("--output", default=None,
                   help="Write annotated result to this path (image or video).")
    p.add_argument("--export", default=None,
                   help="Write per-frame metrics to .jsonl or .csv (by extension).")
    p.add_argument("--export-landmarks", action="store_true",
                   help="Include the 478 landmarks per face in the export.")
    p.add_argument("--no-display", action="store_true",
                   help="Process without opening a window.")
    return p.parse_args(argv)


class FaceState:
    """Per-face stateful helpers (blink tracker + pose smoother)."""

    def __init__(self, ear_threshold: float) -> None:
        self.blink = BlinkTracker(ear_threshold)
        self.smoother = PoseSmoother()


def _process_face(frame, result, idx, opts, states, now_s):
    """Compute + draw metrics for one face; return (panel_lines, record)."""
    h, w = frame.shape[:2]
    face = result.face_landmarks[idx]
    state = states.setdefault(idx, FaceState(opts["ear_threshold"]))
    lines: list[str] = []
    record: dict = {}

    if opts["blink"]:
        bm = state.blink.update(face, w, h, now_s)
        record["ear"] = bm["ear"]
        record["blink"] = {k: bm[k] for k in ("left", "right", "total", "rate_per_min")}
        if idx == 0:
            tag = "closed" if bm["closed"] else "open"
            lines.append(f"EAR {bm['ear']:.2f} ({tag})")
            lines.append(f"Blinks {bm['total']} (L{bm['left']}/R{bm['right']})")
            lines.append(f"Rate {bm['rate_per_min']:.0f}/min")

    if opts["gaze"]:
        gaze = gaze_direction(face)
        draw_gaze(frame, face, gaze)
        record["gaze"] = gaze
        if idx == 0:
            lines.append(f"Gaze {gaze['dir']}")

    if opts["head_pose"] and result.facial_transformation_matrixes:
        matrix = result.facial_transformation_matrixes[idx]
        pitch, yaw, roll = head_pose_from_matrix(matrix)
        if opts["smooth"]:
            pitch, yaw, roll = state.smoother.update(pitch, yaw, roll, now_s)
        nose = face[NOSE_TIP]
        draw_head_pose_axes(frame, (nose.x * w, nose.y * h), head_pose_axes_2d(matrix))
        record["head_pose"] = {
            "pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1),
        }
        if idx == 0:
            lines.append(f"P{pitch:+.0f} Y{yaw:+.0f} R{roll:+.0f}")

    if opts["export_landmarks"]:
        record["landmarks"] = landmarks_to_list(face)

    return lines, record


def _hud(frame, text: str) -> None:
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 120), 1, cv2.LINE_AA)


def _opts_from_args(args) -> dict:
    return {
        "blink": args.blink, "gaze": args.gaze, "head_pose": args.head_pose,
        "smooth": args.smooth, "ear_threshold": args.ear_threshold,
        "export_landmarks": bool(args.export and args.export_landmarks),
    }


def run_image(args, model_path: Path) -> int:
    img = cv2.imread(args.source)
    if img is None:
        print(f"error: could not read image {args.source!r}", file=sys.stderr)
        return 2

    with FaceMeshDetector(
        model_path, running_mode="image", num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        output_blendshapes=args.blendshapes,
        output_transformation_matrixes=args.head_pose,
    ) as det:
        result = det.detect(img)

    features = FEATURE_GROUPS[args.features]
    opts, states = _opts_from_args(args), {}
    panel_lines, records = [], []
    for idx, face in enumerate(result.face_landmarks):
        draw_face_landmarks(img, face, features, draw_points=args.points)
        lines, rec = _process_face(img, result, idx, opts, states, now_s=0.0)
        if idx == 0:
            panel_lines = lines
        records.append(rec)

    if args.blendshapes and result.face_blendshapes:
        draw_blendshapes_overlay(img, result.face_blendshapes[0])
    if panel_lines:
        draw_metrics_panel(img, panel_lines)

    print(f"Detected {len(result.face_landmarks)} face(s).")
    if args.export:
        with create_exporter(args.export) as exp:
            exp.write_frame(0, records)
        print(f"Exported metrics -> {args.export}")

    out = args.output or str(Path(args.source).with_name(Path(args.source).stem + "_mesh.jpg"))
    cv2.imwrite(out, img)
    print(f"Saved -> {out}")

    if not args.no_display:
        try:
            cv2.imshow("Face Mesh (press any key to close)", img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return 0


def run_stream(args, model_path: Path, kind: str, value) -> int:
    cap = cv2.VideoCapture(value)
    if not cap.isOpened():
        print(f"error: could not open source {value!r}", file=sys.stderr)
        return 2

    # LIVE_STREAM only makes sense for a live camera; files use VIDEO mode.
    if args.live and kind != "webcam":
        print("note: --live is ignored for files; using VIDEO mode.", file=sys.stderr)
    running_mode = "live_stream" if (args.live and kind == "webcam") else "video"

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    writer = None
    if args.output:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.output, fourcc, src_fps or 25.0, (w, h))

    exporter = create_exporter(args.export) if args.export else None

    display = not args.no_display
    features = list(FEATURE_GROUPS[args.features])
    feature_idx = _FEATURE_CYCLE.index(args.features) if args.features in _FEATURE_CYCLE else 0
    mesh_on, points_on, blend_on = True, args.points, args.blendshapes
    opts = _opts_from_args(args)
    fps = FPSMeter()
    states: dict[int, FaceState] = {}
    frame_idx = 0
    snapshots = 0
    start = time.monotonic()

    with FaceMeshDetector(
        model_path, running_mode=running_mode, num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        output_blendshapes=args.blendshapes,
        output_transformation_matrixes=args.head_pose,
    ) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if kind == "webcam":
                frame = cv2.flip(frame, 1)

            now_s = time.monotonic() - start
            if kind == "video" and src_fps > 0:
                ts_ms = int(frame_idx / src_fps * 1000)
                now_s = frame_idx / src_fps
            else:
                ts_ms = int(now_s * 1000)

            result = det.detect(frame, timestamp_ms=ts_ms)

            panel_lines, records = [], []
            for idx, face in enumerate(result.face_landmarks):
                if mesh_on:
                    draw_face_landmarks(frame, face, features, draw_points=points_on)
                lines, rec = _process_face(frame, result, idx, opts, states, now_s)
                if idx == 0:
                    panel_lines = lines
                records.append(rec)

            if blend_on and result.face_blendshapes:
                draw_blendshapes_overlay(frame, result.face_blendshapes[0])
            if panel_lines:
                draw_metrics_panel(frame, panel_lines)
            if exporter is not None:
                exporter.write_frame(frame_idx, records)

            current_fps = fps.tick()
            mode_tag = "live" if running_mode == "live_stream" else running_mode
            _hud(frame, f"{current_fps:4.1f} FPS | faces:{len(result.face_landmarks)} "
                        f"| {_FEATURE_CYCLE[feature_idx]} | {mode_tag} | q quit")

            if writer is not None:
                writer.write(frame)

            if display:
                try:
                    cv2.imshow("Face Mesh", frame)
                except cv2.error:
                    display = False
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
                    elif key == ord("h"):
                        opts["head_pose"] = not opts["head_pose"]
                    elif key == ord("e"):
                        opts["blink"] = not opts["blink"]
                    elif key == ord("g"):
                        opts["gaze"] = not opts["gaze"]
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
    if exporter is not None:
        exporter.close()
        print(f"Exported metrics -> {args.export} ({frame_idx} frames)")
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
