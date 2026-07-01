"""Face Mesh — command line entry point.

Examples
--------
    # Everything on, low-latency, smoothed head pose
    python main.py --live --head-pose --blink --gaze --smooth

    # Driver-monitoring style drowsiness detection
    python main.py --blink --drowsiness

    # Calibrate gaze to the screen, then show a gaze cursor
    python main.py --gaze --calibrate --save-calibration cal.json

    # Track several faces with a panel each
    python main.py --num-faces 3 --blink --multiface

    # Export a session with landmarks, then replay it offline:
    python main.py --source clip.mp4 --blink --gaze --head-pose \
        --export session.jsonl --export-landmarks
    python scripts/replay.py session.jsonl

Interactive keys: q/ESC quit · s snapshot · m mesh · f features · p points
· b blendshapes · h head pose · e blink · g gaze · d drowsiness.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from . import (
    BlinkTracker,
    DrowsinessMonitor,
    FaceMeshDetector,
    FaceTracker,
    FPSMeter,
    GazeCalibrator,
    PoseSmoother,
    create_exporter,
    draw_blendshapes_overlay,
    draw_calibration_target,
    draw_drowsiness_banner,
    draw_face_landmarks,
    draw_face_panels,
    draw_gaze,
    draw_gaze_cursor,
    draw_head_pose_axes,
    draw_metrics_panel,
    ensure_model,
    gaze_direction,
    grid_targets,
    head_pose_axes_2d,
    head_pose_from_matrix,
    landmarks_to_list,
    resolve_source,
)
from .drawing import FEATURE_GROUPS
from .metrics import NOSE_TIP, average_ear, mouth_aspect_ratio

_FEATURE_CYCLE = ["all", "tesselation", "contours", "irises"]
_LEVEL_RANK = {"awake": 0, "drowsy": 1, "ALERT": 2}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time face mesh detection (MediaPipe + OpenCV).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="0",
                   help="Webcam index (e.g. 0), or path to an image/video.")
    p.add_argument("--model", default=None,
                   help="Path to face_landmarker.task (auto-downloaded if omitted).")
    p.add_argument("--features", default="all", choices=list(FEATURE_GROUPS))
    p.add_argument("--num-faces", type=int, default=1)
    p.add_argument("--live", action="store_true",
                   help="Low-latency LIVE_STREAM mode (webcam only).")
    p.add_argument("--gpu", action="store_true",
                   help="Try the GPU delegate (falls back to CPU if unavailable).")
    p.add_argument("--points", action="store_true")
    p.add_argument("--blendshapes", action="store_true")
    p.add_argument("--head-pose", action="store_true")
    p.add_argument("--smooth", action="store_true",
                   help="1-Euro filter on head-pose angles.")
    p.add_argument("--blink", action="store_true",
                   help="Per-eye EAR, blink counts and rate.")
    p.add_argument("--gaze", action="store_true")
    p.add_argument("--drowsiness", action="store_true",
                   help="PERCLOS / microsleep / yawn monitoring with alerts.")
    p.add_argument("--multiface", action="store_true",
                   help="Show a metrics panel per face (auto when num-faces>1).")
    p.add_argument("--calibrate", action="store_true",
                   help="Run a 9-point gaze calibration first (webcam only).")
    p.add_argument("--save-calibration", default=None,
                   help="Save the fitted gaze calibration to this JSON file.")
    p.add_argument("--load-calibration", default=None,
                   help="Load a gaze calibration JSON (skips --calibrate).")
    p.add_argument("--ear-threshold", type=float, default=0.21)
    p.add_argument("--det-conf", type=float, default=0.5)
    p.add_argument("--track-conf", type=float, default=0.5)
    p.add_argument("--output", default=None)
    p.add_argument("--export", default=None,
                   help="Per-frame metrics to .jsonl or .csv (by extension).")
    p.add_argument("--export-landmarks", action="store_true")
    p.add_argument("--no-display", action="store_true")
    return p.parse_args(argv)


class FaceState:
    """Per-face stateful helpers."""

    def __init__(self, ear_threshold: float) -> None:
        self.blink = BlinkTracker(ear_threshold)
        self.smoother = PoseSmoother()
        self.drowsiness = DrowsinessMonitor(ear_threshold)


def _worse(a: str, b: str) -> str:
    return a if _LEVEL_RANK[a] >= _LEVEL_RANK[b] else b


def _process_face(frame, result, idx, face_id, opts, states, now_s):
    """Draw + compute metrics for one face.

    Returns ``(title, lines, record, level)``.
    """
    h, w = frame.shape[:2]
    face = result.face_landmarks[idx]
    state = states.setdefault(face_id, FaceState(opts["ear_threshold"]))
    lines: list[str] = []
    record: dict = {}
    level = "awake"

    if opts["blink"]:
        bm = state.blink.update(face, w, h, now_s)
        record["ear"] = bm["ear"]
        record["blink"] = {k: bm[k] for k in ("left", "right", "total", "rate_per_min")}
        tag = "closed" if bm["closed"] else "open"
        lines.append(f"EAR {bm['ear']:.2f} ({tag})")
        lines.append(f"Blinks {bm['total']} (L{bm['left']}/R{bm['right']})")
        lines.append(f"Rate {bm['rate_per_min']:.0f}/min")

    if opts["gaze"]:
        gaze = gaze_direction(face)
        draw_gaze(frame, face, gaze)
        record["gaze"] = dict(gaze)
        lines.append(f"Gaze {gaze['dir']}")
        cal = opts["calibrator"]
        if cal is not None and cal.is_fitted:
            gx, gy = cal.apply(gaze["h"], gaze["v"])
            draw_gaze_cursor(frame, (gx, gy))
            record["gaze"]["screen"] = [round(gx, 1), round(gy, 1)]

    if opts["head_pose"] and result.facial_transformation_matrixes:
        matrix = result.facial_transformation_matrixes[idx]
        pitch, yaw, roll = head_pose_from_matrix(matrix)
        if opts["smooth"]:
            pitch, yaw, roll = state.smoother.update(pitch, yaw, roll, now_s)
        nose = face[NOSE_TIP]
        draw_head_pose_axes(frame, (nose.x * w, nose.y * h), head_pose_axes_2d(matrix))
        record["head_pose"] = {"pitch": round(pitch, 1), "yaw": round(yaw, 1),
                               "roll": round(roll, 1)}
        lines.append(f"P{pitch:+.0f} Y{yaw:+.0f} R{roll:+.0f}")

    if opts["drowsiness"]:
        ear = record.get("ear") or average_ear(face, w, h)
        mar = mouth_aspect_ratio(face, w, h)
        dm = state.drowsiness.update(ear, mar, now_s)
        level = dm["level"]
        record["drowsiness"] = {k: dm[k] for k in ("perclos", "level", "microsleeps", "yawns")}
        lines.append(f"PERCLOS {dm['perclos']*100:.0f}% [{dm['level']}]")

    if opts["export_landmarks"]:
        record["landmarks"] = landmarks_to_list(face)

    return f"Face {face_id}", lines, record, level


def _hud(frame, text: str) -> None:
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 255, 120), 1, cv2.LINE_AA)


def _opts_from_args(args, calibrator) -> dict:
    return {
        "blink": args.blink, "gaze": args.gaze, "head_pose": args.head_pose,
        "smooth": args.smooth, "drowsiness": args.drowsiness,
        "multiface": args.multiface or args.num_faces > 1,
        "ear_threshold": args.ear_threshold, "calibrator": calibrator,
        "export_landmarks": bool(args.export and args.export_landmarks),
    }


def _render_panels(frame, panels, opts, level) -> None:
    if opts["multiface"] and len(panels) > 1:
        draw_face_panels(frame, panels)
    elif panels:
        draw_metrics_panel(frame, panels[0][1])
    if opts["drowsiness"]:
        draw_drowsiness_banner(frame, level)


def run_calibration(model_path, args, cap):
    """Interactive 9-point gaze calibration on the webcam. Returns a fitted
    :class:`GazeCalibrator`, or None if aborted."""
    ok, frame = cap.read()
    if not ok:
        return None
    h, w = cv2.flip(frame, 1).shape[:2]
    cal = GazeCalibrator()
    with FaceMeshDetector(model_path, running_mode="video",
                          min_face_detection_confidence=args.det_conf) as det:
        for (tx, ty) in grid_targets(w, h):
            samples: list[tuple[float, float]] = []
            start = time.monotonic()
            while (elapsed := time.monotonic() - start) < 1.6:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                res = det.detect(frame, timestamp_ms=int(time.monotonic() * 1000))
                if res.face_landmarks and elapsed > 0.5:  # skip the saccade in
                    g = gaze_direction(res.face_landmarks[0])
                    samples.append((g["h"], g["v"]))
                draw_calibration_target(frame, (tx, ty), elapsed / 1.6)
                _hud(frame, "Gaze calibration — look at the dot (ESC to cancel)")
                cv2.imshow("Face Mesh", frame)
                if (cv2.waitKey(1) & 0xFF) == 27:
                    return None
            if samples:
                mh = sum(s[0] for s in samples) / len(samples)
                mv = sum(s[1] for s in samples) / len(samples)
                cal.add_sample(mh, mv, tx, ty)
    if cal.n_samples >= 3:
        cal.fit()
        if args.save_calibration:
            cal.save(args.save_calibration)
            print(f"Saved gaze calibration -> {args.save_calibration}")
        return cal
    print("calibration failed: not enough samples.", file=sys.stderr)
    return None


def run_image(args, model_path: Path) -> int:
    img = cv2.imread(args.source)
    if img is None:
        print(f"error: could not read image {args.source!r}", file=sys.stderr)
        return 2

    calibrator = GazeCalibrator.load(args.load_calibration) if args.load_calibration else None
    with FaceMeshDetector(
        model_path, running_mode="image", num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        output_blendshapes=args.blendshapes,
        output_transformation_matrixes=args.head_pose,
        delegate="gpu" if args.gpu else "cpu",
    ) as det:
        result = det.detect(img)

    features = FEATURE_GROUPS[args.features]
    opts, states = _opts_from_args(args, calibrator), {}
    panels, records, level = [], [], "awake"
    for idx, face in enumerate(result.face_landmarks):
        draw_face_landmarks(img, face, features, draw_points=args.points)
        title, lines, rec, lvl = _process_face(img, result, idx, idx, opts, states, 0.0)
        panels.append((title, lines))
        records.append(rec)
        level = _worse(level, lvl)

    if args.blendshapes and result.face_blendshapes:
        draw_blendshapes_overlay(img, result.face_blendshapes[0])
    _render_panels(img, panels, opts, level)

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

    # Gaze calibration (webcam only, needs a window).
    calibrator = None
    if args.load_calibration:
        calibrator = GazeCalibrator.load(args.load_calibration)
        print(f"Loaded gaze calibration <- {args.load_calibration}")
    elif args.calibrate and kind == "webcam" and not args.no_display:
        calibrator = run_calibration(model_path, args, cap)

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
    opts = _opts_from_args(args, calibrator)
    fps = FPSMeter()
    states: dict[int, FaceState] = {}
    tracker = FaceTracker()
    frame_idx = 0
    snapshots = 0
    start = time.monotonic()

    with FaceMeshDetector(
        model_path, running_mode=running_mode, num_faces=args.num_faces,
        min_face_detection_confidence=args.det_conf,
        min_tracking_confidence=args.track_conf,
        output_blendshapes=args.blendshapes,
        output_transformation_matrixes=args.head_pose,
        delegate="gpu" if args.gpu else "cpu",
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

            face_ids = tracker.update(result.face_landmarks)
            panels, records, level = [], [], "awake"
            for idx, face in enumerate(result.face_landmarks):
                if mesh_on:
                    draw_face_landmarks(frame, face, features, draw_points=points_on)
                title, lines, rec, lvl = _process_face(
                    frame, result, idx, face_ids[idx], opts, states, now_s)
                panels.append((title, lines))
                records.append(rec)
                level = _worse(level, lvl)

            if blend_on and result.face_blendshapes:
                draw_blendshapes_overlay(frame, result.face_blendshapes[0])
            _render_panels(frame, panels, opts, level)
            if exporter is not None:
                exporter.write_frame(frame_idx, records)

            mode_tag = "live" if running_mode == "live_stream" else running_mode
            _hud(frame, f"{fps.tick():4.1f} FPS | faces:{len(result.face_landmarks)} "
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
                    elif key == ord("d"):
                        opts["drowsiness"] = not opts["drowsiness"]
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
