<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python" />
  <img src="https://img.shields.io/badge/Computer%20Vision-MediaPipe-purple?style=flat-square" />
  <img src="https://img.shields.io/badge/Framework-OpenCV-darkgreen?style=flat-square&logo=opencv" />
  <img src="https://img.shields.io/badge/Tracking-478%20Facial%20Landmarks-informational?style=flat-square" />
  <img src="https://img.shields.io/badge/Head%20Pose-Pitch%20%7C%20Yaw%20%7C%20Roll-orange?style=flat-square" />
  <img src="https://img.shields.io/badge/Gaze-Screen%20Calibration-blueviolet?style=flat-square" />
  <img src="https://img.shields.io/badge/Drowsiness-PERCLOS%20Detection-red?style=flat-square" />
  <img src="https://img.shields.io/badge/Multi--Face-Supported-009688?style=flat-square" />
  <img src="https://img.shields.io/badge/Acceleration-GPU%20Delegate-success?style=flat-square" />
  <img src="https://img.shields.io/badge/Export-CSV%20%7C%20JSONL-lightgrey?style=flat-square" />
  <img src="https://img.shields.io/badge/Replay-Offline%20Viewer-795548?style=flat-square" />
  <img src="https://img.shields.io/badge/Status-v1.0.0-brightgreen?style=flat-square" />
</p>

# Face Mesh

Real-time facial landmark detection and analysis in Python, built on
**MediaPipe** (Tasks API) and **OpenCV**. Tracks **478 3D landmarks** per face
and layers on head pose, blink/gaze, drowsiness, and multi-face identity —
from a webcam, image, or video.

**v1.0.0** is the first stable release: an installable package, a test suite in
CI, polynomial gaze calibration with an error readout, session timeline plots,
and stable per-person tracking.

![mesh demo](render_demo.png)

> The image above comes from `scripts/render_demo.py`, a no-camera self-test.
> Real input produces a proper face-shaped mesh.

## Features

- 478-point mesh (tesselation, contours, irises) with selectable feature sets
- Image, video, and low-latency live-stream modes; optional GPU delegate
- Head pose (pitch/yaw/roll) with a 3D gizmo and 1-Euro smoothing
- Per-eye blink counts + blinks-per-minute rate
- Gaze estimation with optional screen calibration (affine or quadratic)
- Drowsiness monitoring — PERCLOS, microsleep and yawn alerts
- Multi-face tracking that keeps metrics attached to the same person
- Export to JSONL / CSV, with an offline replay viewer and timeline plots

## Install

```bash
pip install .                 # or: pip install -e ".[dev]" for tests + plots
```

This installs the `facemesh` command. To run straight from the source tree
without installing, use `python main.py` instead.

## Quick start

```bash
facemesh                                   # live webcam mesh
facemesh --live --head-pose --blink --gaze --smooth   # everything on
facemesh --blink --drowsiness              # driver-monitoring style
facemesh --gaze --calibrate                # 9-point gaze calibration
facemesh --source photo.jpg                # annotate an image
```

Run `facemesh --help` for the full list of options. No webcam? Verify your
install with `python scripts/render_demo.py`.

Interactive keys: `q` quit · `m` mesh · `f` features · `h` head pose ·
`e` blink · `g` gaze · `d` drowsiness · `s` snapshot.

## Architecture

```mermaid
flowchart TD
    SRC["Webcam / Image / Video"] --> DET["FaceMeshDetector<br/>(MediaPipe Tasks)"]
    DET -->|"478 landmarks<br/>+ pose matrix + blendshapes"| TRK["Face Tracker<br/>(stable IDs)"]
    TRK --> PROC{"Per-face processing"}
    PROC --> MET["Metrics<br/>EAR / gaze / MAR / head pose"]
    PROC --> FIL["Filters<br/>1-Euro smoothing"]
    PROC --> CAL["Gaze Calibration<br/>screen mapping"]
    PROC --> DRO["Drowsiness<br/>PERCLOS / microsleep / yawns"]
    MET --> RND["Rendering<br/>mesh / panels / banners / cursor"]
    FIL --> RND
    CAL --> RND
    DRO --> RND
    RND --> DISP["Live display"]
    MET --> EXP["Export<br/>JSONL / CSV"]
    EXP --> RPL["Replay viewer"]
    EXP --> PLT["Timeline plots"]

    classDef input fill:#1565c0,stroke:#0d47a1,color:#fff;
    classDef core fill:#6a1b9a,stroke:#4a148c,color:#fff;
    classDef proc fill:#2e7d32,stroke:#1b5e20,color:#fff;
    classDef out fill:#e65100,stroke:#bf360c,color:#fff;
    class SRC input;
    class DET,TRK core;
    class PROC,MET,FIL,CAL,DRO proc;
    class RND,DISP,EXP,RPL,PLT out;
```

## Analyze a session

Export with landmarks, then replay or plot it offline — no camera or model:

```bash
facemesh --source clip.mp4 --blink --drowsiness \
    --export session.jsonl --export-landmarks
python scripts/replay.py session.jsonl          # re-render the mesh
python scripts/plot_session.py session.jsonl    # EAR / blink / PERCLOS timeline
```

## As a library

```python
import cv2
from facemesh import FaceMeshDetector, draw_face_landmarks, ensure_model

model = ensure_model()                 # downloads/caches the .task bundle
img = cv2.imread("photo.jpg")
with FaceMeshDetector(model, running_mode="image") as det:
    result = det.detect(img)
for face in result.face_landmarks:     # each face = 478 normalized landmarks
    draw_face_landmarks(img, face, ["all"])
cv2.imwrite("out.jpg", img)
```

## Notes

MediaPipe's `FaceLandmarker` returns 478 normalized landmarks per face; the
renderer maps them to pixels over the published connection topologies. The
legacy `mp.solutions.face_mesh` API was removed in MediaPipe 0.10.30+, so this
project targets the current **Tasks API**, which needs the `face_landmarker.task`
bundle — downloaded automatically on first run.

## Roadmap

Shipped through v1.0.0: head pose, blink/gaze, drowsiness, live-stream, GPU
delegate, JSONL/CSV export, replay, calibration, multi-face tracking, tests.
Next: appearance-based re-identification, a validation UI for calibration, and
richer session analytics.

## License

MIT.
