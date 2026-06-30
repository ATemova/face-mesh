# Face Mesh

Real-time facial landmark detection and mesh overlay in Python, built on
**MediaPipe** (Tasks API) and **OpenCV**. Detects **478 3D landmarks** per
face — including refined iris points — and renders the tesselation, feature
contours, and irises live from a webcam, an image, or a video file.

This is **v0.3.0** — adds a low-latency live-stream mode, gaze estimation,
per-eye blink counts with blink-rate, head-pose smoothing, and CSV export.

![mesh demo](render_demo.png)

> The image above is the output of `scripts/render_demo.py`, a no-camera
> self-test that draws synthetic landmarks. Real input produces a proper
> face-shaped mesh.

## Features

- 478-point face mesh (tesselation, contours, irises) with selectable feature sets
- Three running modes: **image**, **video**, and low-latency **live-stream** *(new)*
- Multi-face tracking (`--num-faces`)
- **Head-pose estimation** — pitch / yaw / roll, 3D gizmo, optional 1-Euro smoothing *(smoothing new)*
- **Blink detection** — per-eye counts plus a rolling blinks-per-minute rate *(per-eye + rate new)*
- **Gaze estimation** — direction from iris position relative to the eye *(new)*
- **Per-frame export** to JSONL **or CSV** (metrics, optionally full landmarks) *(CSV new)*
- Optional **expression blendshapes** overlay
- Live FPS meter and an interactive window
- Automatic one-time model download with local caching
- Small, documented, dependency-light codebase

## Requirements

- Python 3.9–3.12
- A webcam (only for live mode)

Install dependencies:

```bash
pip install -r requirements.txt
```

> `requirements.txt` uses `opencv-python` (with GUI support) so the live
> window works. On a headless server, swap it for `opencv-python-headless`
> and run with `--no-display --output ...`.

## Quick start

```bash
# Live webcam (device 0). The model auto-downloads on first run.
python main.py

# Annotate a single image -> writes photo_mesh.jpg next to it
python main.py --source photo.jpg

# Annotate a video -> writes an output file
python main.py --source clip.mp4 --output clip_mesh.mp4

# Show expression blendshapes and track up to two faces
python main.py --blendshapes --num-faces 2

# Just the iris + eye/lip contours, no full tesselation
python main.py --features contours

# Head-pose gizmo + blink counter live
python main.py --head-pose --blink

# Everything on, low-latency live-stream mode, smoothed head pose
python main.py --live --head-pose --blink --gaze --smooth

# Export per-frame metrics from a video to CSV (or .jsonl); add --export-landmarks for points
python main.py --source clip.mp4 --head-pose --blink --gaze --export metrics.csv
```

No webcam handy? Verify your install renders correctly with:

```bash
python scripts/render_demo.py   # writes render_demo.png
```

## Interactive keys (live window)

| Key | Action                          |
|-----|---------------------------------|
| `q` / `Esc` | Quit                    |
| `m` | Toggle the mesh on/off          |
| `f` | Cycle features (all → tesselation → contours → irises) |
| `p` | Toggle landmark points          |
| `b` | Toggle the blendshapes overlay  |
| `h` | Toggle the head-pose gizmo (needs `--head-pose` at launch) |
| `e` | Toggle blink/EAR metrics        |
| `g` | Toggle gaze estimation          |
| `s` | Save a snapshot (`snapshot_NNN.jpg`) |

## Command-line options

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | `0` | Webcam index, or path to image/video (or a stream URL) |
| `--model` | auto | Path to `face_landmarker.task` (downloaded if omitted) |
| `--features` | `all` | `tesselation` · `contours` · `irises` · `all` |
| `--num-faces` | `1` | Max faces to track |
| `--live` | off | Low-latency LIVE_STREAM mode (webcam only) |
| `--points` | off | Plot a dot at every landmark |
| `--blendshapes` | off | Show top expression blendshapes |
| `--head-pose` | off | Estimate pitch/yaw/roll and draw a 3D gizmo |
| `--smooth` | off | 1-Euro filter on head-pose angles |
| `--blink` | off | Per-eye EAR, blink counts and rate |
| `--gaze` | off | Estimate gaze direction from the irises |
| `--ear-threshold` | `0.21` | EAR below this counts as a closed eye |
| `--det-conf` | `0.5` | Min face-detection confidence |
| `--track-conf` | `0.5` | Min tracking confidence (video/live) |
| `--output` | — | Write annotated result to this path |
| `--export` | — | Per-frame metrics to `.jsonl` or `.csv` (by extension) |
| `--export-landmarks` | off | Include the 478 landmarks per face in the export |
| `--no-display` | off | Process without opening a window |

## Project structure

```
face-mesh/
├── main.py                 # CLI entry point (webcam / image / video)
├── requirements.txt
├── scripts/
│   └── render_demo.py      # no-camera render self-test
└── src/
    ├── detector.py         # FaceMeshDetector — image / video / live-stream
    ├── drawing.py          # mesh / gizmo / gaze / panel rendering + styles
    ├── metrics.py          # EAR + blink tracker, gaze, head-pose decomposition
    ├── filters.py          # 1-Euro filter + pose smoother
    ├── export.py           # per-frame JSONL / CSV writers
    ├── model.py            # model download & caching
    └── utils.py            # FPS meter, source resolution
```

## Use it as a library

```python
import cv2
from src import FaceMeshDetector, draw_face_landmarks, ensure_model

model = ensure_model()                      # downloads/caches the .task bundle
img = cv2.imread("photo.jpg")

with FaceMeshDetector(model, running_mode="image") as det:
    result = det.detect(img)

for face in result.face_landmarks:          # each face = 478 normalized landmarks
    draw_face_landmarks(img, face, ["all"])

cv2.imwrite("out.jpg", img)
```

## How it works

MediaPipe's `FaceLandmarker` runs a detection + landmark-regression graph and
returns 478 normalized `(x, y, z)` landmarks per face. The renderer maps those
to pixels and draws lines over the published connection topologies
(`FACE_LANDMARKS_TESSELATION`, `..._LIPS`, `..._LEFT_IRIS`, etc.). Webcam and
video files run in **VIDEO** mode for temporal tracking; images run in
**IMAGE** mode.

> **Note on MediaPipe versions:** the older `mp.solutions.face_mesh` API was
> removed in MediaPipe 0.10.30+. This project targets the current **Tasks API**
> (`mediapipe.tasks.python.vision.FaceLandmarker`), which needs the
> `face_landmarker.task` model bundle — handled automatically by `src/model.py`.

## Roadmap ideas

Done in v0.2: head-pose (pitch/yaw/roll + gizmo), blink/EAR, JSONL export.
Done in v0.3: live-stream mode, gaze direction, per-eye blink + rate, 1-Euro
head-pose smoothing, CSV export.

Next up:

- Calibration step for gaze (map raw ratios to screen coordinates)
- Drowsiness heuristic (sustained low EAR / high blink rate alerts)
- Multi-face metrics panel (not just the primary face)
- Record/replay of exported sessions with a small viewer
- Optional GPU delegate for the model

## License

MIT — do whatever you like; no warranty.
