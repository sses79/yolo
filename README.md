# Roundabout AI

Local computer-vision demo using an Android phone running IP Webcam. Phase 0
provides resilient camera diagnostics, Phase 1 adds local YOLO vehicle/person
detection, and Phase 2 adds scene calibration, ByteTrack IDs, ROI filtering,
and directional line-crossing counts. Phase 3 adds durable CSV events and a
local Streamlit dashboard.

## Project setup

Python 3.14.6 is installed through pyenv on this Mac.

```bash
python3.14 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
```

Run the live diagnostic:

```bash
.venv/bin/roundabout-camera \
  --url http://192.168.1.142:8080/video
```

The URL defaults to `ROUNDABOUT_CAMERA_URL` when that environment variable is
set, otherwise it uses the address above.

Keyboard controls:

- `s`: save a full-resolution snapshot under `data/snapshots/`.
- `r`: start or stop a recording. A manually started recording is capped at
  120 seconds unless `--record-seconds` supplies another limit.
- `q` or Escape: quit cleanly.

Automatically start a five-minute recording:

```bash
.venv/bin/roundabout-camera --record-seconds 300
```

Run without a GUI for a connectivity or soak test:

```bash
.venv/bin/roundabout-camera --headless --duration 600
```

Save a snapshot during a headless test:

```bash
.venv/bin/roundabout-camera \
  --headless \
  --duration 10 \
  --snapshot-after 3
```

Use `--help` for timeout, reconnect, metrics, recording, and output options.

## Phase 3 Streamlit dashboard

Start the primary local dashboard:

```bash
.venv/bin/roundabout-dashboard
```

It binds to `127.0.0.1` by default and opens in the browser. The cached
background worker owns one camera reader and one YOLO model while processing;
normal Streamlit reruns only read its thread-safe latest snapshot. Use the
sidebar to:

- start and stop processing cleanly;
- select the camera, model, device, scene YAML, and event CSV;
- adjust confidence and toggle detection, scene, and metric overlays live;
- explicitly opt into event snapshots (off by default).

The timed live panel shows the annotated newest frame, camera and inference
health, reconnect/failure counters, person warnings, crossing totals, a traffic
chart, and the most recent events. An `Offline`, `Reconnecting`, or `Error`
banner includes the latest available cause. CSV metadata is retained locally;
raw video and event images are not retained by default.

The camera service also runs a wall-clock heartbeat. If all application threads
pause for at least ten seconds, as happens during Mac sleep, the resumed process
records the measured gap and resets the stale camera connection:

```text
system_resume_suspected gap_seconds=958
camera_connection_reset reason=system_resume
```

To select another port while keeping the localhost-only bind:

```bash
.venv/bin/roundabout-dashboard --server.port 8502
```

## Phase 1 vehicle and person detection

Run YOLO26n on the newest camera frame:

```bash
.venv/bin/roundabout-detect
```

The first run downloads `yolo26n.pt`. Model weights are ignored by Git. CPU is
the project default because it was slightly faster than Apple Metal in the
initial benchmark on this Mac. `--device auto` selects Metal (`mps`) when
available and otherwise CPU. The detector keeps only these COCO classes:
person, bicycle, car, motorcycle, bus, and truck.

Useful options:

```bash
.venv/bin/roundabout-detect \
  --device mps \
  --confidence 0.35 \
  --image-size 640
```

Controls:

- `s`: save the current annotated frame under `data/snapshots/`.
- `q` or Escape: quit cleanly.

Run a timed headless detection probe and save an annotated frame:

```bash
.venv/bin/roundabout-detect \
  --headless \
  --duration 30 \
  --snapshot-on-detection
```

Benchmark the same captured frame on CPU and MPS:

```bash
.venv/bin/roundabout-detect \
  --benchmark \
  --benchmark-warmup 2 \
  --benchmark-runs 10
```

Benchmark output reports total predict-call time, including model preprocess,
inference, and postprocess work. MPS is explicitly synchronized before and
after each timed call. An unavailable device is reported and skipped.

Phase 1 metrics separate camera input rate from AI throughput:

- capture FPS;
- inference FPS and latest inference milliseconds;
- frame age after inference;
- frames received, processed, and overwritten;
- current detections by class.

Overwritten frames are expected when inference is slower than the camera. They
show that stale frames are being discarded instead of accumulating latency.

## Phase 2 scene calibration, tracking, and counting

First capture a clean reference frame and interactively mark the road region.
Left-click at least three ROI polygon points, press Enter, then click endpoint
pairs for one or more count lines. Press `s` to save, `u` to undo, or `q` to
cancel:

```bash
.venv/bin/roundabout-calibrate
```

The default outputs are ignored local files:

- `data/calibration/reference.jpg` — the full-resolution calibration frame;
- `data/calibration/scene.yaml` — ROI and count-line coordinates.

An existing image can be calibrated without the live camera:

```bash
.venv/bin/roundabout-calibrate \
  --image data/snapshots/example.jpg \
  --output data/calibration/scene.yaml
```

Run detection with the calibrated scene:

```bash
.venv/bin/roundabout-detect \
  --scene-config data/calibration/scene.yaml
```

Confirmed line crossings are appended as metadata to
`data/events/events.csv`. Use `--event-file PATH` to select another location.
The file is created on the first event and contains UTC time, event type, count
line, object class, direction, track ID, and detection confidence. Plate fields
are reserved but remain empty; no image or raw video is retained by event
storage.

The live command now uses Ultralytics ByteTrack with persistent track IDs.
Detections whose box centre is outside the ROI are excluded. A crossing is
counted only when the tracked centre moves across the finite line segment,
the track is at least three observations old, and that track has not already
crossed that line. Reversing over the same line does not count twice.

Line direction depends on endpoint order. The generated YAML uses
`negative_to_positive` and `positive_to_negative`; rename those values to
scene-specific labels such as `entering` and `leaving` after calibration:

```yaml
count_lines:
  - name: north_entry
    start: [420, 610]
    end: [980, 590]
    negative_to_positive: entering
    positive_to_negative: leaving
```

Coordinates are stored against `reference_size` and scaled if the live frame
size differs. Current cumulative crossing totals are included in periodic
metrics. Each new crossing is also printed with line, direction, class,
track ID, and confidence and written to the event CSV. The Phase 3 Streamlit
dashboard presents the same pipeline through a cached background worker.

Useful tuning options are `--minimum-track-age`, `--maximum-missing-frames`,
and `--tracker-config`. Run without `--scene-config` to see tracking IDs over
the whole frame without filtering or counting.

## Metrics

The command periodically prints:

- actual decoded width and height;
- capture FPS and consumed/display FPS;
- frames received, consumed, and overwritten by the latest-frame buffer;
- failed reads and reconnect count;
- internal frame age and whether the stream has become stale.

`age_ms` measures delay inside this Python process, from capture to consumption.
An MJPEG stream does not provide a source timestamp, so this value cannot measure
the phone's encoding and network delay. A rising age or `STALE` state is still a
useful symptom of a blocked/frozen local pipeline.

## Tests

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pyright
```

The repository recommends the Ruff, Python, and Pylance VS Code extensions.
Workspace settings select `.venv` and the shared Ruff configuration;
`pyproject.toml` sets Pylance/Pyright to `standard` type-checking mode. Pyright
provides the equivalent type check in the terminal.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the dashboard and optional ANPR
phases. See the
[Phase 0 learning guide](docs/phase-0-learning-guide.md) for the architecture,
execution flow, test boundaries, and focused experiments behind that phase. The
[Phase 1 learning guide](docs/phase-1-learning-guide.md) explains the detector
boundary, live inference flow, performance measurements, and path to tracking.
The [Phase 2 learning guide](docs/phase-2-learning-guide.md) traces calibration,
persistent identity, ROI admission, finite-line crossing, deduplication, and the
remaining live accuracy boundary.
The [Phase 3 learning guide](docs/phase-3-learning-guide.md) explains durable
events, single-worker ownership, thread-safe dashboard snapshots, Streamlit
reruns, and sleep/reconnect evidence.
The [Ruff and Pyright learning guide](docs/ruff-pyright-learning-guide.md)
explains shared editor/CLI configuration, static contracts, runtime guards, and
how lint, typing, and tests provide different kinds of evidence.
