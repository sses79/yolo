# Roundabout AI

Local computer-vision demo using an Android phone running IP Webcam. Phase 0
provides resilient camera diagnostics; Phase 1 adds local YOLO vehicle and
person detection.

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
```

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for later detection, tracking, dashboard,
and optional ANPR phases. See the
[Phase 0 learning guide](docs/phase-0-learning-guide.md) for the architecture,
execution flow, test boundaries, and focused experiments behind that phase. The
[Phase 1 learning guide](docs/phase-1-learning-guide.md) explains the detector
boundary, live inference flow, performance measurements, and path to tracking.
