# Roundabout AI

Local computer-vision demo using an Android phone running IP Webcam. Phase 0
provides a resilient camera diagnostic before any AI model is introduced.

## Phase 0 setup

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
execution flow, test boundaries, and focused experiments behind this phase.
