# Roundabout AI

Local computer-vision demo using an Android phone running IP Webcam. Phase 0
provides resilient camera diagnostics, Phase 1 adds local YOLO vehicle/person
detection, and Phase 2 adds scene calibration, ByteTrack IDs, ROI filtering,
and directional line-crossing counts. Phase 3 adds durable CSV events and a
local Streamlit dashboard. Phase 4 adds the evidence gate that decides whether
the camera view is suitable for a later ANPR prototype, Phase 5 provides the
offline prototype, and Phase 6 adds bounded camera-quality recommendations and
opt-in preset control. Phase 7 joins verified camera context to every OCR
outcome and permits only evidence-backed profile mappings to replace the
baseline automatic policy.

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
- explicitly opt into event snapshots and best-of-track vehicle crops (off by
  default).

The timed live panel shows the annotated newest frame, camera and inference
health, reconnect/failure counters, person warnings, crossing totals, a traffic
chart, and the most recent events. Recent events include `slow`, `fast`, or
`unknown` plus the measured relative speed. When event-image saving is enabled,
the table also shows a compact crossing-crop preview after the timestamp. An
`Offline`, `Reconnecting`, or `Error`
banner includes the latest available cause. CSV metadata is retained locally;
raw video and event images are not retained by default.

When **Save event snapshot and vehicle crops** is enabled, every confirmed
crossing saves one full annotated `snapshot` frame. The worker also temporarily
retains a bounded set of raw vehicle crops for each live track and saves
distinct candidates for the exact `crossing` frame, the first suitable
`approach` view before foreground occlusion, the most complete `centred` and
`largest` views, and the `sharpest` moving-vehicle view. Sharpness ranking uses
the vehicle box rather than padded static scenery. Vehicle boxes receive
35% horizontal and 10% vertical padding, and candidates close enough to an
image edge to lose that padding are marked clipped. Boxes smaller than 200x100
pixels are excluded from the crop set; the full event snapshot is still saved.
If one crop wins more than one category, it is saved only once. Raw crops are
taken before drawing detection boxes, ROI lines, or metrics. All files are
written under `data/events/images/` with the track ID and image kind in the
filename. The crop padding and minimum dimensions can be changed under
**Advanced** before starting processing.

For a speed-labelled re-collection, leave the default speed settings initially
and enable event images. The worker measures bottom-centre motion using real
capture timestamps and divides it by vehicle-box height, producing relative
`box heights/second`. A track remains `unknown` until it has at least three
observations spanning 0.5 seconds; values at or above the default `1.0`
threshold are `fast`, and lower values are `slow`. Tune the threshold under
**Advanced** after reviewing representative tracks. This supports sample
stratification but is not mph or km/h; physical speed requires road-plane and
distance calibration. Existing Phase 3 CSV files are upgraded in place with
`unknown` for historical rows when the next event is written.

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

## Phase 6 adaptive camera configuration

Open **Advanced → Adaptive camera (Phase 6)** in the dashboard. Start with
**Recommend only** and enter the IP Webcam base control URL explicitly, for
example `http://192.168.1.142:8080` (without `/video`). The worker reads
`/status.json?show_avail=1`, saves the reported snapshot to
`data/camera/capabilities.json`, and does not write a setting in recommendation
mode.

Every five seconds it measures the calibrated road ROI rather than the whole
image: luminance percentiles, dark/bright clipping, Laplacian sharpness,
directional blur, and noise. Three consecutive matching observations are
required before the dashboard recommends `day`, `glare`, `dusk`, or `night`.
This is a deterministic image-quality baseline, not an LLM decision.

While processing is running, use **Apply profile** to test the recommendation.
The camera client permits only known imaging settings, validates each value
against this phone's reported capabilities, reads the state back after applying
it, and rolls back if verification fails. It never controls torch or flash. A
manual or automatic request waits until the road ROI contains no detected road
user, so it does not change exposure/focus during a crossing. New CSV events
include `camera_profile` and the non-sensitive allowlisted settings snapshot;
historical rows are left blank during automatic schema migration.

Only after manually reviewing all four presets under representative conditions
should you select **Automatic (experimental)** and check the explicit
confirmation box. Automatic switching uses the same three-observation
hysteresis, waits at least five minutes between successful profile changes, and
waits at least one minute after a failed attempt. These timings are configurable
before startup. A guarded recommendation moves only one step through
`night → dusk → day → glare`, refuses to reduce low-light assistance when at
least 10% of ROI pixels are underexposed, and uses recent moving-vehicle
sharpness to avoid adding exposure blur unless underexposure is severe. Moving
sharpness evidence and the temporary block on a recently underexposed profile
expire after 15 minutes. Verified read-back settings are
matched to a known preset after restart, so the guard cannot be bypassed by
losing the in-memory current-profile name. The preset choices are starting hypotheses—validate readable
plate rate and sharpness on separate day, direct-sun, dusk, and night batches.
If night images remain blurred, the controller must report the camera/view limit
rather than lengthening exposure indefinitely.

## Phase 7 OCR-driven camera optimisation

Phase 7 connects each crossing's camera context to its complete OCR outcome.
New event rows distinguish `accepted`, `uncertain`, `no_read`, and `not_run`;
record whether the plate detector found anything; retain detector confidence,
plate size/sharpness, OCR observations/agreement/reasons; and attach the latest
ROI condition and quality measurements. Plate text is still masked at the CSV
boundary. Existing CSV schemas migrate in place, with unavailable historical
measurements left blank and historical blank OCR marked `not_run` rather than
inventing a failure.

The dashboard's **Recent OCR performance by verified profile** table groups its
bounded recent-event window by profile, lighting condition, direction, and
speed class. It shows plate-detection coverage and accepted/uncertain/no-read outcomes. Mean OCR
confidence is calculated only for accepted results and is always shown beside
acceptance coverage; it is not presented as accuracy. The Adaptive camera panel
also shows the actual read-back settings and last verified change time.

Collect comparable manual-profile batches before promoting a Phase 7 mapping.
Copy [`camera_profiles.example.json`](camera_profiles.example.json) to the
ignored `data/camera/` directory, replace its placeholder metrics with measured
coverage and held-out false-read results, set `operator_approved` only after
review, and enter that local path in **Validated OCR profile mapping** before
starting Automatic mode. The loader requires all four conditions, at least 30
samples per condition by default, non-decreasing OCR acceptance coverage, and
non-increasing held-out false reads. Invalid or incomplete mappings stop worker
startup instead of silently controlling the camera.

Without a validated mapping, Automatic mode keeps the Phase 6 identity mapping
(`day -> day`, `glare -> glare`, and so on) and labels it **automatic baseline
mapping**. This permits continued evidence collection without claiming the
baseline presets are OCR-optimal.

## Phase 4 ANPR feasibility gate

Phase 4 does not run OCR. It first tests whether this camera view contains
enough readable plate pixels to justify an ANPR prototype.

The best starting sample is produced by running `roundabout-dashboard` with
**Save event snapshot and vehicle crops** enabled. For each crossing, inspect the
distinct `crossing`, `centred`, `largest`, and `sharpest` files under
`data/events/images/` and use the most readable candidate as one vehicle
observation. The wider horizontal context is intentional because a general
vehicle detector can omit a front or rear bumper and its number plate. Do not
count the alternatives as independent samples because they belong to the same
tracked vehicle.

Create a local CSV template:

```bash
.venv/bin/roundabout-anpr-assess init
```

This creates `data/anpr/annotations.csv`. Add one row per consent-appropriate
local image with:

- an anonymous sample ID and local image path;
- approaching/departing direction, lighting, slow/fast/unknown speed, and
  front/rear;
- `yes`, `no`, or `uncertain` human readability;
- approximate plate width, plate height, and character height in pixels;
- optional quality notes such as blur, glare, skew, or clipping.

There is intentionally no registration-text column. Images, annotations, and
reports under `data/anpr/` are ignored by Git.

Append a validated observation without editing CSV directly:

```bash
.venv/bin/roundabout-anpr-assess add \
  --sample-id sample-001 \
  --image-path data/anpr/images/sample-001.jpg \
  --direction approaching \
  --lighting day \
  --speed slow \
  --plate-side front \
  --human-readable yes \
  --plate-width 92 \
  --plate-height 25 \
  --character-height 18
```

Use `--notes "motion blur"` for relevant quality observations. The `add`
command creates the template automatically if it does not exist and rejects a
duplicate sample ID.

Generate the evidence report:

```bash
.venv/bin/roundabout-anpr-assess report \
  --required-lighting day,shade,glare
```

The default policy requires at least 20 samples, both directions, both speed
groups, and both plate sides. It recommends `proceed` only when at least 80%
are human-readable and median character height is at least 16 pixels;
readability below 20% recommends `stop_at_vehicle_analytics`, while intermediate
results recommend `reposition_first`. Missing quantity or condition coverage
returns `incomplete_sample`. Override these thresholds on the command line when
the project has a documented reason; they are feasibility policy, not an OCR
accuracy guarantee.

Use `speed=unknown` when a still image has no trustworthy speed evidence. Such
rows remain valid observations, but they do not satisfy the report's required
slow/fast coverage.

Typical improvements before collecting a new sample are a lower camera angle,
closer optical view, more light, or a faster shutter. OCR remains Phase 5 and
should begin only after reviewing `data/anpr/report.md`.

## Phase 5 local and live ANPR/OCR

Phase 5 can analyze one image, process saved vehicle crops, or run on confirmed
dashboard crossings. Supply a local YOLO model trained to detect number plates;
`yolo26n.pt` is a general COCO road-user model and cannot be substituted for a
plate detector.

First verify the Python 3.14 OCR installation and PP-OCRv6 small recognizer:

```bash
.venv/bin/roundabout-anpr smoke-test
```

For the simplest one-image check, pass a vehicle crop directly. The JSON result
contains `status`, `ocr_plate`, and `ocr_confidence`. This explicit local command
returns the full normalized `ocr_plate`; do not publish or retain its output
unless that is appropriate for your use. Dashboard event storage remains masked
after the first four characters, for example `AB12***`.

```bash
.venv/bin/roundabout-anpr \
  --image-path data/events/images/example-sharpest.jpg
```

For live use, open the dashboard, enable **Live ANPR/OCR on crossings**, and
start processing. The worker keeps one plate detector and one RapidOCR engine
alive, analyzes the buffered crossing/centred/largest/sharpest crops only after a
confirmed vehicle crossing, and compares the four privacy-visible characters
across frames. It corrects common UK layout ambiguities such as `O`/`0` and
`I`/`1`, requires two agreeing prefixes by default, and accepts a single frame
only when it is UK-format-valid with confidence of at least 0.90.
Accepted results are written to `events.csv` as `ocr_plate` and
`ocr_confidence`. The storage boundary always masks characters after the first
four. Existing `plate_text`/`plate_confidence` CSV columns are upgraded in place
and any historical plate text is masked during migration.

Live ANPR is an experimental collection aid, not proof of identification. It is
opt-in, keeps uncertain/no-read results blank in event rows, and does not bypass
the Phase 4 recommendation or the held-out Phase 5 accuracy evaluation.

Run the gated prototype with the class name exposed by your plate model:

```bash
.venv/bin/roundabout-anpr run \
  --plate-model models/license-plate.pt \
  --plate-class license_plate
```

The command refuses to run unless Phase 4 recommends `proceed`. The current
sample may be explored while it is only `incomplete_sample` with
`--allow-incomplete-feasibility`; that override cannot bypass a `reposition`
or `stop` recommendation.

For each tracked crossing, the prototype detects plates only within the saved
vehicle candidates, rejects small/blurred/skewed/clipped crops, OCRs at most the
three strongest distinct frames, and accepts a plate only when at least two
good observations agree. It uses one long-lived RapidOCR instance in
recognition-only mode. Results remain `uncertain` or `no_read` when evidence is
weak.

Batch console and JSON results redact registration text by default. Explicit local
debugging requires `--store-text`; do not publish that output. To write a
redacted local report:

```bash
.venv/bin/roundabout-anpr run \
  --plate-model models/license-plate.pt \
  --output data/anpr/run.json
```

Held-out evaluation labels are a local, ignored CSV with exactly this header:

```csv
vehicle_id,expected_text
20260817-152243-332697+0100-track-7-line_1,AB12CDE
```

Use the redacted run output to obtain vehicle IDs, keep labelled vehicles out of
tuning, and calculate aggregate metrics:

```bash
.venv/bin/roundabout-anpr evaluate \
  --plate-model models/license-plate.pt \
  --labels data/anpr/held-out.csv \
  --output data/anpr/evaluation.json
```

The evaluation reports exact-plate accuracy, mean character accuracy, no-read
rate, and false-read rate without writing expected or observed plate text. All
models, images, labels, and reports remain under ignored local paths. The
optional PaddleOCR-VL fallback is intentionally not part of this baseline.

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
line, object class, direction, track ID, detection confidence, and relative
speed class/value. Plate fields
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
`--fast-speed-threshold`, `--minimum-speed-observations`,
`--minimum-speed-duration-seconds`, and `--tracker-config`. Run without
`--scene-config` to see tracking IDs over
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
