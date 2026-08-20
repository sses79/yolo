# Local Roundabout AI Camera Demo — Project Plan

## 1. Goal

Build a Python demo that reads the Android **IP Webcam** stream at:

```text
http://192.168.1.142:8080/video
```

and performs all processing locally on the Mac. The first useful version should:

1. keep the live stream running reliably;
2. detect and track road users in a chosen roundabout region;
3. count passing vehicles without counting the same vehicle repeatedly;
4. detect people and show a warning/event in the UI;
5. optionally attempt vehicle registration plate reading when image quality is good enough.
6. present the live demo, controls, counters, and recent events in a local Streamlit dashboard.

The system is a local experiment, not a safety, policing, or identification system. Detection and OCR results will contain errors.

## 2. Recommended scope

### MVP: detection and counting

Start here because it works with ordinary pretrained object detectors and immediately tests whether the camera position is useful.

- Live annotated video in a Streamlit dashboard.
- Detect `car`, `motorcycle`, `bus`, `truck`, `bicycle`, and `person`.
- Restrict analysis to a configurable polygon around the road/roundabout.
- Track each object across frames using a stable track ID.
- Count a vehicle when its tracked centre crosses a configured virtual line.
- Record only a small event row: time, class, direction, confidence, and track ID.
- Show stream FPS, inference FPS, connection state, and counts.

### Optional phase: registration plate recognition (ANPR)

This is a separate pipeline, not a feature supplied by the normal car detector:

```text
vehicle detection -> vehicle tracking -> plate detection -> best plate crop
                  -> rectify/clean crop -> PP-OCRv6 recognition
                  -> format/confidence checks
                  -> agreement across several frames
```

A pretrained general-purpose YOLO model detects vehicles and people, but normally does **not** detect registration plates. Plate reading therefore needs a dedicated plate detector (or a small custom-trained model) plus OCR.

Do not promise ANPR until a short recording proves that plate characters are large and sharp enough. As a practical test, pause several passing cars and inspect the original-resolution frames. If characters cannot be read confidently by a person, OCR will not rescue them reliably.

## 3. Proposed architecture

```text
Android IP Webcam (MJPEG)
          |
          v
background capture/inference worker
OpenCV capture + reconnect loop
          |
          +----> optional rolling/debug recording
          |
          v
latest-frame buffer (drop old frames to avoid lag)
          |
          v
YOLO vehicle/person detector
          |
          v
multi-object tracker (start with ByteTrack)
          |
          +----> ROI filtering + line crossing + event/count store
          |
          +----> optional plate detector -> RapidOCR/PP-OCRv6
                                      -> multi-frame consensus
                                      -> optional VL-1.6 fallback
          |
          v
Streamlit dashboard + local CSV/SQLite output
```

Use a latest-frame buffer rather than processing every MJPEG frame. If inference is slower than the camera, dropping stale frames preserves a live view instead of building an ever-growing delay.

Streamlit is the presentation layer, not the owner of the continuous camera loop. The capture, detection, and tracking pipeline runs in one long-lived background worker. It publishes a thread-safe snapshot containing the latest annotated frame, connection state, FPS, counts, and recent events. A Streamlit fragment refreshes only the live portion of the page at a short interval. This avoids reopening the camera or reloading the model whenever Streamlit reruns the page after a widget interaction.

## 4. Technology choices

- **Python 3.14.6:** primary project runtime. A clean dependency-resolution check on this Apple Silicon Mac succeeded for the MVP and primary OCR stack. Use the normal GIL-enabled build, not free-threaded Python, unless the complete application is separately tested there.
- **OpenCV:** MJPEG capture, frame drawing, image crops, image conversion, and optional recording.
- **Ultralytics YOLO:** use `yolo26n.pt` as the initial local vehicle/person detector. It supplies the COCO classes needed for cars, trucks, buses, motorcycles, bicycles, and people. Benchmark `yolo26s.pt` only if the nano model misses too many small road users. A separate plate-specific YOLO model is still required for ANPR.
- **ByteTrack:** simple, fast initial tracker for a fixed camera.
- **Streamlit:** local web dashboard for the annotated frame, controls, health information, counters, charts, and event table. Use `st.cache_resource` for long-lived model/worker resources and `st.fragment(run_every=...)` for periodic UI refreshes.
- **NumPy:** geometry and frame operations.
- **PyYAML:** editable stream, model, ROI, and line configuration.
- **SQLite (standard library) or CSV:** local event storage; CSV is enough for the MVP.
- **RapidOCR + ONNX Runtime:** primary Python 3.14-compatible OCR path for the optional ANPR phase. Use the PP-OCRv6 small recognition model on an already detected and rectified plate crop, with text detection and orientation classification disabled. This avoids the PaddlePaddle runtime while retaining PP-OCR models.
- **PaddleOCR-VL-1.6 via `llama.cpp`:** optional second-pass experiment for uncertain high-quality crops only. Run the official GGUF model in a separate local native server and call it over localhost. Do not run the 0.9B document VLM on every frame or make it the primary recognizer.
- **pytest:** unit tests for geometry, line crossing, deduplication, reconnect behaviour, and plate validation.

The latest `rapidocr` package metadata does not yet list a Python 3.14 classifier, but it declares Python `<4, >=3.8`, and its full dependency graph resolved successfully under Python 3.14.6 on this Mac. ONNX Runtime supplies a native CPython 3.14/macOS ARM64 wheel. Installation compatibility must still be followed by an import and one-image inference smoke test.

Before distributing the project or using it commercially, review the licences of model weights and libraries, especially the selected Ultralytics licence.

Official implementation references:

- [Ultralytics Python usage](https://docs.ultralytics.com/usage/python/)
- [Ultralytics multi-object tracking](https://docs.ultralytics.com/modes/track/)
- [RapidOCR usage and PP-OCRv6 models](https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/)
- [RapidOCR Python package](https://pypi.org/project/rapidocr/)
- [ONNX Runtime Python package](https://pypi.org/project/onnxruntime/)
- [PaddleOCR-VL-1.6 local backends and GGUF fallback](https://github.com/PaddlePaddle/PaddleOCR/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md)
- [Streamlit fragments and periodic refresh](https://docs.streamlit.io/develop/concepts/architecture/fragments)

## 5. Suggested repository layout

```text
.
├── README.md
├── PROJECT_PLAN.md
├── pyproject.toml
├── config.example.yaml
├── src/
│   └── roundabout_ai/
│       ├── __init__.py
│       ├── dashboard.py        # Streamlit entry point and UI
│       ├── worker.py           # background capture/inference lifecycle
│       ├── shared_state.py     # locked latest frame, status, counts, events
│       ├── capture.py          # camera connection, latest frame, reconnect
│       ├── detector.py         # vehicle/person inference
│       ├── tracker.py          # track adapter and track lifecycle
│       ├── geometry.py         # ROI and line-crossing logic
│       ├── events.py           # deduplication and CSV/SQLite writes
│       ├── overlay.py          # boxes, labels, counters, status
│       └── anpr/               # optional; add only after MVP
│           ├── plate_detector.py
│           ├── ocr.py           # RapidOCR recognition-only adapter
│           ├── vl_fallback.py   # optional localhost llama.cpp client
│           └── consensus.py
├── tests/
│   ├── test_geometry.py
│   ├── test_events.py
│   └── test_plate_consensus.py
├── data/
│   ├── recordings/             # ignored by Git
│   ├── snapshots/              # ignored by Git
│   └── events/                 # ignored by Git
└── models/                     # downloaded/custom weights, ignored by Git
```

Keep the camera URL in local configuration or an environment variable rather than hard-coding it throughout the code.

Example configuration:

```yaml
camera:
  url: "http://192.168.1.142:8080/video"
  reconnect_seconds: 2

detection:
  model: "yolo26n.pt"
  confidence: 0.35
  image_size: 640
  classes: [person, bicycle, car, motorcycle, bus, truck]
  device: cpu

scene:
  roi: []                 # points selected from a captured reference frame
  count_lines: []         # named lines selected from the same frame

storage:
  save_raw_video: false
  save_event_images: false
  event_file: "data/events/events.csv"
  event_crop_horizontal_padding: 0.15
  event_crop_vertical_padding: 0.10
  event_crop_minimum_width: 200
  event_crop_minimum_height: 100

anpr:
  enabled: false
  store_plate_text: false
  ocr_engine: "rapidocr"
  ocr_model: "pp-ocrv6-small"
  use_text_detection: false
  use_orientation_classification: false
  vl_fallback_enabled: false
  vl_fallback_url: "http://127.0.0.1:8111"

dashboard:
  refresh_seconds: 0.2
  bind_address: "127.0.0.1"
```

The vehicle/person model is pinned by name as `yolo26n.pt`; package versions are constrained in `pyproject.toml`. Record a final licence decision before distributing the project or using it commercially.

## 6. Delivery phases

### Phase 0 — Measure the input

**Implementation status:** implemented by the `roundabout-camera` command. It includes GUI and headless modes, reconnect/timeouts, latest-frame buffering, health metrics, keyboard snapshots, bounded MP4 recording, and deterministic reconnect tests. See `README.md` for usage. The remaining acceptance exercise is the full 10-minute run with a deliberate phone stream restart.

Create a small camera diagnostic command that:

- connects and displays frames;
- prints actual resolution, observed FPS, dropped/read failures, and latency symptoms;
- reconnects after the phone/app/Wi-Fi briefly disappears;
- saves a user-triggered full-resolution snapshot;
- optionally records a 2–5 minute sample for repeatable offline development.

**Exit criteria:** a 10-minute run survives at least one deliberate stream restart and does not accumulate increasing delay.

### Phase 1 — Vehicle and human detection

**Implementation status:** implemented by the `roundabout-detect` command. It consumes the newest `FramePacket`, runs `yolo26n.pt`, filters to the six configured road-user/person classes, draws boxes and confidence labels, supports GUI and headless operation, saves annotated snapshots, and reports capture rate, inference rate/time, frame age, processed frames, and overwritten frames. A single-frame benchmark mode compares CPU and available MPS execution with explicit device synchronization.

On the initial 1920×1080 camera benchmark at `imgsz=640` (two warmups and five measured calls), CPU averaged 34.1 ms / 29.3 effective FPS and MPS averaged 36.1 ms / 27.7 effective FPS. CPU is therefore the current default; this is a machine-specific observation, not a general claim about MPS. A 45-second live run processed 841 newest frames while receiving 1,345, demonstrating bounded backpressure rather than a growing queue. The live stream has produced a `person` detection, but representative daylight review of cars and other classes remains an acceptance exercise.

- Run the nano detector on the newest frame.
- Draw class, confidence, and bounding boxes.
- Add a configurable confidence threshold and inference image size.
- Benchmark CPU versus Apple Metal (`mps`) if this is an Apple Silicon Mac.
- Log processing FPS and end-to-end responsiveness.

**Exit criteria:** common vehicles and people in the scene are visibly detected in daylight, and the view remains acceptably live.

### Phase 2 — Scene calibration and tracking

**Implementation status:** implemented. The `roundabout-calibrate` command
captures or opens a clean reference frame and writes a YAML ROI plus one or
more finite count lines. `roundabout-detect` now uses persistent Ultralytics
ByteTrack IDs, scales the calibration to the live resolution, filters on box
centre, labels tracks, and emits a directional crossing once per track/line
after a configurable minimum track age. Geometry, calibration serialization,
ROI filtering, tracker adaptation, direction, finite-line rejection, track
expiry, and deduplication have deterministic tests. The remaining acceptance
exercise is the manual review of at least 100 representative road users.

- Save one clean reference frame.
- Add a small calibration tool for clicking ROI polygon points and count-line endpoints.
- Ignore detections outside the road ROI.
- Track detections with persistent IDs.
- Count only on a genuine line crossing and only once per track per line.
- Require a minimum track age to reject one-frame false positives.
- Infer direction from which side of the line the track occupied before and after crossing.

**Exit criteria:** manually review at least 100 passing road users; report count accuracy and duplicate/missed counts rather than relying on impressions.

### Phase 3 — Events and usable demo

**Implementation status:** implemented by the `roundabout-dashboard` command
and the CSV event sink used by both dashboard and detection CLI. A cached,
long-lived worker owns capture, model inference, tracking, counting, and event
writes; Streamlit reads locked snapshots on a timed fragment. The UI includes
start/stop, live confidence and overlay controls, explicit camera health,
annotated frames, person warnings, metrics, crossing charts, and recent events.
Recent events and CSV rows also carry a timestamp-based relative speed class
(`slow`, `fast`, or `unknown`) and normalized box-heights-per-second value for
sample re-collection; this is explicitly not a physical road-speed estimate.
Storage is metadata-only by default, with event images requiring explicit
opt-in. Each confirmed event saves a full annotated snapshot; the crop selector
also keeps bounded, padded raw crossing, centred, and sharpest candidates per
vehicle and saves distinct, sufficiently large candidates only after a confirmed crossing. Unit/static
checks cover CSV serialization, bounded shared
state, single-worker ownership, dashboard aggregation, and page rendering. The
remaining acceptance exercise is the one-hour live soak test.

- Write crossing-event metadata to CSV. *(Implemented; SQLite remains optional.)*
- Add the Streamlit dashboard as the primary user interface.
- Show the annotated latest frame, connection status, capture/inference FPS, and totals by vehicle type and direction.
- Show recent events in a table and traffic totals in simple charts.
- Add start/stop processing controls and confidence/overlay controls.
- Refresh the live panel with a timed Streamlit fragment while the worker continues independently.
- Cache the model/worker as a resource so normal page reruns do not reload model weights or create duplicate camera readers.
- Add clean shutdown, structured logs, and clear offline/reconnecting status.
- Default to metadata-only storage.
- Add an optional full event snapshot plus raw best-of-track vehicle crops only
  when explicitly enabled.

Suggested event fields:

```text
timestamp, event_type, object_class, direction, track_id,
detection_confidence, plate_text, plate_confidence, speed_class, normalized_speed
```

**Exit criteria:** restartable one-hour local Streamlit demo with stable counts, bounded memory use, no duplicate workers after UI interaction, and no raw-video retention by default.

### Phase 4 — ANPR feasibility gate

**Implementation status:** implemented by `roundabout-anpr-assess`. The command
creates a local annotation CSV, validates sample quantity and coverage, and
generates a Markdown evidence report with a configurable `proceed`,
`reposition_first`, `stop_at_vehicle_analytics`, or `incomplete_sample`
recommendation. It records image references, viewing conditions, human
readability, and approximate plate/character pixel sizes, but deliberately has
no registration-text field. The decision thresholds are visible project policy,
not claims about OCR accuracy. Completing the gate still requires collecting and
reviewing a real consent-appropriate local sample.

Before building OCR, collect a consent-appropriate local sample covering:

- approaching and departing vehicles;
- day, shade, glare, rain, and evening if relevant;
- slow and fast passes;
- front and rear plates at the actual camera angle.

For a labelled sample, record whether a human can read each plate and measure the approximate plate/character size in pixels. Decide one of:

1. **Proceed:** plates are consistently sharp and readable.
2. **Reposition first:** lower angle, optical zoom/closer crop, more light, faster shutter, or better camera is needed.
3. **Stop at vehicle analytics:** the present view cannot support reliable OCR.

This gate prevents spending time tuning OCR against unusable source images.

### Phase 5 — Optional local ANPR prototype

**Implementation status:** implemented as the offline `roundabout-anpr`
prototype. It requires the Phase 4 gate and a user-supplied plate-specific YOLO
model; the general COCO vehicle model is deliberately rejected by class-name
validation. The command groups saved Phase 3 candidates by tracked crossing,
quality-gates plate detections, reuses one RapidOCR PP-OCRv6 small instance,
and reports conservative multi-frame consensus. Registration text is redacted
by default. The optional PaddleOCR-VL fallback remains an extension point and
is not enabled in this local baseline.

- Detect a plate only inside a tracked vehicle crop.
- Reject plates that are too small, blurred, highly skewed, or partly outside the frame.
- Keep several candidate crops for a vehicle and OCR only the sharpest.
- Install `rapidocr` and `onnxruntime` in the main Python 3.14 environment and run an import plus one-image inference smoke test before integration.
- Initialize one long-lived RapidOCR instance; do not reload its recognition model per crop or per Streamlit rerun.
- Use PP-OCRv6 small as the primary recognition model. Because the upstream YOLO stage already found the plate, run recognition-only mode (`use_det=False`, `use_cls=False`, `use_rec=True`) on the rectified crop.
- Benchmark the PP-OCRv6 tiny recognition model only if the small model is too slow; compare accuracy on the same labelled crops before switching.
- Normalize OCR output (uppercase, remove spaces/punctuation) while retaining the raw result for debugging.
- Apply a country-appropriate plate-format check as a weak validation rule, never as proof that OCR is correct.
- Accept a result only after agreement across multiple good frames.
- Associate the accepted text with the vehicle track, not with a single frame.
- Display uncertainty explicitly; do not silently turn a low-confidence guess into a plate number.
- Optionally send only uncertain, otherwise usable crops to a separately running local PaddleOCR-VL-1.6 `llama.cpp` server. Treat its answer as another observation requiring validation and multi-frame agreement, not as ground truth.

**Exit criteria:** evaluate on held-out labelled vehicles and report exact-plate accuracy, character accuracy, no-read rate, and false-read rate. A correct-looking live demo is not sufficient evaluation.

### Phase 6 — Adaptive camera configuration

**Implementation status:** implemented by the Phase 3 dashboard worker. It now
discovers and saves the phone's reported capabilities, measures calibrated-ROI
brightness, clipping, sharpness, directional blur, and noise, and requires
three consistent observations before recommending day, glare, dusk, or night.
The dashboard can explicitly apply or roll back a typed, capability-validated
preset. Experimental automatic mode is separately opt-in and adds empty-ROI,
minimum-dwell, cooldown, read-back, and rollback gates. Every new crossing CSV
row records the active profile and an allowlisted settings snapshot. The exit
criteria still require separately labelled live day, glare, dusk, and night
comparisons; the shipped presets are a deterministic baseline, not a claim of
camera or OCR improvement.

The controller is a bounded feedback loop, not a general AI agent:

```text
road/plate ROI frames + tracked crossing metadata
                    |
                    v
brightness / clipping / blur / noise / plate size / relative speed
                    |
                    v
condition classifier: day / glare / dusk / night
                    |
                    v
choose one allowlisted camera preset
                    |
                    v
IP Webcam HTTP control -> read back state -> cooldown
                    |
                    v
human-readable rate + OCR accepted/uncertain/no-read evidence
```

Prefer a small deterministic state machine initially. A contextual bandit or
Bayesian optimiser becomes useful only after enough labelled events exist to
compare presets under similar lighting and vehicle-speed conditions. An LLM is
not needed in the control loop.

#### Confirmed IP Webcam control surface

Use a configurable base URL such as `http://192.168.1.142:8080`; never derive
control authority from the public MJPEG URL. The installed web page performs
these requests:

| Purpose | HTTP request | Notes from this phone |
|---|---|---|
| Read state and capabilities | `POST /status.json?show_avail=1` | Returns `curvals`, `avail`, and device status without changing settings |
| Generic setting | `POST /settings/<name>?set=<value>` | Used for most enumerated and numeric controls |
| Digital zoom | `POST /ptz?zoom=<index>` | UI indexes the reported zoom choices; reported range is 1.00×–10.00× |
| Stream quality | `POST /settings/quality?set=<1..100>` | Current value observed as 49 |
| Video resolution | `POST /settings/video_size?set=<WxH>` | Includes 3840×2160, 1920×1080, and lower modes |
| Focus | `POST /settings/focusmode?set=<mode>` and `POST /settings/focus_distance?set=<dioptres>` | Modes include auto, macro, continuous-video, continuous-picture, and off; distance range is 0.0–10.0/m |
| Automatic imaging | `POST /settings/scenemode?set=<mode>`, `whitebalance`, and `antibanding` | Includes action/sports/night/HDR scene modes and several white-balance choices |
| Night processing | `POST /settings/night_vision?set=on|off`, plus `night_vision_gain` and `night_vision_average` | Treat as separate presets because averaging can blur moving vehicles |
| Manual sensor | `POST /settings/manual_sensor?set=on|off` | Enables direct ISO/exposure/frame-duration control when supported |
| ISO | `POST /settings/iso?set=<value>` | Reported range is ISO 50–3200 |
| Exposure | `POST /settings/exposure_ns?set=<nanoseconds>` | Values are nanoseconds; validate against device bounds and frame duration |
| Frame duration | `POST /settings/frame_duration?set=<nanoseconds>` | Current 33,333,333 ns is approximately 30 FPS |

The current capability snapshot reported 3840×2160 video, approximately 1.99×
digital zoom, `continuous-picture` focus, automatic scene/white balance,
manual sensor control off, ISO 50, and a 33.3 ms frame duration. Values reported
while automatic sensor control is enabled are observations, not stable manual
settings. The phone exposes only one focal length and aperture, so Phase 6
cannot optimise those mechanically. Digital zoom also cannot create optical
detail that the sensor did not capture.

All writes must use a typed allowlist and values returned by `avail`; never
allow arbitrary endpoint names or raw URLs from a model. After every change,
read `/status.json?show_avail=1` again and verify the requested value. Resolution
changes may interrupt MJPEG and must deliberately trigger the existing capture
reconnect path.

#### Presets and objective measurements

Start with manually reviewed presets rather than per-vehicle adjustments:

- **Day:** short exposure, moderate ISO, calibrated fixed or stable continuous
  focus, and the validated resolution/zoom.
- **Direct sun/glare:** short exposure with highlight clipping constrained;
  retain enough shadow detail for dark vehicles.
- **Dusk:** slightly longer exposure and higher ISO while keeping vehicle blur
  within the labelled acceptance boundary.
- **Night:** the shortest exposure supported by available light, higher ISO,
  and no multi-frame night averaging unless tests show moving plates remain
  sharp.

Measure the road and expected plate region rather than the whole frame. Candidate
features and rewards are:

- ROI luminance percentiles and over/underexposed pixel percentages;
- Laplacian sharpness, directional motion blur, noise, and crop clipping;
- detected plate and character pixel size;
- relative vehicle speed and crossing direction;
- plate-detector confidence;
- human-readable rate and OCR `accepted`, `uncertain`, and `no_read` rates;
- false-read rate from held-out labels, never from live guesses.

Record a `camera_profile` name and a non-sensitive settings snapshot with each
new crossing. Do not infer settings for historical events.

#### Control and safety stages

1. Add a read-only `CameraCapabilities` client and save capability snapshots.
2. Compute quality metrics and show a recommended profile without camera writes.
3. Add explicit operator buttons to apply and roll back allowlisted presets.
4. Collect representative day, glare, dusk, and night evidence for every preset.
5. Enable automatic selection with hysteresis, minimum dwell time, cooldown,
   health checks, and rollback to the last known-good profile.
6. Consider a learned selector only after offline replay beats the deterministic
   baseline on held-out events.

Only one process may own camera configuration. Do not change settings while a
vehicle is crossing, do not continuously chase individual objects, and do not
use torch/flash toward a public road. If light is insufficient, report that the
hardware/view has reached its limit rather than hiding blur with longer
exposure.

**Exit criteria:** across separately labelled day, glare, dusk, and night
batches, automatic preset selection improves or preserves human-readable plate
rate and character sharpness versus the fixed baseline, does not increase false
reads or destabilise capture/counting, survives rejected settings and stream
restarts, and reliably rolls back to the last known-good profile.

## 7. Testing strategy

Develop against both the live stream and saved clips. Saved clips make bugs reproducible and avoid waiting for traffic.

- **Unit tests:** point-in-polygon, side-of-line calculation, crossing state, per-track deduplication, plate normalization, and multi-frame consensus.
- **Recorded integration test:** a short clip with expected crossing counts in a small annotation file.
- **Failure tests:** wrong URL, phone sleeps, Wi-Fi interruption, malformed frame, empty frame, model load failure, and output directory unavailable.
- **Dashboard tests:** repeated widget interactions and page reruns do not reload the model, reopen the stream, reset counts unexpectedly, or start duplicate workers.
- **Performance test:** sustained 30–60 minute run measuring inference FPS, capture failures, memory, and displayed lag.
- **Accuracy review:** manually label a representative sample and calculate errors by daylight/weather/direction rather than quoting model benchmark accuracy.

## 8. Privacy, security, and safe defaults

A number plate can identify or single out a vehicle, and public-road video can capture people and private activity. Before collecting or retaining it, check the rules that apply to the camera location and intended use. The following are engineering safeguards, not legal advice:

- Process on the local Mac; do not upload frames or OCR results.
- Frame only the area needed and mask windows, gardens, pavements, or other irrelevant regions where possible.
- Do not enable face recognition. `person` detection should remain anonymous.
- Default to no raw recording, no event snapshots, and no stored plate text.
- If plate storage is genuinely required, define purpose, access, retention period, deletion, and encryption first.
- Bind the Streamlit dashboard to `127.0.0.1` by default and add authentication before exposing it to the LAN.
- Put phone and Mac on a trusted network; IP Webcam's plain HTTP stream is not encrypted.
- Do not publish sample footage or real registration numbers in the repository.
- Add `data/`, model weights, local config, and logs to `.gitignore`.

## 9. Main risks and mitigations

| Risk | Likely effect | Mitigation |
|---|---|---|
| Camera is high, wide, or oblique | Plates too small/skewed | Feasibility gate; reposition/zoom before OCR work |
| Motion blur or low light | Missed objects and unreadable characters | Improve lighting/exposure, use best-frame selection |
| MJPEG/Wi-Fi interruption | Frozen demo | Capture timeout, reconnect loop, visible health status |
| Inference slower than stream | Growing lag | Latest-frame buffer, smaller model/input, frame skipping |
| Streamlit reruns page code | Model reloads or duplicate camera workers | Cache one worker resource and keep continuous processing outside the page execution loop |
| Occlusion at roundabout | Track ID switches or missed counts | Tune ROI/line placement and tracker using recorded clips |
| Vehicle stops on line | Duplicate counts | Side-change state machine and one event per track/line |
| OCR produces plausible wrong text | Misidentification | Multi-frame agreement, reject uncertainty, measured false-read rate |
| RapidOCR lacks an explicit Python 3.14 classifier | A future release/dependency may regress | Pin the tested versions and keep a Python 3.14 install/import/inference smoke test in CI |
| VL fallback is much heavier than recognition OCR | Lag and excess resource use | Invoke only for selected uncertain crops in a separate local process |
| Model/library licence mismatch | Cannot redistribute/deploy as intended | Review and document licences before distribution |
| Excessive data collection | Privacy/security exposure | Local processing and metadata-only defaults |

## 10. Definition of success

The project is successful when the MVP can run locally for one hour, reconnect to the camera, show a responsive Streamlit dashboard and annotated view, and produce measured vehicle/person events with an agreed count accuracy on labelled roundabout footage.

ANPR is successful only if a separate held-out evaluation meets a deliberately chosen accuracy and false-read threshold. If the camera cannot supply readable plate pixels, finishing with anonymous traffic counts and person detection is still a useful and complete demo.

## 11. Recommended implementation order

1. Scaffold the Python package, configuration, logging, Streamlit entry point, and `.gitignore`.
2. Build and soak-test the camera reader with reconnect and latest-frame buffering.
3. Record a short local development clip.
4. Add vehicle/person detection and benchmark it.
5. Build the ROI/line calibration tool.
6. Add tracking, crossings, counts, and tests.
7. Add metadata-only event output and the Streamlit dashboard with a polished overlay.
8. Evaluate the MVP on manually labelled footage.
9. Run the ANPR feasibility gate.
10. Implement plate detection/OCR only if the source footage passes that gate.
11. Add read-only camera quality recommendations, validate manual presets, and
    enable bounded adaptive camera control only after held-out comparison.
