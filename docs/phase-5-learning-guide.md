# Phase 5 Local ANPR Learning Guide

Phase 5 turns the readable vehicle evidence from Phase 4 into a deliberately
conservative local ANPR prototype. Its purpose is not to produce a registration
guess for every vehicle. It separates plate detection, crop quality, OCR, and
track-level agreement so that weak evidence becomes `uncertain` or `no_read`
instead of a plausible-looking false result.

## The 80/20 View

Six ideas explain most of Phase 5 and its evidence-collection boundary:

1. Phase 4 and a plate-specific detector gate entry into OCR;
2. live capture must preserve useful crossing context before offline analysis;
3. one crossing track, not one JPEG, owns the recognition result;
4. crop quality is checked before expensive or misleading OCR;
5. one OCR engine produces observations, while exact multi-frame agreement
   decides acceptance;
6. redacted output and held-out aggregate metrics make privacy and uncertainty
   structural properties.

The command-line boundary is implemented in
[`anpr_app.py`](../src/roundabout_ai/anpr_app.py), orchestration in
[`anpr_pipeline.py`](../src/roundabout_ai/anpr_pipeline.py), quality and decision
logic in [`anpr.py`](../src/roundabout_ai/anpr.py), and the RapidOCR adapter in
[`ocr.py`](../src/roundabout_ai/ocr.py). The `roundabout-anpr` entry point and
Python 3.14 dependencies are declared in [`pyproject.toml`](../pyproject.toml).

## 1. Keep feasibility, plate detection, and recognition separate

The Phase 5 command accepts saved vehicle crops rather than reopening the
camera; evidence-collection improvements remain in the live worker. Before
loading either AI component,
`_check_feasibility()` reads the Phase 4 annotations and calls
`assess_feasibility()`:

```text
proceed           -> load the models and analyze crops
incomplete_sample -> stop, unless explicitly overridden for experimentation
reposition_first  -> stop
stop_at_vehicle_analytics -> stop
```

The incomplete-sample override is intentionally narrow. It allows development
while condition coverage is unfinished, but it cannot turn evidence that says
“reposition” or “stop” into permission to continue.

The next boundary is model responsibility. `YoloDetector` receives a
user-supplied plate model and requires the configured plate class to appear in
that model's class names. The normal `yolo26n.pt` model detects COCO road users;
it is not silently reused as a plate detector. YOLO answers “where is the
plate?”, while RapidOCR answers “which characters are inside this prepared
plate crop?”

Plate-specific weights under `models/` are local and ignored rather than
versioned with the repository. This keeps the model choice an explicit runtime
input. The OCR runtime can be exercised separately by the synthetic smoke test
before any real-image end-to-end evaluation.

**Transferable lesson:** make every model's responsibility explicit, and stop
before inference when upstream evidence or a required specialist model is
missing.

## 2. Make evidence collection observable before OCR

Phase 5 is offline, but the usefulness of its input is decided in the live
worker. After ByteTrack assigns a persistent ID, `TrackSpeedEstimator` in
[`speed.py`](../src/roundabout_ai/speed.py) stores bottom-centre positions with
the frame's monotonic capture timestamp. It calculates movement in vehicle-box
heights per second:

```text
pixel distance / mean box height / elapsed seconds
```

Normalizing by box height reduces sensitivity to resolution and perspective,
but the result is still image-relative—not mph or km/h. The default policy
needs three observations spanning 0.5 seconds. Insufficient evidence becomes
`unknown`; otherwise a configurable threshold separates `slow` and `fast`.
The classification supports sampling lower- and higher-motion evidence, not
enforcing road speed.

The data follows one explicit path:

```text
TrackSpeedEstimator -> TrackObservation -> CrossingEvent -> EventRecord
                                             ├── CSV speed fields
                                             └── Recent events table
```

Crossing direction has a similar separation between calculation and meaning.
`CrossingCounter` calculates `negative_to_positive` or
`positive_to_negative` from the ordered endpoints. The scene YAML maps those
signs to labels meaningful for that camera, such as `right_to_left` and
`left_to_right`. Reversing endpoints reverses the signs; changing only the YAML
labels preserves the geometry and improves presentation.

When event-image saving is enabled, the worker saves the full annotated
snapshot and distinct raw vehicle candidates. It also encodes one 240-pixel
JPEG data URL for the Recent events table: the raw `crossing` crop is preferred,
with the annotated snapshot as fallback. `EventRecord.preview_image` is
dashboard-only and deliberately omitted by `csv_row()`. Encoding once avoids
rereading 4K JPEGs on every Streamlit refresh, while the bounded recent-event
deque limits memory use.

Older event CSV files are upgraded with `unknown` speed values before new rows
are appended. Historical evidence remains honest; the migration does not infer
motion that was never measured.

**Transferable lesson:** when an offline model depends on live evidence, expose
capture quality and sampling context at collection time without confusing
diagnostic metadata with physical measurements.

## 3. Associate evidence with a vehicle event, not a filename

Phase 3 can save several alternatives for one crossing: `crossing`, `centred`,
`sharpest`, and older `largest` crops. `discover_vehicle_groups()` parses their
timestamp, track ID, line name, and kind. Nearby files with the same track and
line are grouped into one `VehicleImageGroup`.

A tracker ID can be reused after a process restart. Grouping by track ID alone
could therefore combine different vehicles. Phase 5 splits matching files when
their timestamps are more than ten seconds apart and includes the first
timestamp in the generated `vehicle_id`.

Full-frame `snapshot` files are deliberately excluded by the filename pattern.
This preserves the ownership boundary promised by the plan: plate detection
runs only within raw vehicle crops, not across an annotated scene containing
other road users or unrelated areas.

Every group produces one `TrackPlateResult`. Individual filenames survive only
as observation IDs used to explain which candidates contributed to that
result.

**Transferable lesson:** choose the domain entity that owns a decision, then
group raw measurements around that entity without assuming temporary IDs are
globally unique.

## 4. Reject unusable pixels before asking OCR

For every saved vehicle image, the plate detector returns one or more boxes.
`extract_plate_candidate()` pads each box, crops it, and records measurable
quality evidence:

- source width and height;
- Laplacian-variance sharpness;
- width-to-height aspect ratio;
- estimated skew from strong line segments;
- whether padding or the source box touches a crop boundary.

The default `PlateQualityPolicy` rejects a plate when it is too small, blurred,
implausibly shaped, clipped, or more than 15 degrees skewed. These checks are
heuristics, not proof that the remaining characters are readable. Their value
is that every rejection has an explicit reason and poor crops never become OCR
votes.

When one vehicle image contains several accepted plate detections, only the
strongest candidate from that image survives. Accepted candidates from all
images are then ranked by sharpness, detector confidence, and area. At most
three reach OCR. `prepare_plate_crop()` estimates skew, deskews the image, and
enlarges crops shorter than 64 pixels without claiming that interpolation has
created missing character detail.

The defaults in [`config.example.yaml`](../config.example.yaml) document the
intended operating policy. The actual Phase 5 CLI currently receives its
thresholds as command-line arguments; it does not load that YAML section.

**Transferable lesson:** rank only after enforcing a minimum-quality floor, and
record why evidence was excluded so threshold tuning remains explainable.

## 5. OCR creates observations; consensus creates a result

`_run_analysis()` constructs one `RapidOcrRecognizer` before it iterates over
vehicle groups. The recognizer constructs one RapidOCR engine and reuses it for
every selected candidate in that command invocation. This keeps model lifetime
outside the per-crop path.

The adapter selects ONNX Runtime and PP-OCRv6 small. Every call explicitly uses:

```text
use_det=False
use_cls=False
use_rec=True
```

Detection is disabled because the upstream plate model already supplied a
plate crop. Orientation classification is also disabled; Phase 5 performs its
own limited skew preparation. The call returns raw text and confidence.
`normalize_plate_text()` converts Unicode to ASCII where possible, uppercases
the result, and removes everything except letters and digits.

`build_consensus()` then applies the track-level policy:

```text
no observation reaches length/confidence floor -> no_read
one eligible reading only                      -> uncertain
eligible readings conflict                     -> uncertain
fewer than two exact normalized matches        -> uncertain
winning text fails required UK layout check    -> uncertain
two or more exact valid matches                 -> accepted
```

The UK format patterns are weak plausibility checks. They can reject an
obviously malformed string, but a plausible layout is not proof that the
characters are correct. The optional PaddleOCR-VL fallback described in the
project plan is not implemented in this baseline, so it contributes no votes.

**Transferable lesson:** keep probabilistic model output as an observation and
place acceptance in a separate, testable domain policy that can demand repeated
evidence.

## 6. Privacy and evaluation use different output boundaries

`TrackPlateResult` keeps text in memory because consensus and evaluation need
it. `as_dict(include_text=False)` omits the accepted text and also removes raw
and normalized text from every observation. The normal `run` command uses this
redacted representation unless the operator explicitly supplies
`--store-text`. That option affects both console and JSON output and should be
treated as local sensitive data.

The `evaluate` command reads a local CSV with:

```csv
vehicle_id,expected_text
```

It compares only labelled vehicle IDs and writes aggregate metrics:

- exact-plate accuracy;
- mean normalized Levenshtein character accuracy;
- no-read rate;
- false-read rate.

An `uncertain` result counts as a no-read, not as a partially successful read.
A wrong value counts as a false read only when the system marked it
`accepted`. That distinction exposes the central trade-off: stricter consensus
usually reduces false reads while increasing no-reads.

Images, models, ground-truth labels, and reports live under paths ignored by
[`.gitignore`](../.gitignore). The software boundaries reduce accidental
publication, but they do not decide whether collection or retention is lawful
for a particular location and purpose.

**Transferable lesson:** measure wrong confident answers separately from honest
abstentions, and remove sensitive fields at the serialization boundary by
default.

## Execution Flow

```text
roundabout-dashboard
           │
           v
latest frame -> YOLO + ByteTrack -> relative speed estimate
           │                         (slow/fast/unknown)
           v
ROI + finite-line crossing -> configured direction label
           │
           ├── CSV metadata, including relative speed
           ├── bounded Recent events row + crossing thumbnail
           └── opt-in snapshot + raw vehicle candidates
                                      │
                                      v
roundabout-anpr run/evaluate
           │
           v
Phase 4 annotations -> feasibility decision
           │ proceed or explicit incomplete-only override
           v
discover saved vehicle crops
  ├── include crossing/centred/largest/sharpest
  ├── exclude full-frame snapshot
  └── split reused track IDs by timestamp gap
           │
           v
one plate-specific YOLO detector
           │ plate boxes inside each vehicle crop
           v
size + blur + aspect + clipping + skew gates
           │ accepted candidates only
           v
best per image -> best three per vehicle -> deskew/resize
           │
           v
one reused RapidOCR PP-OCRv6 small recognizer
           │ raw text + normalized text + confidence
           v
exact multi-frame consensus + weak UK format check
           │
           ├── accepted
           ├── uncertain
           └── no_read
                    │
                    ├── run: redacted vehicle results by default
                    └── evaluate: aggregate held-out metrics only
```

## What the Tests Prove

Run the focused Phase 5 tests with:

```bash
.venv/bin/python -m pytest \
  tests/test_anpr.py \
  tests/test_anpr_pipeline.py \
  tests/test_ocr.py \
  tests/test_anpr_app.py \
  tests/test_speed.py \
  tests/test_event_images.py \
  tests/test_events.py \
  tests/test_dashboard.py
```

Static checks remain:

```bash
.venv/bin/ruff check .
.venv/bin/pyright
```

The focused tests prove that:

- normalization removes spaces, punctuation, and accents predictably;
- common UK layouts pass the weak format check;
- small, blurred, and clipped synthetic plate crops are rejected;
- a sufficiently sharp synthetic crop can cross the quality boundary;
- exact agreement across two of three observations is accepted;
- a single eligible observation stays uncertain and weak text becomes no-read;
- redacted serialization omits plate and observation text;
- exact accuracy, character accuracy, no-read rate, and false-read rate are
  calculated at vehicle level;
- nearby crop files form one group, reused track IDs split across time, and
  full-frame snapshots are excluded;
- the pipeline supplies three good synthetic frames to consensus;
- the RapidOCR adapter constructs its injected engine once and always requests
  recognition-only mode;
- evaluation labels normalize text and reject duplicate vehicle IDs;
- relative speed stays unknown without enough timed history, separates known
  slow/fast motion, normalizes for box scale, and resets reused stale IDs;
- speed metadata survives crossing and CSV serialization, while legacy CSV
  rows migrate with honest `unknown` values;
- event previews are bounded JPEG data URLs and appear beside timestamp,
  configured direction, and relative speed in the dashboard table.

The full repository currently passes 79 tests, Ruff, and Pyright. The real OCR
smoke command recognized the synthetic `AB12CDE` image with RapidOCR 3.9.2 and
ONNX Runtime 1.27.0 on Python 3.14.

The tests do not prove:

- accuracy of any plate-detector weights on this roundabout view;
- that the current quality thresholds generalize across day, evening, glare,
  rain, front plates, and rear plates;
- that whole-plate sharpness corresponds to character sharpness;
- real multi-frame OCR agreement on the saved event images;
- protection against correlated OCR errors repeated across similar frames;
- end-to-end throughput or memory use on the 3840×2160 stream;
- physical vehicle speed or a universal slow/fast threshold;
- correctness of camera-specific direction names without a human checking the
  line endpoint order and observed traffic;
- legal authority to collect or retain plate data;
- any behavior from the unimplemented optional PaddleOCR-VL fallback.

## Try It

### Verify the real local OCR runtime

```bash
.venv/bin/roundabout-anpr smoke-test
```

Predict first: the command should report RapidOCR and ONNX Runtime versions,
`"recognized": true`, and a confidence value. It exercises real model loading
and one recognition call, but not plate detection or camera imagery.

### Observe the feasibility boundary

With the current local Phase 4 annotations, run:

```bash
.venv/bin/roundabout-anpr run \
  --plate-model models/license-plate.pt
```

Predict first from the latest Phase 4 report: `proceed` loads the local model,
while `incomplete_sample`, `reposition_first`, or
`stop_at_vehicle_analytics` stops before model loading. Adding
`--allow-incomplete-feasibility` moves past only the incomplete-sample case; it
cannot override evidence that says reposition or stop. Plate weights remain a
local, ignored input rather than a repository artifact.

### Change one consensus variable safely

Use fictional observations entirely in memory:

```bash
.venv/bin/python -c 'from roundabout_ai.anpr import OcrObservation,build_consensus; o=lambda i: OcrObservation(str(i),"AB12CDE","AB12CDE",0.9,True); print(build_consensus("demo",(o(1),o(2))).status)'
```

Predict first: two identical eligible observations produce `accepted`. Change
the tuple to `(o(1),)` and predict `uncertain`, because the default policy
requires two agreeing frames. This isolates consensus behavior without reading
or storing any real registration.

### Predict a relative-speed classification

Run the isolated estimator tests:

```bash
.venv/bin/python -m pytest tests/test_speed.py -q
```

Predict first: the same apparent movement at twice the pixel size and twice the
box height produces the same normalized speed. Then inspect
`test_speed_classifies_normalized_track_motion` and change its test-only fast
threshold from `1.0` to `2.0`. Predict which case changes before running it.
This demonstrates why the dashboard threshold must be calibrated against the
observed scene rather than treated as a physical constant.

### Run the focused quality boundary

```bash
.venv/bin/python -m pytest \
  tests/test_anpr.py::test_candidate_quality_rejects_small_blurred_and_clipped_plate \
  tests/test_anpr.py::test_candidate_quality_accepts_a_sharp_plate
```

Predict first: both tests pass because the first synthetic crop records several
rejection reasons while the second crosses an intentionally lower test-only
sharpness threshold. A useful next exercise is to add one synthetic crop whose
only failure is aspect ratio and assert the single reason it produces.

## Continuous-Learning Loop

1. **Define the user-visible goal.** Example: “accept a plate only when several
   useful frames support the same reading.”
2. **Name the enabling concept.** Plate detection localizes evidence, quality
   gates filter it, and track-level consensus controls acceptance.
3. **Implement the smallest useful behavior.** Analyze saved vehicle crops with
   one specialist detector and one reused recognizer; add only the live
   metadata needed to judge and stratify those crops.
4. **Prove it at the cheapest meaningful boundary.** Unit-test crop rejection,
   normalization, consensus, redaction, and metrics; smoke-test the real OCR
   runtime separately.
5. **Explain what failures revealed.** An incomplete Phase 4 sample is not OCR
   failure, an absent plate model is not a recognition failure, and an
   uncertain observation is not a false read.
6. **Record the transferable lesson.** Preserve the difference between input
   feasibility, model observations, domain acceptance, and measured outcomes.

The reusable pattern is:

```text
gate the input -> localize specialist evidence -> reject weak candidates
               -> aggregate repeated observations -> abstain when uncertain
               -> measure confident errors separately
```
