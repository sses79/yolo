# Phase 6 and Live ANPR/OCR Learning Guide

Phase 6 and the live ANPR change form one bounded evidence loop. Phase 6
observes the calibrated road region and can improve the camera's operating
profile; live ANPR waits for a confirmed crossing, analyzes a few in-memory
vehicle crops, and records only an accepted, masked result. Neither subsystem
is allowed to turn weak evidence into a confident identity claim.

This guide focuses on the small set of design choices that explain most of the
behavior. The implementation remains a local experiment, not a safety,
policing, or identification system.

## The 80/20 View

Six ideas explain most of Phase 6 and the new live OCR path:

1. measure the road ROI before deciding which camera condition exists;
2. stabilize recommendations before allowing a state change;
3. treat device-reported capabilities and read-back as the authority;
4. separate recommendation, manual actuation, and automatic actuation;
5. run OCR on crossing-owned evidence, not continuously on every frame;
6. keep full text inside an explicit local diagnostic boundary and mask it at
   the event-storage boundary.

Camera measurement lives in
[`camera_quality.py`](../src/roundabout_ai/camera_quality.py), bounded device
control in [`camera_control.py`](../src/roundabout_ai/camera_control.py), and the
long-lived feedback loop in [`worker.py`](../src/roundabout_ai/worker.py). Live
plate analysis is split across [`anpr_pipeline.py`](../src/roundabout_ai/anpr_pipeline.py),
[`anpr.py`](../src/roundabout_ai/anpr.py), and
[`ocr.py`](../src/roundabout_ai/ocr.py). The privacy boundary is implemented in
[`events.py`](../src/roundabout_ai/events.py), while the deliberately unmasked
single-image command is in [`anpr_app.py`](../src/roundabout_ai/anpr_app.py).

## 1. Observe the useful region, not the whole image

`measure_frame_quality()` converts the frame to grayscale and measures the
configured scene ROI when one exists. It reports:

- 5th, 50th, and 95th luminance percentiles;
- underexposed and overexposed pixel ratios;
- Laplacian-variance sharpness;
- imbalance between horizontal and vertical gradient energy;
- a local high-frequency noise estimate.

The ROI boundary matters more than any individual metric. A bright sky or lamp
outside the road can dominate a whole-frame average while the vehicle region
is dark. The ROI test in
[`test_camera_quality.py`](../tests/test_camera_quality.py) makes that failure
mode concrete by surrounding a dark road patch with white pixels and still
classifying the useful area as `night`.

`classify_condition()` intentionally remains a small deterministic policy:

```text
high clipping or bright upper tail -> glare
median below 45                  -> night
median below 95                  -> dusk
otherwise                        -> day
```

Sharpness, directional blur, and noise are published as diagnostic evidence,
but the current four-way classifier uses luminance and clipping only. That is
an important implementation boundary: the code measures more than it currently
uses to choose a profile. Future decisions should not be credited to those
extra metrics until the classifier or evaluation actually incorporates them.

**Transferable lesson:** measure the part of the input that owns the outcome,
and distinguish observed features from features that currently drive policy.

## 2. Turn noisy measurements into a stable state machine

One frame should not change a camera profile. `CameraProfileAdvisor` tracks a
candidate condition and requires three consecutive matching observations
before publishing a new recommendation. If a different condition interrupts
the sequence, the candidate count resets.

With the default five-second measurement interval, a fresh recommendation
therefore needs roughly three consistent samples rather than one transient
headlight, cloud, or exposure adjustment. This is hysteresis in its simplest
form: require persistence before changing state.

The worker adds two time guards for automatic mode:

- a minimum dwell after a successful profile change, five minutes by default;
- a cooldown after an automatic attempt, one minute by default.

These clocks solve different problems. Dwell prevents repeated successful
switches between nearby states. Cooldown prevents a failing camera endpoint
from being hammered on every frame or quality interval.

The dashboard can still show every quality measurement while the recommended
profile remains unchanged. Observability does not require actuation.

**Transferable lesson:** stabilize an observation in one component, then apply
separate timing guards around the side effect; do not hide both concepts inside
one threshold.

## 3. Make external writes transactional enough to fail safely

The IP Webcam HTTP surface is external mutable state. The controller therefore
does not construct arbitrary setting URLs from user- or model-generated names.
`ALLOWED_SETTINGS` defines a typed allowlist, and `CameraCapabilities.validate()`
also requires the setting to be reported by the phone. Enumerated values must
match the phone's choices; numeric values must remain inside its reported
bounds.

Applying a preset follows a small transaction-like sequence:

```text
read capabilities
      |
      v
validate every requested name and value
      |
      v
snapshot the previous values
      |
      v
write each setting
      |
      v
read state back and verify equivalence
      |
      +---- success -> return applied + previous settings
      |
      `---- failure -> restore previous values and verify restoration
```

This is not a database transaction: the phone may disconnect between writes,
and rollback can also fail. The code reports that second failure explicitly
rather than pretending the camera returned to its old state.

At worker startup, capability discovery is also saved to
`data/camera/capabilities.json`. That ignored local artifact records what the
particular phone claimed to support without making one device's capabilities a
repository-wide assumption.

**Transferable lesson:** before mutating an external device, constrain the
command vocabulary, validate against live capabilities, preserve prior state,
and verify both writes and recovery with a fresh read.

## 4. Separate advice from authority

`ProcessingConfig.camera_adaptation_mode` has three explicit values:

```text
off       -> no capability discovery, measurement, recommendation, or writes
recommend -> discover and measure, but never write automatically
automatic -> allow guarded automatic preset application
```

Automatic mode also requires a separate confirmation flag at configuration
validation time. Selecting the mode alone is insufficient.

Manual and automatic requests share the same final safety gate: the calibrated
ROI must contain no detected road user. If a vehicle is present, a manual
request is put back on the queue and the UI reports that the change is waiting.
This avoids changing exposure or focus in the middle of the very crossing whose
evidence will be evaluated.

Successful changes publish the current profile, keep the prior values for an
explicit rollback, and attach the active profile plus an allowlisted settings
snapshot to later event rows. Camera state is therefore evidence attached to
the observation, not just invisible process state.

Recommend-only is the correct starting mode because Phase 6's exit criterion
is not “the API accepted a preset.” The still-open evidence question is whether
the preset improves readable-plate rate and OCR outcomes on separately labelled
day, glare, dusk, and night samples.

**Transferable lesson:** a recommendation is information; a device write is
authority. Give them distinct modes, validation, UI language, and tests.

## 5. Trigger live OCR from a domain event and reuse its evidence

Live ANPR does not run OCR on every video frame. The worker loads one plate
detector and one `RapidOcrRecognizer` when processing starts, then keeps both
alive. `VehicleCandidateBuffer` retains useful vehicle crops as tracking
progresses. Only a confirmed line crossing triggers plate analysis for that
track.

This connects computation to the entity that owns the answer:

```text
latest frame
    |
    v
vehicle detector + tracker
    |                     \
    |                      `-> candidate buffer by track ID
    v                                      |
CrossingCounter emits one crossing          |
    |                                      |
    `--------------------------------------'
                       |
                       v
crossing / centred / largest / sharpest in-memory crops
                       |
                       v
plate detector -> quality gate -> best three candidates
                       |
                       v
one reused RapidOCR recognizer -> normalized observations
                       |
                       v
live consensus -> accepted / uncertain / no_read
```

The live path calls `analyze_images()` rather than writing crops to disk and
calling the offline discovery pipeline. Saving event images remains a separate
opt-in. This both reduces I/O and ensures that enabling OCR does not silently
enable image retention.

The live quality policy is deliberately more permissive than the offline Phase
5 defaults: it accepts smaller crops, more skew, and lower sharpness before OCR.
The pipeline still ranks candidates and limits recognition to three. Policy
injection allows the offline evaluator to retain exact, conservative semantics
while the live collection aid can tolerate the real camera's smaller plate
evidence.

**Transferable lesson:** trigger expensive specialist inference at the domain
event that needs it, reuse already buffered evidence, and inject context-specific
policy without duplicating the pipeline.

## 6. Put uncertainty and privacy at explicit boundaries

OCR first normalizes text and canonicalizes common UK letter/digit ambiguities
against plausible layouts. A format match is weak supporting evidence, never
proof of identity.

Offline Phase 5 consensus requires exact full-text agreement. Live consensus
instead groups eligible observations by the first four characters—the same
prefix that may be stored. By default it accepts either:

- two eligible observations with the same four-character prefix; or
- one UK-format-valid observation with confidence at least `0.90`.

Conflicting prefixes, insufficient agreement, or a failed required format
check remain `uncertain`; no eligible observation becomes `no_read`. The worker
adds only `accepted` results to the event map, so uncertain/no-read crossings
keep blank OCR fields.

`CsvEventStore.write_all()` is the privacy boundary. It always calls
`mask_plate_text()` before constructing an `EventRecord`, exposing the first
four normalized characters and replacing the remainder with `*`. It also
migrates legacy `plate_text` rows through the same mask. The Recent events table
therefore receives masked data from storage rather than relying on display-only
redaction.

The explicit local command has a different contract:

```bash
.venv/bin/roundabout-anpr --image-path PATH
```

It returns the full normalized `ocr_plate` for one deliberately supplied local
image. That makes diagnosis usable when a clear crop produces no live value,
but its output is sensitive and should not be copied into reports, commits, or
shared logs. Event CSV and dashboard output remain masked regardless of this
diagnostic behavior.

**Transferable lesson:** decide where sensitive data may exist, enforce masking
at the durable boundary, and make any unmasked diagnostic path narrow and
explicit.

## End-to-End Execution Flow

The whole Phase 6 plus live OCR path is owned by one long-lived worker:

```text
Dashboard inputs
    |
    v
ProcessingConfig validates camera mode, auto confirmation, timings,
plate model, OCR thresholds, and candidate limit
    |
    v
worker startup
    ├── vehicle detector
    ├── optional long-lived plate detector + RapidOCR
    └── optional capability discovery + local capability snapshot
    |
    v
newest frame only
    ├── ROI quality -> advisor -> recommendation
    │                         |
    │                         v
    │            manual/automatic request + empty ROI + timers
    │                         |
    │                         v
    │            validate -> apply -> read back -> rollback on failure
    │
    └── detect -> track -> buffer crops -> crossing
                                      |
                                      v
                         live plate/OCR consensus
                                      |
                                      v
                 accepted full result in process memory
                                      |
                                      v
 CsvEventStore masks plate + writes OCR confidence, relative speed,
 camera profile, and settings -> dashboard receives masked EventRecord
```

The shared worker matters. Streamlit reruns do not reload the models, reopen the
camera, or create a second control loop. Camera recommendations, crossing state,
candidate buffers, and OCR model lifetime remain coherent inside one processing
session.

## What the Tests Prove

Run the focused tests with:

```bash
.venv/bin/python -m pytest \
  tests/test_camera_quality.py \
  tests/test_camera_control.py \
  tests/test_anpr.py \
  tests/test_anpr_pipeline.py \
  tests/test_ocr.py \
  tests/test_anpr_app.py \
  tests/test_events.py \
  tests/test_dashboard.py
```

Static checks remain:

```bash
.venv/bin/ruff check .
.venv/bin/pyright
```

The focused tests prove that:

- synthetic day, glare, dusk, and night frames cross the intended boundaries;
- ROI measurement ignores misleading bright surroundings;
- a recommendation needs repeated observations and resets on interruption;
- arbitrary setting names, unsupported choices, and out-of-range numbers fail;
- a preset is read back after application and restored after a mismatch;
- common UK OCR ambiguities are canonicalized deterministically;
- two matching privacy-visible prefixes can form live consensus;
- a single format-valid observation must reach the strict `0.90` fallback;
- live in-memory crops pass through the shared plate-analysis pipeline;
- event writes and legacy migration mask all characters after the first four;
- dashboard table rows expose masked OCR text and confidence;
- the `--image-path` shorthand routes to the one-image command.

The tests do not prove:

- that the four presets improve plate readability on this physical camera;
- that brightness-only condition thresholds generalize across weather and
  seasons;
- recovery from a real phone disconnect halfway through a multi-setting write;
- that the empty-ROI gate never misses a road user;
- live end-to-end OCR accuracy or false-read rate on held-out vehicles;
- protection from correlated OCR errors repeated across similar frames;
- that prefix agreement identifies a complete registration;
- throughput and memory behavior during dense traffic at the full stream
  resolution;
- legal authority to collect, retain, or use registration data.

## Try It

### Observe recommendations without changing the camera

Run the dashboard:

```bash
.venv/bin/roundabout-dashboard --server.port 8502
```

Open **Advanced -> Adaptive camera (Phase 6)**, choose **Recommend only**, and
enter the IP Webcam base URL without `/video`. Predict first: capability state
is saved locally, quality metrics update about every five seconds, and no camera
setting request is sent. Covering the calibrated road region should eventually
recommend `night`, but only after three consistent observations.

This is the safest real-device experiment because it exercises observation and
classification without authorizing a write.

### Isolate the three-observation rule

Run the focused advisor test:

```bash
.venv/bin/python -m pytest \
  tests/test_camera_quality.py::test_advisor_requires_repeated_observations_and_resets_candidate \
  -q
```

Predict first: an intervening `day` observation breaks the first `night`
sequence, so the advisor returns `night` only after the final three consecutive
night observations. As a safe code-reading exercise, change the test-only
`required_observations` from `3` to `2`, predict which assertion fails, then
restore it.

### Diagnose one local image without masking

Use a consent-appropriate ignored local crop:

```bash
.venv/bin/roundabout-anpr \
  --image-path data/events/images/example-sharpest.jpg
```

Predict first: the JSON contains `status`, full `ocr_plate`, and
`ocr_confidence`. A missing plate detection, rejected quality gate, weak OCR, or
format failure should be distinguishable from a successful read. Treat the
full plate as sensitive; do not redirect this output into a tracked file.

### Exercise live consensus with fictional data

```bash
.venv/bin/python -c 'from roundabout_ai.anpr import OcrObservation,build_live_consensus; o=lambda t,c: OcrObservation(t,t,t,c,True); print(build_live_consensus("demo",(o("AB12CDE",0.8),o("AB12CDF",0.85))).status.value)'
```

Predict first: the result is `accepted` because both fictional readings share
the stored four-character prefix. Remove the second observation and keep
confidence at `0.8`; predict `uncertain`. Raise the single confidence to `0.95`;
predict `accepted` through the strict single-frame fallback.

This experiment exposes the tradeoff directly: live acceptance is useful for a
masked collection aid, but it is not exact full-plate agreement.

### Verify masking at durable storage

```bash
.venv/bin/python -m pytest \
  tests/test_events.py::test_csv_event_store_masks_ocr_and_migrates_legacy_plate_columns \
  -q
```

Predict first: both the newly supplied fictional plate and the legacy full-text
column become four visible characters plus `*`. The test demonstrates why a UI
bug cannot accidentally reveal full text already stored in `EventRecord`.

## Continuous-Learning Loop

1. **Define the outcome.** Example: “increase accepted masked OCR events without
   increasing held-out false reads.”
2. **Record context with the event.** Keep lighting condition, relative speed,
   active profile, allowlisted settings, OCR status, and confidence associated
   with each crossing.
3. **Change one bounded policy.** Adjust one preset, quality threshold, or
   consensus threshold; do not change camera position, exposure, and acceptance
   policy in the same comparison.
4. **Compare like with like.** Evaluate day against day and similar speed and
   direction strata on vehicles not used for tuning.
5. **Measure abstention and error separately.** Report exact accuracy,
   character accuracy, no-read rate, and false-read rate; an increased acceptance
   rate is not automatically an improvement.
6. **Promote only supported behavior.** Keep recommendation mode until labelled
   evidence supports a preset, and keep uncertain observations blank even when
   a plausible guess would look better in the dashboard.

The reusable pattern is:

```text
observe locally -> stabilize the recommendation -> constrain the side effect
                -> verify external state -> attach context to the event
                -> aggregate evidence -> mask at storage -> evaluate held out
```
