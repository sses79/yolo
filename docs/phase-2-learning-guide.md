# Phase 2 Scene Calibration, Tracking, and Counting Learning Guide

Phase 2 turns repeated frame-level detections into anonymous traffic events. The important change is state: the application now knows where analysis is allowed, which observations belong to the same road user, and whether that tracked centre genuinely crossed a configured line.

## The 80/20 View

Five ideas explain most of Phase 2:

1. calibration is a coordinate contract between one reference image and every live frame;
2. ByteTrack supplies identity, while project dataclasses isolate its framework output;
3. the ROI is an admission gate, not a counting mechanism;
4. a crossing is a state transition across a finite segment, not contact with a line;
5. live counts are useful evidence, but they are not durable events or proof of accuracy.

The two commands are registered in [`pyproject.toml`](../pyproject.toml). [`src/roundabout_ai/calibration_app.py`](../src/roundabout_ai/calibration_app.py) creates the scene, while [`src/roundabout_ai/detection_app.py`](../src/roundabout_ai/detection_app.py) applies it to tracked detections. Scene conversion lives in [`src/roundabout_ai/scene.py`](../src/roundabout_ai/scene.py), and the deterministic crossing state machine lives in [`src/roundabout_ai/geometry.py`](../src/roundabout_ai/geometry.py).

## 1. Calibration creates a reusable coordinate contract

`roundabout-calibrate` obtains one full-resolution frame from either the camera or `--image`. `CalibrationSession` then has two explicit modes:

- `roi`: every click adds a polygon vertex; Enter requires at least three vertices and moves to line selection;
- `lines`: each pair of clicks becomes a named `CountLine`; `s` requires at least one complete line before saving.

The command writes two local artifacts:

```text
data/calibration/reference.jpg
data/calibration/scene.yaml
```

The reference image explains what the coordinates mean. The YAML records `reference_size`, ROI points, and count-line endpoints. Both stay outside version control through [`.gitignore`](../.gitignore), because the image may contain public-road or private-scene detail.

`Scene.scaled()` makes the calibration reusable if live frames have a different resolution. It independently multiplies x coordinates by `live_width / reference_width` and y coordinates by `live_height / reference_height`. The ROI and count lines therefore continue to describe the same relative parts of the picture.

This scaling assumes that the camera view has the same composition and aspect. Moving, rotating, digitally zooming, or substantially cropping the phone invalidates the geometric meaning even if the numeric scaling succeeds. Recalibrate after any such change.

**Transferable lesson:** when users draw geometry on an image, store both the coordinates and the coordinate system that gives them meaning.

## 2. Identity is added at the model boundary

Phase 1 used `YoloDetector.predict()` and produced independent `Detection` values. Phase 2 calls `YoloDetector.track()` instead. Internally it invokes Ultralytics with:

```text
persist=True, tracker="bytetrack.yaml"
```

`persist=True` tells Ultralytics that consecutive calls belong to one sequence. ByteTrack can then associate boxes across frames and expose `Results.boxes.id`.

The adapter immediately converts that framework value into `Detection.track_id: int | None`. Downstream scene and geometry code still receives plain immutable dataclasses rather than tensors or Ultralytics `Results`. A box may legitimately have no track ID, so `track_observations()` excludes it from crossing logic while it can still remain visible as a detection.

The detector is constructed once before the live loop. Recreating it per frame would also recreate tracker state, making persistent IDs impossible. The runtime dependency `lap` supplies the assignment solver used by Ultralytics tracking; version `0.5.13` is pinned because the project runs on CPython 3.14.

Track identity is an estimate, not a physical identifier. Occlusion, detector misses, and similar vehicles passing close together can cause a missed track or ID switch. Phase 2 deliberately stores no faces, plates, or other identifying data.

**Transferable lesson:** stateful framework features must have one long-lived owner, but their output should still be translated into application-owned types at the boundary.

## 3. The ROI decides which observations participate

The ROI answers one question: “Is this detection inside the road area we intend to analyse?” It does not count anything.

`filter_detections_by_roi()` calculates the centre of each `xyxy` box and sends that point to `point_in_polygon()`. A centre inside or exactly on the polygon boundary is admitted. A centre outside is removed before track observations reach the counter.

```text
detection box -> box centre -> point-in-polygon -> keep or ignore
```

Using one representative point makes the rule deterministic. It also creates a visible boundary condition: a large box can overlap the ROI while its centre remains outside, so it is ignored until the centre enters. This is simpler and more stable than counting any overlap, but ROI placement should leave enough margin for expected road-user boxes.

An empty ROI means “do not filter” in `point_in_polygon()` and `filter_detections_by_roi()`. Running without `--scene-config` goes further: tracking IDs remain enabled, but both ROI filtering and crossing lines are disabled. That mode is useful for inspecting tracker behavior before calibrating.

**Transferable lesson:** reduce fuzzy spatial intent to one documented admission rule, then test its boundary behavior explicitly.

## 4. A crossing is a guarded state transition

`CrossingCounter` owns a small state record per track ID:

```text
age + last seen frame + last centre + last side per line + counted lines
```

For an ordered count line from `start` to `end`, `signed_side()` uses a 2-D cross product to classify a point as negative, on the line, or positive. A candidate crossing requires the previous and current non-zero sides to differ.

That side change alone is insufficient because an infinite mathematical line extends across the whole image. `segments_intersect()` additionally checks that the motion from the previous centre to the current centre intersects the actual finite count-line segment between its two endpoints.

The complete event gate is:

```text
same persistent track ID
  AND previous/current centres lie on opposite non-zero sides
  AND the centre-motion segment intersects the finite count segment
  AND track age >= --minimum-track-age
  AND this track has not already counted on this line
```

The default minimum age is three tracked observations. This rejects a new two-frame candidate that appears across a line before establishing a useful history. `counted_lines` prevents a vehicle that stops, jitters, or reverses over the line from incrementing the same line twice.

Tracks not observed for more than `--maximum-missing-frames` are removed from the counter. This bounds memory and prevents old state from surviving indefinitely. If ByteTrack later reuses the numeric ID, it starts a fresh counter lifecycle after expiry.

Direction comes from line endpoint order. A negative-to-positive transition gets the line's `negative_to_positive` label; the reverse gets `positive_to_negative`. These default mathematical names can be edited in the generated YAML to scene names such as `entering` and `leaving` without changing the geometry.

**Transferable lesson:** reliable event detection usually combines identity, geometry, temporal history, validation gates, and explicit deduplication—not one threshold or one frame.

## 5. Keep live state separate from durable evidence

For each accepted transition, `CrossingCounter.update()` returns a `CrossingEvent` containing line name, track ID, class, direction, and confidence. It also increments an in-memory counter keyed by line, direction, and class.

`run_live()` immediately prints each new event and includes cumulative crossing totals in periodic metrics. The annotated frame draws the ROI, count lines, detection boxes, and track IDs. This makes the logic observable while calibrating and reviewing the scene.

The counter is intentionally process-local. Restarting `roundabout-detect` resets it, and Phase 2 does not write events to CSV or SQLite. Durable event storage and the Streamlit dashboard belong to Phase 3. Keeping that boundary explicit prevents a display counter from being mistaken for an audited history.

Likewise, deterministic geometry tests do not establish real-world count accuracy. The Phase 2 exit criterion in [`PROJECT_PLAN.md`](../PROJECT_PLAN.md) still requires manually reviewing at least 100 passing road users and reporting correct, missed, and duplicate counts.

**Transferable lesson:** distinguish an observable runtime state from a persisted record, and distinguish software correctness from measured model/system accuracy.

## Execution Flow

Calibration runs once whenever the camera composition changes:

```text
roundabout-calibrate
        │
        ├── camera frame or --image
        v
 CalibrationSession
        │ click ROI points -> Enter -> click line endpoint pairs
        v
 Scene(reference_size, roi, count_lines)
        │
        ├── reference.jpg
        └── scene.yaml
```

The live path applies that contract to every newest frame:

```text
CameraCapture -> LatestFrameStore -> newest FramePacket
                                         │
                                         v
                            YoloDetector.track(frame)
                                         │
                               Detection(track_id)
                                         │
                  load + scale Scene -> ROI centre filter
                                         │
                                         v
                                  TrackObservation
                                         │
                                         v
                                  CrossingCounter
                         ┌───────────────┴───────────────┐
                         v                               v
             new CrossingEvent                 cumulative counts
                         │                               │
                         └───────────────┬───────────────┘
                                         v
                           console metrics + overlay
```

The latest-frame buffer from earlier phases remains upstream. Dropping stale frames keeps the display responsive, while ByteTrack associates the sequence that the inference loop actually receives.

## What the Tests Prove

Run the verified suite with:

```bash
.venv/bin/python -m pytest
```

The full suite currently contains 31 passing tests. Phase 2 coverage spans [`tests/test_calibration_app.py`](../tests/test_calibration_app.py), [`tests/test_scene.py`](../tests/test_scene.py), [`tests/test_geometry.py`](../tests/test_geometry.py), plus tracker and CLI checks in [`tests/test_detector.py`](../tests/test_detector.py) and [`tests/test_detection_app.py`](../tests/test_detection_app.py).

These tests prove that:

- calibration requires a valid ROI and constructs endpoint pairs as named lines;
- scene YAML survives a save/load round trip;
- coordinates scale with frame resolution;
- duplicate line names and malformed scene structures are rejected;
- ROI membership includes polygon boundaries and uses the box centre;
- tracker IDs are adapted into `Detection` values;
- `persist=True` and the requested tracker configuration reach Ultralytics;
- signed-side direction depends on endpoint order;
- crossing the finite segment emits the expected direction once;
- crossing only an infinite extension is rejected;
- young tracks are rejected by the minimum-age gate;
- the same track cannot count twice on the same line;
- missing track state expires;
- scene annotation draws on a copy instead of mutating the capture frame;
- CLI defaults and metrics expose the Phase 2 controls and counts.

A real `yolo26n.pt` call with ByteTrack also completed successfully on a synthetic blank frame after installing `lap`. That smoke test proves the installed framework and assignment dependency can initialize together under Python 3.14; it does not prove tracking quality.

The tests do not prove:

- that the selected ROI matches the intended road surface;
- that the line is placed where vehicle centres reliably cross it;
- that ByteTrack preserves IDs through real occlusion;
- the best minimum age or missing-frame limit for this camera;
- accuracy across vehicle classes, directions, weather, or lighting;
- the 100-road-user acceptance criterion;
- persistence across process restarts.

The attempted live smoke test could not add evidence because the phone endpoint returned `connection refused`. Treat the next successful live run and manual review as acceptance work, not as optional polish.

## Try It

### Calibrate the live view

```bash
.venv/bin/roundabout-calibrate
```

Predict first: detections will later be admitted only when their box centres fall inside the yellow ROI. Place the orange count line so expected centre trajectories cross between its endpoints, with room for natural path variation.

Use a saved frame when the camera is unavailable:

```bash
.venv/bin/roundabout-calibrate \
  --image data/snapshots/example.jpg \
  --output data/calibration/scene.yaml
```

### Observe tracking and counting

```bash
.venv/bin/roundabout-detect \
  --scene-config data/calibration/scene.yaml
```

Watch a single road user from ROI entry to exit. Its label should retain one `#track_id`. When its centre crosses the finite line, the console should print one event and the cumulative crossing metric should increase once.

### Experiment: change only minimum track age

Run comparable observations with two values:

```bash
.venv/bin/roundabout-detect \
  --scene-config data/calibration/scene.yaml \
  --minimum-track-age 2

.venv/bin/roundabout-detect \
  --scene-config data/calibration/scene.yaml \
  --minimum-track-age 6
```

Predict first: age 2 should accept short-lived tracks more readily but may admit transient false positives. Age 6 should demand more evidence but may miss objects first detected close to the line. Compare against manually observed crossings rather than assuming either total is more accurate.

### Experiment: reverse direction semantics safely

Make a copy of the local scene YAML and swap the count line's `start` and `end` points. Predict first: the same physical crossing should still count, but its positive/negative direction label should reverse. This demonstrates that direction is defined by ordered geometry, not by camera intuition. Restore the original file after the experiment.

## How Phase 2 Enables Phase 3

Phase 3 can consume `CrossingEvent` without reimplementing tracking or geometry:

```text
CrossingEvent -> event store -> shared worker snapshot -> Streamlit dashboard
```

The next smallest useful boundary is an event sink that accepts a crossing and writes a stable row with a timestamp. The worker can then publish current counts and recent events independently of Streamlit reruns. Tests should first prove event serialization and idempotent worker ownership, then use a short recorded clip before a one-hour live run.

## Continuous-Learning Loop

Use this loop for Phase 3 and later computer-vision changes:

1. **Define the user-visible goal.** Show one trustworthy passing-road-user event.
2. **Name the enabling concept.** Separate observation, identity, geometry, and persistence.
3. **Implement the smallest useful behavior.** Persist one existing `CrossingEvent` without moving crossing logic into storage or UI code.
4. **Prove it at the cheapest meaningful boundary.** Test one deterministic trajectory and one serialized event, then replay a short labelled clip.
5. **Explain what failures revealed.** Classify errors as detector misses, tracker ID switches, ROI exclusion, line geometry, deduplication, or storage/UI lifecycle bugs.
6. **Record the transferable lesson.** Every layer should own one kind of truth and expose a small boundary to the next.

The reusable pattern remains:

```text
goal -> core principle -> smallest change -> focused proof
     -> failure lesson -> measured next step
```
