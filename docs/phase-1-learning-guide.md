# Phase 1 Vehicle and Person Detection Learning Guide

Phase 1 turns fresh camera frames into local AI observations. Its most important achievement is not drawing rectangles: it adds inference without weakening Phase 0's low-latency capture design, and it defines a clean boundary between raw YOLO output and the tracking, counting, and dashboard features that come later.

## The 80/20 View

Five ideas explain most of Phase 1:

1. AI is a consumer of the existing latest-frame boundary;
2. an adapter keeps Ultralytics objects out of application logic;
3. model metadata, not assumed numeric IDs, defines the class filter;
4. device choices come from synchronized measurements on the real workload;
5. a detection describes one frame, not a persistent vehicle or event.

The installed `roundabout-detect` command is registered in [`pyproject.toml`](../pyproject.toml). Its camera/inference loop lives in [`src/roundabout_ai/detection_app.py`](../src/roundabout_ai/detection_app.py), while model adaptation, annotation, and benchmarking live in [`src/roundabout_ai/detector.py`](../src/roundabout_ai/detector.py).

## 1. Keep inference downstream of the latest-frame boundary

Phase 1 reuses `CameraCapture` and `LatestFrameStore` from [`src/roundabout_ai/capture.py`](../src/roundabout_ai/capture.py). The capture thread continues reading the MJPEG stream independently. The detection loop calls `consume_latest()` only when it is ready for another inference.

```text
camera at ~30 FPS -> one-frame store -> detector at its available rate
                          │
                          └── replaces an old unprocessed frame
```

This behavior is visible in the live metrics. A 45-second validation run received 1,345 frames, processed 841, and overwrote 503. Those overwrites are the mechanism preventing a queue of increasingly stale frames. The detector processed fewer frames than the camera produced, but it continued working on recent input.

`frame_age` is calculated after prediction completes:

```text
prediction completion time - FramePacket.captured_at
```

It therefore includes local waiting and prediction time. As in Phase 0, it does not include time spent encoding on the phone or travelling over the network before OpenCV produced the frame.

**Transferable lesson:** add expensive processing after a bounded freshness boundary; do not let a slower model become the owner of camera pacing.

## 2. Translate framework output at one adapter boundary

`YoloDetector` owns the Ultralytics model. The rest of the project receives immutable application types:

- `Detection`: class ID, label, confidence, and integer `xyxy` coordinates;
- `DetectionBatch`: a tuple of detections, total elapsed time, and device;
- `BenchmarkResult`: summary timings without framework objects.

Ultralytics returns tensors through `Results.boxes`. `YoloDetector.predict()` moves their values to CPU-visible lists, converts coordinates and class IDs to ordinary Python values, and returns project dataclasses. That conversion happens in one place.

This boundary provides two benefits. First, drawing, reporting, future tracking, and tests do not need to know the Ultralytics tensor API. Second, a later model implementation can produce the same `DetectionBatch` without forcing changes throughout the application.

The model is constructed once in `YoloDetector.__init__()`. `run_live()` creates one detector before starting the camera loop, so model weights are not reloaded for every frame. This same ownership rule will matter when Streamlit reruns page code in Phase 3.

**Transferable lesson:** translate third-party return types into small domain types at the integration edge, and keep the expensive resource behind that edge.

## 3. Filter semantic class names through model metadata

The project is interested in six classes:

```text
person, bicycle, car, motorcycle, bus, truck
```

They are defined by `ROAD_USER_CLASSES` and configurable through `--classes`. `YoloDetector` reads `model.names`, normalizes either its dictionary or list representation, and derives the matching numeric IDs. It passes only those IDs to `model.predict()`.

This is safer than embedding remembered COCO numbers in application code. The adapter validates that every requested name exists and fails early with a useful error if a different model lacks one.

The default prediction call also carries the two main accuracy/performance controls:

- `conf=0.35`: discard lower-confidence candidates;
- `imgsz=640`: resize input for model inference.

These values affect different failure modes. Lower confidence may recover distant road users but increase false positives. A larger image may preserve more small-object detail but costs additional inference time. Neither setting should be chosen from intuition alone; compare them on the same labelled roundabout footage.

**Transferable lesson:** configure models with semantic names at the application boundary, then resolve framework-specific identifiers from the loaded artifact.

## 4. Benchmark the workload, not the hardware label

`--benchmark` captures one 1920×1080 frame, stops the camera, and uses that same frame for every requested device. Each detector performs warmup calls before measured calls. `benchmark_detector()` reports mean, median, minimum, maximum, and effective FPS.

MPS operations can be asynchronous. Timing only the Python call could stop the clock before the GPU finishes. `_synchronize()` calls `torch.mps.synchronize()` immediately before and after each MPS prediction, making the measured interval comparable with synchronous CPU work.

The initial project benchmark used `yolo26n.pt`, `imgsz=640`, two warmups, and five measured calls:

| Device | Mean | Effective FPS |
|---|---:|---:|
| CPU | 34.1 ms | 29.3 |
| MPS | 36.1 ms | 27.7 |

CPU was slightly faster in this small sample, so [`config.example.yaml`](../config.example.yaml) and the CLI default to CPU. This result is local and workload-specific. Another Mac, model size, image size, thermal state, or longer benchmark may change the outcome. `--device auto` remains available and selects MPS when PyTorch reports it available.

The live processing rate was lower and more variable than the isolated benchmark, commonly around 16–21 FPS. That is expected because the live loop also contends with capture, drawing, metrics, scheduling, and occasional latency spikes. “Effective FPS” from a repeated frozen frame is a model-call measurement, not a promise about the complete application.

**Transferable lesson:** synchronize asynchronous devices, hold the input constant, warm up first, and keep microbenchmark claims separate from end-to-end throughput.

## 5. Detection is observation, not identity

`DetectionBatch.counts` summarizes labels in the current frame. If the same car appears in 30 frames, Phase 1 observes it up to 30 times. Nothing in this phase assigns a track ID, remembers a trajectory, detects a line crossing, or writes an event.

That boundary prevents a tempting but serious mistake: adding per-frame `car` counts together and calling the result “cars passed.” Persistent identity belongs to Phase 2.

Presentation is also downstream of detection. `annotate_detections()` copies the source frame, clamps coordinates to image bounds, draws class-specific boxes, and adds labels such as `car 0.87`. `draw_overlay()` then adds performance metrics. The original capture frame remains unchanged for another consumer or later processing stage.

The live command can show this result, save it with `s`, or save the first positive frame using `--snapshot-on-detection`. Headless mode runs the same inference path without creating a GUI window. Snapshots and downloaded model weights are excluded by [`.gitignore`](../.gitignore).

**Transferable lesson:** name data by what it proves—a frame-level observation is not a unique object, crossing, or historical event.

## Execution Flow

```text
shell: .venv/bin/roundabout-detect
                    │
                    v
          detection_app.main()
                    │
       parse model/device/classes/settings
                    │
                    v
          construct YoloDetector once
                    │
                    v
        CameraCapture background thread
                    │
                    v
             LatestFrameStore
                    │ consume newest FramePacket
                    v
         YoloDetector.predict(frame)
                    │
        ┌───────────┼────────────┐
        v           v            v
 preprocess      YOLO26n     postprocess
                                 │
                                 v
                        DetectionBatch
                                 │
             ┌───────────────────┼─────────────────┐
             v                   v                 v
       boxes + labels      timing metrics    current counts
             │                   │                 │
             └───────────────────┴─────────────────┘
                                 │
                                 v
                  GUI / console / optional snapshot
```

Benchmark mode branches after argument parsing. It captures one frame, stops the live source, creates one detector per available requested device, warms each detector, and prints timing summaries.

## What the Tests Prove

Run the verified suite with:

```bash
.venv/bin/python -m pytest
```

The full suite currently contains 19 passing tests. The eight Phase 1-focused tests in [`tests/test_detector.py`](../tests/test_detector.py) and [`tests/test_detection_app.py`](../tests/test_detection_app.py) prove:

- class strings are normalized and deduplicated;
- requested names are converted to model-provided class IDs;
- unknown classes fail during detector construction;
- confidence, image size, device, class IDs, and quiet mode reach the model call;
- tensor-like results become the expected immutable `Detection` values;
- per-frame class counts are derived correctly;
- annotation creates a modified copy without altering the source frame;
- warmups are excluded from benchmark statistics;
- benchmark summaries calculate their timing and FPS values correctly;
- CLI defaults match the documented Phase 1 configuration;
- invalid confidence and benchmark-device arguments are rejected;
- metrics include device, inference rate, responsiveness, and class counts.

The fake model in `test_yolo_detector_adapts_model_results` is the important isolation boundary. It proves the application/framework contract without downloading weights or depending on model accuracy.

The tests do not prove:

- that YOLO correctly recognizes every roundabout road user;
- precision or recall at the current camera angle and lighting;
- that a detection remains associated with the same object in another frame;
- the best confidence or image size for this scene;
- long-run thermal performance;
- CPU/MPS performance on another machine;
- line crossing, deduplication, or vehicle counts.

Real-model validation detected four people and one bus in Ultralytics' standard test image, and the live stream produced a person detection. Representative daylight review across cars, trucks, buses, motorcycles, bicycles, and people remains the Phase 1 acceptance exercise recorded in [`PROJECT_PLAN.md`](../PROJECT_PLAN.md).

## Try It

### Observe live inference

```bash
.venv/bin/roundabout-detect
```

Predict first: capture FPS should remain close to the camera rate, inference FPS may be lower, and `overwritten` should rise whenever prediction cannot consume every frame. That rising counter is healthy if `frame_age` stays bounded.

### Capture the first positive observation

```bash
.venv/bin/roundabout-detect \
  --headless \
  --duration 60 \
  --snapshot-on-detection
```

This saves at most one annotated snapshot. Without a snapshot option, headless detection stores no frames.

### Experiment: change the confidence threshold

Run two short observations at different thresholds:

```bash
.venv/bin/roundabout-detect --headless --duration 30 --confidence 0.25
.venv/bin/roundabout-detect --headless --duration 30 --confidence 0.60
```

Predict first: `0.25` should admit more weak candidates and may detect smaller or less clear objects, but it may also create more false positives. `0.60` should report fewer, stronger detections. Do not decide which is better from total counts alone; inspect the same saved clip or manually reviewed scene.

### Experiment: trade detail for speed

Benchmark the same setup twice:

```bash
.venv/bin/roundabout-detect --benchmark --image-size 480 --benchmark-runs 10
.venv/bin/roundabout-detect --benchmark --image-size 640 --benchmark-runs 10
```

Predict first: 480 should usually be faster, while 640 has more input detail available for small distant road users. The command captures a new frame for each invocation, so this is a practical comparison rather than a perfectly controlled scientific one. For a strict comparison, extend benchmark mode later to accept one saved image path.

## How Phase 1 Enables Phase 2

Phase 2 should consume the plain `Detection` values and add state across frames:

```text
DetectionBatch -> tracker -> track ID + trajectory
                           -> ROI membership
                           -> line-side history
                           -> one crossing event
```

The detector should not learn about crossing lines or event storage. The next smallest useful behavior is a tracker adapter that accepts consecutive detections and returns persistent IDs. Its cheapest proof is a deterministic sequence of synthetic boxes that move across frames without requiring a live camera.

## Continuous-Learning Loop

Use this loop for tracking and every later phase:

1. **Define the user-visible goal.** Count one passing car once.
2. **Name the enabling concept.** Persistent identity across consecutive frames.
3. **Implement the smallest useful behavior.** Convert `Detection` boxes into tracker input and return track IDs.
4. **Prove it at the cheapest meaningful boundary.** Use a synthetic multi-frame path, then a short recorded clip.
5. **Explain what failures revealed.** Separate detector misses, tracker ID switches, ROI mistakes, and crossing-state errors.
6. **Record the transferable lesson.** Keep frame observation, identity, geometry, and event persistence as distinct responsibilities.

The reusable pattern remains:

```text
goal -> core principle -> smallest change -> focused proof
     -> failure lesson -> next-phase takeaway
```
