# Phase 0 Camera Diagnostic Learning Guide

Phase 0 answers a foundational question before adding YOLO, tracking, or Streamlit: can the application receive a fresh camera frame reliably, observe failures, and preserve useful samples without building hidden delay? The implementation is deliberately small, but its boundaries become the foundation for every later phase.

## The 80/20 View

Five ideas explain most of Phase 0:

1. capture and consumption have different owners;
2. freshness matters more than processing every frame;
3. reconnecting is normal runtime behavior;
4. useful metrics distinguish source health from consumer performance;
5. recordings must follow elapsed time, not source-frame count.

The command-line entry point is registered as `roundabout-camera` in [`pyproject.toml`](../pyproject.toml). Its orchestration lives in [`src/roundabout_ai/diagnostic.py`](../src/roundabout_ai/diagnostic.py), while continuous capture and shared state live in [`src/roundabout_ai/capture.py`](../src/roundabout_ai/capture.py).

## 1. Separate continuous capture from frame consumption

`CameraCapture` owns the OpenCV connection and reads it on a daemon thread. The main diagnostic loop owns display, snapshots, recording, keyboard input, and periodic reporting. They communicate through `LatestFrameStore`.

This split matters because `cv2.VideoCapture.read()` may block or fail independently of the work performed on a frame. Later, YOLO inference and the Streamlit dashboard can replace the current consumer without taking ownership of the camera connection.

The lifecycle is explicit:

- `CameraCapture.start()` creates at most one capture thread.
- `_run()` owns each `cv2.VideoCapture` instance and releases it in `finally`.
- `CameraCapture.stop()` signals the thread, waits for it, and publishes the final `stopped` state.
- `run()` restores signal handlers, closes recordings, stops capture, and destroys GUI windows in its own `finally` block.

`VideoCaptureLike` and the injected `capture_factory` form a narrow test boundary. Tests can substitute a fake camera without teaching production code about mocks.

**Transferable lesson:** give long-lived external resources one clear owner, then expose a small data boundary to the rest of the application.

## 2. A one-frame buffer protects real-time freshness

An ordinary queue would preserve every frame. That sounds safe, but it is harmful for a live system: if inference takes longer than capture, queued frames become increasingly old.

`LatestFrameStore.publish()` replaces the previous unconsumed packet. `consume_latest()` returns a sequence only once. When replacement happens before consumption, `frames_overwritten` increases.

```text
camera:     frame 41 ── frame 42 ── frame 43 ── frame 44
                         overwrite    overwrite       │
latest slot:                 42   ─────── 43   ────── 44
                                                       │
consumer:                                             read
```

The overwritten count is intentional backpressure telemetry, not automatically an error. When later YOLO inference is slower than the roughly 30 FPS source, the consumer should see the newest available frame and the count should reveal how much input it skipped.

[`tests/test_capture.py`](../tests/test_capture.py) proves that publishing frames 1 and 2 before consuming returns frame 2, records one overwrite, and never returns the same sequence twice.

**Transferable lesson:** for live perception, bound the buffer and make dropped work observable; freshness is often more valuable than completeness.

## 3. Reconnection is a state machine, not an exceptional exit

`CameraCapture._run()` treats camera loss as expected:

```text
starting
   │
   v
connecting ── open failed ──> offline ── wait ──┐
   │                                             │
   └── opened ──> connected ── frame ──> live   │
                        │                        │
                        └── repeated read failure
                                  │
                                  v
                             reconnecting ───────┘
```

Open and read timeout properties are requested from OpenCV, and buffer size is requested as one frame. These settings are backend-dependent: OpenCV may return `False` when a backend does not support them. The outer reconnect loop is therefore the actual reliability guarantee.

Two consecutive failed reads trigger reconnection by default. Every capture object is released before the retry wait. The wait uses `Event.wait()` instead of `sleep()`, allowing shutdown to interrupt it immediately.

The fake-capture test in [`tests/test_capture.py`](../tests/test_capture.py) makes the first open fail and the second succeed. It proves the retry loop creates another capture and eventually publishes a frame. It does not prove how the real Android app, Wi-Fi, or OpenCV backend behaves during a long outage.

**Transferable lesson:** model recoverable infrastructure loss as visible state and retry policy; do not scatter reconnection across consumers.

## 4. Measure capture, consumption, and staleness separately

One FPS number cannot explain a live pipeline. Phase 0 reports:

- `capture_fps`: frames published by the capture thread over a rolling window;
- `consumed_fps`: fresh packets accepted by the main loop;
- `received`, `consumed`, and `overwritten`: the flow through the one-frame boundary;
- `read_failures` and `reconnects`: source reliability;
- `age`: time from local capture publication to the current consumer observation;
- `LIVE` or `STALE`: whether any new frame arrived within the configured threshold.

Both FPS meters and frame age use `time.monotonic()`. Wall-clock adjustments cannot make durations jump backward or forward. Wall time is used only for human-readable filenames.

There is an important boundary: MJPEG frames do not carry a source timestamp here. `age` measures delay inside this Python process; it does not include phone encoding or network transit before OpenCV returns the frame. The guide in [`README.md`](../README.md) documents that limit explicitly.

The completed live smoke test observed 1920×1080 input at about 30 FPS, zero read failures, and low local frame age. That is evidence that the short happy path works on this setup, not evidence for the ten-minute restart acceptance criterion in [`PROJECT_PLAN.md`](../PROJECT_PLAN.md).

**Transferable lesson:** give every latency measurement named endpoints, and report producer and consumer rates independently.

## 5. Recording cadence follows elapsed time

The camera currently supplies about 30 FPS, while recordings default to 15 FPS. Writing every captured frame into a 15 FPS container would create slow-motion playback: three seconds of 30 FPS input would become about six seconds on disk.

`Recorder.write()` avoids that by comparing monotonic elapsed time with the next output-frame deadline:

```text
next_frame_at = frames_written / output_fps
```

Only a frame arriving at or after that deadline is written. Recording duration is bounded to five minutes by CLI validation, output directories are created lazily, and `stop()` releases the writer once.

[`tests/test_diagnostic.py`](../tests/test_diagnostic.py) replaces OpenCV's writer and monotonic clock. At 10 output FPS, frames offered at 0.00, 0.05, and 0.10 seconds produce two writes, not three. A live file probe also verified that a requested three-second, 15 FPS recording produced approximately three seconds of playable video rather than slow motion.

Snapshots take a different path: `save_snapshot()` writes the unscaled source frame at JPEG quality 95, and its test reads the image back to confirm the dimensions are preserved.

**Transferable lesson:** media timestamps and output cadence determine playback duration; frame count alone is not a clock.

## Execution Flow

```text
shell: .venv/bin/roundabout-camera
                 │
                 v
pyproject console script -> diagnostic.main()
                 │
          parse and validate CLI
                 │
                 v
             diagnostic.run()
                 │
       ┌─────────┴──────────┐
       v                    v
CameraCapture thread    main diagnostic loop
       │                    │
OpenCV VideoCapture         │
       │                    │
read/reconnect              │
       │                    │
       v                    │
LatestFrameStore ── consume newest packet
                            │
              ┌─────────────┼──────────────┐
              v             v              v
        metrics/overlay  snapshot JPEG  sampled MP4
```

In GUI mode the final branch also handles `s`, `r`, `q`, and Escape. In headless mode the same main loop runs without OpenCV windows, which makes connectivity checks and timed soak tests scriptable.

## What the Tests Prove

Run the verified suite with:

```bash
.venv/bin/python -m pytest
```

The 11 tests prove:

- latest-frame replacement and overwrite accounting;
- one-time consumption by sequence number;
- rolling capture and consumption rate calculations;
- status, failure, and reconnect counters;
- retry after an injected open failure;
- CLI recording-duration bounds and default URL;
- stale metric formatting;
- full-resolution snapshot persistence;
- elapsed-time sampling by the recorder;
- idempotent recorder shutdown.

They do not prove:

- that every OpenCV backend honors timeout or buffer properties;
- that the Android camera will recover after a real app or Wi-Fi restart;
- ten-minute stability, memory behavior, or absence of long-term lag;
- visual quality in every lighting condition;
- end-to-end phone/network latency;
- AI inference behavior, because Phase 0 intentionally contains no model.

The remaining Phase 0 acceptance test is manual because it requires changing external state: run for ten minutes, stop IP Webcam briefly, restart it, and observe a return from `offline` or `reconnecting` to `live` without rising frame age.

## Try It

### Observe the happy path

```bash
.venv/bin/roundabout-camera \
  --url http://192.168.1.142:8080/video
```

Press `s` for a full-resolution snapshot, `r` to toggle a bounded recording, and `q` to quit.

### Run a reproducible headless probe

```bash
.venv/bin/roundabout-camera \
  --headless \
  --duration 10 \
  --snapshot-after 3 \
  --metrics-interval 1
```

Predict first: after the first decoded frame, status should become `live`; resolution should become 1920×1080 on the current phone configuration; `received` and `consumed` should rise together when the consumer keeps up.

### Experiment: make backpressure visible

This is a safe unit-level experiment that does not modify production code or connect to the camera.

1. Open [`tests/test_capture.py`](../tests/test_capture.py).
2. In `test_latest_frame_store_overwrites_unconsumed_frame`, publish a third distinct frame before `consume_latest()`.
3. Predict the result: sequence 3 should be returned and `frames_overwritten` should be 2.
4. Change only the expected assertions and run:

   ```bash
   .venv/bin/python -m pytest tests/test_capture.py
   ```

5. Revert the temporary exercise change afterward.

This isolates the core design rule without needing a live camera or timing-sensitive test.

### Experiment: observe the real reconnect boundary

```bash
.venv/bin/roundabout-camera --headless --duration 600
```

Predict first: stopping IP Webcam should produce `offline` or `reconnecting`, increment failure/reconnect telemetry, and mark the stream `STALE`. Restarting the server should return to `live` and resume increasing `received`. This experiment writes no media unless recording or snapshot options are supplied.

## How Phase 0 Enables Phase 1

Phase 1 should consume `FramePacket` objects from `LatestFrameStore`, run YOLO on the newest packet, and publish inference FPS separately from capture FPS. The camera thread should remain unchanged. That preserves the ownership boundary and makes slower inference appear as overwrites rather than accumulated latency.

The smallest next proof is not a polished dashboard. It is one detector invocation on a consumed frame with measured inference time, while confirming capture continues near its existing rate.

## Continuous-Learning Loop

Use this loop for Phase 1 and later work:

1. **Define the user-visible goal.** Example: show current vehicle/person boxes without visible lag.
2. **Name the enabling concept.** Consume the newest frame and keep inference independent of capture.
3. **Implement the smallest useful behavior.** Run `yolo26n.pt` on one `FramePacket` and draw its detections.
4. **Prove it at the cheapest meaningful boundary.** Unit-test result adaptation, then use a short saved clip or live smoke test.
5. **Explain what failures revealed.** Separate model misses, slow inference, stale input, and camera failure using distinct metrics.
6. **Record the transferable lesson.** Preserve the principle that mattered—for example, bounded buffering—rather than only the final parameter value.

The reusable pattern is:

```text
goal -> core principle -> smallest change -> focused proof
     -> failure lesson -> next-phase takeaway
```
