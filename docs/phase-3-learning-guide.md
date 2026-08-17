# Phase 3 Events and Streamlit Dashboard Learning Guide

Phase 3 turns the tracking-and-counting pipeline into a usable local service. Confirmed crossings now survive process restarts as CSV metadata, while a Streamlit dashboard observes one long-lived background worker without taking ownership of the camera or model.

## The 80/20 View

Five ideas explain most of Phase 3:

1. one long-lived worker owns every stateful runtime resource;
2. immutable, locked snapshots separate processing from presentation;
3. Streamlit reruns control the page, while a timed fragment controls only display cadence;
4. a confirmed crossing becomes durable metadata at one explicit boundary;
5. health counters and heartbeat logs turn sleep and camera failures into evidence.

The implementation is split across [`worker.py`](../src/roundabout_ai/worker.py), [`shared_state.py`](../src/roundabout_ai/shared_state.py), [`events.py`](../src/roundabout_ai/events.py), and [`dashboard.py`](../src/roundabout_ai/dashboard.py). The `roundabout-dashboard` entry point is registered in [`pyproject.toml`](../pyproject.toml).

## 1. One worker owns the stateful pipeline

`DetectionWorker` owns the resources whose state must survive ordinary UI reruns:

```text
one worker
  ├── one camera capture service
  ├── one YOLO detector and ByteTrack session
  ├── one crossing counter
  ├── one CSV event store
  └── one inference thread
```

Streamlit reruns the page script after widget interaction. If the page directly created the model or camera, moving a slider could reload model weights, open another stream connection, and reset tracker identity.

`get_worker()` uses `st.cache_resource`, so normal reruns recover the same process-wide `DetectionWorker`. `DetectionWorker.start()` also checks its lifecycle lock and refuses to create a second live thread. These are complementary guards: Streamlit retains the owner, and the owner protects its own invariant.

`ProcessingConfig` contains structural choices such as camera URL, model, device, scene, and output path. Those controls are disabled while processing because changing them requires rebuilding part of the pipeline. `OverlayControls` contains confidence and display toggles; the worker reads those under a separate lock on every processed frame, so they can change live.

Stopping sets a shared event, asks the camera service to stop, and joins the inference thread. Cleanup also runs in `finally` and is registered with `atexit`. Normal UI stops, failures, and interpreter shutdown therefore share the same cleanup path.

**Transferable lesson:** give stateful external resources one explicit, long-lived owner, then make duplicate ownership impossible at both the framework and application boundaries.

## 2. Snapshots are the concurrency boundary

The worker produces data continuously, while Streamlit reads it periodically. `DashboardState` is the narrow bridge between them. Every mutation and snapshot is protected by one lock, so the UI cannot combine a frame from one iteration with counters from another.

`DashboardSnapshot` is frozen and contains presentation-ready state: connection status, annotated frame, health metrics, object and crossing counts, recent events, and a person-visible flag. `snapshot()` copies the image and mappings before returning them. Without the image copy, the UI and worker could observe or mutate the same NumPy buffer concurrently. Recent events use a bounded `deque`, so leaving the dashboard running does not make that list grow forever.

This boundary also explains why the browser can look slower than `roundabout-detect`. The worker processes the newest camera frame independently, but the dashboard displays only the latest copied snapshot on its next refresh. Streamlit must then encode and send that image to the browser. With the default `0.2` second refresh, visible updates are capped near five per second even when capture or inference is faster.

The upstream `LatestFrameStore` follows the same low-latency idea: it overwrites an unconsumed frame instead of building a queue. A rising overwrite count is expected when capture is faster than inference; it shows that stale frames were dropped, not necessarily an error.

**Transferable lesson:** cross a thread or process boundary with a small immutable snapshot, and measure deliberate data loss when freshness matters more than completeness.

## 3. Streamlit reruns and processing have different clocks

The dashboard has three different lifecycles:

```text
widget interaction -> full page rerun
timed fragment     -> live panel rerun every refresh_seconds
worker thread      -> continuous capture and inference until Stop
```

The timed `live_panel()` fragment reads a snapshot and renders status, metrics, frame, object table, crossing chart, and recent events. It does not run inference. Changing refresh from `0.2` to `1.0` seconds should make the browser update less often without reducing the worker's inference rate.

Start and Stop expose an important Streamlit detail. Button disabled states are calculated from the snapshot taken at the beginning of a page run. After `worker.start()` or `worker.stop()` changes the background state, `st.rerun()` immediately starts a new page pass so both buttons are recalculated. Without that explicit rerun, Stop could remain disabled until a later interaction even though the worker was already running.

`dashboard.main()` launches the packaged command and binds to `127.0.0.1`. The dashboard is local-only by default. Binding to a LAN address would be a separate security decision and should add authentication before exposing camera-derived information.

**Transferable lesson:** in reactive applications, separate computation cadence, presentation cadence, and widget-state recalculation; each has a different trigger and performance cost.

## 4. Persistence begins only after a crossing is confirmed

Phase 2's `CrossingCounter` remains the authority on whether a crossing occurred. Phase 3 does not move geometry into storage. It passes each returned `CrossingEvent` to `CsvEventStore.write_all()` immediately after the counter update.

The stable CSV schema is:

```text
timestamp,event_type,line_name,object_class,direction,track_id,
detection_confidence,plate_text,plate_confidence
```

The store creates the parent directory and header on the first non-empty batch, appends rows thereafter, and rejects an existing file with a different header. Timestamps are converted to explicit UTC with millisecond precision; naive datetimes are rejected. A lock protects writes made through one store instance.

`plate_text` and `plate_confidence` are reserved for a later, consent-gated ANPR phase and remain blank now. By default, an event stores metadata only. Raw video is never retained by the event store. When the user explicitly enables event images, each crossing saves one full annotated snapshot, while a bounded in-memory selector also saves distinct padded crossing, centred, and sharpest vehicle crops that satisfy a minimum vehicle-box size.

Both `roundabout-detect` and the dashboard use the same event sink. This keeps persistence independent of presentation. The guarantee is intentionally local and single-process: the CSV writer is not a multi-process database lock, and a row proves that software emitted an event, not that a human-labelled review agrees with it.

**Transferable lesson:** persist domain events after the decision boundary, with a stable schema and privacy-preserving defaults, instead of treating UI state or raw observations as the record.

## 5. Observability distinguishes sleep from camera failure

On macOS, the phone may keep serving IP Webcam while the Mac sleeps. The Python process cannot read frames or execute timers during that sleep. After wake, OpenCV may still be blocked in a stale MJPEG read before the normal reconnect loop can progress.

`CameraCapture` runs a one-second wall-clock heartbeat. `ResumeGapDetector` reports a gap of at least ten seconds. After a resume it records:

```text
system_resume_suspected gap_seconds=958
camera_connection_reset reason=system_resume
```

The first line is evidence that the application stopped executing for roughly 958 seconds. The second says the capture service requested a fresh connection rather than trusting the old one. Status, read-failure, and reconnect counters then expose recovery to the dashboard.

This is evidence-based diagnosis, not absolute proof of sleep: a process pause, severe system stall, or debugger stop could also create a wall-clock gap. Correlating it with `pmset -g log` strengthens the conclusion. In the observed session, seven OpenCV failures aligned with Mac sleep/wake intervals of roughly 1.7 to 17 minutes while the phone stream remained available.

The heartbeat cannot run while the Mac is asleep; it detects the pause only after scheduling resumes. OpenCV timeout properties are backend-dependent, and the local backend did not accept the attempted open/read timeout settings, so the configured five-second values are not guaranteed interruption deadlines.

**Transferable lesson:** log the causal transition you can observe, preserve uncertainty about its cause, and document where third-party timeout guarantees end.

## Execution Flow

```text
roundabout-dashboard
        │
        v
Streamlit sidebar -> cached DetectionWorker
                           │
       CameraCapture -> LatestFrameStore -> newest FramePacket
                                              │
                                      YoloDetector.track
                                              │
                                scale Scene -> ROI filter
                                              │
                                      CrossingCounter
                                  ┌───────────┴───────────┐
                                  v                       v
                           CsvEventStore     optional snapshot + vehicle crops
                                  │
                                  └──> DashboardState.publish_frame
                                               │
                                      locked copied snapshot
                                               │
                                               v
                                  timed Streamlit fragment
                               frame + metrics + chart + table
```

Resume recovery is a separate path:

```text
large heartbeat gap -> system_resume_suspected -> reconnect request
  -> camera_connection_reset -> release stale capture -> reconnect
```

## What the Tests Prove

Run the verified checks with:

```bash
.venv/bin/python -m pytest
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pyright
```

The full suite currently contains 43 passing tests. Phase 3 evidence is concentrated in [`test_events.py`](../tests/test_events.py), [`test_shared_state.py`](../tests/test_shared_state.py), [`test_worker.py`](../tests/test_worker.py), [`test_dashboard.py`](../tests/test_dashboard.py), and [`test_capture.py`](../tests/test_capture.py).

These tests prove that:

- CSV rows use stable fields, one header, UTC timestamps, and empty future plate fields;
- an empty event batch creates no file and an incompatible header is rejected;
- snapshots are atomic, recent events are bounded, and returned frames are copied;
- one worker rejects a duplicate start and publishes processed state using fakes;
- stopping the worker produces a clean stopped state;
- dashboard charts aggregate class and direction correctly;
- the Streamlit page renders without starting a real camera;
- Start immediately enables Stop, and Stop immediately re-enables Start;
- a large synthetic clock gap emits resume evidence and requests a reset;
- the camera retries after an open failure and publishes a frame.

The tests do not prove:

- one-hour live stability or bounded memory over that duration;
- real browser frame rate or end-to-end camera-to-screen latency;
- recovery from every sleep or network failure;
- that OpenCV honours timeout properties on the active backend;
- multi-process CSV safety or crash-atomic durability;
- model, tracker, line-crossing, or person-warning accuracy in this scene;
- safe multi-user semantics if the dashboard is exposed remotely.

The observed live session adds operational evidence: real car crossings were persisted, optional event images were written, and capture reconnected after failures correlated with Mac sleep. The current implementation improves those images by selecting raw best-of-track vehicle crops. This still does not replace the Phase 3 one-hour acceptance run or labelled accuracy review.

## Try It

### Run the local dashboard

```bash
.venv/bin/roundabout-dashboard
```

Start processing and confirm that status progresses through model loading and connection to Live. Move a live overlay control and predict first: the next processed frame should reflect it without reloading the model or resetting cumulative counts.

### Inspect the durable boundary

After a crossing, inspect the last rows:

```bash
tail -n 5 data/events/events.csv
```

Match the row with the recent-event table. Restart the dashboard and predict first: the table and in-memory totals start fresh, but existing CSV rows remain and new rows append.

### Change only dashboard cadence

Set **Dashboard refresh seconds** to `1.0` before starting and compare it with `0.2`.

Predict first: visible frames and metrics update less often, while inference continues independently. `Inference FPS` should remain broadly similar. This isolates presentation cadence from model throughput.

### Gather sleep/resume evidence

Leave the dashboard terminal visible, allow the Mac to sleep, then wake it. Look for the paired heartbeat/reset messages and a later connection message. Correlate the time with:

```bash
pmset -g log
```

Predict first: the logged gap should approximate the interval during which the process was not scheduled, followed by a reset. If reset is delayed, record the OpenCV warning duration too; it is evidence about backend blocking.

### Run the acceptance soak

Run for one hour and record start/end values for processed frames, overwrite count, failures, reconnects, CSV rows, and process memory. Note sleep or Wi-Fi interruptions separately so planned interruptions are not confused with unexplained instability.

## How Phase 3 Enables Phase 4

Phase 4 can enrich the existing event boundary without coupling OCR to Streamlit:

```text
confirmed CrossingEvent
  -> consent-approved plate crop and OCR feasibility gate
  -> populate plate fields only if justified
  -> existing event store and dashboard
```

The blank plate columns are extension points, not evidence that ANPR exists. Before populating them, measure whether plates are human-readable at the actual distance and angle, define retention rules, and evaluate OCR on a consent-appropriate labelled sample.

## Continuous-Learning Loop

1. **Define one observable claim.** Example: “UI reruns never create a second camera reader.”
2. **Name the owner and boundary.** Worker owns runtime state; snapshots cross into Streamlit.
3. **Choose the cheapest proof.** Use a fake-worker test before a live one-hour run.
4. **Instrument the real path.** Record status, gaps, reconnects, frame age, and event rows.
5. **Separate behavior from accuracy.** A row proves persistence; labelled review evaluates correctness.
6. **Classify failures.** Separate camera/network, sleep, model, tracker, geometry, storage, and UI failures.
7. **Record the next prediction.** Change one variable, predict its effect, then compare the metric.

The reusable pattern is:

```text
single owner -> explicit snapshot/event boundary -> focused test
             -> live evidence -> measured limitation -> next experiment
```
