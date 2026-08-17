# Phase 4 ANPR Feasibility Learning Guide

Phase 4 answers a narrower and more valuable question than “which OCR model should we use?” It asks whether the real camera view contains enough human-readable plate information to justify OCR work at all. The phase improves event-image evidence, turns human review into validated metadata, and applies an explicit decision policy before Phase 5 can begin.

## The 80/20 View

Five ideas explain most of Phase 4:

1. camera feasibility is a separate gate before OCR accuracy;
2. one tracked vehicle, not one saved image, is the sampling unit;
3. bounded best-of-track candidates are retained in memory and written only after a confirmed crossing;
4. quantity, condition coverage, readability, and character size are independent requirements;
5. privacy is enforced by defaults and schema boundaries, not only by user intent.

The capture path is implemented in [`event_images.py`](../src/roundabout_ai/event_images.py) and connected to the long-lived dashboard worker in [`worker.py`](../src/roundabout_ai/worker.py). The assessment path is implemented in [`anpr_feasibility.py`](../src/roundabout_ai/anpr_feasibility.py) and exposed as `roundabout-anpr-assess` through [`pyproject.toml`](../pyproject.toml).

## 1. Gate camera evidence before choosing OCR

An OCR engine cannot recover characters that the camera never resolved. Phase 4 therefore contains no OCR model. It records whether a human can read a plate at the original pixels and approximately how many pixels describe the plate and its characters.

`FeasibilitySample` captures the evidence needed for that decision:

```text
anonymous sample ID + local image path
direction + lighting + speed + plate side
human readability
plate width + plate height + character height
quality notes
```

The schema deliberately has no registration-text field. A readable label means a reviewer could read the plate; it does not require storing what the plate says.

The default policy asks for at least 20 observations, an 80% readable rate, and a median character height of at least 16 pixels. These are project thresholds, not claims that an OCR engine will achieve a particular accuracy. Even `proceed` means only that Phase 5 is worth evaluating with a separate held-out dataset.

**Transferable lesson:** test whether the input contains the information a downstream model needs before tuning or replacing that model.

## 2. One vehicle is one observation

One crossing can produce several files:

```text
snapshot  -> full annotated crossing frame
crossing  -> raw vehicle crop from the crossing frame
centred   -> raw crop with the strongest completeness/edge-clearance rank
sharpest  -> raw crop with the highest whole-crop Laplacian variance
```

These are alternative views of one tracked vehicle, not independent samples. Counting all three would make one easy vehicle worth three votes and inflate both sample quantity and readability.

The human workflow is therefore:

1. group files by track ID;
2. inspect that track’s `crossing`, `centred`, and `sharpest` candidates;
3. choose the most readable candidate;
4. add one annotation row for that vehicle.

The implementation removes duplicate candidate frames by `frame_number`. If one frame wins multiple categories, `VehicleCandidateBuffer.select()` returns it only once. The annotation CLI separately rejects duplicate `sample_id` values, but it does not parse filenames to prove that two different IDs refer to different tracks. One-vehicle-per-row remains an important review rule.

“Sharpest” is relative, not absolute. The score covers the entire padded vehicle crop, not the plate region. Headlights, body edges, or background texture can win the score while the plate remains motion-blurred. If every observed frame is blurry, the best candidate is still blurry.

**Transferable lesson:** define the independent experimental unit before collecting metrics, and distinguish a ranking heuristic from a quality guarantee.

## 3. Buffer first, persist only confirmed evidence

`VehicleCandidateBuffer` observes raw tracked vehicle detections on every processed frame. It ignores people, unsupported classes, and detections without a track ID. Each accepted crop is copied before boxes, ROI lines, or metric overlays are drawn.

For every live track, the buffer keeps at most three candidates:

- `crossing` is continually replaced by the newest observation;
- `centred` prefers suitable, unclipped crops with more edge clearance, then larger area and sharpness;
- `sharpest` prefers suitable, unclipped crops with higher Laplacian variance, then area.

The name `centred` is shorthand for a completeness heuristic. Its rank uses distance from the left or right image edge, not Euclidean distance from the image centre.

Memory is bounded in two ways. A track expires after the configured missing-frame limit, and the buffer keeps at most 32 tracks by default. This bounds growth, although 4K crops are individually large and can still create meaningful memory pressure.

The worker’s order is the important ownership boundary:

```text
raw frame
  -> YOLO + ByteTrack detections
  -> candidate_buffer.observe(raw frame, detections)
  -> ROI filter
  -> CrossingCounter.update(...)
  -> CsvEventStore.write_all(...)
  -> only for confirmed event records:
       save annotated snapshot
       select and save raw candidate crops
```

The buffer may observe vehicles outside the ROI, but nothing is written for them unless the filtered counting path emits a confirmed crossing event. Image retention is also opt-in: the buffer is not constructed unless **Save event snapshot and vehicle crops** was enabled when the worker started.

Crop padding and minimum source-box dimensions are configurable in the dashboard. Padding preserves context around a plate; minimum dimensions avoid filling the assessment set with obviously tiny vehicles. A vehicle that fails the crop threshold can still receive a full event snapshot.

**Transferable lesson:** keep provisional evidence bounded and ephemeral, then cross the persistence boundary only after the domain event is confirmed.

## 4. A complete sample and a good sample are different things

`assess_feasibility()` evaluates the policy in a deliberate order:

```text
insufficient quantity or missing coverage -> incomplete_sample
else readable rate below 20%               -> stop_at_vehicle_analytics
else readability >= 80% and median
     character height >= 16px              -> proceed
else                                        -> reposition_first
```

This precedence prevents a small or one-sided sample from producing a confident decision. Twenty excellent front plates in one direction do not show that rear plates, the opposite direction, or faster vehicles will work.

By default, coverage requires:

- both `approaching` and `departing` directions;
- both `slow` and `fast` speeds;
- both `front` and `rear` plate sides.

Relevant lighting conditions can be added with `--required-lighting`. `speed=unknown` is valid because a still image does not prove speed, but it intentionally satisfies neither slow nor fast coverage. Speed coverage needs trustworthy video review or another measurement source; guessing from a still would make the report look complete without adding evidence.

Readability and pixel size are also separate. A human may infer a small plate from context, while OCR remains fragile. Conversely, a large plate can be unreadable because of motion blur, glare, clipping, or focus. The median character height makes the policy less sensitive to one unusually close vehicle.

**Transferable lesson:** separate coverage, quantity, and quality checks, and make incomplete evidence a first-class result instead of treating it as failure or success.

## 5. Privacy is a structural property

Event images are disabled by default in [`dashboard.py`](../src/roundabout_ai/dashboard.py). When enabled, files remain local under `data/events/images/`. Feasibility annotations and reports live under `data/anpr/`. Both directories are ignored by Git through [`.gitignore`](../.gitignore), so normal commits do not publish vehicle images or local review data.

The feasibility CSV stores anonymous IDs, local paths, conditions, measurements, and notes. It stores no registration text. `render_report()` aggregates counts and medians without copying images or plate content into the Markdown report.

These boundaries reduce accidental retention, but they do not replace operational responsibility. A local image can still contain identifiable information. Collection must remain consent-appropriate, access should remain limited, and images should be deleted when they are no longer needed.

**Transferable lesson:** privacy controls are strongest when defaults, schemas, file locations, and version-control rules all reinforce the same policy.

## Execution Flow

The capture and assessment paths meet through local files, not through OCR:

```text
IP Webcam frame
      │
      v
YOLO + ByteTrack -> raw best-of-track candidate buffer
      │                         │
      v                         │ bounded and provisional
ROI + CrossingCounter           │
      │                         │
confirmed crossing ─────────────┘
      │
      ├── CSV event metadata
      ├── annotated snapshot             opt-in
      └── distinct raw vehicle crops     opt-in
                    │
                    v
        human selects one image per track
                    │
                    v
roundabout-anpr-assess add -> validated annotations.csv
                    │
                    v
roundabout-anpr-assess report
      │
      ├── incomplete_sample
      ├── reposition_first
      ├── stop_at_vehicle_analytics
      └── proceed -> only then evaluate OCR in Phase 5
```

## What the Tests Prove

Run the focused Phase 4 tests with:

```bash
.venv/bin/python -m pytest tests/test_event_images.py tests/test_anpr_feasibility.py
```

Static checks remain:

```bash
.venv/bin/ruff check src tests
.venv/bin/ruff format --check src tests
.venv/bin/pyright
```

Evidence is concentrated in [`test_event_images.py`](../tests/test_event_images.py) and [`test_anpr_feasibility.py`](../tests/test_anpr_feasibility.py). These tests prove that:

- the buffer selects distinct crossing, centred, and sharpest frames;
- people and untracked detections are ignored;
- stale tracks expire and total track state is bounded;
- crops are padded, edge clipping is recorded, and undersized vehicle boxes are rejected;
- snapshots and raw candidates use event-specific filenames and retain expected dimensions;
- annotation files round-trip through a stable CSV schema;
- invalid schemas and duplicate sample IDs are rejected;
- missing condition coverage returns `incomplete_sample`;
- synthetic complete samples exercise `proceed`, `reposition_first`, and `stop_at_vehicle_analytics`;
- the report explains its policy and does not introduce a plate-text field.

The tests do not prove:

- that a saved crop contains a visible plate;
- that `sharpest` is the best plate frame;
- that an annotation path points to an existing image;
- that a human-readable label or pixel measurement is correct;
- that the local sample is unbiased or consent-appropriate;
- that 16-pixel characters guarantee OCR accuracy;
- real low-light, rain, glare, motion, or 4K memory performance;
- any OCR model’s accuracy or false-read rate.

The local post-tuning report adds operational evidence, not a test guarantee: 25 of 28 reviewed vehicles were marked human-readable (89.3%), with balanced approaching/departing and front/rear coverage. Median character height was 14.5 pixels, and every speed remained `unknown`. The report therefore remained `incomplete_sample` because slow/fast coverage had not been established; the character-size result also identifies a remaining camera-quality risk.

## Try It

### Collect real candidates

Run the dashboard:

```bash
.venv/bin/roundabout-dashboard
```

Enable **Save event snapshot and vehicle crops** before pressing Start. After a confirmed crossing, group the resulting files by track ID and compare the `crossing`, `centred`, and `sharpest` images.

Predict first: `sharpest` will often be useful, but a `centred` or `crossing` candidate can show a clearer plate because the score is not plate-specific.

### Exercise the policy without real plate data

Use temporary metadata and fictional image paths so the experiment retains no camera information:

```bash
.venv/bin/roundabout-anpr-assess init \
  --output /tmp/roundabout-anpr-learning.csv \
  --overwrite

.venv/bin/roundabout-anpr-assess add \
  --annotations /tmp/roundabout-anpr-learning.csv \
  --sample-id approaching-demo \
  --image-path images/approaching-demo.jpg \
  --direction approaching \
  --lighting day \
  --speed slow \
  --plate-side front \
  --human-readable yes \
  --plate-width 90 \
  --plate-height 24 \
  --character-height 18

.venv/bin/roundabout-anpr-assess add \
  --annotations /tmp/roundabout-anpr-learning.csv \
  --sample-id departing-demo \
  --image-path images/departing-demo.jpg \
  --direction departing \
  --lighting day \
  --speed fast \
  --plate-side rear \
  --human-readable yes \
  --plate-width 90 \
  --plate-height 24 \
  --character-height 18

.venv/bin/roundabout-anpr-assess report \
  --annotations /tmp/roundabout-anpr-learning.csv \
  --output /tmp/roundabout-anpr-learning-report.md \
  --minimum-samples 2
```

Predict first: this balanced synthetic sample should produce `proceed`. Then run the report again with `--minimum-character-height 20`. Only one policy variable changes, so the expected recommendation is `reposition_first`.

### Separate daylight and evening evidence

Do not mix a new exposure or zoom configuration silently into an earlier sample. Archive the previous annotations, collect a new group of distinct tracks, and require the relevant lighting conditions:

```bash
.venv/bin/roundabout-anpr-assess report \
  --required-lighting day,evening \
  --output data/anpr/report-day-evening.md
```

Predict first: a dataset containing only daylight rows becomes `incomplete_sample`, even if every plate is readable. This is the desired behavior because the evidence does not cover the intended evening condition.

## How Phase 4 Enables Phase 5

Phase 5 should start only after the feasibility report supports the intended operating conditions. Its job is different:

```text
Phase 4: does the camera preserve readable characters?
Phase 5: how accurately does a chosen OCR pipeline recognize them?
```

Phase 5 will need consent-appropriate ground truth, a held-out evaluation set, normalization rules, false-read analysis, and retention controls. Passing Phase 4 does not supply any of those automatically; it prevents starting them with inadequate camera evidence.

## Continuous-Learning Loop

1. **Define the user-visible goal.** Example: “decide whether this fixed view justifies an OCR prototype.”
2. **Name the enabling concept.** The camera is an information boundary; best-frame selection and labelled measurements expose its limits.
3. **Implement the smallest useful behavior.** Save bounded candidates only for confirmed crossings and record one anonymous row per vehicle.
4. **Prove it at the cheapest meaningful boundary.** Unit-test selection and policy logic before collecting real images.
5. **Explain what failures revealed.** Missing speed labels mean incomplete coverage; readable but small characters mean camera improvement may still be needed.
6. **Record the transferable lesson.** Separate input feasibility from model accuracy, and require evidence across the conditions that matter.

The reusable pattern is:

```text
bounded raw evidence -> one independent observation -> explicit policy
                     -> honest incomplete state -> next controlled experiment
```
