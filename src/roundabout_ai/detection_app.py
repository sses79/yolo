"""Live vehicle/person detection, tracking, ROI filtering, and counting."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from collections import Counter, deque
from collections.abc import Sequence
from pathlib import Path

import cv2

from roundabout_ai.capture import CameraCapture, CaptureConfig, Frame, LatestFrameStore
from roundabout_ai.detector import (
    ROAD_USER_CLASSES,
    YoloDetector,
    annotate_detections,
    available_devices,
    benchmark_detector,
    parse_class_names,
)
from roundabout_ai.diagnostic import DEFAULT_URL, draw_overlay, save_snapshot
from roundabout_ai.events import CsvEventStore
from roundabout_ai.geometry import CrossingCounter
from roundabout_ai.scene import (
    Scene,
    annotate_scene,
    filter_detections_by_roi,
    load_scene,
    track_observations,
)

WINDOW_NAME = "Roundabout AI detection — s snapshot, q quit"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def confidence_value(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def class_names_value(value: str) -> tuple[str, ...]:
    try:
        return parse_class_names(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def benchmark_devices_value(value: str) -> tuple[str, ...]:
    devices = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    invalid = sorted(set(devices) - {"cpu", "mps"})
    if not devices or invalid:
        suffix = f": {', '.join(invalid)}" if invalid else ""
        raise argparse.ArgumentTypeError(f"devices must be cpu and/or mps{suffix}")
    return devices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run local YOLO vehicle/person detection on the newest IP Webcam frame."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ROUNDABOUT_CAMERA_URL", DEFAULT_URL),
        help="MJPEG stream URL (default: %(default)s)",
    )
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--confidence", type=confidence_value, default=0.35)
    parser.add_argument("--image-size", type=positive_int, default=640)
    parser.add_argument(
        "--scene-config",
        type=Path,
        help="YAML ROI and count-line calibration from roundabout-calibrate",
    )
    parser.add_argument("--tracker-config", default="bytetrack.yaml")
    parser.add_argument("--minimum-track-age", type=positive_int, default=3)
    parser.add_argument("--maximum-missing-frames", type=nonnegative_int, default=30)
    parser.add_argument(
        "--classes",
        type=class_names_value,
        default=ROAD_USER_CLASSES,
        help="comma-separated model class names",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=positive_float)
    parser.add_argument("--metrics-interval", type=positive_float, default=2.0)
    parser.add_argument("--reconnect-seconds", type=positive_float, default=2.0)
    parser.add_argument("--open-timeout-seconds", type=positive_float, default=5.0)
    parser.add_argument("--read-timeout-seconds", type=positive_float, default=5.0)
    parser.add_argument(
        "--snapshot-directory", type=Path, default=Path("data/snapshots")
    )
    parser.add_argument(
        "--event-file",
        type=Path,
        default=Path("data/events/events.csv"),
        help="append crossing-event metadata to this CSV file",
    )
    parser.add_argument(
        "--snapshot-after",
        type=positive_float,
        help="save one annotated snapshot this many seconds after first inference",
    )
    parser.add_argument(
        "--snapshot-on-detection",
        action="store_true",
        help="save the first annotated frame containing a requested object",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="capture one frame, benchmark devices, and exit",
    )
    parser.add_argument(
        "--benchmark-devices",
        type=benchmark_devices_value,
        default=("cpu", "mps"),
    )
    parser.add_argument("--benchmark-warmup", type=nonnegative_int, default=2)
    parser.add_argument("--benchmark-runs", type=positive_int, default=10)
    parser.add_argument("--first-frame-timeout", type=positive_float, default=15.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


class InferenceRate:
    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = window_seconds
        self._completed_at: deque[float] = deque()

    def tick(self, completed_at: float) -> None:
        self._completed_at.append(completed_at)
        cutoff = completed_at - self.window_seconds
        while self._completed_at and self._completed_at[0] < cutoff:
            self._completed_at.popleft()

    @property
    def fps(self) -> float:
        if len(self._completed_at) < 2:
            return 0.0
        duration = self._completed_at[-1] - self._completed_at[0]
        return (len(self._completed_at) - 1) / duration if duration > 0 else 0.0


def format_detection_metrics(
    *,
    capture_fps: float,
    inference_fps: float,
    inference_ms: float,
    frame_age_ms: float,
    received: int,
    processed: int,
    overwritten: int,
    device: str,
    counts: dict[str, int],
    crossings: dict[str, int] | None = None,
) -> str:
    counts_text = (
        ",".join(f"{name}:{count}" for name, count in sorted(counts.items())) or "none"
    )
    crossings_text = (
        ",".join(f"{name}:{count}" for name, count in sorted((crossings or {}).items()))
        or "none"
    )
    return (
        f"device={device} capture_fps={capture_fps:.1f} "
        f"inference_fps={inference_fps:.1f} inference={inference_ms:.1f}ms "
        f"frame_age={frame_age_ms:.1f}ms received={received} "
        f"processed={processed} overwritten={overwritten} objects={counts_text} "
        f"crossings={crossings_text}"
    )


def wait_for_frame(
    store: LatestFrameStore,
    *,
    timeout_seconds: float,
) -> Frame:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        packet = store.consume_latest()
        if packet is not None:
            return packet.frame
        time.sleep(0.01)
    raise TimeoutError(f"no camera frame received within {timeout_seconds:g} seconds")


def run_benchmark(args: argparse.Namespace) -> int:
    store = LatestFrameStore()
    camera = CameraCapture(
        CaptureConfig(
            url=args.url,
            reconnect_seconds=args.reconnect_seconds,
            open_timeout_seconds=args.open_timeout_seconds,
            read_timeout_seconds=args.read_timeout_seconds,
        ),
        store,
    )
    camera.start()
    try:
        frame = wait_for_frame(store, timeout_seconds=args.first_frame_timeout)
    finally:
        camera.stop()

    supported = set(available_devices())
    print(
        f"Benchmark frame: {frame.shape[1]}x{frame.shape[0]}, "
        f"model={args.model}, image_size={args.image_size}, confidence={args.confidence:g}",
        flush=True,
    )
    for device in args.benchmark_devices:
        if device not in supported:
            print(f"{device}: unavailable (skipped)", flush=True)
            continue
        detector = YoloDetector(
            args.model,
            confidence=args.confidence,
            image_size=args.image_size,
            device=device,
            class_names=args.classes,
        )
        result = benchmark_detector(
            detector,
            frame,
            warmup_runs=args.benchmark_warmup,
            measured_runs=args.benchmark_runs,
        )
        print(
            f"{result.device}: runs={result.runs} mean={result.mean_ms:.1f}ms "
            f"median={result.median_ms:.1f}ms min={result.minimum_ms:.1f}ms "
            f"max={result.maximum_ms:.1f}ms effective_fps={result.fps:.1f}",
            flush=True,
        )
    return 0


def run_live(args: argparse.Namespace) -> int:
    logger = logging.getLogger("roundabout_ai.detection")
    print(
        f"Loading model={args.model} device={args.device} image_size={args.image_size} "
        f"confidence={args.confidence:g} classes={','.join(args.classes)}",
        flush=True,
    )
    detector = YoloDetector(
        args.model,
        confidence=args.confidence,
        image_size=args.image_size,
        device=args.device,
        class_names=args.classes,
    )
    configured_scene = load_scene(args.scene_config) if args.scene_config else None
    event_store = CsvEventStore(args.event_file)
    counter: CrossingCounter | None = None
    counter_size: tuple[int, int] | None = None
    print(f"Model ready on {detector.device}. Camera URL: {args.url}", flush=True)
    if configured_scene:
        print(
            f"Scene ready: roi_points={len(configured_scene.roi)} "
            f"count_lines={len(configured_scene.count_lines)}",
            flush=True,
        )
    else:
        print(
            "No scene configuration: tracking enabled, ROI/counting disabled.",
            flush=True,
        )

    store = LatestFrameStore()
    camera = CameraCapture(
        CaptureConfig(
            url=args.url,
            reconnect_seconds=args.reconnect_seconds,
            open_timeout_seconds=args.open_timeout_seconds,
            read_timeout_seconds=args.read_timeout_seconds,
        ),
        store,
        logger=logger,
    )
    camera.start()
    rate = InferenceRate()
    processed = 0
    latest_annotated: Frame | None = None
    latest_metrics = "waiting for first frame"
    first_inference_at: float | None = None
    snapshot_saved = False
    started_at = time.monotonic()
    next_metrics_at = started_at
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    try:
        while not stopping:
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break
            packet = store.consume_latest()
            if packet is None:
                if now >= next_metrics_at:
                    stats = store.stats(now=now)
                    since_frame = (
                        "n/a"
                        if stats.seconds_since_frame is None
                        else f"{stats.seconds_since_frame:.1f}s"
                    )
                    print(
                        f"waiting status={stats.status} received={stats.frames_received} "
                        f"read_failures={stats.read_failures} reconnects={stats.reconnects} "
                        f"since_frame={since_frame}",
                        flush=True,
                    )
                    next_metrics_at = now + args.metrics_interval
                if not args.headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
                time.sleep(0.005)
                continue

            batch = detector.track(packet.frame, tracker_config=args.tracker_config)
            completed_at = time.monotonic()
            rate.tick(completed_at)
            processed += 1
            if first_inference_at is None:
                first_inference_at = completed_at
            stats = store.stats(now=completed_at)
            scene: Scene | None = None
            detections = batch.detections
            if configured_scene:
                scene = configured_scene.scaled(
                    packet.frame.shape[1], packet.frame.shape[0]
                )
                detections = filter_detections_by_roi(detections, scene.roi)
            frame_size = (packet.frame.shape[1], packet.frame.shape[0])
            if counter is None or counter_size != frame_size:
                counter = CrossingCounter(
                    scene.count_lines if scene else (),
                    minimum_track_age=args.minimum_track_age,
                    maximum_missing_frames=args.maximum_missing_frames,
                )
                counter_size = frame_size
            crossing_events = counter.update(track_observations(detections))
            event_store.write_all(crossing_events)
            for event in crossing_events:
                print(
                    f"crossing line={event.line_name} direction={event.direction} "
                    f"class={event.label} track_id={event.track_id} "
                    f"confidence={event.confidence:.2f}",
                    flush=True,
                )
            latest_metrics = format_detection_metrics(
                capture_fps=stats.capture_fps,
                inference_fps=rate.fps,
                inference_ms=batch.elapsed_seconds * 1000,
                frame_age_ms=(completed_at - packet.captured_at) * 1000,
                received=stats.frames_received,
                processed=processed,
                overwritten=stats.frames_overwritten,
                device=batch.device,
                counts=dict(Counter(detection.label for detection in detections)),
                crossings=counter.counts,
            )
            scene_frame = annotate_scene(packet.frame, scene) if scene else packet.frame
            latest_annotated = draw_overlay(
                annotate_detections(scene_frame, detections), latest_metrics
            )

            if completed_at >= next_metrics_at:
                print(latest_metrics, flush=True)
                next_metrics_at = completed_at + args.metrics_interval

            timed_snapshot_due = (
                args.snapshot_after is not None
                and first_inference_at is not None
                and completed_at - first_inference_at >= args.snapshot_after
            )
            detection_snapshot_due = args.snapshot_on_detection and bool(detections)
            if not snapshot_saved and (timed_snapshot_due or detection_snapshot_due):
                path = save_snapshot(latest_annotated, args.snapshot_directory)
                snapshot_saved = True
                print(f"Annotated snapshot saved: {path}", flush=True)

            if not args.headless:
                cv2.imshow(WINDOW_NAME, latest_annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = save_snapshot(latest_annotated, args.snapshot_directory)
                    print(f"Annotated snapshot saved: {path}", flush=True)
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
    finally:
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        print(
            f"Stopped after processing {processed} frames. Last metrics: {latest_metrics}",
            flush=True,
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        return run_benchmark(args) if args.benchmark else run_live(args)
    except (TimeoutError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
