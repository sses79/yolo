"""Long-lived capture and inference worker for the Streamlit dashboard."""

from __future__ import annotations

import atexit
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread, current_thread
from typing import Protocol

from roundabout_ai.capture import (
    CameraCapture,
    CaptureConfig,
    ConsumptionRate,
    Frame,
    LatestFrameStore,
)
from roundabout_ai.detector import (
    ROAD_USER_CLASSES,
    DetectionBatch,
    YoloDetector,
    annotate_detections,
)
from roundabout_ai.diagnostic import draw_overlay, save_snapshot
from roundabout_ai.events import CsvEventStore
from roundabout_ai.geometry import CrossingCounter
from roundabout_ai.scene import (
    Scene,
    annotate_scene,
    filter_detections_by_roi,
    load_scene,
    track_observations,
)
from roundabout_ai.shared_state import DashboardSnapshot, DashboardState


@dataclass(frozen=True, slots=True)
class ProcessingConfig:
    url: str
    model: str = "yolo26n.pt"
    device: str = "cpu"
    confidence: float = 0.35
    image_size: int = 640
    classes: tuple[str, ...] = ROAD_USER_CLASSES
    scene_config: Path | None = None
    tracker_config: str = "bytetrack.yaml"
    minimum_track_age: int = 3
    maximum_missing_frames: int = 30
    reconnect_seconds: float = 2.0
    open_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    event_file: Path = Path("data/events/events.csv")
    save_event_images: bool = False
    event_image_directory: Path = Path("data/events/images")

    def __post_init__(self) -> None:
        if not self.url.strip():
            raise ValueError("camera URL must not be empty")
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.image_size <= 0:
            raise ValueError("image size must be positive")
        if self.minimum_track_age <= 0:
            raise ValueError("minimum track age must be positive")
        if self.maximum_missing_frames < 0:
            raise ValueError("maximum missing frames must be nonnegative")


@dataclass(frozen=True, slots=True)
class OverlayControls:
    confidence: float = 0.35
    show_detections: bool = True
    show_scene: bool = True
    show_metrics: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")


class DetectorLike(Protocol):
    confidence: float
    device: str

    def track(
        self, frame: Frame, *, tracker_config: str = "bytetrack.yaml"
    ) -> DetectionBatch: ...


class CameraLike(Protocol):
    def start(self) -> None: ...

    def stop(self, *, timeout: float = 7.0) -> None: ...


DetectorFactory = Callable[[ProcessingConfig], DetectorLike]
CameraFactory = Callable[[ProcessingConfig, LatestFrameStore], CameraLike]
EventStoreFactory = Callable[[Path], CsvEventStore]


def _create_detector(config: ProcessingConfig) -> DetectorLike:
    return YoloDetector(
        config.model,
        confidence=config.confidence,
        image_size=config.image_size,
        device=config.device,
        class_names=config.classes,
    )


def _create_camera(config: ProcessingConfig, store: LatestFrameStore) -> CameraLike:
    return CameraCapture(
        CaptureConfig(
            url=config.url,
            reconnect_seconds=config.reconnect_seconds,
            open_timeout_seconds=config.open_timeout_seconds,
            read_timeout_seconds=config.read_timeout_seconds,
        ),
        store,
        logger=logging.getLogger("roundabout_ai.dashboard.capture"),
    )


class DetectionWorker:
    """Own exactly one model, camera reader, and inference loop while running."""

    def __init__(
        self,
        *,
        detector_factory: DetectorFactory = _create_detector,
        camera_factory: CameraFactory = _create_camera,
        event_store_factory: EventStoreFactory = CsvEventStore,
        state: DashboardState | None = None,
    ) -> None:
        self.state = state or DashboardState()
        self._detector_factory = detector_factory
        self._camera_factory = camera_factory
        self._event_store_factory = event_store_factory
        self._logger = logging.getLogger("roundabout_ai.dashboard.worker")
        self._stop = Event()
        self._lifecycle_lock = Lock()
        self._controls_lock = Lock()
        self._controls = OverlayControls()
        self._thread: Thread | None = None
        self._camera: CameraLike | None = None
        atexit.register(self.stop)

    @property
    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> DashboardSnapshot:
        return self.state.snapshot()

    def set_controls(self, controls: OverlayControls) -> None:
        with self._controls_lock:
            self._controls = controls

    def _current_controls(self) -> OverlayControls:
        with self._controls_lock:
            return self._controls

    def start(self, config: ProcessingConfig) -> bool:
        """Start processing once; return False if a worker is already alive."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            with self._controls_lock:
                self._controls = OverlayControls(
                    confidence=config.confidence,
                    show_detections=self._controls.show_detections,
                    show_scene=self._controls.show_scene,
                    show_metrics=self._controls.show_metrics,
                )
            self.state.begin()
            self._thread = Thread(
                target=self._run,
                args=(config,),
                name="roundabout-inference",
                daemon=True,
            )
            self._thread.start()
        return True

    def stop(self, *, timeout: float = 10.0) -> bool:
        """Request a clean stop; return whether a running worker was found."""

        with self._lifecycle_lock:
            thread = self._thread
            camera = self._camera
            if thread is None or not thread.is_alive():
                return False
            self._stop.set()
            self.state.set_status("stopping")
        if camera is not None:
            camera.stop(timeout=min(timeout, 7.0))
        if thread is not current_thread():
            thread.join(timeout=timeout)
        return True

    def _run(self, config: ProcessingConfig) -> None:
        camera: CameraLike | None = None
        error: str | None = None
        try:
            scene = load_scene(config.scene_config) if config.scene_config else None
            self.state.set_status("loading_model", f"Loading {config.model}")
            detector = self._detector_factory(config)
            event_store = self._event_store_factory(config.event_file)
            frame_store = LatestFrameStore()
            camera = self._camera_factory(config, frame_store)
            with self._lifecycle_lock:
                self._camera = camera
            self._logger.info(
                "worker_started model=%s device=%s url=%s",
                config.model,
                detector.device,
                config.url,
            )
            camera.start()
            self._process_frames(config, detector, event_store, frame_store, scene)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self._logger.exception("worker_failed error=%s", error)
        finally:
            if camera is not None:
                camera.stop()
            with self._lifecycle_lock:
                self._camera = None
            self.state.finish(error=error)
            self._logger.info("worker_stopped error=%s", error or "none")

    def _process_frames(
        self,
        config: ProcessingConfig,
        detector: DetectorLike,
        event_store: CsvEventStore,
        frame_store: LatestFrameStore,
        configured_scene: Scene | None,
    ) -> None:
        rate = ConsumptionRate()
        processed = 0
        counter: CrossingCounter | None = None
        counter_size: tuple[int, int] | None = None

        while not self._stop.is_set():
            packet = frame_store.consume_latest()
            if packet is None:
                self.state.publish_capture(frame_store.stats())
                self._stop.wait(0.01)
                continue

            controls = self._current_controls()
            detector.confidence = controls.confidence
            batch = detector.track(packet.frame, tracker_config=config.tracker_config)
            completed_at = time.monotonic()
            rate.tick(completed_at)
            processed += 1
            stats = frame_store.stats(now=completed_at)
            width, height = packet.frame.shape[1], packet.frame.shape[0]
            scene = configured_scene.scaled(width, height) if configured_scene else None
            detections = batch.detections
            if scene:
                detections = filter_detections_by_roi(detections, scene.roi)

            frame_size = (width, height)
            if counter is None or counter_size != frame_size:
                counter = CrossingCounter(
                    scene.count_lines if scene else (),
                    minimum_track_age=config.minimum_track_age,
                    maximum_missing_frames=config.maximum_missing_frames,
                )
                counter_size = frame_size
            crossing_events = counter.update(track_observations(detections))
            event_records = event_store.write_all(crossing_events)

            annotated = packet.frame
            if controls.show_scene and scene:
                annotated = annotate_scene(annotated, scene)
            if controls.show_detections:
                annotated = annotate_detections(annotated, detections)
            if controls.show_metrics:
                annotated = draw_overlay(
                    annotated,
                    (
                        f"{batch.device} capture={stats.capture_fps:.1f}fps "
                        f"inference={rate.fps:.1f}fps/{batch.elapsed_seconds * 1000:.1f}ms "
                        f"processed={processed} crossings={sum(counter.counts.values())}"
                    ),
                )
            if event_records and config.save_event_images:
                path = save_snapshot(annotated, config.event_image_directory)
                self._logger.info("event_snapshot_saved path=%s", path)

            for record in event_records:
                self._logger.info(
                    "crossing_event line=%s direction=%s class=%s track_id=%d confidence=%.3f",
                    record.line_name,
                    record.direction,
                    record.object_class,
                    record.track_id,
                    record.detection_confidence,
                )
            self.state.publish_frame(
                annotated,
                capture_stats=stats,
                inference_fps=rate.fps,
                inference_ms=batch.elapsed_seconds * 1000,
                frame_age_ms=(completed_at - packet.captured_at) * 1000,
                frames_processed=processed,
                object_counts=Counter(detection.label for detection in detections),
                crossing_counts=counter.counts,
                events=event_records,
            )
