"""Long-lived capture and inference worker for the Streamlit dashboard."""

from __future__ import annotations

import atexit
import json
import logging
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Event, Lock, Thread, current_thread
from typing import Protocol

from roundabout_ai.anpr import (
    ConsensusPolicy,
    PlateQualityPolicy,
    PlateStatus,
    build_live_consensus,
)
from roundabout_ai.anpr_pipeline import (
    PlateDetectorLike,
    PlateRecognizerLike,
    analyze_images,
)
from roundabout_ai.camera_control import (
    ALLOWED_SETTINGS,
    CAMERA_PRESETS,
    CameraCapabilities,
    CameraControlError,
    IpWebcamControlClient,
    identify_camera_profile,
    load_validated_profile_mapping,
)
from roundabout_ai.camera_quality import (
    CameraProfileAdvisor,
    FrameQuality,
    classify_condition,
    guard_profile_transition,
    measure_frame_quality,
)
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
from roundabout_ai.diagnostic import draw_overlay
from roundabout_ai.event_images import (
    VEHICLE_CLASSES,
    VehicleCandidate,
    VehicleCandidateBuffer,
    event_preview_data_url,
    save_event_candidates,
    save_event_snapshot,
)
from roundabout_ai.events import CameraEventEvidence, CsvEventStore, OcrEventEvidence
from roundabout_ai.geometry import CrossingCounter
from roundabout_ai.ocr import RapidOcrRecognizer
from roundabout_ai.scene import (
    Scene,
    annotate_scene,
    filter_detections_by_roi,
    load_scene,
    track_observations,
)
from roundabout_ai.shared_state import DashboardSnapshot, DashboardState
from roundabout_ai.speed import TrackSpeedEstimator


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
    fast_speed_threshold: float = 1.0
    minimum_speed_observations: int = 3
    minimum_speed_duration_seconds: float = 0.5
    reconnect_seconds: float = 2.0
    open_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    event_file: Path = Path("data/events/events.csv")
    save_event_images: bool = False
    event_image_directory: Path = Path("data/events/images")
    event_crop_horizontal_padding: float = 0.35
    event_crop_vertical_padding: float = 0.10
    event_crop_minimum_width: int = 200
    event_crop_minimum_height: int = 100
    camera_adaptation_mode: str = "off"
    camera_control_url: str | None = None
    camera_control_timeout_seconds: float = 2.0
    camera_quality_interval_seconds: float = 5.0
    camera_minimum_dwell_seconds: float = 300.0
    camera_switch_cooldown_seconds: float = 60.0
    camera_automatic_confirmed: bool = False
    camera_capabilities_file: Path = Path("data/camera/capabilities.json")
    camera_validated_profiles_file: Path | None = None
    camera_minimum_profile_samples: int = 30
    live_anpr: bool = False
    anpr_plate_model: Path = Path("models/license-plate.pt")
    anpr_plate_class: str = "license_plate"
    anpr_detector_confidence: float = 0.35
    anpr_image_size: int = 1280
    anpr_minimum_ocr_confidence: float = 0.5
    anpr_minimum_agreement: int = 2
    anpr_maximum_ocr_candidates: int = 3

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
        if self.fast_speed_threshold <= 0:
            raise ValueError("fast speed threshold must be positive")
        if self.minimum_speed_observations < 2:
            raise ValueError("minimum speed observations must be at least 2")
        if self.minimum_speed_duration_seconds <= 0:
            raise ValueError("minimum speed duration must be positive")
        if (
            self.event_crop_horizontal_padding < 0
            or self.event_crop_vertical_padding < 0
        ):
            raise ValueError("event crop padding must be nonnegative")
        if self.event_crop_minimum_width <= 0 or self.event_crop_minimum_height <= 0:
            raise ValueError("minimum event crop dimensions must be positive")
        if self.camera_adaptation_mode not in {"off", "recommend", "automatic"}:
            raise ValueError(
                "camera adaptation mode must be off, recommend, or automatic"
            )
        if self.camera_adaptation_mode != "off" and not (
            self.camera_control_url and self.camera_control_url.strip()
        ):
            raise ValueError("camera control URL is required for camera adaptation")
        if self.camera_adaptation_mode == "automatic" and not (
            self.camera_automatic_confirmed
        ):
            raise ValueError(
                "automatic camera adaptation requires explicit confirmation"
            )
        if self.camera_control_timeout_seconds <= 0:
            raise ValueError("camera control timeout must be positive")
        if self.camera_quality_interval_seconds <= 0:
            raise ValueError("camera quality interval must be positive")
        if self.camera_minimum_dwell_seconds < 0:
            raise ValueError("camera minimum dwell must be nonnegative")
        if self.camera_switch_cooldown_seconds < 0:
            raise ValueError("camera switch cooldown must be nonnegative")
        if self.camera_minimum_profile_samples <= 0:
            raise ValueError("minimum profile samples must be positive")
        if not self.anpr_plate_class.strip():
            raise ValueError("ANPR plate class must not be empty")
        if not 0 <= self.anpr_detector_confidence <= 1:
            raise ValueError("ANPR detector confidence must be between 0 and 1")
        if self.anpr_image_size <= 0:
            raise ValueError("ANPR image size must be positive")
        if not 0 <= self.anpr_minimum_ocr_confidence <= 1:
            raise ValueError("minimum OCR confidence must be between 0 and 1")
        if self.anpr_minimum_agreement <= 0:
            raise ValueError("minimum OCR agreement must be positive")
        if self.anpr_maximum_ocr_candidates <= 0:
            raise ValueError("maximum OCR candidates must be positive")
        if self.live_anpr and not self.anpr_plate_model.is_file():
            raise ValueError(
                f"ANPR plate model does not exist: {self.anpr_plate_model}"
            )


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
ControlClientFactory = Callable[[ProcessingConfig], IpWebcamControlClient]
PlateDetectorFactory = Callable[[ProcessingConfig], PlateDetectorLike]
PlateRecognizerFactory = Callable[[], PlateRecognizerLike]


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


def _create_control_client(config: ProcessingConfig) -> IpWebcamControlClient:
    assert config.camera_control_url is not None
    return IpWebcamControlClient(
        config.camera_control_url,
        timeout_seconds=config.camera_control_timeout_seconds,
    )


def _create_plate_detector(config: ProcessingConfig) -> PlateDetectorLike:
    return YoloDetector(
        str(config.anpr_plate_model),
        confidence=config.anpr_detector_confidence,
        image_size=config.anpr_image_size,
        device=config.device,
        class_names=(config.anpr_plate_class,),
    )


class DetectionWorker:
    """Own exactly one model, camera reader, and inference loop while running."""

    def __init__(
        self,
        *,
        detector_factory: DetectorFactory = _create_detector,
        camera_factory: CameraFactory = _create_camera,
        event_store_factory: EventStoreFactory = CsvEventStore,
        control_client_factory: ControlClientFactory = _create_control_client,
        plate_detector_factory: PlateDetectorFactory = _create_plate_detector,
        plate_recognizer_factory: PlateRecognizerFactory = RapidOcrRecognizer,
        state: DashboardState | None = None,
    ) -> None:
        self.state = state or DashboardState()
        self._detector_factory = detector_factory
        self._camera_factory = camera_factory
        self._event_store_factory = event_store_factory
        self._control_client_factory = control_client_factory
        self._plate_detector_factory = plate_detector_factory
        self._plate_recognizer_factory = plate_recognizer_factory
        self._logger = logging.getLogger("roundabout_ai.dashboard.worker")
        self._stop = Event()
        self._lifecycle_lock = Lock()
        self._controls_lock = Lock()
        self._controls = OverlayControls()
        self._thread: Thread | None = None
        self._camera: CameraLike | None = None
        self._camera_commands: SimpleQueue[str] = SimpleQueue()
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

    def request_camera_profile(self, profile: str) -> None:
        if profile not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera profile: {profile}")
        self._camera_commands.put(profile)

    def request_camera_rollback(self) -> None:
        self._camera_commands.put("rollback")

    def _current_controls(self) -> OverlayControls:
        with self._controls_lock:
            return self._controls

    def start(self, config: ProcessingConfig) -> bool:
        """Start processing once; return False if a worker is already alive."""

        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop.clear()
            while True:
                try:
                    self._camera_commands.get_nowait()
                except Empty:
                    break
            with self._controls_lock:
                self._controls = OverlayControls(
                    confidence=config.confidence,
                    show_detections=self._controls.show_detections,
                    show_scene=self._controls.show_scene,
                    show_metrics=self._controls.show_metrics,
                )
            self.state.begin()
            self.state.configure_camera_adaptation(config.camera_adaptation_mode)
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
            plate_detector: PlateDetectorLike | None = None
            plate_recognizer: PlateRecognizerLike | None = None
            if config.live_anpr:
                self.state.set_status(
                    "loading_model", f"Loading live ANPR {config.anpr_plate_model}"
                )
                plate_detector = self._plate_detector_factory(config)
                plate_recognizer = self._plate_recognizer_factory()
            event_store = self._event_store_factory(config.event_file)
            control_client: IpWebcamControlClient | None = None
            capabilities: CameraCapabilities | None = None
            profile_mapping = {condition: condition for condition in CAMERA_PRESETS}
            validated_mapping = False
            if (
                config.camera_adaptation_mode == "automatic"
                and config.camera_validated_profiles_file is not None
            ):
                profile_mapping = load_validated_profile_mapping(
                    config.camera_validated_profiles_file,
                    minimum_samples=config.camera_minimum_profile_samples,
                )
                validated_mapping = True
            if config.camera_adaptation_mode != "off":
                control_client = self._control_client_factory(config)
                try:
                    capabilities = control_client.save_capabilities(
                        config.camera_capabilities_file
                    )
                    self.state.publish_camera_adaptation(
                        status="ready; recommendation only"
                        if config.camera_adaptation_mode == "recommend"
                        else (
                            "ready; validated OCR profile mapping armed"
                            if validated_mapping
                            else "ready; automatic baseline mapping armed"
                        ),
                        current_settings=capabilities.snapshot(
                            sorted(ALLOWED_SETTINGS)
                        ),
                    )
                except CameraControlError as exc:
                    self.state.publish_camera_adaptation(
                        status=f"capability discovery failed: {exc}"
                    )
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
            self._process_frames(
                config,
                detector,
                event_store,
                frame_store,
                scene,
                control_client,
                capabilities,
                plate_detector,
                plate_recognizer,
                profile_mapping,
            )
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
        control_client: IpWebcamControlClient | None,
        capabilities: CameraCapabilities | None,
        plate_detector: PlateDetectorLike | None,
        plate_recognizer: PlateRecognizerLike | None,
        profile_mapping: dict[str, str],
    ) -> None:
        rate = ConsumptionRate()
        processed = 0
        counter: CrossingCounter | None = None
        counter_size: tuple[int, int] | None = None
        speed_estimator = TrackSpeedEstimator(
            fast_threshold=config.fast_speed_threshold,
            minimum_observations=config.minimum_speed_observations,
            minimum_duration_seconds=config.minimum_speed_duration_seconds,
        )
        candidate_buffer = (
            VehicleCandidateBuffer(
                maximum_missing_frames=config.maximum_missing_frames,
                horizontal_padding_ratio=config.event_crop_horizontal_padding,
                vertical_padding_ratio=config.event_crop_vertical_padding,
                minimum_vehicle_width=config.event_crop_minimum_width,
                minimum_vehicle_height=config.event_crop_minimum_height,
            )
            if config.save_event_images or config.live_anpr
            else None
        )
        advisor = CameraProfileAdvisor()
        latest_quality: FrameQuality | None = None
        last_quality_at = float("-inf")
        last_profile_change_at = float("-inf")
        last_auto_attempt_at = float("-inf")
        current_profile = (
            identify_camera_profile(capabilities.current) if capabilities else None
        )
        if current_profile is not None:
            self.state.publish_camera_adaptation(current_profile=current_profile)
        current_settings = (
            capabilities.snapshot(sorted(ALLOWED_SETTINGS)) if capabilities else {}
        )
        rollback_settings: dict[str, str] = {}
        recommended_profile: str | None = None
        latest_vehicle_sharpness: float | None = None
        latest_vehicle_sharpness_at = float("-inf")
        profile_underexposure: dict[str, tuple[float, float]] = {}

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
            speed_estimates = speed_estimator.update(
                detections, captured_at=packet.captured_at
            )
            if candidate_buffer is not None:
                candidate_buffer.observe(packet.frame, detections)
            if scene:
                detections = filter_detections_by_roi(detections, scene.roi)

            if config.camera_adaptation_mode != "off" and (
                completed_at - last_quality_at >= config.camera_quality_interval_seconds
            ):
                quality = measure_frame_quality(
                    packet.frame, scene.roi if scene else ()
                )
                latest_quality = quality
                if current_profile is not None:
                    profile_underexposure[current_profile] = (
                        quality.underexposed_ratio,
                        completed_at,
                    )
                condition = advisor.observe(quality)
                mapped_profile = (
                    None if condition is None else profile_mapping[condition]
                )
                recent_vehicle_sharpness = (
                    latest_vehicle_sharpness
                    if completed_at - latest_vehicle_sharpness_at <= 900.0
                    else None
                )
                recommended_profile = guard_profile_transition(
                    current_profile,
                    mapped_profile,
                    quality,
                    moving_vehicle_sharpness=recent_vehicle_sharpness,
                    known_profile_underexposure={
                        profile: ratio
                        for profile, (
                            ratio,
                            observed_at,
                        ) in profile_underexposure.items()
                        if completed_at - observed_at <= 900.0
                    },
                )
                self.state.publish_camera_adaptation(
                    quality=quality.as_dict(),
                    recommended_profile=recommended_profile,
                )
                last_quality_at = completed_at

            requested_profile: str | None = None
            try:
                requested_profile = self._camera_commands.get_nowait()
            except Empty:
                pass
            no_vehicle_present = not any(
                detection.label in ROAD_USER_CLASSES for detection in detections
            )
            automatic_due = (
                config.camera_adaptation_mode == "automatic"
                and recommended_profile is not None
                and recommended_profile != current_profile
                and completed_at - last_profile_change_at
                >= config.camera_minimum_dwell_seconds
                and completed_at - last_auto_attempt_at
                >= config.camera_switch_cooldown_seconds
            )
            if requested_profile is not None and not no_vehicle_present:
                self._camera_commands.put(requested_profile)
                self.state.publish_camera_adaptation(
                    status="camera change waiting for an empty ROI"
                )
            elif (
                requested_profile is not None or automatic_due
            ) and no_vehicle_present:
                target = requested_profile or recommended_profile
                if automatic_due:
                    last_auto_attempt_at = completed_at
                if control_client is None or capabilities is None:
                    self.state.publish_camera_adaptation(
                        status="camera control unavailable; recommendation retained"
                    )
                else:
                    try:
                        if target == "rollback":
                            if not rollback_settings:
                                raise CameraControlError(
                                    "no previous settings to restore"
                                )
                            result = control_client.rollback(rollback_settings)
                            current_profile = None
                        else:
                            assert target is not None
                            result = control_client.apply_preset(
                                CAMERA_PRESETS[target], capabilities=capabilities
                            )
                            current_profile = result.profile
                        rollback_settings = dict(result.previous)
                        capabilities = control_client.capabilities()
                        current_settings = capabilities.snapshot(
                            sorted(ALLOWED_SETTINGS)
                        )
                        last_profile_change_at = completed_at
                        self.state.publish_camera_adaptation(
                            current_profile=current_profile or "baseline",
                            status=f"applied and verified: {target}",
                            current_settings=current_settings,
                            last_changed_at=datetime.now(UTC).isoformat(
                                timespec="seconds"
                            ),
                        )
                        self._logger.info(
                            "camera_profile_applied profile=%s settings=%s",
                            target,
                            json.dumps(result.applied, sort_keys=True),
                        )
                    except CameraControlError as exc:
                        self.state.publish_camera_adaptation(
                            status=f"camera change failed; rolled back: {exc}"
                        )
                        self._logger.warning("camera_profile_failed error=%s", exc)

            frame_size = (width, height)
            if counter is None or counter_size != frame_size:
                counter = CrossingCounter(
                    scene.count_lines if scene else (),
                    minimum_track_age=config.minimum_track_age,
                    maximum_missing_frames=config.maximum_missing_frames,
                )
                counter_size = frame_size
            crossing_events = counter.update(
                track_observations(detections, speed_estimates)
            )
            candidates_by_track: dict[
                int, tuple[tuple[str, VehicleCandidate], ...]
            ] = {}
            if crossing_events and candidate_buffer is not None:
                for event in crossing_events:
                    if event.label not in VEHICLE_CLASSES:
                        continue
                    candidates = candidate_buffer.select(event.track_id)
                    candidates_by_track[event.track_id] = candidates
                    if candidates:
                        latest_vehicle_sharpness = max(
                            candidate.sharpness for _kind, candidate in candidates
                        )
                        latest_vehicle_sharpness_at = completed_at
            ocr_evidence: dict[int, OcrEventEvidence] = {}
            if (
                crossing_events
                and candidate_buffer is not None
                and plate_detector is not None
                and plate_recognizer is not None
            ):
                for event in crossing_events:
                    if event.label not in VEHICLE_CLASSES:
                        continue
                    candidates = candidates_by_track[event.track_id]
                    analysis = analyze_images(
                        f"live-track-{event.track_id}",
                        tuple(
                            (f"{kind}-track-{event.track_id}", candidate.crop)
                            for kind, candidate in candidates
                        ),
                        plate_detector,
                        plate_recognizer,
                        quality_policy=PlateQualityPolicy(
                            minimum_width=48,
                            minimum_height=12,
                            minimum_sharpness=20.0,
                            minimum_aspect_ratio=1.4,
                            maximum_skew_degrees=25.0,
                            edge_margin=1,
                        ),
                        consensus_policy=ConsensusPolicy(
                            minimum_confidence=config.anpr_minimum_ocr_confidence,
                            minimum_agreement=config.anpr_minimum_agreement,
                        ),
                        maximum_ocr_candidates=config.anpr_maximum_ocr_candidates,
                        consensus_builder=build_live_consensus,
                    )
                    result = analysis.result
                    rejection_reasons = tuple(
                        f"rejected:{reason}:{count}"
                        for reason, count in sorted(analysis.rejection_reasons.items())
                    )
                    ocr_evidence[event.track_id] = OcrEventEvidence(
                        status=result.status.value,
                        plate_text=result.plate_text
                        if result.status is PlateStatus.ACCEPTED
                        else "",
                        confidence=result.confidence,
                        plate_detected=analysis.detections > 0,
                        plate_detector_confidence=analysis.best_detector_confidence,
                        plate_width=analysis.best_plate_width,
                        plate_height=analysis.best_plate_height,
                        plate_sharpness=analysis.best_plate_sharpness,
                        observation_count=result.observation_count,
                        agreement=result.agreement,
                        reasons=(*result.reasons, *rejection_reasons),
                    )
                    self._logger.info(
                        "live_anpr track_id=%d status=%s confidence=%s "
                        "observations=%d reasons=%s",
                        event.track_id,
                        result.status.value,
                        "none"
                        if result.confidence is None
                        else f"{result.confidence:.3f}",
                        result.observation_count,
                        ",".join(result.reasons) or "none",
                    )
            event_records = event_store.write_all(
                crossing_events,
                camera_profile=current_profile or "",
                camera_settings=json.dumps(current_settings, sort_keys=True)
                if current_settings
                else "",
                camera_evidence=CameraEventEvidence(
                    condition=""
                    if latest_quality is None
                    else classify_condition(latest_quality),
                    luminance_median=None
                    if latest_quality is None
                    else latest_quality.luminance_median,
                    sharpness=None
                    if latest_quality is None
                    else latest_quality.sharpness,
                    directional_blur=None
                    if latest_quality is None
                    else latest_quality.directional_blur,
                    noise=None if latest_quality is None else latest_quality.noise,
                    underexposed_ratio=None
                    if latest_quality is None
                    else latest_quality.underexposed_ratio,
                    overexposed_ratio=None
                    if latest_quality is None
                    else latest_quality.overexposed_ratio,
                ),
                ocr_evidence=ocr_evidence,
            )

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
            dashboard_records = list(event_records)
            if config.save_event_images and candidate_buffer is not None:
                for record_index, record in enumerate(event_records):
                    snapshot_path = save_event_snapshot(
                        annotated,
                        config.event_image_directory,
                        record,
                    )
                    self._logger.info(
                        "event_snapshot_saved path=%s track_id=%d",
                        snapshot_path,
                        record.track_id,
                    )
                    candidates = candidates_by_track.get(
                        record.track_id, candidate_buffer.select(record.track_id)
                    )
                    preview_frame = next(
                        (
                            candidate.crop
                            for kind, candidate in candidates
                            if kind == "crossing"
                        ),
                        annotated,
                    )
                    dashboard_records[record_index] = replace(
                        record,
                        preview_image=event_preview_data_url(preview_frame),
                    )
                    if not candidates:
                        self._logger.info(
                            "event_candidates_skipped track_id=%d reason=no_suitable_crop "
                            "minimum_width=%d minimum_height=%d",
                            record.track_id,
                            config.event_crop_minimum_width,
                            config.event_crop_minimum_height,
                        )
                    paths = save_event_candidates(
                        candidates,
                        config.event_image_directory,
                        record,
                    )
                    for (kind, candidate), path in zip(candidates, paths, strict=True):
                        self._logger.info(
                            "event_candidate_saved path=%s kind=%s track_id=%d "
                            "width=%d height=%d source_width=%d source_height=%d "
                            "sharpness=%.1f clipped=%s edge_clearance=%d",
                            path,
                            kind,
                            record.track_id,
                            candidate.crop.shape[1],
                            candidate.crop.shape[0],
                            candidate.source_width,
                            candidate.source_height,
                            candidate.sharpness,
                            candidate.clipped,
                            candidate.edge_clearance,
                        )

            for record in event_records:
                self._logger.info(
                    "crossing_event line=%s direction=%s class=%s track_id=%d "
                    "confidence=%.3f speed=%s normalized_speed=%s",
                    record.line_name,
                    record.direction,
                    record.object_class,
                    record.track_id,
                    record.detection_confidence,
                    record.speed_class,
                    (
                        "unknown"
                        if record.normalized_speed is None
                        else f"{record.normalized_speed:.3f}"
                    ),
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
                events=tuple(dashboard_records),
            )
