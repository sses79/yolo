"""Thread-safe snapshots shared by the processing worker and dashboard."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from threading import Lock

from roundabout_ai.capture import CaptureStats, Frame
from roundabout_ai.events import EventRecord


@dataclass(frozen=True, slots=True)
class DashboardSnapshot:
    running: bool
    status: str
    status_message: str | None
    latest_frame: Frame | None
    started_at: str | None
    capture_fps: float
    inference_fps: float
    inference_ms: float
    frame_age_ms: float | None
    frames_received: int
    frames_processed: int
    frames_overwritten: int
    read_failures: int
    reconnects: int
    object_counts: Mapping[str, int]
    crossing_counts: Mapping[str, int]
    recent_events: tuple[EventRecord, ...]
    person_visible: bool

    @property
    def total_crossings(self) -> int:
        return sum(self.crossing_counts.values())


class DashboardState:
    """Publish atomic worker snapshots without exposing mutable worker state."""

    def __init__(self, *, recent_event_limit: int = 100) -> None:
        if recent_event_limit <= 0:
            raise ValueError("recent event limit must be positive")
        self._lock = Lock()
        self._recent_events: deque[EventRecord] = deque(maxlen=recent_event_limit)
        self._running = False
        self._status = "stopped"
        self._status_message: str | None = None
        self._latest_frame: Frame | None = None
        self._started_at: str | None = None
        self._capture_fps = 0.0
        self._inference_fps = 0.0
        self._inference_ms = 0.0
        self._frame_age_ms: float | None = None
        self._frames_received = 0
        self._frames_processed = 0
        self._frames_overwritten = 0
        self._read_failures = 0
        self._reconnects = 0
        self._object_counts: dict[str, int] = {}
        self._crossing_counts: dict[str, int] = {}
        self._person_visible = False

    def begin(self) -> None:
        with self._lock:
            self._running = True
            self._status = "loading_model"
            self._status_message = None
            self._latest_frame = None
            self._started_at = datetime.now(UTC).isoformat(timespec="seconds")
            self._capture_fps = 0.0
            self._inference_fps = 0.0
            self._inference_ms = 0.0
            self._frame_age_ms = None
            self._frames_received = 0
            self._frames_processed = 0
            self._frames_overwritten = 0
            self._read_failures = 0
            self._reconnects = 0
            self._object_counts = {}
            self._crossing_counts = {}
            self._recent_events.clear()
            self._person_visible = False

    def set_status(self, status: str, message: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._status_message = message

    def publish_capture(self, stats: CaptureStats) -> None:
        with self._lock:
            self._status = stats.status
            self._status_message = stats.last_error
            self._capture_fps = stats.capture_fps
            self._frames_received = stats.frames_received
            self._frames_overwritten = stats.frames_overwritten
            self._read_failures = stats.read_failures
            self._reconnects = stats.reconnects

    def publish_frame(
        self,
        frame: Frame,
        *,
        capture_stats: CaptureStats,
        inference_fps: float,
        inference_ms: float,
        frame_age_ms: float,
        frames_processed: int,
        object_counts: Mapping[str, int],
        crossing_counts: Mapping[str, int],
        events: tuple[EventRecord, ...],
    ) -> None:
        with self._lock:
            self._status = "live"
            self._status_message = capture_stats.last_error
            self._latest_frame = frame
            self._capture_fps = capture_stats.capture_fps
            self._inference_fps = inference_fps
            self._inference_ms = inference_ms
            self._frame_age_ms = frame_age_ms
            self._frames_received = capture_stats.frames_received
            self._frames_processed = frames_processed
            self._frames_overwritten = capture_stats.frames_overwritten
            self._read_failures = capture_stats.read_failures
            self._reconnects = capture_stats.reconnects
            self._object_counts = dict(object_counts)
            self._crossing_counts = dict(crossing_counts)
            for event in events:
                self._recent_events.appendleft(event)
            self._person_visible = self._object_counts.get("person", 0) > 0

    def finish(self, *, error: str | None = None) -> None:
        with self._lock:
            self._running = False
            self._status = "error" if error else "stopped"
            self._status_message = error
            self._person_visible = False

    def snapshot(self) -> DashboardSnapshot:
        with self._lock:
            return DashboardSnapshot(
                running=self._running,
                status=self._status,
                status_message=self._status_message,
                latest_frame=None
                if self._latest_frame is None
                else self._latest_frame.copy(),
                started_at=self._started_at,
                capture_fps=self._capture_fps,
                inference_fps=self._inference_fps,
                inference_ms=self._inference_ms,
                frame_age_ms=self._frame_age_ms,
                frames_received=self._frames_received,
                frames_processed=self._frames_processed,
                frames_overwritten=self._frames_overwritten,
                read_failures=self._read_failures,
                reconnects=self._reconnects,
                object_counts=dict(self._object_counts),
                crossing_counts=dict(self._crossing_counts),
                recent_events=tuple(self._recent_events),
                person_visible=self._person_visible,
            )
