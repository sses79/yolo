"""Resilient, low-latency capture for an IP Webcam MJPEG stream."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from threading import Event, Lock, Thread
import time
from typing import Callable, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

Frame = NDArray[np.uint8]


class VideoCaptureLike(Protocol):
    def isOpened(self) -> bool: ...

    def open(self, source: str) -> bool: ...

    def read(self) -> tuple[bool, Frame | None]: ...

    def release(self) -> None: ...

    def set(self, prop_id: int, value: float) -> bool: ...


CaptureFactory = Callable[[], VideoCaptureLike]


@dataclass(frozen=True, slots=True)
class FramePacket:
    frame: Frame
    sequence: int
    captured_at: float


@dataclass(frozen=True, slots=True)
class CaptureStats:
    status: str
    width: int
    height: int
    capture_fps: float
    frames_received: int
    frames_consumed: int
    frames_overwritten: int
    read_failures: int
    reconnects: int
    seconds_since_frame: float | None
    last_error: str | None


class LatestFrameStore:
    """A one-frame buffer that makes backpressure visible instead of adding lag."""

    def __init__(self, *, rate_window_seconds: float = 5.0) -> None:
        self._lock = Lock()
        self._packet: FramePacket | None = None
        self._last_consumed_sequence = 0
        self._sequence = 0
        self._status = "starting"
        self._width = 0
        self._height = 0
        self._frames_received = 0
        self._frames_consumed = 0
        self._frames_overwritten = 0
        self._read_failures = 0
        self._reconnects = 0
        self._last_frame_at: float | None = None
        self._last_error: str | None = None
        self._rate_window_seconds = rate_window_seconds
        self._capture_times: deque[float] = deque()

    def publish(self, frame: Frame, *, captured_at: float | None = None) -> None:
        now = time.monotonic() if captured_at is None else captured_at
        height, width = frame.shape[:2]
        with self._lock:
            if (
                self._packet is not None
                and self._packet.sequence > self._last_consumed_sequence
            ):
                self._frames_overwritten += 1
            self._sequence += 1
            self._packet = FramePacket(frame, self._sequence, now)
            self._width = width
            self._height = height
            self._frames_received += 1
            self._last_frame_at = now
            self._status = "live"
            self._last_error = None
            self._capture_times.append(now)
            cutoff = now - self._rate_window_seconds
            while self._capture_times and self._capture_times[0] < cutoff:
                self._capture_times.popleft()

    def consume_latest(self) -> FramePacket | None:
        with self._lock:
            packet = self._packet
            if packet is None or packet.sequence <= self._last_consumed_sequence:
                return None
            self._last_consumed_sequence = packet.sequence
            self._frames_consumed += 1
            return packet

    def set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self._status = status
            self._last_error = error

    def note_read_failure(self, error: str = "frame read failed") -> None:
        with self._lock:
            self._read_failures += 1
            self._last_error = error

    def note_reconnect(self) -> None:
        with self._lock:
            self._reconnects += 1

    def stats(self, *, now: float | None = None) -> CaptureStats:
        observed_at = time.monotonic() if now is None else now
        with self._lock:
            if len(self._capture_times) > 1:
                duration = self._capture_times[-1] - self._capture_times[0]
                fps = (len(self._capture_times) - 1) / duration if duration > 0 else 0.0
            else:
                fps = 0.0
            since_frame = (
                None
                if self._last_frame_at is None
                else max(0.0, observed_at - self._last_frame_at)
            )
            return CaptureStats(
                status=self._status,
                width=self._width,
                height=self._height,
                capture_fps=fps,
                frames_received=self._frames_received,
                frames_consumed=self._frames_consumed,
                frames_overwritten=self._frames_overwritten,
                read_failures=self._read_failures,
                reconnects=self._reconnects,
                seconds_since_frame=since_frame,
                last_error=self._last_error,
            )


@dataclass(frozen=True, slots=True)
class CaptureConfig:
    url: str
    reconnect_seconds: float = 2.0
    open_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 5.0
    failures_before_reconnect: int = 2


class CameraCapture:
    """Read frames continuously, reconnecting whenever the stream disappears."""

    def __init__(
        self,
        config: CaptureConfig,
        store: LatestFrameStore,
        *,
        capture_factory: CaptureFactory | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self.store = store
        self._capture_factory = capture_factory or cv2.VideoCapture
        self._logger = logger or logging.getLogger(__name__)
        self._stop = Event()
        self._thread: Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def stop(self, *, timeout: float = 7.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                self._logger.warning("capture thread did not stop before timeout")
        self.store.set_status("stopped")

    @property
    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _new_capture(self) -> VideoCaptureLike:
        capture = self._capture_factory()
        # These properties are backend-dependent. Unsupported backends simply
        # return False; the reconnect loop remains the fallback.
        capture.set(
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            self.config.open_timeout_seconds * 1000,
        )
        capture.set(
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            self.config.read_timeout_seconds * 1000,
        )
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def _run(self) -> None:
        first_attempt = True
        while not self._stop.is_set():
            self.store.set_status("connecting")
            capture = self._new_capture()
            try:
                if not capture.open(self.config.url) or not capture.isOpened():
                    self.store.set_status("offline", "could not open camera stream")
                    self._logger.warning("could not open camera stream; retrying")
                    if not first_attempt:
                        self.store.note_reconnect()
                    first_attempt = False
                    self._stop.wait(self.config.reconnect_seconds)
                    continue

                if not first_attempt:
                    self.store.note_reconnect()
                first_attempt = False
                self.store.set_status("connected")
                self._logger.info("connected to camera stream")
                consecutive_failures = 0

                while not self._stop.is_set():
                    ok, frame = capture.read()
                    if ok and frame is not None and frame.size > 0:
                        consecutive_failures = 0
                        self.store.publish(frame)
                        continue

                    consecutive_failures += 1
                    self.store.note_read_failure()
                    if consecutive_failures >= self.config.failures_before_reconnect:
                        self.store.set_status("reconnecting", "camera reads failed")
                        self._logger.warning("camera reads failed; reconnecting")
                        break
            except Exception as exc:  # camera backends can raise native wrappers
                message = f"{type(exc).__name__}: {exc}"
                self.store.note_read_failure(message)
                self.store.set_status("reconnecting", message)
                self._logger.exception("capture error; reconnecting")
            finally:
                capture.release()

            if not self._stop.is_set():
                self._stop.wait(self.config.reconnect_seconds)


class ConsumptionRate:
    """Rolling FPS meter for frames actually consumed by the diagnostic."""

    def __init__(self, window_seconds: float = 5.0) -> None:
        self.window_seconds = window_seconds
        self._times: deque[float] = deque()

    def tick(self, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else now
        self._times.append(timestamp)
        cutoff = timestamp - self.window_seconds
        while self._times and self._times[0] < cutoff:
            self._times.popleft()

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        duration = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / duration if duration > 0 else 0.0
