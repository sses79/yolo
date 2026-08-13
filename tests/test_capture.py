from __future__ import annotations

import time

import numpy as np

from roundabout_ai.capture import (
    CameraCapture,
    CaptureConfig,
    ConsumptionRate,
    LatestFrameStore,
)


def frame(value: int = 0) -> np.ndarray:
    return np.full((2, 3, 3), value, dtype=np.uint8)


def test_latest_frame_store_overwrites_unconsumed_frame() -> None:
    store = LatestFrameStore()

    store.publish(frame(1), captured_at=10.0)
    store.publish(frame(2), captured_at=10.1)

    packet = store.consume_latest()
    assert packet is not None
    assert packet.sequence == 2
    assert packet.frame[0, 0, 0] == 2

    stats = store.stats(now=10.2)
    assert stats.frames_received == 2
    assert stats.frames_consumed == 1
    assert stats.frames_overwritten == 1
    assert stats.width == 3
    assert stats.height == 2


def test_store_returns_each_sequence_only_once() -> None:
    store = LatestFrameStore()
    store.publish(frame(), captured_at=5.0)

    assert store.consume_latest() is not None
    assert store.consume_latest() is None
    assert store.stats(now=5.1).frames_consumed == 1


def test_capture_fps_uses_observation_window() -> None:
    store = LatestFrameStore(rate_window_seconds=5.0)
    for timestamp in (1.0, 1.5, 2.0):
        store.publish(frame(), captured_at=timestamp)

    assert store.stats(now=2.0).capture_fps == 2.0


def test_failure_and_reconnect_counters_are_reported() -> None:
    store = LatestFrameStore()
    store.note_read_failure("bad frame")
    store.note_reconnect()
    store.set_status("reconnecting", "bad frame")

    stats = store.stats(now=time.monotonic())
    assert stats.status == "reconnecting"
    assert stats.read_failures == 1
    assert stats.reconnects == 1
    assert stats.last_error == "bad frame"


def test_consumption_rate() -> None:
    meter = ConsumptionRate(window_seconds=5.0)
    meter.tick(1.0)
    meter.tick(1.25)
    meter.tick(1.5)

    assert meter.fps == 4.0


def test_camera_retries_after_open_failure_and_publishes_frame() -> None:
    created = 0

    class FakeCapture:
        def __init__(self, opens: bool) -> None:
            self.opens = opens
            self.opened = False
            self.reads = 0

        def set(self, _prop_id: int, _value: float) -> bool:
            return True

        def open(self, _source: str) -> bool:
            self.opened = self.opens
            return self.opened

        def isOpened(self) -> bool:
            return self.opened

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return True, frame(9)
            return False, None

        def release(self) -> None:
            self.opened = False

    def factory() -> FakeCapture:
        nonlocal created
        created += 1
        return FakeCapture(opens=created > 1)

    store = LatestFrameStore()
    camera = CameraCapture(
        CaptureConfig(
            url="http://camera.invalid/video",
            reconnect_seconds=0.001,
            failures_before_reconnect=1,
        ),
        store,
        capture_factory=factory,
    )

    camera.start()
    deadline = time.monotonic() + 1.0
    while store.stats().frames_received == 0 and time.monotonic() < deadline:
        time.sleep(0.001)
    camera.stop()

    stats = store.stats()
    assert created >= 2
    assert stats.frames_received >= 1
    assert stats.reconnects >= 1
