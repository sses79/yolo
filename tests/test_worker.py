from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from roundabout_ai.capture import Frame, LatestFrameStore
from roundabout_ai.detector import Detection, DetectionBatch
from roundabout_ai.worker import DetectionWorker, ProcessingConfig


class FakeDetector:
    device = "cpu"

    def __init__(self) -> None:
        self.confidence = 0.35
        self.calls = 0

    def track(
        self, frame: Frame, *, tracker_config: str = "bytetrack.yaml"
    ) -> DetectionBatch:
        assert frame.shape == (4, 6, 3)
        assert tracker_config == "bytetrack.yaml"
        self.calls += 1
        return DetectionBatch(
            (Detection(0, "person", 0.8, (0, 0, 2, 2), track_id=4),),
            0.01,
            self.device,
        )


class FakeCamera:
    def __init__(self, store: LatestFrameStore) -> None:
        self.store = store
        self.stops = 0

    def start(self) -> None:
        self.store.publish(np.zeros((4, 6, 3), dtype=np.uint8))

    def stop(self, *, timeout: float = 7.0) -> None:
        assert timeout > 0
        self.stops += 1


def test_worker_reuses_one_thread_and_publishes_dashboard_frame(
    tmp_path: Path,
) -> None:
    detector = FakeDetector()
    cameras: list[FakeCamera] = []

    def camera_factory(
        _config: ProcessingConfig, store: LatestFrameStore
    ) -> FakeCamera:
        camera = FakeCamera(store)
        cameras.append(camera)
        return camera

    worker = DetectionWorker(
        detector_factory=lambda _config: detector,
        camera_factory=camera_factory,
    )
    config = ProcessingConfig(
        url="http://camera.invalid/video",
        event_file=tmp_path / "events.csv",
    )

    assert worker.start(config)
    assert not worker.start(config)
    deadline = time.monotonic() + 1.0
    while worker.snapshot().frames_processed == 0 and time.monotonic() < deadline:
        time.sleep(0.001)

    snapshot = worker.snapshot()
    assert snapshot.frames_processed == 1
    assert snapshot.object_counts == {"person": 1}
    assert snapshot.person_visible
    assert snapshot.latest_frame is not None
    assert detector.calls == 1
    assert len(cameras) == 1

    assert worker.stop()
    assert not worker.snapshot().running
    assert worker.snapshot().status == "stopped"
