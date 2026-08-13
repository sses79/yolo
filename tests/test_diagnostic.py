from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from roundabout_ai.capture import CaptureStats
from roundabout_ai.diagnostic import (
    Recorder,
    build_parser,
    format_metrics,
    recording_seconds,
    save_snapshot,
)


def test_recording_duration_is_bounded() -> None:
    assert recording_seconds("300") == 300
    try:
        recording_seconds("301")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("duration over five minutes should be rejected")


def test_default_camera_url() -> None:
    args = build_parser().parse_args([])
    assert args.url == "http://192.168.1.142:8080/video"


def test_metrics_report_stale_and_counters() -> None:
    stats = CaptureStats(
        status="reconnecting",
        width=1920,
        height=1080,
        capture_fps=14.9,
        frames_received=100,
        frames_consumed=80,
        frames_overwritten=20,
        read_failures=2,
        reconnects=1,
        seconds_since_frame=3.0,
        last_error="camera reads failed",
    )

    text = format_metrics(
        stats,
        consumed_fps=12.0,
        frame_age_seconds=0.25,
        stale_after_seconds=2.0,
        recording=False,
    )

    assert "[STALE]" in text
    assert "resolution=1920x1080" in text
    assert "overwritten=20" in text
    assert "age=250ms" in text


def test_snapshot_preserves_full_resolution(tmp_path: Path) -> None:
    source = np.full((12, 20, 3), 127, dtype=np.uint8)
    path = save_snapshot(source, tmp_path)

    assert path.exists()
    import cv2

    saved = cv2.imread(str(path))
    assert saved.shape == source.shape


def test_recorder_samples_frames_at_output_rate(
    tmp_path: Path, monkeypatch
) -> None:
    import roundabout_ai.diagnostic as diagnostic

    class FakeWriter:
        def __init__(self) -> None:
            self.writes = 0
            self.released = False

        def isOpened(self) -> bool:
            return True

        def write(self, _frame: np.ndarray) -> None:
            self.writes += 1

        def release(self) -> None:
            self.released = True

    writer = FakeWriter()
    clock = [100.0]
    monkeypatch.setattr(diagnostic.cv2, "VideoWriter", lambda *args: writer)
    monkeypatch.setattr(diagnostic.cv2, "VideoWriter_fourcc", lambda *args: 0)
    monkeypatch.setattr(diagnostic.time, "monotonic", lambda: clock[0])

    recorder = Recorder(tmp_path, fps=10.0)
    image = np.zeros((10, 20, 3), dtype=np.uint8)
    recorder.start(image, limit_seconds=1.0)

    recorder.write(image)
    clock[0] = 100.05
    recorder.write(image)
    clock[0] = 100.1
    recorder.write(image)

    assert writer.writes == 2
    assert recorder.frames_written == 2
    assert recorder.stop() is not None
    assert recorder.stop() is None
    assert writer.released is True
