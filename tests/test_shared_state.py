from __future__ import annotations

from dataclasses import replace

import numpy as np

from roundabout_ai.capture import CaptureStats
from roundabout_ai.events import EventRecord
from roundabout_ai.shared_state import DashboardState


def capture_stats() -> CaptureStats:
    return CaptureStats(
        status="live",
        width=3,
        height=2,
        capture_fps=25.0,
        frames_received=12,
        frames_consumed=5,
        frames_overwritten=6,
        read_failures=1,
        reconnects=2,
        seconds_since_frame=0.01,
        last_error=None,
    )


def event(track_id: int) -> EventRecord:
    return EventRecord(
        "2026-08-15T18:30:01.234Z",
        "line_crossing",
        "entry",
        "car",
        "entering",
        track_id,
        0.9,
    )


def test_dashboard_state_publishes_atomic_bounded_snapshots() -> None:
    state = DashboardState(recent_event_limit=2)
    state.begin()
    frame = np.zeros((2, 3, 3), dtype=np.uint8)

    state.publish_frame(
        frame,
        capture_stats=capture_stats(),
        inference_fps=10.0,
        inference_ms=100.0,
        frame_age_ms=120.0,
        frames_processed=5,
        object_counts={"person": 1, "car": 2},
        crossing_counts={"entry:entering:car": 3},
        events=(event(1), event(2), event(3)),
    )
    snapshot = state.snapshot()

    assert snapshot.running
    assert snapshot.status == "live"
    assert snapshot.person_visible
    assert snapshot.total_crossings == 3
    assert [item.track_id for item in snapshot.recent_events] == [3, 2]
    assert snapshot.latest_frame is not None
    snapshot.latest_frame[0, 0, 0] = 255
    unchanged_frame = state.snapshot().latest_frame
    assert unchanged_frame is not None
    assert unchanged_frame[0, 0, 0] == 0


def test_dashboard_state_exposes_offline_and_clean_stop_status() -> None:
    state = DashboardState()
    state.begin()
    state.publish_capture(
        replace(
            capture_stats(),
            status="reconnecting",
            last_error="camera reads failed",
        )
    )

    assert state.snapshot().status == "reconnecting"
    assert state.snapshot().status_message == "camera reads failed"
    state.finish()
    assert not state.snapshot().running
    assert state.snapshot().status == "stopped"
