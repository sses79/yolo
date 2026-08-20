from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import roundabout_ai.dashboard
from roundabout_ai.dashboard import (
    adaptive_camera_performance_rows,
    crossing_chart_rows,
    event_table_rows,
)
from roundabout_ai.events import EventRecord
from roundabout_ai.shared_state import DashboardSnapshot


def snapshot() -> DashboardSnapshot:
    event = EventRecord(
        "2026-08-15T18:30:01.234Z",
        "line_crossing",
        "north",
        "car",
        "entering",
        7,
        0.9,
        ocr_plate="AB12***",
        ocr_confidence=0.91,
        ocr_status="accepted",
        plate_detected=True,
        plate_detector_confidence=0.87,
        plate_sharpness=102.0,
        speed_class="fast",
        normalized_speed=1.25,
        camera_profile="day",
        camera_condition="day",
        preview_image="data:image/jpeg;base64,preview",
    )
    return DashboardSnapshot(
        running=True,
        status="live",
        status_message=None,
        latest_frame=None,
        started_at="2026-08-15T18:00:00+00:00",
        capture_fps=25.0,
        inference_fps=12.0,
        inference_ms=80.0,
        frame_age_ms=95.0,
        frames_received=100,
        frames_processed=40,
        frames_overwritten=59,
        read_failures=0,
        reconnects=0,
        object_counts={"car": 1},
        crossing_counts={
            "north:entering:car": 2,
            "south:entering:car": 1,
            "north:leaving:bus": 1,
        },
        recent_events=(event,),
        person_visible=False,
    )


def test_dashboard_rows_aggregate_crossings_and_serialize_events() -> None:
    current = snapshot()

    assert crossing_chart_rows(current) == [
        {"category": "bus · leaving", "count": 1},
        {"category": "car · entering", "count": 3},
    ]
    assert event_table_rows(current)[0]["track_id"] == 7
    assert event_table_rows(current)[0]["direction"] == "entering"
    assert event_table_rows(current)[0]["speed_class"] == "fast"
    assert event_table_rows(current)[0]["normalized_speed"] == 1.25
    assert event_table_rows(current)[0]["ocr_plate"] == "AB12***"
    assert event_table_rows(current)[0]["ocr_confidence"] == 0.91
    assert event_table_rows(current)[0]["ocr_status"] == "accepted"
    assert (
        event_table_rows(current)[0]["preview_image"]
        == "data:image/jpeg;base64,preview"
    )


def test_dashboard_aggregates_profile_performance_with_coverage() -> None:
    accepted = snapshot().recent_events[0]
    no_read = replace(
        accepted,
        track_id=8,
        ocr_plate="",
        ocr_confidence=None,
        ocr_status="no_read",
        plate_detected=False,
        plate_detector_confidence=None,
        plate_sharpness=None,
    )
    not_run = replace(
        accepted,
        track_id=9,
        ocr_plate="",
        ocr_confidence=None,
        ocr_status="not_run",
        plate_detected=None,
    )
    current = replace(snapshot(), recent_events=(accepted, no_read, not_run))

    rows = adaptive_camera_performance_rows(current)

    assert len(rows) == 1
    assert rows[0]["crossings"] == 3
    assert rows[0]["direction"] == "entering"
    assert rows[0]["ocr_eligible"] == 2
    assert rows[0]["plate_detection_rate"] == 0.5
    assert rows[0]["ocr_accepted"] == 1
    assert rows[0]["ocr_no_read"] == 1
    assert rows[0]["ocr_acceptance_rate"] == 0.5
    assert rows[0]["mean_accepted_ocr_confidence"] == 0.91


def test_streamlit_page_renders_without_starting_camera() -> None:
    module_path = Path(roundabout_ai.dashboard.__file__)
    app = AppTest.from_file(module_path.with_name("dashboard_app.py"))

    app.run(timeout=10)

    assert not app.exception
    assert app.title[0].value == "Roundabout AI"
    assert app.button[0].label == "Start"


def test_start_rerun_enables_stop_button(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeWorker:
        def __init__(self) -> None:
            self.running = False

        def snapshot(self) -> DashboardSnapshot:
            return replace(
                snapshot(),
                running=self.running,
                status="live" if self.running else "stopped",
            )

        def set_controls(self, _controls: object) -> None:
            pass

        def start(self, _config: object) -> bool:
            self.running = True
            return True

        def stop(self) -> bool:
            self.running = False
            return True

    fake_worker = FakeWorker()
    monkeypatch.setattr(roundabout_ai.dashboard, "get_worker", lambda: fake_worker)
    module_path = Path(roundabout_ai.dashboard.__file__)
    app = AppTest.from_file(module_path.with_name("dashboard_app.py"))

    app.run(timeout=10)
    assert app.button[0].disabled is False
    assert app.button[1].disabled is True

    app.button[0].click().run(timeout=10)

    assert app.button[0].disabled is True
    assert app.button[1].disabled is False

    app.button[1].click().run(timeout=10)

    assert app.button[0].disabled is False
    assert app.button[1].disabled is True
