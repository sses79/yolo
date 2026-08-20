from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from roundabout_ai.events import (
    EVENT_FIELDS,
    LEGACY_EVENT_FIELDS,
    PHASE6_EVENT_FIELDS,
    SPEED_EVENT_FIELDS,
    CameraEventEvidence,
    CsvEventStore,
    OcrEventEvidence,
    format_timestamp,
)
from roundabout_ai.geometry import CrossingEvent


def crossing_event(
    *, line_name: str = "north_entry", track_id: int = 7
) -> CrossingEvent:
    return CrossingEvent(line_name, track_id, "car", "entering", 0.87654321)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_csv_event_store_appends_stable_metadata_rows(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "events.csv"
    timestamp = datetime(2026, 8, 15, 18, 30, 1, 234567, tzinfo=UTC)
    store = CsvEventStore(path)

    first_records = store.write_all((crossing_event(),), timestamp=timestamp)
    second_records = store.write_all(
        (crossing_event(line_name="south_exit", track_id=8),),
        timestamp=timestamp,
    )

    assert len(first_records) == 1
    assert first_records[0].track_id == 7
    assert len(second_records) == 1

    assert path.read_text(encoding="utf-8").count(",event_type,") == 1
    rows = read_rows(path)
    assert len(rows) == 2
    assert rows[0]["timestamp"] == "2026-08-15T18:30:01.234Z"
    assert rows[0]["line_name"] == "north_entry"
    assert rows[0]["track_id"] == "7"
    assert rows[0]["detection_confidence"] == "0.876543"
    assert rows[0]["ocr_status"] == "not_run"
    assert rows[0]["plate_detected"] == ""
    assert rows[0]["speed_class"] == "unknown"
    assert rows[0]["camera_condition"] == ""
    assert rows[1]["line_name"] == "south_exit"
    assert rows[1]["track_id"] == "8"


def test_csv_event_store_does_not_create_file_for_empty_batch(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"

    assert CsvEventStore(path).write_all(()) == ()
    assert not path.exists()


def test_csv_event_store_rejects_an_incompatible_existing_file(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.csv"
    path.write_text("wrong,header\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected CSV header"):
        CsvEventStore(path).write_all((crossing_event(),))


def test_csv_event_store_upgrades_legacy_event_file(tmp_path: Path) -> None:
    path = tmp_path / "events.csv"
    legacy_header = PHASE6_EVENT_FIELDS
    path.write_text(
        ",".join(legacy_header)
        + "\n"
        + "2026-08-15T18:30:01.234Z,line_crossing,north,car,entering,6,"
        "0.9,AB12***,0.8,fast,1.2,day,{}\n",
        encoding="utf-8",
    )

    CsvEventStore(path).write_all((crossing_event(),))

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == EVENT_FIELDS
    migrated = dict(zip(EVENT_FIELDS, rows[1], strict=True))
    assert migrated["ocr_status"] == "accepted"
    assert migrated["ocr_plate"] == "AB12***"
    assert migrated["speed_class"] == "fast"
    assert migrated["camera_condition"] == ""


def test_csv_event_store_records_profile_and_upgrades_speed_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.csv"
    speed_header = SPEED_EVENT_FIELDS
    path.write_text(",".join(speed_header) + "\n", encoding="utf-8")

    records = CsvEventStore(path).write_all(
        (crossing_event(),),
        camera_profile="day",
        camera_settings='{"focusmode": "continuous-video"}',
    )

    rows = read_rows(path)
    assert rows[0]["camera_profile"] == "day"
    assert rows[0]["camera_settings"] == '{"focusmode": "continuous-video"}'
    assert records[0].camera_profile == "day"


def test_csv_event_store_records_phase7_ocr_and_camera_evidence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.csv"

    records = CsvEventStore(path).write_all(
        (crossing_event(),),
        camera_profile="dusk",
        camera_settings='{"scenemode": "sports"}',
        camera_evidence=CameraEventEvidence(
            condition="dusk",
            luminance_median=72.0,
            sharpness=81.25,
            directional_blur=0.2,
            noise=1.5,
            underexposed_ratio=0.03,
            overexposed_ratio=0.01,
        ),
        ocr_evidence={
            7: OcrEventEvidence(
                status="uncertain",
                plate_text="AB12CDE",
                confidence=0.74,
                plate_detected=True,
                plate_detector_confidence=0.88,
                plate_width=96,
                plate_height=24,
                plate_sharpness=105.5,
                observation_count=2,
                agreement=1,
                reasons=("insufficient_masked_prefix_agreement",),
            )
        },
    )

    row = read_rows(path)[0]
    assert row["ocr_status"] == "uncertain"
    assert row["ocr_plate"] == ""
    assert row["ocr_confidence"] == "0.740000"
    assert row["plate_detected"] == "true"
    assert row["plate_detector_confidence"] == "0.880000"
    assert row["plate_width"] == "96"
    assert row["plate_sharpness"] == "105.500000"
    assert row["ocr_observation_count"] == "2"
    assert row["ocr_reasons"] == "insufficient_masked_prefix_agreement"
    assert row["camera_condition"] == "dusk"
    assert row["camera_sharpness"] == "81.250000"
    assert records[0].ocr_status == "uncertain"


def test_csv_event_store_masks_ocr_and_migrates_legacy_plate_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.csv"
    path.write_text(
        ",".join(LEGACY_EVENT_FIELDS)
        + "\n"
        + "2026-08-15T18:30:01.234Z,line_crossing,north,car,entering,6,"
        "0.9,XY99XYZ,0.8,fast,1.2,,\n",
        encoding="utf-8",
    )

    records = CsvEventStore(path).write_all(
        (crossing_event(),), ocr_results={7: ("AB12CDE", 0.9123456)}
    )

    rows = read_rows(path)
    assert rows[0]["ocr_plate"] == "XY99***"
    assert rows[0]["ocr_confidence"] == "0.8"
    assert rows[1]["ocr_plate"] == "AB12***"
    assert rows[1]["ocr_confidence"] == "0.912346"
    assert records[0].ocr_plate == "AB12***"


def test_timestamp_requires_timezone_and_schema_has_unique_fields() -> None:
    with pytest.raises(ValueError, match="timezone"):
        format_timestamp(datetime(2026, 8, 15))

    assert len(EVENT_FIELDS) == len(set(EVENT_FIELDS))
