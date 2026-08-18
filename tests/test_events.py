from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

import pytest

from roundabout_ai.events import EVENT_FIELDS, CsvEventStore, format_timestamp
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
    assert rows == [
        {
            "timestamp": "2026-08-15T18:30:01.234Z",
            "event_type": "line_crossing",
            "line_name": "north_entry",
            "object_class": "car",
            "direction": "entering",
            "track_id": "7",
            "detection_confidence": "0.876543",
            "plate_text": "",
            "plate_confidence": "",
            "speed_class": "unknown",
            "normalized_speed": "",
        },
        {
            "timestamp": "2026-08-15T18:30:01.234Z",
            "event_type": "line_crossing",
            "line_name": "south_exit",
            "object_class": "car",
            "direction": "entering",
            "track_id": "8",
            "detection_confidence": "0.876543",
            "plate_text": "",
            "plate_confidence": "",
            "speed_class": "unknown",
            "normalized_speed": "",
        },
    ]


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
    legacy_header = EVENT_FIELDS[:-2]
    path.write_text(",".join(legacy_header) + "\n", encoding="utf-8")

    CsvEventStore(path).write_all((crossing_event(),))

    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert tuple(rows[0]) == EVENT_FIELDS
    assert rows[1][-2:] == ["unknown", ""]


def test_timestamp_requires_timezone_and_schema_has_unique_fields() -> None:
    with pytest.raises(ValueError, match="timezone"):
        format_timestamp(datetime(2026, 8, 15))

    assert len(EVENT_FIELDS) == len(set(EVENT_FIELDS))
