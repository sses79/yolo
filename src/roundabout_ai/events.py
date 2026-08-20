"""Durable metadata-only storage for roundabout events."""

from __future__ import annotations

import csv
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from roundabout_ai.geometry import CrossingEvent

BASE_EVENT_FIELDS = (
    "timestamp",
    "event_type",
    "line_name",
    "object_class",
    "direction",
    "track_id",
    "detection_confidence",
    "plate_text",
    "plate_confidence",
)
SPEED_EVENT_FIELDS = (
    *BASE_EVENT_FIELDS,
    "speed_class",
    "normalized_speed",
)
EVENT_FIELDS = (
    *SPEED_EVENT_FIELDS,
    "camera_profile",
    "camera_settings",
)


@dataclass(frozen=True, slots=True)
class EventRecord:
    timestamp: str
    event_type: str
    line_name: str
    object_class: str
    direction: str
    track_id: int
    detection_confidence: float
    plate_text: str = ""
    plate_confidence: float | None = None
    speed_class: str = "unknown"
    normalized_speed: float | None = None
    camera_profile: str = ""
    camera_settings: str = ""
    preview_image: str | None = None

    def csv_row(self) -> tuple[str, ...]:
        return (
            self.timestamp,
            self.event_type,
            self.line_name,
            self.object_class,
            self.direction,
            str(self.track_id),
            f"{self.detection_confidence:.6f}",
            self.plate_text,
            "" if self.plate_confidence is None else f"{self.plate_confidence:.6f}",
            self.speed_class,
            "" if self.normalized_speed is None else f"{self.normalized_speed:.6f}",
            self.camera_profile,
            self.camera_settings,
        )

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "timestamp": self.timestamp,
            "preview_image": self.preview_image,
            "event_type": self.event_type,
            "line_name": self.line_name,
            "object_class": self.object_class,
            "direction": self.direction,
            "track_id": self.track_id,
            "detection_confidence": self.detection_confidence,
            "speed_class": self.speed_class,
            "normalized_speed": self.normalized_speed,
            "plate_text": self.plate_text,
            "plate_confidence": self.plate_confidence,
            "camera_profile": self.camera_profile,
            "camera_settings": self.camera_settings,
        }


def utc_now() -> datetime:
    return datetime.now(UTC)


def format_timestamp(value: datetime) -> str:
    """Return an unambiguous UTC timestamp suitable for event interchange."""

    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("event timestamp must include a timezone")
    return (
        value.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


class CsvEventStore:
    """Append crossing-event metadata to a CSV file with one stable header."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self.path = path
        self._clock = clock
        self._lock = threading.Lock()

    def write_all(
        self,
        events: Iterable[CrossingEvent],
        *,
        timestamp: datetime | None = None,
        camera_profile: str = "",
        camera_settings: str = "",
    ) -> tuple[EventRecord, ...]:
        """Write one frame's events and return their timestamped records."""

        pending = tuple(events)
        if not pending:
            return ()
        observed_at = format_timestamp(timestamp or self._clock())
        records = tuple(
            EventRecord(
                observed_at,
                "line_crossing",
                event.line_name,
                event.label,
                event.direction,
                event.track_id,
                event.confidence,
                speed_class=event.speed_class,
                normalized_speed=event.normalized_speed,
                camera_profile=camera_profile,
                camera_settings=camera_settings,
            )
            for event in pending
        )

        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a+", encoding="utf-8", newline="") as handle:
                    handle.seek(0)
                    existing_header = next(csv.reader(handle), None)
                    header = tuple(existing_header) if existing_header else None
                    supported_headers = {
                        None,
                        EVENT_FIELDS,
                        SPEED_EVENT_FIELDS,
                        BASE_EVENT_FIELDS,
                    }
                    if header not in supported_headers:
                        raise ValueError(
                            f"event file has an unexpected CSV header: {self.path}"
                        )
                    handle.seek(0, 2)
                    writer = csv.writer(handle)
                    if existing_header is None:
                        writer.writerow(EVENT_FIELDS)
                    if header in {BASE_EVENT_FIELDS, SPEED_EVENT_FIELDS}:
                        handle.seek(0)
                        legacy_rows = list(csv.reader(handle))[1:]
                        handle.seek(0)
                        handle.truncate()
                        writer.writerow(EVENT_FIELDS)
                        if header == BASE_EVENT_FIELDS:
                            writer.writerows(
                                (*row, "unknown", "", "", "") for row in legacy_rows
                            )
                        else:
                            writer.writerows((*row, "", "") for row in legacy_rows)
                    handle.seek(0, 2)
                    writer.writerows(record.csv_row() for record in records)
        except OSError as exc:
            raise RuntimeError(
                f"could not write event file {self.path}: {exc}"
            ) from exc
        return records
