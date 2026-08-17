"""Durable metadata-only storage for roundabout events."""

from __future__ import annotations

import csv
import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from roundabout_ai.geometry import CrossingEvent

EVENT_FIELDS = (
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
        )

    def as_dict(self) -> dict[str, str | int | float | None]:
        return {
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "line_name": self.line_name,
            "object_class": self.object_class,
            "direction": self.direction,
            "track_id": self.track_id,
            "detection_confidence": self.detection_confidence,
            "plate_text": self.plate_text,
            "plate_confidence": self.plate_confidence,
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
            )
            for event in pending
        )

        try:
            with self._lock:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a+", encoding="utf-8", newline="") as handle:
                    handle.seek(0)
                    existing_header = next(csv.reader(handle), None)
                    if (
                        existing_header is not None
                        and tuple(existing_header) != EVENT_FIELDS
                    ):
                        raise ValueError(
                            f"event file has an unexpected CSV header: {self.path}"
                        )
                    handle.seek(0, 2)
                    writer = csv.writer(handle)
                    if existing_header is None:
                        writer.writerow(EVENT_FIELDS)
                    writer.writerows(record.csv_row() for record in records)
        except OSError as exc:
            raise RuntimeError(
                f"could not write event file {self.path}: {exc}"
            ) from exc
        return records
