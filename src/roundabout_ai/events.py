"""Durable metadata-only storage for roundabout events."""

from __future__ import annotations

import csv
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from roundabout_ai.anpr import mask_plate_text
from roundabout_ai.geometry import CrossingEvent

LEGACY_BASE_EVENT_FIELDS = (
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
LEGACY_SPEED_EVENT_FIELDS = (
    *LEGACY_BASE_EVENT_FIELDS,
    "speed_class",
    "normalized_speed",
)
LEGACY_EVENT_FIELDS = (
    *LEGACY_SPEED_EVENT_FIELDS,
    "camera_profile",
    "camera_settings",
)
BASE_EVENT_FIELDS = (
    "timestamp",
    "event_type",
    "line_name",
    "object_class",
    "direction",
    "track_id",
    "detection_confidence",
    "ocr_plate",
    "ocr_confidence",
)
SPEED_EVENT_FIELDS = (*BASE_EVENT_FIELDS, "speed_class", "normalized_speed")
PHASE6_EVENT_FIELDS = (*SPEED_EVENT_FIELDS, "camera_profile", "camera_settings")
EVENT_FIELDS = (
    *BASE_EVENT_FIELDS,
    "ocr_status",
    "plate_detected",
    "plate_detector_confidence",
    "plate_width",
    "plate_height",
    "plate_sharpness",
    "ocr_observation_count",
    "ocr_agreement",
    "ocr_reasons",
    "speed_class",
    "normalized_speed",
    "camera_profile",
    "camera_settings",
    "camera_condition",
    "camera_luminance_median",
    "camera_sharpness",
    "camera_directional_blur",
    "camera_noise",
    "camera_underexposed_ratio",
    "camera_overexposed_ratio",
)

OCR_STATUSES = frozenset(("accepted", "uncertain", "no_read", "not_run"))


@dataclass(frozen=True, slots=True)
class OcrEventEvidence:
    status: str = "not_run"
    plate_text: str = ""
    confidence: float | None = None
    plate_detected: bool | None = None
    plate_detector_confidence: float | None = None
    plate_width: int | None = None
    plate_height: int | None = None
    plate_sharpness: float | None = None
    observation_count: int = 0
    agreement: int = 0
    reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.status not in OCR_STATUSES:
            raise ValueError(f"unsupported OCR status: {self.status}")
        if self.observation_count < 0 or self.agreement < 0:
            raise ValueError("OCR observation and agreement counts must be nonnegative")


@dataclass(frozen=True, slots=True)
class CameraEventEvidence:
    condition: str = ""
    luminance_median: float | None = None
    sharpness: float | None = None
    directional_blur: float | None = None
    noise: float | None = None
    underexposed_ratio: float | None = None
    overexposed_ratio: float | None = None


@dataclass(frozen=True, slots=True)
class EventRecord:
    timestamp: str
    event_type: str
    line_name: str
    object_class: str
    direction: str
    track_id: int
    detection_confidence: float
    ocr_plate: str = ""
    ocr_confidence: float | None = None
    ocr_status: str = "not_run"
    plate_detected: bool | None = None
    plate_detector_confidence: float | None = None
    plate_width: int | None = None
    plate_height: int | None = None
    plate_sharpness: float | None = None
    ocr_observation_count: int = 0
    ocr_agreement: int = 0
    ocr_reasons: str = ""
    speed_class: str = "unknown"
    normalized_speed: float | None = None
    camera_profile: str = ""
    camera_settings: str = ""
    camera_condition: str = ""
    camera_luminance_median: float | None = None
    camera_sharpness: float | None = None
    camera_directional_blur: float | None = None
    camera_noise: float | None = None
    camera_underexposed_ratio: float | None = None
    camera_overexposed_ratio: float | None = None
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
            self.ocr_plate,
            "" if self.ocr_confidence is None else f"{self.ocr_confidence:.6f}",
            self.ocr_status,
            "" if self.plate_detected is None else str(self.plate_detected).lower(),
            _number_text(self.plate_detector_confidence),
            "" if self.plate_width is None else str(self.plate_width),
            "" if self.plate_height is None else str(self.plate_height),
            _number_text(self.plate_sharpness),
            str(self.ocr_observation_count),
            str(self.ocr_agreement),
            self.ocr_reasons,
            self.speed_class,
            "" if self.normalized_speed is None else f"{self.normalized_speed:.6f}",
            self.camera_profile,
            self.camera_settings,
            self.camera_condition,
            _number_text(self.camera_luminance_median),
            _number_text(self.camera_sharpness),
            _number_text(self.camera_directional_blur),
            _number_text(self.camera_noise),
            _number_text(self.camera_underexposed_ratio),
            _number_text(self.camera_overexposed_ratio),
        )

    def as_dict(self) -> dict[str, str | int | float | bool | None]:
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
            "ocr_plate": self.ocr_plate,
            "ocr_confidence": self.ocr_confidence,
            "ocr_status": self.ocr_status,
            "plate_detected": self.plate_detected,
            "plate_detector_confidence": self.plate_detector_confidence,
            "plate_width": self.plate_width,
            "plate_height": self.plate_height,
            "plate_sharpness": self.plate_sharpness,
            "ocr_observation_count": self.ocr_observation_count,
            "ocr_agreement": self.ocr_agreement,
            "ocr_reasons": self.ocr_reasons,
            "camera_profile": self.camera_profile,
            "camera_settings": self.camera_settings,
            "camera_condition": self.camera_condition,
            "camera_luminance_median": self.camera_luminance_median,
            "camera_sharpness": self.camera_sharpness,
            "camera_directional_blur": self.camera_directional_blur,
            "camera_noise": self.camera_noise,
            "camera_underexposed_ratio": self.camera_underexposed_ratio,
            "camera_overexposed_ratio": self.camera_overexposed_ratio,
        }


def _number_text(value: float | None) -> str:
    return "" if value is None else f"{value:.6f}"


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
        camera_evidence: CameraEventEvidence = CameraEventEvidence(),
        ocr_evidence: Mapping[int, OcrEventEvidence] | None = None,
        ocr_results: Mapping[int, tuple[str, float | None]] | None = None,
    ) -> tuple[EventRecord, ...]:
        """Write one frame's events and return their timestamped records."""

        pending = tuple(events)
        if not pending:
            return ()
        observed_at = format_timestamp(timestamp or self._clock())
        if ocr_evidence is not None and ocr_results is not None:
            raise ValueError("pass OCR evidence or legacy OCR results, not both")
        evidence_by_track = dict(ocr_evidence or {})
        for track_id, (plate_text, confidence) in (ocr_results or {}).items():
            evidence_by_track[track_id] = OcrEventEvidence(
                status="accepted",
                plate_text=plate_text,
                confidence=confidence,
                plate_detected=True,
            )
        records = tuple(
            self._record(
                event,
                observed_at,
                camera_profile,
                camera_settings,
                camera_evidence,
                evidence_by_track.get(event.track_id, OcrEventEvidence()),
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
                        PHASE6_EVENT_FIELDS,
                        SPEED_EVENT_FIELDS,
                        BASE_EVENT_FIELDS,
                        LEGACY_EVENT_FIELDS,
                        LEGACY_SPEED_EVENT_FIELDS,
                        LEGACY_BASE_EVENT_FIELDS,
                    }
                    if header not in supported_headers:
                        raise ValueError(
                            f"event file has an unexpected CSV header: {self.path}"
                        )
                    handle.seek(0, 2)
                    writer = csv.writer(handle)
                    if existing_header is None:
                        writer.writerow(EVENT_FIELDS)
                    if header is not None and header != EVENT_FIELDS:
                        handle.seek(0)
                        legacy_rows = list(csv.reader(handle))[1:]
                        handle.seek(0)
                        handle.truncate()
                        writer.writerow(EVENT_FIELDS)
                        for row in legacy_rows:
                            values = dict(zip(header, row, strict=True))
                            writer.writerow(_migrate_row(values))
                    handle.seek(0, 2)
                    writer.writerows(record.csv_row() for record in records)
        except OSError as exc:
            raise RuntimeError(
                f"could not write event file {self.path}: {exc}"
            ) from exc
        return records

    @staticmethod
    def _record(
        event: CrossingEvent,
        observed_at: str,
        camera_profile: str,
        camera_settings: str,
        camera: CameraEventEvidence,
        ocr: OcrEventEvidence,
    ) -> EventRecord:
        return EventRecord(
            observed_at,
            "line_crossing",
            event.line_name,
            event.label,
            event.direction,
            event.track_id,
            event.confidence,
            ocr_plate=mask_plate_text(
                ocr.plate_text if ocr.status == "accepted" else ""
            ),
            ocr_confidence=ocr.confidence,
            ocr_status=ocr.status,
            plate_detected=ocr.plate_detected,
            plate_detector_confidence=ocr.plate_detector_confidence,
            plate_width=ocr.plate_width,
            plate_height=ocr.plate_height,
            plate_sharpness=ocr.plate_sharpness,
            ocr_observation_count=ocr.observation_count,
            ocr_agreement=ocr.agreement,
            ocr_reasons=";".join(ocr.reasons),
            speed_class=event.speed_class,
            normalized_speed=event.normalized_speed,
            camera_profile=camera_profile,
            camera_settings=camera_settings,
            camera_condition=camera.condition,
            camera_luminance_median=camera.luminance_median,
            camera_sharpness=camera.sharpness,
            camera_directional_blur=camera.directional_blur,
            camera_noise=camera.noise,
            camera_underexposed_ratio=camera.underexposed_ratio,
            camera_overexposed_ratio=camera.overexposed_ratio,
        )


def _migrate_row(values: Mapping[str, str]) -> tuple[str, ...]:
    migrated = {field: values.get(field, "") for field in EVENT_FIELDS}
    stored_plate = values.get("ocr_plate", "")
    legacy_plate = values.get("plate_text", "")
    plate = stored_plate or legacy_plate
    migrated["ocr_plate"] = stored_plate or mask_plate_text(legacy_plate)
    migrated["ocr_confidence"] = values.get("ocr_confidence") or values.get(
        "plate_confidence", ""
    )
    migrated["ocr_status"] = values.get("ocr_status") or (
        "accepted" if plate else "not_run"
    )
    migrated["speed_class"] = values.get("speed_class") or "unknown"
    migrated["ocr_observation_count"] = values.get("ocr_observation_count") or "0"
    migrated["ocr_agreement"] = values.get("ocr_agreement") or "0"
    return tuple(migrated[field] for field in EVENT_FIELDS)
