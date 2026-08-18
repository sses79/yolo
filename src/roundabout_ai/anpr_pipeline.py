"""Offline plate detection and OCR over saved vehicle-track candidates."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Protocol, cast

import cv2

from roundabout_ai.anpr import (
    ConsensusPolicy,
    OcrObservation,
    PlateCandidate,
    PlateQualityPolicy,
    TrackPlateResult,
    build_consensus,
    extract_plate_candidate,
    prepare_plate_crop,
)
from roundabout_ai.capture import Frame
from roundabout_ai.detector import DetectionBatch

_EVENT_IMAGE = re.compile(
    r"^event-(?P<stamp>\d{8}-\d{6}-\d{6}[+-]\d{4})-track-"
    r"(?P<track>\d+)-(?P<line>.+)-"
    r"(?P<kind>crossing|centred|largest|sharpest)\.jpg$"
)


class PlateDetectorLike(Protocol):
    def predict(self, frame: Frame) -> DetectionBatch: ...


class PlateRecognizerLike(Protocol):
    def recognize(self, image: Frame, *, image_id: str) -> OcrObservation: ...


@dataclass(frozen=True, slots=True)
class VehicleImageGroup:
    vehicle_id: str
    paths: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class VehicleAnalysis:
    result: TrackPlateResult
    image_count: int
    detections: int
    accepted_candidates: int
    rejection_reasons: dict[str, int]


def discover_vehicle_groups(
    directory: Path, *, maximum_event_gap_seconds: float = 10.0
) -> tuple[VehicleImageGroup, ...]:
    """Group candidate images without conflating reused tracker IDs."""

    if maximum_event_gap_seconds <= 0:
        raise ValueError("maximum event gap must be positive")
    parsed: list[tuple[int, str, datetime, Path]] = []
    for path in sorted(directory.glob("event-*.jpg")):
        match = _EVENT_IMAGE.fullmatch(path.name)
        if match is None:
            continue
        stamp = datetime.strptime(match["stamp"], "%Y%m%d-%H%M%S-%f%z")
        parsed.append((int(match["track"]), match["line"], stamp, path))

    groups: list[VehicleImageGroup] = []
    for track_line in sorted({(track, line) for track, line, _, _ in parsed}):
        track, line = track_line
        matching = sorted(
            (
                (stamp, path)
                for item_track, item_line, stamp, path in parsed
                if (item_track, item_line) == track_line
            ),
            key=lambda item: item[0],
        )
        current: list[tuple[datetime, Path]] = []
        for stamp, path in matching:
            if (
                current
                and (stamp - current[-1][0]).total_seconds() > maximum_event_gap_seconds
            ):
                groups.append(_make_group(track, line, current))
                current = []
            current.append((stamp, path))
        if current:
            groups.append(_make_group(track, line, current))
    return tuple(sorted(groups, key=lambda group: group.vehicle_id))


def _make_group(
    track: int, line: str, items: Sequence[tuple[datetime, Path]]
) -> VehicleImageGroup:
    first_stamp = items[0][0].strftime("%Y%m%d-%H%M%S-%f%z")
    return VehicleImageGroup(
        vehicle_id=f"{first_stamp}-track-{track}-{line}",
        paths=tuple(path for _, path in items),
    )


def analyze_vehicle(
    group: VehicleImageGroup,
    detector: PlateDetectorLike,
    recognizer: PlateRecognizerLike,
    *,
    quality_policy: PlateQualityPolicy = PlateQualityPolicy(),
    consensus_policy: ConsensusPolicy = ConsensusPolicy(),
    maximum_ocr_candidates: int = 3,
) -> VehicleAnalysis:
    """Detect plates in vehicle crops and form one conservative track result."""

    if maximum_ocr_candidates <= 0:
        raise ValueError("maximum OCR candidates must be positive")
    candidates: list[PlateCandidate] = []
    rejection_reasons: Counter[str] = Counter()
    detection_count = 0
    image_count = 0
    for path in group.paths:
        loaded = cv2.imread(str(path))
        if loaded is None:
            rejection_reasons["unreadable_image"] += 1
            continue
        frame = cast(Frame, loaded)
        image_count += 1
        batch = detector.predict(frame)
        detection_count += len(batch.detections)
        image_candidates: list[PlateCandidate] = []
        for detection in batch.detections:
            candidate = extract_plate_candidate(
                group.vehicle_id,
                path.name,
                frame,
                detection.xyxy,
                detection.confidence,
                quality_policy,
            )
            if candidate is None:
                rejection_reasons["invalid_box"] += 1
            elif candidate.quality.accepted:
                image_candidates.append(candidate)
            else:
                rejection_reasons.update(candidate.quality.reasons)
        if image_candidates:
            candidates.append(max(image_candidates, key=_candidate_rank))

    chosen = sorted(candidates, key=_candidate_rank, reverse=True)[
        :maximum_ocr_candidates
    ]
    observations = tuple(
        recognizer.recognize(
            prepare_plate_crop(candidate.crop), image_id=candidate.image_id
        )
        for candidate in chosen
    )
    result = build_consensus(group.vehicle_id, observations, consensus_policy)
    return VehicleAnalysis(
        result=result,
        image_count=image_count,
        detections=detection_count,
        accepted_candidates=len(candidates),
        rejection_reasons=dict(rejection_reasons),
    )


def analyze_groups(
    groups: Iterable[VehicleImageGroup],
    detector: PlateDetectorLike,
    recognizer: PlateRecognizerLike,
    *,
    quality_policy: PlateQualityPolicy = PlateQualityPolicy(),
    consensus_policy: ConsensusPolicy = ConsensusPolicy(),
    maximum_ocr_candidates: int = 3,
) -> tuple[VehicleAnalysis, ...]:
    return tuple(
        analyze_vehicle(
            group,
            detector,
            recognizer,
            quality_policy=quality_policy,
            consensus_policy=consensus_policy,
            maximum_ocr_candidates=maximum_ocr_candidates,
        )
        for group in groups
    )


def _candidate_rank(candidate: PlateCandidate) -> tuple[float, float, int]:
    quality = candidate.quality
    return (
        quality.sharpness,
        candidate.detector_confidence,
        quality.width * quality.height,
    )
