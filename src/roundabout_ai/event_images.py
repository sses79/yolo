"""Bounded best-of-track vehicle crops saved only after confirmed crossings."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2

from roundabout_ai.capture import Frame
from roundabout_ai.detector import Detection
from roundabout_ai.events import EventRecord

VEHICLE_CLASSES = frozenset(("car", "motorcycle", "bus", "truck"))
DEFAULT_HORIZONTAL_PADDING_RATIO = 0.15
DEFAULT_VERTICAL_PADDING_RATIO = 0.10
DEFAULT_MINIMUM_VEHICLE_WIDTH = 200
DEFAULT_MINIMUM_VEHICLE_HEIGHT = 100


@dataclass(frozen=True, slots=True)
class VehicleCandidate:
    track_id: int
    frame_number: int
    crop: Frame
    area: int
    sharpness: float
    clipped: bool
    edge_clearance: int
    source_width: int
    source_height: int
    suitable: bool


@dataclass(slots=True)
class _CandidateState:
    last_seen_frame: int
    crossing: VehicleCandidate
    centred: VehicleCandidate
    sharpest: VehicleCandidate


def _candidate(
    frame: Frame,
    detection: Detection,
    frame_number: int,
    *,
    horizontal_padding_ratio: float,
    vertical_padding_ratio: float,
    minimum_vehicle_width: int,
    minimum_vehicle_height: int,
) -> VehicleCandidate | None:
    if detection.track_id is None or detection.label not in VEHICLE_CLASSES:
        return None
    height, width = frame.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = detection.xyxy
    source_width = raw_x2 - raw_x1
    source_height = raw_y2 - raw_y1
    if source_width < 2 or source_height < 2:
        return None
    horizontal_padding = round(source_width * horizontal_padding_ratio)
    vertical_padding = round(source_height * vertical_padding_ratio)
    intended = (
        raw_x1 - horizontal_padding,
        raw_y1 - vertical_padding,
        raw_x2 + horizontal_padding,
        raw_y2 + vertical_padding,
    )
    x1 = max(0, min(intended[0], width))
    y1 = max(0, min(intended[1], height))
    x2 = max(0, min(intended[2], width))
    y2 = max(0, min(intended[3], height))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return VehicleCandidate(
        track_id=detection.track_id,
        frame_number=frame_number,
        crop=crop,
        area=(x2 - x1) * (y2 - y1),
        sharpness=sharpness,
        clipped=(x1, y1, x2, y2) != intended,
        edge_clearance=min(x1, width - x2),
        source_width=source_width,
        source_height=source_height,
        suitable=(
            source_width >= minimum_vehicle_width
            and source_height >= minimum_vehicle_height
        ),
    )


def _centred_rank(candidate: VehicleCandidate) -> tuple[bool, bool, int, int, float]:
    return (
        candidate.suitable,
        not candidate.clipped,
        candidate.edge_clearance,
        candidate.area,
        candidate.sharpness,
    )


def _sharpest_rank(candidate: VehicleCandidate) -> tuple[bool, bool, float, int]:
    return (
        candidate.suitable,
        not candidate.clipped,
        candidate.sharpness,
        candidate.area,
    )


class VehicleCandidateBuffer:
    """Keep at most three raw crops for each of a bounded set of live tracks."""

    def __init__(
        self,
        *,
        maximum_missing_frames: int = 30,
        maximum_tracks: int = 32,
        horizontal_padding_ratio: float = DEFAULT_HORIZONTAL_PADDING_RATIO,
        vertical_padding_ratio: float = DEFAULT_VERTICAL_PADDING_RATIO,
        minimum_vehicle_width: int = DEFAULT_MINIMUM_VEHICLE_WIDTH,
        minimum_vehicle_height: int = DEFAULT_MINIMUM_VEHICLE_HEIGHT,
    ) -> None:
        if maximum_missing_frames < 0:
            raise ValueError("maximum missing frames must be nonnegative")
        if maximum_tracks <= 0:
            raise ValueError("maximum tracks must be positive")
        if horizontal_padding_ratio < 0 or vertical_padding_ratio < 0:
            raise ValueError("candidate padding ratios must be nonnegative")
        if minimum_vehicle_width <= 0 or minimum_vehicle_height <= 0:
            raise ValueError("minimum vehicle dimensions must be positive")
        self.maximum_missing_frames = maximum_missing_frames
        self.maximum_tracks = maximum_tracks
        self.horizontal_padding_ratio = horizontal_padding_ratio
        self.vertical_padding_ratio = vertical_padding_ratio
        self.minimum_vehicle_width = minimum_vehicle_width
        self.minimum_vehicle_height = minimum_vehicle_height
        self._frame_number = 0
        self._tracks: dict[int, _CandidateState] = {}

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def observe(self, frame: Frame, detections: Iterable[Detection]) -> None:
        self._frame_number += 1
        for detection in detections:
            current = _candidate(
                frame,
                detection,
                self._frame_number,
                horizontal_padding_ratio=self.horizontal_padding_ratio,
                vertical_padding_ratio=self.vertical_padding_ratio,
                minimum_vehicle_width=self.minimum_vehicle_width,
                minimum_vehicle_height=self.minimum_vehicle_height,
            )
            if current is None:
                continue
            state = self._tracks.get(current.track_id)
            if state is None:
                self._tracks[current.track_id] = _CandidateState(
                    last_seen_frame=self._frame_number,
                    crossing=current,
                    centred=current,
                    sharpest=current,
                )
                continue
            state.last_seen_frame = self._frame_number
            state.crossing = current
            if _centred_rank(current) > _centred_rank(state.centred):
                state.centred = current
            if _sharpest_rank(current) > _sharpest_rank(state.sharpest):
                state.sharpest = current

        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if self._frame_number - state.last_seen_frame > self.maximum_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]
        while len(self._tracks) > self.maximum_tracks:
            oldest = min(
                self._tracks,
                key=lambda track_id: self._tracks[track_id].last_seen_frame,
            )
            del self._tracks[oldest]

    def select(self, track_id: int) -> tuple[tuple[str, VehicleCandidate], ...]:
        """Return distinct, suitably sized crossing/centred/sharpest crops."""

        state = self._tracks.get(track_id)
        if state is None:
            return ()
        selected: list[tuple[str, VehicleCandidate]] = []
        used_frames: set[int] = set()
        for name, candidate in (
            ("crossing", state.crossing),
            ("centred", state.centred),
            ("sharpest", state.sharpest),
        ):
            if candidate.suitable and candidate.frame_number not in used_frames:
                selected.append((name, candidate))
                used_frames.add(candidate.frame_number)
        return tuple(selected)


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return normalized or "unnamed"


def _event_path(
    directory: Path,
    event: EventRecord,
    kind: str,
    observed_at: datetime,
) -> Path:
    stamp = observed_at.strftime("%Y%m%d-%H%M%S-%f%z")
    return directory / (
        f"event-{stamp}-track-{event.track_id}-{_slug(event.line_name)}-"
        f"{_slug(kind)}.jpg"
    )


def _save_jpeg(frame: Frame, path: Path) -> None:
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"OpenCV failed to save event image to {path}")


def save_event_snapshot(
    frame: Frame,
    directory: Path,
    event: EventRecord,
    *,
    clock: Callable[[], datetime] | None = None,
) -> Path:
    """Save the full annotated frame at the confirmed crossing."""

    observed_at = (clock or (lambda: datetime.now().astimezone()))()
    directory.mkdir(parents=True, exist_ok=True)
    path = _event_path(directory, event, "snapshot", observed_at)
    _save_jpeg(frame, path)
    return path


def save_event_candidates(
    candidates: Iterable[tuple[str, VehicleCandidate]],
    directory: Path,
    event: EventRecord,
    *,
    clock: Callable[[], datetime] | None = None,
) -> tuple[Path, ...]:
    """Save raw vehicle crops using one timestamped, event-specific filename set."""

    observed_at = (clock or (lambda: datetime.now().astimezone()))()
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for kind, candidate in candidates:
        path = _event_path(directory, event, kind, observed_at)
        _save_jpeg(candidate.crop, path)
        paths.append(path)
    return tuple(paths)
