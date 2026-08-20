"""Timestamp-based relative speed estimates for tracked road users."""

from __future__ import annotations

import math
import statistics
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass, field
from itertools import pairwise

from roundabout_ai.detector import Detection

VEHICLE_CLASSES = frozenset({"bicycle", "car", "motorcycle", "bus", "truck"})


@dataclass(frozen=True, slots=True)
class SpeedEstimate:
    """A relative speed expressed in vehicle-box heights per second."""

    speed_class: str
    normalized_speed: float | None


@dataclass(frozen=True, slots=True)
class _TimedPosition:
    captured_at: float
    bottom_centre: tuple[float, float]
    box_height: float


@dataclass(slots=True)
class _SpeedTrack:
    observations: deque[_TimedPosition] = field(default_factory=deque)
    last_seen_at: float | None = None


class TrackSpeedEstimator:
    """Classify track motion as slow/fast without claiming physical speed."""

    def __init__(
        self,
        *,
        fast_threshold: float = 1.0,
        minimum_observations: int = 3,
        minimum_duration_seconds: float = 0.5,
        history_seconds: float = 2.0,
        stale_after_seconds: float = 2.0,
    ) -> None:
        if fast_threshold <= 0:
            raise ValueError("fast speed threshold must be positive")
        if minimum_observations < 2:
            raise ValueError("minimum speed observations must be at least 2")
        if minimum_duration_seconds <= 0:
            raise ValueError("minimum speed duration must be positive")
        if history_seconds < minimum_duration_seconds:
            raise ValueError("speed history must cover the minimum duration")
        if stale_after_seconds <= 0:
            raise ValueError("speed track stale time must be positive")
        self.fast_threshold = fast_threshold
        self.minimum_observations = minimum_observations
        self.minimum_duration_seconds = minimum_duration_seconds
        self.history_seconds = history_seconds
        self.stale_after_seconds = stale_after_seconds
        self._tracks: dict[int, _SpeedTrack] = {}

    def update(
        self, detections: Iterable[Detection], *, captured_at: float
    ) -> dict[int, SpeedEstimate]:
        """Update vehicle tracks and return estimates for this observation time."""

        stale_ids = [
            track_id
            for track_id, track in self._tracks.items()
            if track.last_seen_at is not None
            and captured_at - track.last_seen_at > self.stale_after_seconds
        ]
        for track_id in stale_ids:
            del self._tracks[track_id]

        estimates: dict[int, SpeedEstimate] = {}
        for detection in detections:
            if detection.track_id is None or detection.label not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = detection.xyxy
            box_height = float(y2 - y1)
            if box_height <= 0:
                continue
            track = self._tracks.setdefault(detection.track_id, _SpeedTrack())
            if track.last_seen_at is not None and captured_at <= track.last_seen_at:
                estimates[detection.track_id] = self._estimate(track)
                continue
            track.observations.append(
                _TimedPosition(
                    captured_at,
                    ((x1 + x2) / 2, float(y2)),
                    box_height,
                )
            )
            track.last_seen_at = captured_at
            cutoff = captured_at - self.history_seconds
            while (
                len(track.observations) > 1
                and track.observations[0].captured_at < cutoff
            ):
                track.observations.popleft()
            estimates[detection.track_id] = self._estimate(track)

        return estimates

    def _estimate(self, track: _SpeedTrack) -> SpeedEstimate:
        observations = track.observations
        if (
            len(observations) < self.minimum_observations
            or observations[-1].captured_at - observations[0].captured_at
            < self.minimum_duration_seconds
        ):
            return SpeedEstimate("unknown", None)

        segment_speeds: list[float] = []
        for previous, current in pairwise(observations):
            elapsed = current.captured_at - previous.captured_at
            scale = (previous.box_height + current.box_height) / 2
            if elapsed <= 0 or scale <= 0:
                continue
            distance = math.dist(previous.bottom_centre, current.bottom_centre)
            segment_speeds.append(distance / scale / elapsed)
        if not segment_speeds:
            return SpeedEstimate("unknown", None)
        speed = statistics.median(segment_speeds)
        speed_class = "fast" if speed >= self.fast_threshold else "slow"
        return SpeedEstimate(speed_class, speed)
