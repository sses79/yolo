"""Scene geometry and deterministic line-crossing state."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

Point = tuple[float, float]


def signed_side(point: Point, start: Point, end: Point) -> float:
    """Return the 2-D cross product for a point and an ordered line."""

    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (
        point[0] - start[0]
    )


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Return True for points inside or on the edge of a polygon."""

    if len(polygon) < 3:
        return True
    x, y = point
    inside = False
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        cross = signed_side(point, previous, current)
        if (
            abs(cross) <= 1e-6
            and min(x1, x2) <= x <= max(x1, x2)
            and min(y1, y2) <= y <= max(y1, y2)
        ):
            return True
        if (y1 > y) != (y2 > y):
            intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < intersection_x:
                inside = not inside
        previous = current
    return inside


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return whether two closed line segments intersect."""

    ab_c = signed_side(c, a, b)
    ab_d = signed_side(d, a, b)
    cd_a = signed_side(a, c, d)
    cd_b = signed_side(b, c, d)
    return ab_c * ab_d <= 0 and cd_a * cd_b <= 0


@dataclass(frozen=True, slots=True)
class CountLine:
    name: str
    start: Point
    end: Point
    negative_to_positive: str = "negative_to_positive"
    positive_to_negative: str = "positive_to_negative"

    def __post_init__(self) -> None:
        if self.start == self.end:
            raise ValueError(f"count line {self.name!r} must have distinct endpoints")


@dataclass(frozen=True, slots=True)
class TrackObservation:
    track_id: int
    label: str
    confidence: float
    centre: Point
    speed_class: str = "unknown"
    normalized_speed: float | None = None


@dataclass(frozen=True, slots=True)
class CrossingEvent:
    line_name: str
    track_id: int
    label: str
    direction: str
    confidence: float
    speed_class: str = "unknown"
    normalized_speed: float | None = None


@dataclass(slots=True)
class _TrackState:
    age: int
    last_seen_frame: int
    last_point: Point
    line_sides: dict[str, int]
    counted_lines: set[str] = field(default_factory=set)


class CrossingCounter:
    """Count a finite-line crossing once per persistent track and line."""

    def __init__(
        self,
        lines: Iterable[CountLine],
        *,
        minimum_track_age: int = 3,
        maximum_missing_frames: int = 30,
        side_epsilon: float = 1e-6,
    ) -> None:
        if minimum_track_age <= 0:
            raise ValueError("minimum track age must be positive")
        if maximum_missing_frames < 0:
            raise ValueError("maximum missing frames must be nonnegative")
        self.lines = tuple(lines)
        self.minimum_track_age = minimum_track_age
        self.maximum_missing_frames = maximum_missing_frames
        self.side_epsilon = side_epsilon
        self._frame_number = 0
        self._tracks: dict[int, _TrackState] = {}
        self._counts: Counter[tuple[str, str, str]] = Counter()

    @property
    def counts(self) -> dict[str, int]:
        return {
            f"{line}:{direction}:{label}": count
            for (line, direction, label), count in sorted(self._counts.items())
        }

    def _side(self, point: Point, line: CountLine) -> int:
        value = signed_side(point, line.start, line.end)
        if abs(value) <= self.side_epsilon:
            return 0
        return 1 if value > 0 else -1

    def update(
        self, observations: Iterable[TrackObservation]
    ) -> tuple[CrossingEvent, ...]:
        self._frame_number += 1
        events: list[CrossingEvent] = []
        for observation in observations:
            state = self._tracks.get(observation.track_id)
            current_sides = {
                line.name: self._side(observation.centre, line) for line in self.lines
            }
            if state is None:
                self._tracks[observation.track_id] = _TrackState(
                    age=1,
                    last_seen_frame=self._frame_number,
                    last_point=observation.centre,
                    line_sides=current_sides,
                )
                continue

            state.age += 1
            state.last_seen_frame = self._frame_number
            for line in self.lines:
                previous_side = state.line_sides.get(line.name, 0)
                current_side = current_sides[line.name]
                crossed = (
                    previous_side != 0
                    and current_side != 0
                    and previous_side != current_side
                    and segments_intersect(
                        state.last_point, observation.centre, line.start, line.end
                    )
                )
                if (
                    crossed
                    and state.age >= self.minimum_track_age
                    and line.name not in state.counted_lines
                ):
                    direction = (
                        line.negative_to_positive
                        if previous_side < current_side
                        else line.positive_to_negative
                    )
                    event = CrossingEvent(
                        line.name,
                        observation.track_id,
                        observation.label,
                        direction,
                        observation.confidence,
                        observation.speed_class,
                        observation.normalized_speed,
                    )
                    events.append(event)
                    state.counted_lines.add(line.name)
                    self._counts[(line.name, direction, observation.label)] += 1
                if current_side != 0:
                    state.line_sides[line.name] = current_side
            state.last_point = observation.centre

        expired = [
            track_id
            for track_id, state in self._tracks.items()
            if self._frame_number - state.last_seen_frame > self.maximum_missing_frames
        ]
        for track_id in expired:
            del self._tracks[track_id]
        return tuple(events)
