from __future__ import annotations

from roundabout_ai.geometry import (
    CountLine,
    CrossingCounter,
    TrackObservation,
    point_in_polygon,
    segments_intersect,
    signed_side,
)


def observation(track_id: int, centre: tuple[float, float]) -> TrackObservation:
    return TrackObservation(track_id, "car", 0.9, centre)


def test_point_in_polygon_includes_boundary_and_empty_roi() -> None:
    square = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))

    assert point_in_polygon((5.0, 5.0), square)
    assert point_in_polygon((0.0, 5.0), square)
    assert not point_in_polygon((11.0, 5.0), square)
    assert point_in_polygon((999.0, 999.0), ())


def test_line_geometry_uses_order_and_finite_segments() -> None:
    assert signed_side((5.0, 2.0), (0.0, 0.0), (10.0, 0.0)) > 0
    assert signed_side((5.0, -2.0), (0.0, 0.0), (10.0, 0.0)) < 0
    assert segments_intersect((5.0, -2.0), (5.0, 2.0), (0.0, 0.0), (10.0, 0.0))
    assert not segments_intersect(
        (15.0, -2.0), (15.0, 2.0), (0.0, 0.0), (10.0, 0.0)
    )


def test_crossing_is_directional_and_counted_once_per_track_line() -> None:
    line = CountLine(
        "entry", (0.0, 0.0), (10.0, 0.0), "entering", "leaving"
    )
    counter = CrossingCounter((line,), minimum_track_age=2)

    assert counter.update((observation(7, (5.0, -2.0)),)) == ()
    events = counter.update((observation(7, (5.0, 2.0)),))
    assert len(events) == 1
    assert events[0].direction == "entering"
    assert events[0].track_id == 7
    assert counter.counts == {"entry:entering:car": 1}

    assert counter.update((observation(7, (5.0, -2.0)),)) == ()
    assert counter.counts == {"entry:entering:car": 1}


def test_crossing_rejects_young_track_and_line_extension() -> None:
    line = CountLine("entry", (0.0, 0.0), (10.0, 0.0))
    counter = CrossingCounter((line,), minimum_track_age=3)

    counter.update((observation(1, (5.0, -1.0)), observation(2, (15.0, -1.0))))
    assert counter.update(
        (observation(1, (5.0, 1.0)), observation(2, (15.0, 1.0)))
    ) == ()
    assert counter.counts == {}


def test_expired_track_id_starts_a_new_lifecycle() -> None:
    line = CountLine("entry", (0.0, 0.0), (10.0, 0.0))
    counter = CrossingCounter(
        (line,), minimum_track_age=2, maximum_missing_frames=1
    )
    counter.update((observation(4, (5.0, -1.0)),))
    counter.update(())
    counter.update(())

    assert counter.update((observation(4, (5.0, 1.0)),)) == ()
