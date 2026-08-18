from __future__ import annotations

import pytest

from roundabout_ai.detector import Detection
from roundabout_ai.speed import TrackSpeedEstimator


def vehicle(track_id: int, bottom_x: int, *, height: int = 20) -> Detection:
    return Detection(
        2, "car", 0.9, (bottom_x - 5, 100 - height, bottom_x + 5, 100), track_id
    )


def test_speed_is_unknown_until_timed_history_is_sufficient() -> None:
    estimator = TrackSpeedEstimator(
        minimum_observations=3, minimum_duration_seconds=0.5
    )

    estimator.update((vehicle(1, 0),), captured_at=1.0)
    estimate = estimator.update((vehicle(1, 2),), captured_at=1.6)[1]

    assert estimate.speed_class == "unknown"
    assert estimate.normalized_speed is None


@pytest.mark.parametrize(
    ("positions", "expected_class"),
    [((0, 4, 8), "slow"), ((0, 15, 30), "fast")],
)
def test_speed_classifies_normalized_track_motion(
    positions: tuple[int, int, int], expected_class: str
) -> None:
    estimator = TrackSpeedEstimator(
        fast_threshold=1.0,
        minimum_observations=3,
        minimum_duration_seconds=1.0,
    )

    estimates = [
        estimator.update((vehicle(1, position),), captured_at=captured_at)[1]
        for captured_at, position in zip((0.0, 0.5, 1.0), positions)
    ]

    assert estimates[-1].speed_class == expected_class


def test_box_height_normalizes_resolution_and_perspective() -> None:
    small = TrackSpeedEstimator(minimum_duration_seconds=1.0)
    large = TrackSpeedEstimator(minimum_duration_seconds=1.0)

    small_estimates = [
        small.update((vehicle(1, position, height=20),), captured_at=captured_at)[1]
        for captured_at, position in zip((0.0, 0.5, 1.0), (0, 10, 20))
    ]
    large_estimates = [
        large.update((vehicle(1, position * 2, height=40),), captured_at=captured_at)[1]
        for captured_at, position in zip((0.0, 0.5, 1.0), (0, 10, 20))
    ]

    assert small_estimates[-1].normalized_speed == pytest.approx(
        large_estimates[-1].normalized_speed
    )


def test_stale_track_id_starts_with_unknown_speed() -> None:
    estimator = TrackSpeedEstimator(
        minimum_observations=3,
        minimum_duration_seconds=0.5,
        stale_after_seconds=1.0,
    )
    estimator.update((vehicle(1, 0),), captured_at=0.0)
    estimator.update((vehicle(1, 10),), captured_at=0.5)
    estimator.update((vehicle(1, 20),), captured_at=1.0)

    estimate = estimator.update((vehicle(1, 20),), captured_at=3.0)[1]

    assert estimate.speed_class == "unknown"
