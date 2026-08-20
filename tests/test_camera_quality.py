from __future__ import annotations

import numpy as np
import pytest

from roundabout_ai.camera_quality import (
    CameraProfileAdvisor,
    classify_condition,
    guard_profile_transition,
    measure_frame_quality,
)


@pytest.mark.parametrize(
    ("level", "condition"),
    ((20, "night"), (70, "dusk"), (140, "day"), (255, "glare")),
)
def test_quality_classifies_lighting(level: int, condition: str) -> None:
    frame = np.full((20, 30, 3), level, dtype=np.uint8)

    quality = measure_frame_quality(frame)

    assert classify_condition(quality) == condition
    assert quality.luminance_median == level


def test_quality_uses_scene_roi_instead_of_bright_surroundings() -> None:
    frame = np.full((20, 20, 3), 255, dtype=np.uint8)
    frame[5:15, 5:15] = 20
    roi = ((5.0, 5.0), (14.0, 5.0), (14.0, 14.0), (5.0, 14.0))

    quality = measure_frame_quality(frame, roi)

    assert classify_condition(quality) == "night"


def test_advisor_requires_repeated_observations_and_resets_candidate() -> None:
    advisor = CameraProfileAdvisor(required_observations=3)
    night = measure_frame_quality(np.full((10, 10, 3), 20, dtype=np.uint8))
    day = measure_frame_quality(np.full((10, 10, 3), 140, dtype=np.uint8))

    assert advisor.observe(night) is None
    assert advisor.observe(day) is None
    assert advisor.observe(night) is None
    assert advisor.observe(night) is None
    assert advisor.observe(night) == "night"


def test_transition_guard_steps_from_night_to_day_through_dusk() -> None:
    night_brightened_scene = measure_frame_quality(
        np.full((10, 10, 3), 115, dtype=np.uint8)
    )

    guarded = guard_profile_transition(
        "night",
        "day",
        night_brightened_scene,
        moving_vehicle_sharpness=21.4,
    )

    assert guarded == "dusk"


def test_transition_guard_does_not_make_underexposed_scene_darker() -> None:
    underexposed = measure_frame_quality(np.zeros((10, 10, 3), dtype=np.uint8))

    assert guard_profile_transition("night", "day", underexposed) == "night"


def test_transition_guard_allows_more_light_for_severe_underexposure() -> None:
    underexposed = measure_frame_quality(np.zeros((10, 10, 3), dtype=np.uint8))

    assert (
        guard_profile_transition(
            "day", "night", underexposed, moving_vehicle_sharpness=8.1
        )
        == "dusk"
    )


def test_transition_guard_avoids_more_blur_without_underexposure() -> None:
    adequately_exposed = measure_frame_quality(np.full((10, 10, 3), 70, dtype=np.uint8))

    assert (
        guard_profile_transition(
            "day", "night", adequately_exposed, moving_vehicle_sharpness=20.0
        )
        == "day"
    )


def test_transition_guard_does_not_retry_recently_underexposed_profile() -> None:
    adequately_exposed = measure_frame_quality(np.full((10, 10, 3), 70, dtype=np.uint8))

    assert (
        guard_profile_transition(
            "dusk",
            "day",
            adequately_exposed,
            moving_vehicle_sharpness=20.0,
            known_profile_underexposure={"day": 0.40},
        )
        == "dusk"
    )
