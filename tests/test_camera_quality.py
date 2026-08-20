from __future__ import annotations

import numpy as np
import pytest

from roundabout_ai.camera_quality import (
    CameraProfileAdvisor,
    classify_condition,
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
