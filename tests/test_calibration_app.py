from __future__ import annotations

import pytest

from roundabout_ai.calibration_app import CalibrationSession


def test_calibration_session_builds_roi_and_named_lines() -> None:
    session = CalibrationSession()
    for point in ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0)):
        session.add_point(point)
    session.finish_roi()
    session.add_point((10.0, 50.0))
    session.add_point((90.0, 50.0))

    scene = session.scene(100, 100)

    assert scene.reference_size == (100, 100)
    assert scene.count_lines[0].name == "line_1"


def test_calibration_requires_complete_roi_and_line() -> None:
    session = CalibrationSession()
    session.add_point((0.0, 0.0))
    with pytest.raises(ValueError, match="three"):
        session.finish_roi()
