from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from roundabout_ai.detector import Detection
from roundabout_ai.geometry import CountLine
from roundabout_ai.scene import (
    Scene,
    annotate_scene,
    filter_detections_by_roi,
    load_scene,
    save_scene,
    track_observations,
)


def test_scene_yaml_round_trip_and_scaling(tmp_path: Path) -> None:
    path = tmp_path / "scene.yaml"
    original = Scene(
        (100, 50),
        ((0.0, 0.0), (100.0, 0.0), (100.0, 50.0)),
        (CountLine("entry", (10.0, 20.0), (90.0, 20.0), "in", "out"),),
    )

    save_scene(original, path)
    loaded = load_scene(path)
    scaled = loaded.scaled(200, 100)

    assert loaded == original
    assert scaled.roi[1] == (200.0, 0.0)
    assert scaled.count_lines[0].start == (20.0, 40.0)
    assert scaled.count_lines[0].negative_to_positive == "in"


def test_scene_validation_rejects_duplicate_line_names(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        """reference_size: [100, 100]
roi: []
count_lines:
  - {name: entry, start: [0, 1], end: [2, 1]}
  - {name: entry, start: [0, 2], end: [2, 2]}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate"):
        load_scene(path)


def test_roi_filter_and_track_observation_use_box_centre() -> None:
    inside = Detection(2, "car", 0.9, (10, 10, 20, 20), 11)
    outside = Detection(2, "car", 0.8, (110, 10, 120, 20), 12)
    roi = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))

    filtered = filter_detections_by_roi((inside, outside), roi)
    observations = track_observations(filtered)

    assert filtered == (inside,)
    assert observations[0].track_id == 11
    assert observations[0].centre == (15.0, 15.0)


def test_scene_annotation_returns_a_drawn_copy() -> None:
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    scene = Scene(
        (100, 100),
        ((5.0, 5.0), (90.0, 5.0), (90.0, 90.0)),
        (CountLine("entry", (10.0, 50.0), (90.0, 50.0)),),
    )

    annotated = annotate_scene(source, scene)

    assert not np.shares_memory(source, annotated)
    assert np.count_nonzero(source) == 0
    assert np.count_nonzero(annotated) > 0
