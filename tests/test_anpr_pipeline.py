from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from roundabout_ai.anpr import OcrObservation, PlateQualityPolicy, PlateStatus
from roundabout_ai.anpr_pipeline import (
    VehicleImageGroup,
    analyze_images,
    analyze_vehicle,
    discover_vehicle_groups,
)
from roundabout_ai.detector import Detection, DetectionBatch


class FakeDetector:
    def predict(self, frame: object) -> DetectionBatch:
        return DetectionBatch(
            (Detection(0, "license_plate", 0.9, (30, 30, 180, 70)),),
            0.01,
            "cpu",
        )


class FakeRecognizer:
    def __init__(self) -> None:
        self.image_ids: list[str] = []

    def recognize(self, image: object, *, image_id: str) -> OcrObservation:
        self.image_ids.append(image_id)
        return OcrObservation(image_id, "AB12CDE", "AB12CDE", 0.95, True)


def write_vehicle_image(path: Path) -> None:
    image = np.full((100, 240, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "AB12CDE",
        (38, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    assert cv2.imwrite(str(path), image)


def test_discovery_groups_nearby_candidates_but_splits_reused_track_ids(
    tmp_path: Path,
) -> None:
    names = (
        "event-20260817-120000-000000+0100-track-7-line_1-crossing.jpg",
        "event-20260817-120000-010000+0100-track-7-line_1-sharpest.jpg",
        "event-20260817-120000-020000+0100-track-7-line_1-snapshot.jpg",
        "event-20260817-130000-000000+0100-track-7-line_1-crossing.jpg",
    )
    for name in names:
        (tmp_path / name).touch()
    groups = discover_vehicle_groups(tmp_path)
    assert [len(group.paths) for group in groups] == [2, 1]
    assert groups[0].vehicle_id != groups[1].vehicle_id


def test_pipeline_uses_multiple_good_vehicle_crops_for_consensus(
    tmp_path: Path,
) -> None:
    paths = tuple(tmp_path / f"candidate-{number}.jpg" for number in range(3))
    for path in paths:
        write_vehicle_image(path)
    recognizer = FakeRecognizer()
    analysis = analyze_vehicle(
        VehicleImageGroup("vehicle-1", paths),
        FakeDetector(),
        recognizer,
        quality_policy=PlateQualityPolicy(minimum_sharpness=10),
    )
    assert analysis.result.status is PlateStatus.ACCEPTED
    assert analysis.result.agreement == 3
    assert len(recognizer.image_ids) == 3
    assert analysis.best_detector_confidence == 0.9
    assert analysis.best_plate_width == 150
    assert analysis.best_plate_height == 40
    assert analysis.best_plate_sharpness is not None


def test_pipeline_analyzes_live_in_memory_vehicle_crops() -> None:
    frame = np.full((100, 240, 3), 255, dtype=np.uint8)
    cv2.putText(
        frame,
        "AB12CDE",
        (38, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    recognizer = FakeRecognizer()

    analysis = analyze_images(
        "live-track-7",
        (("crossing", frame), ("sharpest", frame.copy())),
        FakeDetector(),
        recognizer,
        quality_policy=PlateQualityPolicy(minimum_sharpness=10),
    )

    assert analysis.result.status is PlateStatus.ACCEPTED
    assert analysis.result.plate_text == "AB12CDE"
    assert recognizer.image_ids == ["crossing", "sharpest"]
