from __future__ import annotations

import numpy as np
import pytest

from roundabout_ai.detector import (
    Detection,
    DetectionBatch,
    YoloDetector,
    annotate_detections,
    benchmark_detector,
    parse_class_names,
)


class FakeTensor:
    def __init__(self, values: list[object]) -> None:
        self.values = values

    def cpu(self) -> "FakeTensor":
        return self

    def tolist(self) -> list[object]:
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10.2, 20.4, 100.6, 120.8]])
    conf = FakeTensor([0.876])
    cls = FakeTensor([2.0])


class FakeResult:
    boxes = FakeBoxes()


class FakeModel:
    names = {0: "person", 1: "bicycle", 2: "car"}

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def predict(self, **kwargs: object) -> list[FakeResult]:
        self.calls.append(kwargs)
        return [FakeResult()]


def test_parse_class_names_normalizes_and_deduplicates() -> None:
    assert parse_class_names(" Car,person,car ") == ("car", "person")
    with pytest.raises(ValueError, match="at least one"):
        parse_class_names(" , ")


def test_yolo_detector_adapts_model_results() -> None:
    model = FakeModel()
    detector = YoloDetector(
        "fake.pt",
        confidence=0.4,
        image_size=320,
        device="cpu",
        class_names=("person", "car"),
        model_factory=lambda _path: model,
    )
    frame = np.zeros((200, 300, 3), dtype=np.uint8)

    batch = detector.predict(frame)

    assert batch.device == "cpu"
    assert batch.counts == {"car": 1}
    assert batch.detections == (
        Detection(2, "car", pytest.approx(0.876), (10, 20, 101, 121)),
    )
    assert model.calls[0]["source"] is frame
    assert model.calls[0]["classes"] == [0, 2]
    assert model.calls[0]["conf"] == 0.4
    assert model.calls[0]["imgsz"] == 320
    assert model.calls[0]["verbose"] is False


def test_yolo_detector_rejects_unknown_requested_class() -> None:
    with pytest.raises(ValueError, match="truck"):
        YoloDetector(
            "fake.pt",
            device="cpu",
            class_names=("truck",),
            model_factory=lambda _path: FakeModel(),
        )


def test_annotation_returns_copy_and_draws_box() -> None:
    source = np.zeros((100, 160, 3), dtype=np.uint8)
    detection = Detection(2, "car", 0.9, (10, 20, 100, 80))

    annotated = annotate_detections(source, (detection,))

    assert not np.shares_memory(source, annotated)
    assert np.count_nonzero(source) == 0
    assert np.count_nonzero(annotated) > 0


def test_benchmark_summarizes_measured_runs() -> None:
    class FakeDetector:
        device = "cpu"

        def __init__(self) -> None:
            self.calls = 0

        def predict(self, _frame: np.ndarray) -> DetectionBatch:
            self.calls += 1
            elapsed = (self.calls - 1) * 0.01
            return DetectionBatch((), elapsed, self.device)

    detector = FakeDetector()
    result = benchmark_detector(
        detector,  # type: ignore[arg-type]
        np.zeros((2, 2, 3), dtype=np.uint8),
        warmup_runs=1,
        measured_runs=3,
    )

    assert detector.calls == 4
    assert result.mean_ms == pytest.approx(20.0)
    assert result.median_ms == pytest.approx(20.0)
    assert result.minimum_ms == pytest.approx(10.0)
    assert result.maximum_ms == pytest.approx(30.0)
    assert result.fps == pytest.approx(50.0)
