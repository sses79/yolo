"""YOLO vehicle/person detection with framework-independent results."""

from __future__ import annotations

import statistics
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, cast

import cv2

from roundabout_ai.capture import Frame

ROAD_USER_CLASSES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
)


class ModelLike(Protocol):
    @property
    def names(self) -> Mapping[int, str] | Sequence[str]: ...

    def predict(self, **kwargs: object) -> Sequence[object]: ...

    def track(self, **kwargs: object) -> Sequence[object]: ...


ModelFactory = Callable[[str], ModelLike]


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    detections: tuple[Detection, ...]
    elapsed_seconds: float
    device: str

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(detection.label for detection in self.detections))


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    device: str
    runs: int
    mean_ms: float
    median_ms: float
    minimum_ms: float
    maximum_ms: float
    fps: float


def parse_class_names(value: str) -> tuple[str, ...]:
    names = tuple(
        dict.fromkeys(name.strip().lower() for name in value.split(",") if name.strip())
    )
    if not names:
        raise ValueError("at least one detection class is required")
    return names


def available_devices() -> tuple[str, ...]:
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return tuple(devices)


def resolve_device(requested: str) -> str:
    normalized = requested.lower()
    if normalized == "auto":
        return "mps" if "mps" in available_devices() else "cpu"
    if normalized not in {"cpu", "mps"}:
        raise ValueError("device must be one of: auto, cpu, mps")
    if normalized == "mps" and "mps" not in available_devices():
        raise ValueError("MPS was requested but is not available")
    return normalized


def _synchronize(device: str) -> None:
    if device == "mps":
        import torch

        torch.mps.synchronize()


def _as_list(value: object) -> list[object]:
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if isinstance(converted, Iterable):
            return list(converted)
        raise TypeError("tensor tolist() result is not iterable")
    if isinstance(value, Iterable):
        return list(value)
    raise TypeError("model output is not iterable")


def _number(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"model output is not numeric: {type(value).__name__}")


def _normalized_names(
    names: Mapping[int, str] | Sequence[str],
) -> dict[int, str]:
    if isinstance(names, Mapping):
        return {int(class_id): str(label) for class_id, label in names.items()}
    return {class_id: str(label) for class_id, label in enumerate(names)}


class YoloDetector:
    """Load YOLO once and expose only the detections this project needs."""

    def __init__(
        self,
        model_path: str = "yolo26n.pt",
        *,
        confidence: float = 0.35,
        image_size: int = 640,
        device: str = "auto",
        class_names: Iterable[str] = ROAD_USER_CLASSES,
        model_factory: ModelFactory | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if image_size <= 0:
            raise ValueError("image size must be greater than zero")
        if model_factory is None:
            from ultralytics import YOLO

            self._model = cast(ModelLike, YOLO(model_path))
        else:
            self._model = model_factory(model_path)

        self.model_path = model_path
        self.confidence = confidence
        self.image_size = image_size
        self.device = resolve_device(device)
        self.class_names = tuple(dict.fromkeys(name.lower() for name in class_names))
        self._names = _normalized_names(self._model.names)
        wanted = set(self.class_names)
        self._class_ids = tuple(
            class_id
            for class_id, label in self._names.items()
            if label.lower() in wanted
        )
        found = {self._names[class_id].lower() for class_id in self._class_ids}
        missing = sorted(wanted - found)
        if missing:
            raise ValueError(
                f"model does not provide requested classes: {', '.join(missing)}"
            )

    def _run(
        self,
        frame: Frame,
        *,
        track: bool,
        tracker_config: str = "bytetrack.yaml",
    ) -> DetectionBatch:
        _synchronize(self.device)
        started_at = time.perf_counter()
        kwargs: dict[str, object] = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "classes": list(self._class_ids),
            "device": self.device,
            "verbose": False,
        }
        if track:
            kwargs.update(persist=True, tracker=tracker_config)
            results = self._model.track(**kwargs)
        else:
            results = self._model.predict(**kwargs)
        _synchronize(self.device)
        elapsed = time.perf_counter() - started_at

        detections: list[Detection] = []
        if results:
            boxes = getattr(results[0], "boxes", None)
            if boxes is not None:
                coordinates = _as_list(boxes.xyxy)
                confidences = _as_list(boxes.conf)
                class_ids = _as_list(boxes.cls)
                raw_track_ids = getattr(boxes, "id", None)
                track_ids: list[object]
                if raw_track_ids is None:
                    track_ids = [None] * len(coordinates)
                else:
                    track_ids = _as_list(raw_track_ids)
                for coordinates_row, confidence, class_id, track_id in zip(
                    coordinates, confidences, class_ids, track_ids, strict=True
                ):
                    if not isinstance(coordinates_row, Sequence):
                        raise TypeError("model box coordinates are not a sequence")
                    coordinate_values = tuple(
                        round(_number(value)) for value in coordinates_row
                    )
                    if len(coordinate_values) != 4:
                        raise ValueError("model box must contain four coordinates")
                    numeric_class_id = int(_number(class_id))
                    x1, y1, x2, y2 = coordinate_values
                    detections.append(
                        Detection(
                            class_id=numeric_class_id,
                            label=self._names[numeric_class_id],
                            confidence=_number(confidence),
                            xyxy=(x1, y1, x2, y2),
                            track_id=None
                            if track_id is None
                            else int(_number(track_id)),
                        )
                    )
        return DetectionBatch(tuple(detections), elapsed, self.device)

    def predict(self, frame: Frame) -> DetectionBatch:
        return self._run(frame, track=False)

    def track(
        self, frame: Frame, *, tracker_config: str = "bytetrack.yaml"
    ) -> DetectionBatch:
        """Run Ultralytics tracking while retaining state between frames."""

        return self._run(frame, track=True, tracker_config=tracker_config)


def benchmark_detector(
    detector: YoloDetector,
    frame: Frame,
    *,
    warmup_runs: int,
    measured_runs: int,
) -> BenchmarkResult:
    if warmup_runs < 0 or measured_runs <= 0:
        raise ValueError("warmup runs must be nonnegative and measured runs positive")
    for _ in range(warmup_runs):
        detector.predict(frame)
    samples = [
        detector.predict(frame).elapsed_seconds * 1000 for _ in range(measured_runs)
    ]
    mean_ms = statistics.fmean(samples)
    return BenchmarkResult(
        device=detector.device,
        runs=measured_runs,
        mean_ms=mean_ms,
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        fps=1000 / mean_ms if mean_ms > 0 else 0.0,
    )


COLORS: dict[str, tuple[int, int, int]] = {
    "person": (60, 220, 255),
    "bicycle": (255, 180, 60),
    "car": (80, 220, 80),
    "motorcycle": (255, 80, 220),
    "bus": (60, 150, 255),
    "truck": (80, 80, 255),
}


def annotate_detections(frame: Frame, detections: Iterable[Detection]) -> Frame:
    annotated = frame.copy()
    height, width = annotated.shape[:2]
    for detection in detections:
        x1, y1, x2, y2 = detection.xyxy
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width - 1))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height - 1))
        color = COLORS.get(detection.label, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        track = "" if detection.track_id is None else f" #{detection.track_id}"
        text = f"{detection.label}{track} {detection.confidence:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 6)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(width - 1, x1 + text_width + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 4, max(text_height + 1, y1 - baseline - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated
