"""Load, scale, filter, and draw roundabout scene calibration."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from roundabout_ai.capture import Frame
from roundabout_ai.detector import Detection
from roundabout_ai.geometry import CountLine, Point, TrackObservation, point_in_polygon


@dataclass(frozen=True, slots=True)
class Scene:
    reference_size: tuple[int, int]
    roi: tuple[Point, ...]
    count_lines: tuple[CountLine, ...]

    def scaled(self, width: int, height: int) -> Scene:
        reference_width, reference_height = self.reference_size
        scale_x = width / reference_width
        scale_y = height / reference_height

        def scale(point: Point) -> Point:
            return point[0] * scale_x, point[1] * scale_y

        return Scene(
            (width, height),
            tuple(scale(point) for point in self.roi),
            tuple(
                CountLine(
                    line.name,
                    scale(line.start),
                    scale(line.end),
                    line.negative_to_positive,
                    line.positive_to_negative,
                )
                for line in self.count_lines
            ),
        )


def _point(value: object, field_name: str) -> Point:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{field_name} must be an [x, y] point")
    return float(value[0]), float(value[1])


def load_scene(path: Path) -> Scene:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("scene configuration must be a YAML mapping")
    size = data.get("reference_size")
    if not isinstance(size, (list, tuple)) or len(size) != 2:
        raise ValueError("reference_size must be [width, height]")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("reference_size values must be positive")
    roi = tuple(_point(value, "roi") for value in data.get("roi", ()))
    if roi and len(roi) < 3:
        raise ValueError("roi must be empty or contain at least three points")
    raw_lines = data.get("count_lines", ())
    if not isinstance(raw_lines, list):
        raise ValueError("count_lines must be a list")
    lines: list[CountLine] = []
    names: set[str] = set()
    for index, raw in enumerate(raw_lines, start=1):
        if not isinstance(raw, dict):
            raise ValueError("each count line must be a mapping")
        name = str(raw.get("name", f"line_{index}"))
        if name in names:
            raise ValueError(f"duplicate count line name: {name}")
        names.add(name)
        lines.append(
            CountLine(
                name=name,
                start=_point(raw.get("start"), f"{name}.start"),
                end=_point(raw.get("end"), f"{name}.end"),
                negative_to_positive=str(
                    raw.get("negative_to_positive", "negative_to_positive")
                ),
                positive_to_negative=str(
                    raw.get("positive_to_negative", "positive_to_negative")
                ),
            )
        )
    return Scene((width, height), roi, tuple(lines))


def save_scene(scene: Scene, path: Path) -> None:
    data: dict[str, Any] = {
        "reference_size": list(scene.reference_size),
        "roi": [list(point) for point in scene.roi],
        "count_lines": [
            {
                "name": line.name,
                "start": list(line.start),
                "end": list(line.end),
                "negative_to_positive": line.negative_to_positive,
                "positive_to_negative": line.positive_to_negative,
            }
            for line in scene.count_lines
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def detection_centre(detection: Detection) -> Point:
    x1, y1, x2, y2 = detection.xyxy
    return (x1 + x2) / 2, (y1 + y2) / 2


def filter_detections_by_roi(
    detections: Iterable[Detection], roi: tuple[Point, ...]
) -> tuple[Detection, ...]:
    if not roi:
        return tuple(detections)
    return tuple(
        detection
        for detection in detections
        if point_in_polygon(detection_centre(detection), roi)
    )


def track_observations(detections: Iterable[Detection]) -> tuple[TrackObservation, ...]:
    return tuple(
        TrackObservation(
            detection.track_id,
            detection.label,
            detection.confidence,
            detection_centre(detection),
        )
        for detection in detections
        if detection.track_id is not None
    )


def annotate_scene(frame: Frame, scene: Scene) -> Frame:
    annotated = frame.copy()
    if scene.roi:
        points = [(round(x), round(y)) for x, y in scene.roi]
        cv2.polylines(annotated, [np.array(points)], True, (0, 255, 255), 2)
    for line in scene.count_lines:
        start = (round(line.start[0]), round(line.start[1]))
        end = (round(line.end[0]), round(line.end[1]))
        cv2.line(annotated, start, end, (0, 80, 255), 3)
        cv2.putText(
            annotated,
            line.name,
            start,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 80, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated
