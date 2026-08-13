from __future__ import annotations

import pytest

from roundabout_ai.detection_app import build_parser, format_detection_metrics


def test_detection_defaults_match_phase_one_plan() -> None:
    args = build_parser().parse_args([])

    assert args.model == "yolo26n.pt"
    assert args.device == "cpu"
    assert args.confidence == 0.35
    assert args.image_size == 640
    assert args.snapshot_on_detection is False
    assert args.classes == (
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "bus",
        "truck",
    )


def test_detection_parser_validates_confidence_and_devices() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--confidence", "1.1"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--benchmark-devices", "cuda"])


def test_detection_metrics_include_performance_and_counts() -> None:
    metrics = format_detection_metrics(
        capture_fps=30.0,
        inference_fps=12.5,
        inference_ms=80.0,
        frame_age_ms=95.0,
        received=100,
        processed=40,
        overwritten=59,
        device="mps",
        counts={"person": 1, "car": 2},
    )

    assert "device=mps" in metrics
    assert "inference_fps=12.5" in metrics
    assert "frame_age=95.0ms" in metrics
    assert "objects=car:2,person:1" in metrics
