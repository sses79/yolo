"""Command-line entry point for the gated, offline Phase 5 ANPR prototype."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np

from roundabout_ai.anpr import (
    ConsensusPolicy,
    PlateQualityPolicy,
    evaluate_results,
    normalize_plate_text,
)
from roundabout_ai.anpr_feasibility import (
    Recommendation,
    assess_feasibility,
    load_samples,
)
from roundabout_ai.anpr_pipeline import (
    VehicleAnalysis,
    analyze_groups,
    discover_vehicle_groups,
)
from roundabout_ai.detector import YoloDetector
from roundabout_ai.ocr import RapidOcrRecognizer

LABEL_FIELDS = ("vehicle_id", "expected_text")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _check_feasibility(path: Path, *, allow_incomplete: bool) -> str:
    if not path.exists():
        if allow_incomplete:
            return "missing (experimental override)"
        raise ValueError(
            f"Phase 4 annotations do not exist: {path}; run "
            "roundabout-anpr-assess first"
        )
    assessment = assess_feasibility(load_samples(path))
    recommendation = assessment.recommendation
    if recommendation is Recommendation.PROCEED:
        return recommendation.value
    if allow_incomplete and recommendation is Recommendation.INCOMPLETE_SAMPLE:
        return f"{recommendation.value} (experimental override)"
    raise ValueError(
        f"Phase 4 recommendation is {recommendation.value}; improve the camera/sample "
        "before OCR. Only incomplete_sample can be overridden for experimentation."
    )


def _policies(args: argparse.Namespace) -> tuple[PlateQualityPolicy, ConsensusPolicy]:
    return (
        PlateQualityPolicy(
            minimum_width=args.minimum_plate_width,
            minimum_height=args.minimum_plate_height,
            minimum_sharpness=args.minimum_sharpness,
            maximum_skew_degrees=args.maximum_skew,
        ),
        ConsensusPolicy(
            minimum_confidence=args.minimum_ocr_confidence,
            minimum_agreement=args.minimum_agreement,
            require_uk_format=not args.no_uk_format_check,
        ),
    )


def _run_analysis(args: argparse.Namespace) -> tuple[str, tuple[VehicleAnalysis, ...]]:
    gate = _check_feasibility(
        args.feasibility_annotations,
        allow_incomplete=args.allow_incomplete_feasibility,
    )
    groups = discover_vehicle_groups(args.images)
    if not groups:
        raise ValueError(f"no Phase 3 event images found under {args.images}")
    detector = YoloDetector(
        str(args.plate_model),
        confidence=args.detector_confidence,
        image_size=args.image_size,
        device=args.device,
        class_names=(args.plate_class,),
    )
    recognizer = RapidOcrRecognizer()
    quality_policy, consensus_policy = _policies(args)
    analyses = analyze_groups(
        groups,
        detector,
        recognizer,
        quality_policy=quality_policy,
        consensus_policy=consensus_policy,
        maximum_ocr_candidates=args.maximum_ocr_candidates,
    )
    return gate, analyses


def _payload(
    gate: str, analyses: tuple[VehicleAnalysis, ...], *, include_text: bool
) -> dict[str, object]:
    counts: dict[str, int] = {}
    vehicles: list[dict[str, object]] = []
    for analysis in analyses:
        status = analysis.result.status.value
        counts[status] = counts.get(status, 0) + 1
        vehicles.append(
            {
                "result": analysis.result.as_dict(include_text=include_text),
                "image_count": analysis.image_count,
                "plate_detections": analysis.detections,
                "accepted_candidates": analysis.accepted_candidates,
                "rejection_reasons": analysis.rejection_reasons,
            }
        )
    return {
        "privacy": "plate text included by explicit opt-in"
        if include_text
        else "plate text redacted",
        "feasibility_gate": gate,
        "vehicle_count": len(analyses),
        "status_counts": counts,
        "vehicles": vehicles,
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_evaluation_labels(path: Path) -> dict[str, str]:
    """Read local vehicle-level ground truth used only by the evaluate command."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != LABEL_FIELDS:
            raise ValueError(f"labels must have header: {','.join(LABEL_FIELDS)}")
        labels: dict[str, str] = {}
        for row_number, row in enumerate(reader, start=2):
            vehicle_id = row["vehicle_id"].strip()
            expected = normalize_plate_text(row["expected_text"])
            if not vehicle_id or not expected:
                raise ValueError(f"row {row_number}: both fields are required")
            if vehicle_id in labels:
                raise ValueError(f"row {row_number}: duplicate vehicle_id {vehicle_id}")
            labels[vehicle_id] = expected
    return labels


def _command_run(args: argparse.Namespace) -> int:
    gate, analyses = _run_analysis(args)
    payload = _payload(gate, analyses, include_text=args.store_text)
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        _write_json(args.output, payload)
        print(f"Local report written: {args.output}")
    return 0


def _command_evaluate(args: argparse.Namespace) -> int:
    labels = load_evaluation_labels(args.labels)
    gate, analyses = _run_analysis(args)
    metrics = evaluate_results(labels, (item.result for item in analyses))
    payload: dict[str, object] = {
        "privacy": "evaluation report contains aggregate metrics only",
        "feasibility_gate": gate,
        "metrics": asdict(metrics),
    }
    print(json.dumps(payload, indent=2))
    if args.output is not None:
        _write_json(args.output, payload)
        print(f"Aggregate evaluation written: {args.output}")
    return 0


def _command_smoke_test(_: argparse.Namespace) -> int:
    image = np.full((64, 260, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "AB12CDE",
        (8, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.25,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    observation = RapidOcrRecognizer().recognize(image, image_id="smoke-test")
    print(
        json.dumps(
            {
                "rapidocr": version("rapidocr"),
                "onnxruntime": version("onnxruntime"),
                "recognized": observation.normalized_text == "AB12CDE",
                "confidence": observation.confidence,
            },
            indent=2,
        )
    )
    return 0 if observation.normalized_text == "AB12CDE" else 1


def _add_analysis_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--images", type=Path, default=Path("data/events/images"))
    parser.add_argument("--plate-model", type=Path, required=True)
    parser.add_argument("--plate-class", default="license_plate")
    parser.add_argument(
        "--feasibility-annotations",
        type=Path,
        default=Path("data/anpr/annotations.csv"),
    )
    parser.add_argument("--allow-incomplete-feasibility", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="cpu")
    parser.add_argument("--detector-confidence", type=float, default=0.35)
    parser.add_argument("--image-size", type=positive_int, default=640)
    parser.add_argument("--minimum-plate-width", type=positive_int, default=64)
    parser.add_argument("--minimum-plate-height", type=positive_int, default=16)
    parser.add_argument("--minimum-sharpness", type=float, default=40.0)
    parser.add_argument("--maximum-skew", type=float, default=15.0)
    parser.add_argument("--minimum-ocr-confidence", type=float, default=0.5)
    parser.add_argument("--minimum-agreement", type=positive_int, default=2)
    parser.add_argument("--maximum-ocr-candidates", type=positive_int, default=3)
    parser.add_argument("--no-uk-format-check", action="store_true")
    parser.add_argument("--output", type=Path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Gated offline ANPR over saved vehicle crops."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke-test", help="test the local OCR runtime")
    smoke.set_defaults(handler=_command_smoke_test)

    run = subparsers.add_parser("run", help="analyze local event images")
    _add_analysis_arguments(run)
    run.add_argument(
        "--store-text",
        action="store_true",
        help="opt in to showing and storing registration text locally",
    )
    run.set_defaults(handler=_command_run)

    evaluate = subparsers.add_parser(
        "evaluate", help="calculate held-out aggregate metrics"
    )
    _add_analysis_arguments(evaluate)
    evaluate.add_argument("--labels", type=Path, required=True)
    evaluate.set_defaults(handler=_command_evaluate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
