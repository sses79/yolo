from __future__ import annotations

import cv2
import numpy as np
import pytest

from roundabout_ai.anpr import (
    OcrObservation,
    PlateQualityPolicy,
    PlateStatus,
    TrackPlateResult,
    build_consensus,
    character_accuracy,
    evaluate_results,
    extract_plate_candidate,
    matches_uk_plate_format,
    normalize_plate_text,
)


def observation(text: str, confidence: float = 0.9) -> OcrObservation:
    normalized = normalize_plate_text(text)
    return OcrObservation(
        image_id=f"{text}.jpg",
        raw_text=text,
        normalized_text=normalized,
        confidence=confidence,
        format_valid=matches_uk_plate_format(normalized),
    )


def test_normalizes_and_weakly_validates_uk_plate_text() -> None:
    assert normalize_plate_text(" ab12 c-dé ") == "AB12CDE"
    assert matches_uk_plate_format("AB12 CDE")
    assert not matches_uk_plate_format("NOT-A-PLATE")


def test_candidate_quality_rejects_small_blurred_and_clipped_plate() -> None:
    frame = np.full((80, 180, 3), 127, dtype=np.uint8)
    candidate = extract_plate_candidate(
        "vehicle-1",
        "image.jpg",
        frame,
        (0, 10, 45, 24),
        0.8,
    )
    assert candidate is not None
    assert not candidate.quality.accepted
    assert set(candidate.quality.reasons) >= {"too_small", "blurred", "clipped"}


def test_candidate_quality_accepts_a_sharp_plate() -> None:
    frame = np.full((100, 240, 3), 255, dtype=np.uint8)
    cv2.putText(
        frame,
        "AB12CDE",
        (52, 61),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        2,
    )
    candidate = extract_plate_candidate(
        "vehicle-1",
        "image.jpg",
        frame,
        (45, 35, 190, 70),
        0.9,
        PlateQualityPolicy(minimum_sharpness=10),
    )
    assert candidate is not None
    assert candidate.quality.accepted


def test_consensus_requires_exact_multi_frame_agreement() -> None:
    result = build_consensus(
        "vehicle-1",
        (observation("AB12 CDE"), observation("AB12CDE"), observation("AB12CDF")),
    )
    assert result.status is PlateStatus.ACCEPTED
    assert result.plate_text == "AB12CDE"
    assert result.agreement == 2


def test_consensus_preserves_uncertainty_and_no_read() -> None:
    uncertain = build_consensus("vehicle-1", (observation("AB12CDE"),))
    no_read = build_consensus("vehicle-2", (observation("?", 0.2),))
    assert uncertain.status is PlateStatus.UNCERTAIN
    assert "insufficient_multi_frame_agreement" in uncertain.reasons
    assert no_read.status is PlateStatus.NO_READ


def test_result_redacts_text_by_default() -> None:
    result = build_consensus(
        "vehicle-1", (observation("AB12CDE"), observation("AB12CDE"))
    )
    redacted = result.as_dict(include_text=False)
    assert "plate_text" not in redacted
    assert "normalized_text" not in redacted["observations"][0]  # type: ignore[index]


def test_evaluation_reports_exact_character_no_read_and_false_read_rates() -> None:
    accepted = lambda vehicle, text: TrackPlateResult(  # noqa: E731
        vehicle,
        PlateStatus.ACCEPTED,
        text,
        0.9,
        2,
        2,
        True,
        (),
        (),
    )
    metrics = evaluate_results(
        {"one": "AB12CDE", "two": "XY99XYZ", "three": "CD34EFG"},
        (accepted("one", "AB12CDE"), accepted("two", "XY99XYA")),
    )
    assert metrics.exact_plate_accuracy == pytest.approx(1 / 3)
    assert metrics.no_read_rate == pytest.approx(1 / 3)
    assert metrics.false_read_rate == pytest.approx(1 / 3)
    assert character_accuracy("AB12CDE", "AB12CDF") == pytest.approx(6 / 7)
