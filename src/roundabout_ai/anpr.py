"""Quality gates, normalization, and multi-frame consensus for local ANPR."""

from __future__ import annotations

import re
import statistics
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import cast

import cv2
import numpy as np

from roundabout_ai.capture import Frame


class PlateStatus(StrEnum):
    ACCEPTED = "accepted"
    UNCERTAIN = "uncertain"
    NO_READ = "no_read"


@dataclass(frozen=True, slots=True)
class PlateQualityPolicy:
    minimum_width: int = 64
    minimum_height: int = 16
    minimum_sharpness: float = 40.0
    minimum_aspect_ratio: float = 2.0
    maximum_aspect_ratio: float = 7.0
    maximum_skew_degrees: float = 15.0
    edge_margin: int = 2
    padding_ratio: float = 0.05

    def __post_init__(self) -> None:
        if self.minimum_width <= 0 or self.minimum_height <= 0:
            raise ValueError("minimum plate dimensions must be positive")
        if self.minimum_sharpness < 0:
            raise ValueError("minimum sharpness must be nonnegative")
        if not 0 < self.minimum_aspect_ratio < self.maximum_aspect_ratio:
            raise ValueError("plate aspect-ratio range is invalid")
        if not 0 <= self.maximum_skew_degrees <= 45:
            raise ValueError("maximum skew must be between 0 and 45 degrees")
        if self.edge_margin < 0 or self.padding_ratio < 0:
            raise ValueError("edge margin and padding must be nonnegative")


@dataclass(frozen=True, slots=True)
class PlateQuality:
    width: int
    height: int
    sharpness: float
    aspect_ratio: float
    skew_degrees: float | None
    clipped: bool
    accepted: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PlateCandidate:
    vehicle_id: str
    image_id: str
    detector_confidence: float
    crop: Frame
    quality: PlateQuality


@dataclass(frozen=True, slots=True)
class OcrObservation:
    image_id: str
    raw_text: str
    normalized_text: str
    confidence: float
    format_valid: bool


@dataclass(frozen=True, slots=True)
class ConsensusPolicy:
    minimum_confidence: float = 0.5
    minimum_agreement: int = 2
    minimum_length: int = 4
    require_uk_format: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.minimum_confidence <= 1:
            raise ValueError("minimum OCR confidence must be between 0 and 1")
        if self.minimum_agreement <= 0 or self.minimum_length <= 0:
            raise ValueError("agreement and text length must be positive")


@dataclass(frozen=True, slots=True)
class TrackPlateResult:
    vehicle_id: str
    status: PlateStatus
    plate_text: str
    confidence: float | None
    agreement: int
    observation_count: int
    format_valid: bool
    reasons: tuple[str, ...]
    observations: tuple[OcrObservation, ...]

    def as_dict(self, *, include_text: bool) -> dict[str, object]:
        observations: list[dict[str, object]] = []
        for item in self.observations:
            observation: dict[str, object] = {
                "image_id": item.image_id,
                "confidence": item.confidence,
                "format_valid": item.format_valid,
            }
            if include_text:
                observation.update(
                    raw_text=item.raw_text,
                    normalized_text=item.normalized_text,
                )
            observations.append(observation)
        result: dict[str, object] = {
            "vehicle_id": self.vehicle_id,
            "status": self.status.value,
            "confidence": self.confidence,
            "agreement": self.agreement,
            "observation_count": self.observation_count,
            "format_valid": self.format_valid,
            "reasons": list(self.reasons),
            "observations": observations,
        }
        if include_text:
            result["plate_text"] = self.plate_text
        return result


_UK_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"^[A-Z]{2}[0-9]{2}[A-Z]{3}$",  # current, e.g. AB12CDE
        r"^[A-Z][0-9]{1,3}[A-Z]{3}$",  # prefix
        r"^[A-Z]{3}[0-9]{1,3}[A-Z]$",  # suffix
        r"^[A-Z]{1,3}[0-9]{1,4}$",  # dateless letters first
        r"^[0-9]{1,4}[A-Z]{1,3}$",  # dateless numbers first
    )
)


def normalize_plate_text(value: str) -> str:
    """Return uppercase ASCII letters and digits only."""

    ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore")
    return re.sub(r"[^A-Z0-9]", "", ascii_value.decode("ascii").upper())


def matches_uk_plate_format(value: str) -> bool:
    """Apply common UK layouts as a weak plausibility check."""

    normalized = normalize_plate_text(value)
    return any(pattern.fullmatch(normalized) for pattern in _UK_PATTERNS)


def estimate_skew_degrees(image: Frame) -> float | None:
    """Estimate horizontal skew from strong line segments in a plate crop."""

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 180)
    minimum_length = max(12, image.shape[1] // 4)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(12, image.shape[1] // 8),
        minLineLength=minimum_length,
        maxLineGap=max(3, image.shape[1] // 20),
    )
    if lines is None:
        return None
    angles: list[float] = []
    for raw_line in lines:
        flattened = np.asarray(raw_line).reshape(-1)
        if len(flattened) != 4:
            continue
        x1, y1, x2, y2 = (int(value) for value in flattened)
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        while angle <= -90:
            angle += 180
        while angle > 90:
            angle -= 180
        if abs(angle) <= 45:
            angles.append(angle)
    return statistics.median(angles) if angles else None


def _deskew(image: Frame, skew_degrees: float | None) -> Frame:
    if skew_degrees is None or abs(skew_degrees) < 0.5:
        return image.copy()
    height, width = image.shape[:2]
    transform = cv2.getRotationMatrix2D((width / 2, height / 2), skew_degrees, 1.0)
    return cast(
        Frame,
        cv2.warpAffine(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE,
        ),
    )


def prepare_plate_crop(image: Frame, *, target_height: int = 64) -> Frame:
    """Deskew and enlarge a plate crop without inventing character detail."""

    if target_height <= 0:
        raise ValueError("target height must be positive")
    skew = estimate_skew_degrees(image)
    prepared = _deskew(image, skew)
    height, width = prepared.shape[:2]
    if height >= target_height:
        return prepared
    scale = target_height / height
    return cast(
        Frame,
        cv2.resize(
            prepared,
            (max(1, round(width * scale)), target_height),
            interpolation=cv2.INTER_CUBIC,
        ),
    )


def extract_plate_candidate(
    vehicle_id: str,
    image_id: str,
    frame: Frame,
    xyxy: tuple[int, int, int, int],
    detector_confidence: float,
    policy: PlateQualityPolicy = PlateQualityPolicy(),
) -> PlateCandidate | None:
    """Crop one detected plate and attach explicit quality evidence."""

    frame_height, frame_width = frame.shape[:2]
    raw_x1, raw_y1, raw_x2, raw_y2 = xyxy
    raw_width = raw_x2 - raw_x1
    raw_height = raw_y2 - raw_y1
    if raw_width < 2 or raw_height < 2:
        return None
    pad_x = round(raw_width * policy.padding_ratio)
    pad_y = round(raw_height * policy.padding_ratio)
    intended = (
        raw_x1 - pad_x,
        raw_y1 - pad_y,
        raw_x2 + pad_x,
        raw_y2 + pad_y,
    )
    x1 = max(0, min(frame_width, intended[0]))
    y1 = max(0, min(frame_height, intended[1]))
    x2 = max(0, min(frame_width, intended[2]))
    y2 = max(0, min(frame_height, intended[3]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    crop = frame[y1:y2, x1:x2].copy()
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    aspect_ratio = raw_width / raw_height
    skew = estimate_skew_degrees(crop)
    clipped = (
        intended != (x1, y1, x2, y2)
        or raw_x1 <= policy.edge_margin
        or raw_y1 <= policy.edge_margin
        or raw_x2 >= frame_width - policy.edge_margin
        or raw_y2 >= frame_height - policy.edge_margin
    )
    reasons: list[str] = []
    if raw_width < policy.minimum_width or raw_height < policy.minimum_height:
        reasons.append("too_small")
    if sharpness < policy.minimum_sharpness:
        reasons.append("blurred")
    if not policy.minimum_aspect_ratio <= aspect_ratio <= policy.maximum_aspect_ratio:
        reasons.append("implausible_aspect_ratio")
    if clipped:
        reasons.append("clipped")
    if skew is not None and abs(skew) > policy.maximum_skew_degrees:
        reasons.append("highly_skewed")
    quality = PlateQuality(
        width=raw_width,
        height=raw_height,
        sharpness=sharpness,
        aspect_ratio=aspect_ratio,
        skew_degrees=skew,
        clipped=clipped,
        accepted=not reasons,
        reasons=tuple(reasons),
    )
    return PlateCandidate(
        vehicle_id=vehicle_id,
        image_id=image_id,
        detector_confidence=detector_confidence,
        crop=crop,
        quality=quality,
    )


def build_consensus(
    vehicle_id: str,
    observations: Sequence[OcrObservation],
    policy: ConsensusPolicy = ConsensusPolicy(),
) -> TrackPlateResult:
    """Accept text only after exact agreement across several good frames."""

    eligible = tuple(
        item
        for item in observations
        if len(item.normalized_text) >= policy.minimum_length
        and item.confidence >= policy.minimum_confidence
    )
    if not eligible:
        return TrackPlateResult(
            vehicle_id,
            PlateStatus.NO_READ,
            "",
            None,
            0,
            len(observations),
            False,
            ("no_observation_met_text_and_confidence_thresholds",),
            tuple(observations),
        )

    counts = Counter(item.normalized_text for item in eligible)
    highest_count = max(counts.values())
    leaders = sorted(text for text, count in counts.items() if count == highest_count)
    selected_text = leaders[0]
    selected = tuple(item for item in eligible if item.normalized_text == selected_text)
    confidence = statistics.fmean(item.confidence for item in selected)
    format_valid = matches_uk_plate_format(selected_text)
    reasons: list[str] = []
    if len(leaders) > 1:
        reasons.append("conflicting_observations")
    if highest_count < policy.minimum_agreement:
        reasons.append("insufficient_multi_frame_agreement")
    if policy.require_uk_format and not format_valid:
        reasons.append("uk_format_check_failed")
    status = PlateStatus.ACCEPTED if not reasons else PlateStatus.UNCERTAIN
    return TrackPlateResult(
        vehicle_id,
        status,
        selected_text,
        confidence,
        highest_count,
        len(observations),
        format_valid,
        tuple(reasons),
        tuple(observations),
    )


def character_accuracy(expected: str, observed: str) -> float:
    """Return normalized Levenshtein character accuracy in the range [0, 1]."""

    left = normalize_plate_text(expected)
    right = normalize_plate_text(observed)
    denominator = max(len(left), len(right))
    if denominator == 0:
        return 1.0
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return max(0.0, 1 - previous[-1] / denominator)


@dataclass(frozen=True, slots=True)
class EvaluationMetrics:
    vehicle_count: int
    exact_matches: int
    wrong_reads: int
    no_reads: int
    exact_plate_accuracy: float
    mean_character_accuracy: float
    no_read_rate: float
    false_read_rate: float


def evaluate_results(
    expected_by_vehicle: dict[str, str],
    results: Iterable[TrackPlateResult],
) -> EvaluationMetrics:
    """Measure accepted results against held-out vehicle-level ground truth."""

    result_by_vehicle = {result.vehicle_id: result for result in results}
    exact_matches = 0
    wrong_reads = 0
    no_reads = 0
    character_scores: list[float] = []
    for vehicle_id, expected in expected_by_vehicle.items():
        result = result_by_vehicle.get(vehicle_id)
        observed = "" if result is None else result.plate_text
        accepted = result is not None and result.status is PlateStatus.ACCEPTED
        if not accepted:
            no_reads += 1
            observed = ""
        elif normalize_plate_text(observed) == normalize_plate_text(expected):
            exact_matches += 1
        else:
            wrong_reads += 1
        character_scores.append(character_accuracy(expected, observed))
    count = len(expected_by_vehicle)
    return EvaluationMetrics(
        vehicle_count=count,
        exact_matches=exact_matches,
        wrong_reads=wrong_reads,
        no_reads=no_reads,
        exact_plate_accuracy=exact_matches / count if count else 0.0,
        mean_character_accuracy=statistics.fmean(character_scores)
        if character_scores
        else 0.0,
        no_read_rate=no_reads / count if count else 0.0,
        false_read_rate=wrong_reads / count if count else 0.0,
    )
