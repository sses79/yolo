"""ROI image-quality measurements and deterministic camera recommendations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import cv2
import numpy as np

from roundabout_ai.capture import Frame
from roundabout_ai.geometry import Point

ADAPTIVE_PROFILE_ORDER = ("night", "dusk", "day", "glare")
DEFAULT_MAXIMUM_UNDEREXPOSED_RATIO = 0.10
DEFAULT_MINIMUM_MOVING_SHARPNESS = 30.0


@dataclass(frozen=True, slots=True)
class FrameQuality:
    luminance_p05: float
    luminance_median: float
    luminance_p95: float
    underexposed_ratio: float
    overexposed_ratio: float
    sharpness: float
    directional_blur: float
    noise: float

    def as_dict(self) -> dict[str, float]:
        return {
            "luminance_p05": self.luminance_p05,
            "luminance_median": self.luminance_median,
            "luminance_p95": self.luminance_p95,
            "underexposed_ratio": self.underexposed_ratio,
            "overexposed_ratio": self.overexposed_ratio,
            "sharpness": self.sharpness,
            "directional_blur": self.directional_blur,
            "noise": self.noise,
        }


def measure_frame_quality(frame: Frame, roi: tuple[Point, ...] = ()) -> FrameQuality:
    if frame.size == 0:
        raise ValueError("cannot measure an empty frame")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pixels = _roi_pixels(gray, roi)
    p05, median, p95 = np.percentile(pixels, (5, 50, 95))
    measured = _masked_image(gray, roi)
    laplacian = cv2.Laplacian(measured, cv2.CV_64F)
    gradient_x = cv2.Sobel(measured, cv2.CV_64F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(measured, cv2.CV_64F, 0, 1, ksize=3)
    energy_x = float(np.mean(np.abs(gradient_x)))
    energy_y = float(np.mean(np.abs(gradient_y)))
    noise_residual = measured.astype(np.float32) - cv2.GaussianBlur(
        measured, (3, 3), 0
    ).astype(np.float32)
    return FrameQuality(
        float(p05),
        float(median),
        float(p95),
        float(np.mean(pixels <= 16)),
        float(np.mean(pixels >= 239)),
        float(laplacian.var()),
        abs(energy_x - energy_y) / max(energy_x + energy_y, 1e-6),
        float(np.median(np.abs(noise_residual))),
    )


def classify_condition(quality: FrameQuality) -> str:
    if quality.overexposed_ratio >= 0.08 or (
        quality.luminance_p95 >= 250 and quality.luminance_median >= 100
    ):
        return "glare"
    if quality.luminance_median < 45:
        return "night"
    if quality.luminance_median < 95:
        return "dusk"
    return "day"


class CameraProfileAdvisor:
    """Require repeated observations before changing a recommendation."""

    def __init__(self, *, required_observations: int = 3) -> None:
        if required_observations <= 0:
            raise ValueError("required observations must be positive")
        self.required_observations = required_observations
        self.recommendation: str | None = None
        self._candidate: str | None = None
        self._candidate_count = 0

    def observe(self, quality: FrameQuality) -> str | None:
        candidate = classify_condition(quality)
        if candidate == self.recommendation:
            self._candidate = None
            self._candidate_count = 0
            return self.recommendation
        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_count = 1
        else:
            self._candidate_count += 1
        if self._candidate_count >= self.required_observations:
            self.recommendation = candidate
            self._candidate = None
            self._candidate_count = 0
        return self.recommendation


def guard_profile_transition(
    current_profile: str | None,
    requested_profile: str | None,
    quality: FrameQuality,
    *,
    moving_vehicle_sharpness: float | None = None,
    known_profile_underexposure: Mapping[str, float] | None = None,
    maximum_underexposed_ratio: float = DEFAULT_MAXIMUM_UNDEREXPOSED_RATIO,
    minimum_moving_sharpness: float = DEFAULT_MINIMUM_MOVING_SHARPNESS,
) -> str | None:
    """Bound automatic changes using exposure and moving-subject evidence."""

    if requested_profile is None or current_profile is None:
        return requested_profile
    if requested_profile == current_profile:
        return requested_profile
    if current_profile not in ADAPTIVE_PROFILE_ORDER:
        return requested_profile
    if requested_profile not in ADAPTIVE_PROFILE_ORDER:
        return requested_profile
    if maximum_underexposed_ratio < 0 or minimum_moving_sharpness < 0:
        raise ValueError("adaptive transition thresholds must be nonnegative")

    current_index = ADAPTIVE_PROFILE_ORDER.index(current_profile)
    requested_index = ADAPTIVE_PROFILE_ORDER.index(requested_profile)
    direction = 1 if requested_index > current_index else -1
    adjacent_profile = ADAPTIVE_PROFILE_ORDER[current_index + direction]

    # Do not repeatedly re-enter a profile that recently proved unusably dark.
    # The caller bounds the age of this evidence so ambient changes can be
    # reconsidered later.
    adjacent_underexposure = (known_profile_underexposure or {}).get(adjacent_profile)
    if (
        adjacent_underexposure is not None
        and adjacent_underexposure >= maximum_underexposed_ratio
    ):
        return current_profile

    # Moving right in the profile order reduces low-light assistance. Do not
    # make an already underexposed image darker merely because the active night
    # profile raised its median luminance into the nominal day band.
    if direction > 0 and (quality.underexposed_ratio >= maximum_underexposed_ratio):
        return current_profile

    # Moving left increases low-light assistance and can lengthen exposure.
    # Severe underexposure takes precedence; otherwise retain the faster
    # profile when recent moving vehicles are already blurred.
    if (
        direction < 0
        and quality.underexposed_ratio < maximum_underexposed_ratio
        and moving_vehicle_sharpness is not None
        and moving_vehicle_sharpness < minimum_moving_sharpness
    ):
        return current_profile

    return adjacent_profile


def _roi_pixels(gray: np.ndarray, roi: tuple[Point, ...]) -> np.ndarray:
    if not roi:
        return gray.reshape(-1)
    mask = np.zeros(gray.shape, dtype=np.uint8)
    points = np.array([(round(x), round(y)) for x, y in roi], dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    pixels = gray[mask > 0]
    if pixels.size == 0:
        raise ValueError("ROI contains no pixels")
    return pixels


def _masked_image(gray: np.ndarray, roi: tuple[Point, ...]) -> np.ndarray:
    if not roi:
        return gray
    x, y, width, height = cv2.boundingRect(
        np.array([(round(px), round(py)) for px, py in roi], dtype=np.int32)
    )
    return gray[y : y + height, x : x + width]
