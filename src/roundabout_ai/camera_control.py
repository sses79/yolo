"""Typed, bounded control client for the Android IP Webcam HTTP API."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

ALLOWED_SETTINGS = frozenset(
    {
        "antibanding",
        "exposure_ns",
        "focus_distance",
        "focusmode",
        "frame_duration",
        "iso",
        "manual_sensor",
        "night_vision",
        "night_vision_average",
        "night_vision_gain",
        "quality",
        "scenemode",
        "video_size",
        "whitebalance",
    }
)
NUMERIC_SETTINGS = frozenset(
    {
        "exposure_ns",
        "focus_distance",
        "frame_duration",
        "iso",
        "night_vision_average",
        "night_vision_gain",
        "quality",
    }
)


class CameraControlError(RuntimeError):
    """The camera rejected a safe control operation or returned invalid state."""


@dataclass(frozen=True, slots=True)
class CameraCapabilities:
    current: Mapping[str, str]
    available: Mapping[str, object]

    @classmethod
    def from_payload(cls, payload: object) -> CameraCapabilities:
        if not isinstance(payload, dict):
            raise CameraControlError("camera status must be a JSON object")
        current = payload.get("curvals")
        available = payload.get("avail")
        if not isinstance(current, dict) or not isinstance(available, dict):
            raise CameraControlError("camera status is missing curvals or avail")
        return cls(
            {str(key): str(value) for key, value in current.items()},
            {str(key): value for key, value in available.items()},
        )

    def validate(self, name: str, value: str | int | float) -> str:
        if name not in ALLOWED_SETTINGS:
            raise CameraControlError(f"camera setting is not allowlisted: {name}")
        if name not in self.available:
            raise CameraControlError(f"camera does not report setting: {name}")
        requested = str(value)
        available = self.available[name]
        if name in NUMERIC_SETTINGS:
            number = _number(requested, name)
            bounds = _numeric_bounds(available)
            if bounds is not None and not bounds[0] <= number <= bounds[1]:
                raise CameraControlError(
                    f"{name}={requested} is outside camera range "
                    f"{bounds[0]:g}..{bounds[1]:g}"
                )
        else:
            choices = _choices(available)
            if choices and requested not in choices:
                raise CameraControlError(
                    f"camera does not support {name}={requested}; "
                    f"available: {', '.join(choices)}"
                )
        return requested

    def snapshot(self, settings: Sequence[str]) -> dict[str, str]:
        return {name: self.current[name] for name in settings if name in self.current}


@dataclass(frozen=True, slots=True)
class CameraPreset:
    name: str
    settings: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class CameraApplyResult:
    profile: str
    previous: Mapping[str, str]
    applied: Mapping[str, str]
    requires_reconnect: bool


PostJson = Callable[[str, float], object]
Post = Callable[[str, float], None]


def _default_post_json(url: str, timeout: float) -> object:
    request = Request(url, data=b"", method="POST")
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _default_post(url: str, timeout: float) -> None:
    request = Request(url, data=b"", method="POST")
    with urlopen(request, timeout=timeout) as response:
        response.read()


class IpWebcamControlClient:
    """Discover, validate, apply, verify, and roll back camera settings."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 2.0,
        post_json: PostJson = _default_post_json,
        post: Post = _default_post,
    ) -> None:
        if not base_url.strip():
            raise ValueError("camera control URL must not be empty")
        if timeout_seconds <= 0:
            raise ValueError("camera control timeout must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._post_json = post_json
        self._post = post

    def capabilities(self) -> CameraCapabilities:
        try:
            payload = self._post_json(
                f"{self.base_url}/status.json?show_avail=1", self.timeout_seconds
            )
        except (OSError, TimeoutError, ValueError) as exc:
            raise CameraControlError(
                f"could not read camera capabilities: {exc}"
            ) from exc
        return CameraCapabilities.from_payload(payload)

    def save_capabilities(self, path: Path) -> CameraCapabilities:
        capabilities = self.capabilities()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "control_url": self.base_url,
                    "curvals": dict(capabilities.current),
                    "avail": dict(capabilities.available),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return capabilities

    def apply_preset(
        self,
        preset: CameraPreset,
        *,
        capabilities: CameraCapabilities | None = None,
    ) -> CameraApplyResult:
        before = capabilities or self.capabilities()
        requested = {
            name: before.validate(name, value) for name, value in preset.settings
        }
        previous = before.snapshot(tuple(requested))
        applied: dict[str, str] = {}
        try:
            for name, value in requested.items():
                self._set(name, value)
                applied[name] = value
            after = self.capabilities()
            for name, value in requested.items():
                actual = after.current.get(name)
                if actual is None or not _equivalent(actual, value, name):
                    raise CameraControlError(
                        f"camera read-back mismatch for {name}: "
                        f"requested {value}, got {actual or 'missing'}"
                    )
        except Exception as exc:
            rollback_error = self._restore(previous, before)
            if rollback_error is not None:
                raise CameraControlError(
                    f"camera preset {preset.name} failed ({exc}); "
                    f"rollback also failed ({rollback_error})"
                ) from exc
            if isinstance(exc, CameraControlError):
                raise
            raise CameraControlError(
                f"camera rejected preset {preset.name}: {exc}"
            ) from exc
        return CameraApplyResult(
            preset.name,
            previous,
            requested,
            "video_size" in requested,
        )

    def rollback(self, settings: Mapping[str, str]) -> CameraApplyResult:
        preset = CameraPreset("rollback", tuple(settings.items()))
        return self.apply_preset(preset)

    def _set(self, name: str, value: str) -> None:
        url = (
            f"{self.base_url}/settings/{quote(name, safe='')}?"
            f"{urlencode({'set': value})}"
        )
        try:
            self._post(url, self.timeout_seconds)
        except (OSError, TimeoutError) as exc:
            raise CameraControlError(f"could not set {name}: {exc}") from exc

    def _restore(
        self, previous: Mapping[str, str], capabilities: CameraCapabilities
    ) -> str | None:
        for name, value in reversed(tuple(previous.items())):
            try:
                capabilities.validate(name, value)
                self._set(name, value)
            except CameraControlError as exc:
                return str(exc)
        try:
            restored = self.capabilities()
        except CameraControlError as exc:
            return str(exc)
        for name, value in previous.items():
            actual = restored.current.get(name)
            if actual is None or not _equivalent(actual, value, name):
                return (
                    f"read-back mismatch for {name}: requested {value}, "
                    f"got {actual or 'missing'}"
                )
        return None


def _choices(value: object) -> tuple[str, ...]:
    if isinstance(value, dict):
        choices = value.get("values") or value.get("choices")
        return _choices(choices)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    return ()


def _numeric_bounds(value: object) -> tuple[float, float] | None:
    if isinstance(value, dict):
        minimum = value.get("min")
        maximum = value.get("max")
        if minimum is not None and maximum is not None:
            return float(minimum), float(maximum)
        value = value.get("values") or value.get("choices")
    if isinstance(value, (list, tuple)) and value:
        try:
            numbers = tuple(float(item) for item in value)
        except TypeError, ValueError:
            return None
        return min(numbers), max(numbers)
    return None


def _number(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError as exc:
        raise CameraControlError(f"{name} must be numeric") from exc


def _equivalent(actual: str, requested: str, name: str) -> bool:
    if name not in NUMERIC_SETTINGS:
        return actual == requested
    try:
        expected_number = float(requested)
        tolerance = max(abs(expected_number) * 0.001, 1e-6)
        return abs(float(actual) - expected_number) <= tolerance
    except ValueError:
        return actual == requested


CAMERA_PRESETS: Mapping[str, CameraPreset] = {
    "day": CameraPreset(
        "day",
        (
            ("manual_sensor", "off"),
            ("night_vision", "off"),
            ("focusmode", "continuous-video"),
            ("scenemode", "action"),
            ("whitebalance", "auto"),
        ),
    ),
    "glare": CameraPreset(
        "glare",
        (
            ("manual_sensor", "off"),
            ("night_vision", "off"),
            ("focusmode", "continuous-video"),
            ("scenemode", "beach"),
            ("whitebalance", "daylight"),
        ),
    ),
    "dusk": CameraPreset(
        "dusk",
        (
            ("manual_sensor", "off"),
            ("night_vision", "off"),
            ("focusmode", "continuous-video"),
            ("scenemode", "sports"),
            ("whitebalance", "auto"),
        ),
    ),
    "night": CameraPreset(
        "night",
        (
            ("manual_sensor", "off"),
            ("night_vision", "off"),
            ("focusmode", "continuous-video"),
            ("scenemode", "night"),
            ("whitebalance", "auto"),
        ),
    ),
}
