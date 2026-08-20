from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from roundabout_ai.camera_control import (
    CameraCapabilities,
    CameraControlError,
    CameraPreset,
    IpWebcamControlClient,
    identify_camera_profile,
    load_validated_profile_mapping,
)


def payload() -> dict[str, object]:
    return {
        "curvals": {
            "manual_sensor": "off",
            "focusmode": "continuous-picture",
            "iso": "50",
        },
        "avail": {
            "manual_sensor": ["on", "off"],
            "focusmode": ["auto", "continuous-video", "continuous-picture"],
            "iso": {"min": 50, "max": 3200},
        },
    }


def test_capabilities_validate_allowlist_choices_and_numeric_bounds() -> None:
    capabilities = CameraCapabilities.from_payload(payload())

    assert capabilities.validate("focusmode", "continuous-video") == (
        "continuous-video"
    )
    assert capabilities.validate("iso", 800) == "800"
    with pytest.raises(CameraControlError, match="not allowlisted"):
        capabilities.validate("torch", "on")
    with pytest.raises(CameraControlError, match="does not support"):
        capabilities.validate("focusmode", "invalid")
    with pytest.raises(CameraControlError, match="outside camera range"):
        capabilities.validate("iso", 6400)


def test_client_applies_and_verifies_preset_and_saves_snapshot(
    tmp_path: Path,
) -> None:
    state = payload()
    posts: list[str] = []

    def post_json(_url: str, _timeout: float) -> object:
        return state

    def post(url: str, _timeout: float) -> None:
        posts.append(url)
        parsed = urlparse(url)
        name = parsed.path.rsplit("/", maxsplit=1)[-1]
        current = state["curvals"]
        assert isinstance(current, dict)
        current[name] = parse_qs(parsed.query)["set"][0]

    client = IpWebcamControlClient(
        "http://camera.invalid/", post_json=post_json, post=post
    )
    result = client.apply_preset(
        CameraPreset(
            "day", (("manual_sensor", "off"), ("focusmode", "continuous-video"))
        )
    )
    snapshot = tmp_path / "capabilities.json"
    client.save_capabilities(snapshot)

    assert result.profile == "day"
    assert result.previous["focusmode"] == "continuous-picture"
    assert state["curvals"]["focusmode"] == "continuous-video"  # type: ignore[index]
    assert len(posts) == 2
    assert '"continuous-video"' in snapshot.read_text(encoding="utf-8")


def test_client_rolls_back_when_readback_does_not_match() -> None:
    state = payload()
    writes: list[str] = []

    def post(_url: str, _timeout: float) -> None:
        writes.append(_url)

    client = IpWebcamControlClient(
        "http://camera.invalid", post_json=lambda _url, _timeout: state, post=post
    )

    with pytest.raises(CameraControlError, match="read-back mismatch"):
        client.apply_preset(CameraPreset("day", (("focusmode", "continuous-video"),)))

    assert len(writes) == 2


def test_identifies_verified_preset_after_worker_restart() -> None:
    current = {
        "manual_sensor": "off",
        "night_vision": "off",
        "focusmode": "continuous-video",
        "scenemode": "night",
        "whitebalance": "auto",
        "iso": "50",
    }

    assert identify_camera_profile(current) == "night"
    assert identify_camera_profile({**current, "scenemode": "portrait"}) is None


def validated_mapping_payload() -> dict[str, object]:
    return {
        "version": 1,
        "conditions": {
            condition: {
                "profile": profile,
                "sample_count": 40,
                "operator_approved": True,
                "ocr_acceptance_rate": 0.6,
                "baseline_ocr_acceptance_rate": 0.5,
                "held_out_false_read_rate": 0.02,
                "baseline_held_out_false_read_rate": 0.03,
            }
            for condition, profile in {
                "day": "day",
                "glare": "glare",
                "dusk": "day",
                "night": "dusk",
            }.items()
        },
    }


def test_loads_only_evidence_backed_operator_approved_profile_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(validated_mapping_payload()), encoding="utf-8")

    mapping = load_validated_profile_mapping(path)

    assert mapping["night"] == "dusk"
    assert mapping["dusk"] == "day"


def test_rejects_profile_mapping_with_weak_samples_or_regressed_false_reads(
    tmp_path: Path,
) -> None:
    payload = validated_mapping_payload()
    conditions = payload["conditions"]
    assert isinstance(conditions, dict)
    night = conditions["night"]
    assert isinstance(night, dict)
    night["sample_count"] = 5
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CameraControlError, match="requires 30"):
        load_validated_profile_mapping(path)

    night["sample_count"] = 40
    night["held_out_false_read_rate"] = 0.04
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CameraControlError, match="increases held-out false reads"):
        load_validated_profile_mapping(path)
