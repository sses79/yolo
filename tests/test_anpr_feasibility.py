from __future__ import annotations

import csv
from pathlib import Path

import pytest

from roundabout_ai.anpr_feasibility import (
    SAMPLE_FIELDS,
    FeasibilityPolicy,
    FeasibilitySample,
    Recommendation,
    append_sample,
    assess_feasibility,
    load_samples,
    main,
    render_report,
    write_template,
)


def sample(
    sample_id: str,
    *,
    readable: str = "yes",
    character_height: float | None = 18.0,
    direction: str = "approaching",
    speed: str = "slow",
    plate_side: str = "front",
    lighting: str = "day",
) -> FeasibilitySample:
    return FeasibilitySample(
        sample_id=sample_id,
        image_path=f"data/anpr/images/{sample_id}.jpg",
        direction=direction,
        lighting=lighting,
        speed=speed,
        plate_side=plate_side,
        human_readable=readable,
        plate_width_px=90,
        plate_height_px=24,
        character_height_px=character_height,
    )


def complete_samples(
    *, readable: str = "yes", height: float = 18.0
) -> tuple[FeasibilitySample, ...]:
    return (
        sample("1", readable=readable, character_height=height),
        sample(
            "2",
            readable=readable,
            character_height=height,
            direction="departing",
            speed="fast",
            plate_side="rear",
        ),
    )


def policy() -> FeasibilityPolicy:
    return FeasibilityPolicy(
        minimum_samples=2,
        minimum_readable_rate=0.8,
        stop_readable_rate=0.2,
        minimum_median_character_height_px=16,
    )


def test_template_and_loader_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"
    write_template(path)
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            (
                "sample-1",
                "images/one.jpg",
                "approaching",
                "shade",
                "slow",
                "front",
                "uncertain",
                "88",
                "23",
                "",
                "motion blur",
            )
        )

    loaded = load_samples(path)

    assert loaded[0].sample_id == "sample-1"
    assert loaded[0].character_height_px is None
    assert loaded[0].notes == "motion blur"
    with pytest.raises(FileExistsError):
        write_template(path)


def test_append_sample_creates_file_and_rejects_duplicate(tmp_path: Path) -> None:
    path = tmp_path / "annotations.csv"

    append_sample(path, sample("one"))

    assert load_samples(path) == (sample("one"),)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        append_sample(path, sample("one"))


def test_loader_rejects_bad_schema_and_duplicate_ids(tmp_path: Path) -> None:
    bad_schema = tmp_path / "bad.csv"
    bad_schema.write_text("wrong,header\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected CSV header"):
        load_samples(bad_schema)

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(
        ",".join(SAMPLE_FIELDS)
        + "\n"
        + "same,image.jpg,approaching,day,slow,front,yes,80,20,14,\n"
        + "same,image2.jpg,departing,day,fast,rear,no,80,20,14,\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_samples(duplicate)


def test_assessment_requires_quantity_and_condition_coverage() -> None:
    assessment = assess_feasibility((sample("1", speed="unknown"),), policy())

    assert assessment.recommendation is Recommendation.INCOMPLETE_SAMPLE
    assert "direction:departing" in assessment.missing_coverage
    assert "speed:fast" in assessment.missing_coverage
    assert "plate_side:rear" in assessment.missing_coverage


@pytest.mark.parametrize(
    ("samples", "expected"),
    (
        (complete_samples(), Recommendation.PROCEED),
        (
            complete_samples(height=10),
            Recommendation.REPOSITION_FIRST,
        ),
        (
            complete_samples(readable="no"),
            Recommendation.STOP_AT_VEHICLE_ANALYTICS,
        ),
    ),
)
def test_assessment_recommendations(
    samples: tuple[FeasibilitySample, ...], expected: Recommendation
) -> None:
    assert assess_feasibility(samples, policy()).recommendation is expected


def test_report_explains_policy_and_never_contains_plate_text() -> None:
    assessment = assess_feasibility(complete_samples(), policy())

    report = render_report(
        assessment, policy(), annotations_path=Path("data/anpr/annotations.csv")
    )

    assert "Recommendation: **proceed**" in report
    assert "thresholds are project policy" in report
    assert "No registration text is collected" in report


def test_cli_creates_template_and_report(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"
    report = tmp_path / "report.md"
    assert main(("init", "--output", str(annotations))) == 0
    with annotations.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for item in complete_samples():
            writer.writerow(
                (
                    item.sample_id,
                    item.image_path,
                    item.direction,
                    item.lighting,
                    item.speed,
                    item.plate_side,
                    item.human_readable,
                    item.plate_width_px,
                    item.plate_height_px,
                    item.character_height_px,
                    item.notes,
                )
            )

    assert (
        main(
            (
                "report",
                "--annotations",
                str(annotations),
                "--output",
                str(report),
                "--minimum-samples",
                "2",
            )
        )
        == 0
    )
    assert "**proceed**" in report.read_text(encoding="utf-8")


def test_cli_adds_a_validated_observation(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations.csv"

    assert (
        main(
            (
                "add",
                "--annotations",
                str(annotations),
                "--sample-id",
                "one",
                "--image-path",
                "images/one.jpg",
                "--direction",
                "approaching",
                "--lighting",
                "day",
                "--speed",
                "slow",
                "--plate-side",
                "front",
                "--human-readable",
                "yes",
                "--plate-width",
                "90",
                "--plate-height",
                "24",
                "--character-height",
                "18",
            )
        )
        == 0
    )
    assert load_samples(annotations)[0].sample_id == "one"
