"""Local, metadata-only assessment for the ANPR feasibility gate."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

SAMPLE_FIELDS = (
    "sample_id",
    "image_path",
    "direction",
    "lighting",
    "speed",
    "plate_side",
    "human_readable",
    "plate_width_px",
    "plate_height_px",
    "character_height_px",
    "notes",
)

DIRECTIONS = ("approaching", "departing")
LIGHTING_CONDITIONS = ("day", "shade", "glare", "rain", "evening", "other")
SPEEDS = ("slow", "fast", "unknown")
REQUIRED_SPEEDS = ("slow", "fast")
PLATE_SIDES = ("front", "rear")
READABILITY_VALUES = ("yes", "no", "uncertain")


class Recommendation(StrEnum):
    PROCEED = "proceed"
    REPOSITION_FIRST = "reposition_first"
    STOP_AT_VEHICLE_ANALYTICS = "stop_at_vehicle_analytics"
    INCOMPLETE_SAMPLE = "incomplete_sample"


@dataclass(frozen=True, slots=True)
class FeasibilitySample:
    sample_id: str
    image_path: str
    direction: str
    lighting: str
    speed: str
    plate_side: str
    human_readable: str
    plate_width_px: int
    plate_height_px: int
    character_height_px: float | None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.sample_id.strip():
            raise ValueError("sample_id must not be empty")
        if not self.image_path.strip():
            raise ValueError("image_path must not be empty")
        for value, allowed, name in (
            (self.direction, DIRECTIONS, "direction"),
            (self.lighting, LIGHTING_CONDITIONS, "lighting"),
            (self.speed, SPEEDS, "speed"),
            (self.plate_side, PLATE_SIDES, "plate side"),
            (self.human_readable, READABILITY_VALUES, "human readability"),
        ):
            if value not in allowed:
                raise ValueError(f"invalid {name}: {value}")
        if self.plate_width_px <= 0 or self.plate_height_px <= 0:
            raise ValueError("plate dimensions must be positive")
        if self.character_height_px is not None and self.character_height_px <= 0:
            raise ValueError("character height must be positive")

    @property
    def is_readable(self) -> bool:
        return self.human_readable == "yes"


@dataclass(frozen=True, slots=True)
class FeasibilityPolicy:
    minimum_samples: int = 20
    minimum_readable_rate: float = 0.8
    stop_readable_rate: float = 0.2
    minimum_median_character_height_px: float = 16.0
    required_directions: tuple[str, ...] = DIRECTIONS
    required_speeds: tuple[str, ...] = REQUIRED_SPEEDS
    required_plate_sides: tuple[str, ...] = PLATE_SIDES
    required_lighting: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.minimum_samples <= 0:
            raise ValueError("minimum samples must be positive")
        if not 0 <= self.stop_readable_rate <= self.minimum_readable_rate <= 1:
            raise ValueError("readable rates must satisfy 0 <= stop <= minimum <= 1")
        if self.minimum_median_character_height_px <= 0:
            raise ValueError("minimum character height must be positive")
        _validate_required(self.required_directions, DIRECTIONS, "direction")
        _validate_required(self.required_speeds, SPEEDS, "speed")
        _validate_required(self.required_plate_sides, PLATE_SIDES, "plate side")
        _validate_required(self.required_lighting, LIGHTING_CONDITIONS, "lighting")


@dataclass(frozen=True, slots=True)
class FeasibilityAssessment:
    recommendation: Recommendation
    sample_count: int
    readable_count: int
    uncertain_count: int
    readable_rate: float
    median_plate_width_px: float | None
    median_plate_height_px: float | None
    median_character_height_px: float | None
    missing_coverage: tuple[str, ...]
    reasons: tuple[str, ...]
    distributions: dict[str, dict[str, int]]


def _validate_required(
    values: Iterable[str], allowed: tuple[str, ...], field_name: str
) -> None:
    invalid = sorted(set(values) - set(allowed))
    if invalid:
        raise ValueError(f"invalid required {field_name}: {', '.join(invalid)}")


def _positive_int(value: str, *, field_name: str, row_number: int) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field_name} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"row {row_number}: {field_name} must be positive")
    return parsed


def _optional_positive_float(
    value: str, *, field_name: str, row_number: int
) -> float | None:
    if not value:
        return None
    try:
        parsed = float(value)
    except ValueError as exc:
        raise ValueError(f"row {row_number}: {field_name} must be numeric") from exc
    if parsed <= 0:
        raise ValueError(f"row {row_number}: {field_name} must be positive")
    return parsed


def _choice(
    value: str,
    *,
    allowed: tuple[str, ...],
    field_name: str,
    row_number: int,
) -> str:
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"row {row_number}: {field_name} must be one of {', '.join(allowed)}"
        )
    return normalized


def write_template(path: Path, *, overwrite: bool = False) -> None:
    """Create an empty local annotation CSV with the stable Phase 4 schema."""

    if path.exists() and not overwrite:
        raise FileExistsError(f"annotation file already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(SAMPLE_FIELDS)


def append_sample(path: Path, sample: FeasibilitySample) -> None:
    """Append one validated observation, creating the template when needed."""

    if not path.exists():
        write_template(path)
    existing = load_samples(path)
    if any(item.sample_id == sample.sample_id for item in existing):
        raise ValueError(f"duplicate sample_id {sample.sample_id}")
    with path.open("a", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(
            (
                sample.sample_id,
                sample.image_path,
                sample.direction,
                sample.lighting,
                sample.speed,
                sample.plate_side,
                sample.human_readable,
                sample.plate_width_px,
                sample.plate_height_px,
                ""
                if sample.character_height_px is None
                else f"{sample.character_height_px:g}",
                sample.notes,
            )
        )


def load_samples(path: Path) -> tuple[FeasibilitySample, ...]:
    """Load and validate feasibility observations without reading image content."""

    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SAMPLE_FIELDS:
            raise ValueError(f"annotation file has an unexpected CSV header: {path}")
        samples: list[FeasibilitySample] = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            sample_id = row["sample_id"].strip()
            if not sample_id:
                raise ValueError(f"row {row_number}: sample_id must not be empty")
            if sample_id in seen_ids:
                raise ValueError(f"row {row_number}: duplicate sample_id {sample_id}")
            seen_ids.add(sample_id)
            image_path = row["image_path"].strip()
            if not image_path:
                raise ValueError(f"row {row_number}: image_path must not be empty")
            samples.append(
                FeasibilitySample(
                    sample_id=sample_id,
                    image_path=image_path,
                    direction=_choice(
                        row["direction"],
                        allowed=DIRECTIONS,
                        field_name="direction",
                        row_number=row_number,
                    ),
                    lighting=_choice(
                        row["lighting"],
                        allowed=LIGHTING_CONDITIONS,
                        field_name="lighting",
                        row_number=row_number,
                    ),
                    speed=_choice(
                        row["speed"],
                        allowed=SPEEDS,
                        field_name="speed",
                        row_number=row_number,
                    ),
                    plate_side=_choice(
                        row["plate_side"],
                        allowed=PLATE_SIDES,
                        field_name="plate_side",
                        row_number=row_number,
                    ),
                    human_readable=_choice(
                        row["human_readable"],
                        allowed=READABILITY_VALUES,
                        field_name="human_readable",
                        row_number=row_number,
                    ),
                    plate_width_px=_positive_int(
                        row["plate_width_px"],
                        field_name="plate_width_px",
                        row_number=row_number,
                    ),
                    plate_height_px=_positive_int(
                        row["plate_height_px"],
                        field_name="plate_height_px",
                        row_number=row_number,
                    ),
                    character_height_px=_optional_positive_float(
                        row["character_height_px"].strip(),
                        field_name="character_height_px",
                        row_number=row_number,
                    ),
                    notes=row["notes"].strip(),
                )
            )
    return tuple(samples)


def _missing_values(
    samples: Sequence[FeasibilitySample],
    *,
    attribute: str,
    required: tuple[str, ...],
) -> tuple[str, ...]:
    observed = {str(getattr(sample, attribute)) for sample in samples}
    return tuple(f"{attribute}:{value}" for value in required if value not in observed)


def assess_feasibility(
    samples: Sequence[FeasibilitySample],
    policy: FeasibilityPolicy = FeasibilityPolicy(),
) -> FeasibilityAssessment:
    """Apply an explicit project policy to human-labelled image observations."""

    sample_count = len(samples)
    readable_count = sum(sample.is_readable for sample in samples)
    uncertain_count = sum(sample.human_readable == "uncertain" for sample in samples)
    readable_rate = readable_count / sample_count if sample_count else 0.0
    character_heights = [
        sample.character_height_px
        for sample in samples
        if sample.character_height_px is not None
    ]
    missing_coverage = (
        _missing_values(
            samples, attribute="direction", required=policy.required_directions
        )
        + _missing_values(samples, attribute="speed", required=policy.required_speeds)
        + _missing_values(
            samples, attribute="plate_side", required=policy.required_plate_sides
        )
        + _missing_values(
            samples, attribute="lighting", required=policy.required_lighting
        )
    )
    median_character_height = (
        statistics.median(character_heights) if character_heights else None
    )
    reasons: list[str] = []
    if sample_count < policy.minimum_samples:
        reasons.append(
            f"sample has {sample_count} rows; policy requires {policy.minimum_samples}"
        )
    if missing_coverage:
        reasons.append(f"missing coverage: {', '.join(missing_coverage)}")
    if sample_count < policy.minimum_samples or missing_coverage:
        recommendation = Recommendation.INCOMPLETE_SAMPLE
    elif readable_rate < policy.stop_readable_rate:
        recommendation = Recommendation.STOP_AT_VEHICLE_ANALYTICS
        reasons.append(
            f"readable rate {readable_rate:.1%} is below stop threshold "
            f"{policy.stop_readable_rate:.1%}"
        )
    elif (
        readable_rate >= policy.minimum_readable_rate
        and median_character_height is not None
        and median_character_height >= policy.minimum_median_character_height_px
    ):
        recommendation = Recommendation.PROCEED
        reasons.append("readability and character-size thresholds are satisfied")
    else:
        recommendation = Recommendation.REPOSITION_FIRST
        if readable_rate < policy.minimum_readable_rate:
            reasons.append(
                f"readable rate {readable_rate:.1%} is below proceed threshold "
                f"{policy.minimum_readable_rate:.1%}"
            )
        if median_character_height is None:
            reasons.append("no character-height measurements were provided")
        elif median_character_height < policy.minimum_median_character_height_px:
            reasons.append(
                f"median character height {median_character_height:g}px is below "
                f"{policy.minimum_median_character_height_px:g}px"
            )

    distributions = {
        name: dict(
            sorted(Counter(str(getattr(sample, name)) for sample in samples).items())
        )
        for name in ("direction", "lighting", "speed", "plate_side", "human_readable")
    }
    return FeasibilityAssessment(
        recommendation=recommendation,
        sample_count=sample_count,
        readable_count=readable_count,
        uncertain_count=uncertain_count,
        readable_rate=readable_rate,
        median_plate_width_px=statistics.median(
            sample.plate_width_px for sample in samples
        )
        if samples
        else None,
        median_plate_height_px=statistics.median(
            sample.plate_height_px for sample in samples
        )
        if samples
        else None,
        median_character_height_px=median_character_height,
        missing_coverage=missing_coverage,
        reasons=tuple(reasons),
        distributions=distributions,
    )


def render_report(
    assessment: FeasibilityAssessment,
    policy: FeasibilityPolicy,
    *,
    annotations_path: Path,
) -> str:
    """Render a reviewable report without copying images or registration text."""

    def pixels(value: float | None) -> str:
        return "not measured" if value is None else f"{value:g}px"

    lines = [
        "# ANPR Feasibility Report",
        "",
        f"- Recommendation: **{assessment.recommendation.value}**",
        f"- Annotation file: `{annotations_path}`",
        f"- Samples: {assessment.sample_count}",
        f"- Human-readable: {assessment.readable_count} ({assessment.readable_rate:.1%})",
        f"- Uncertain: {assessment.uncertain_count}",
        f"- Median plate size: {pixels(assessment.median_plate_width_px)} x "
        f"{pixels(assessment.median_plate_height_px)}",
        f"- Median character height: {pixels(assessment.median_character_height_px)}",
        "",
        "## Decision policy",
        "",
        f"- Minimum samples: {policy.minimum_samples}",
        f"- Proceed readable rate: {policy.minimum_readable_rate:.1%}",
        f"- Stop readable rate: below {policy.stop_readable_rate:.1%}",
        f"- Minimum median character height: "
        f"{policy.minimum_median_character_height_px:g}px",
        "",
        "These thresholds are project policy, not an OCR accuracy guarantee.",
        "",
        "## Reasons",
        "",
        *(f"- {reason}" for reason in assessment.reasons),
        "",
        "## Coverage",
        "",
    ]
    for category, counts in assessment.distributions.items():
        summary = ", ".join(f"{name}={count}" for name, count in counts.items())
        lines.append(f"- {category}: {summary or 'none'}")
    lines.extend(
        (
            "",
            "## Interpretation",
            "",
            "- `proceed`: the labelled sample meets the configured feasibility policy; Phase 5 still requires held-out OCR evaluation.",
            "- `reposition_first`: improve angle, distance, lighting, or shutter speed, then collect a new sample.",
            "- `stop_at_vehicle_analytics`: the present view does not justify OCR work.",
            "- `incomplete_sample`: collect the missing quantity or condition coverage before deciding.",
            "",
            "No registration text is collected by this Phase 4 report.",
            "",
        )
    )
    return "\n".join(lines)


def _rate(value: str) -> float:
    parsed = float(value)
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_count(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _lighting_list(value: str) -> tuple[str, ...]:
    conditions = tuple(
        dict.fromkeys(item.strip().lower() for item in value.split(",") if item.strip())
    )
    try:
        _validate_required(conditions, LIGHTING_CONDITIONS, "lighting")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create and assess a local, metadata-only ANPR feasibility sample."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    initialize = subparsers.add_parser("init", help="create an empty annotation CSV")
    initialize.add_argument(
        "--output", type=Path, default=Path("data/anpr/annotations.csv")
    )
    initialize.add_argument("--overwrite", action="store_true")

    add = subparsers.add_parser("add", help="append one labelled observation")
    add.add_argument(
        "--annotations", type=Path, default=Path("data/anpr/annotations.csv")
    )
    add.add_argument("--sample-id", required=True)
    add.add_argument("--image-path", required=True)
    add.add_argument("--direction", choices=DIRECTIONS, required=True)
    add.add_argument("--lighting", choices=LIGHTING_CONDITIONS, required=True)
    add.add_argument("--speed", choices=SPEEDS, required=True)
    add.add_argument("--plate-side", choices=PLATE_SIDES, required=True)
    add.add_argument("--human-readable", choices=READABILITY_VALUES, required=True)
    add.add_argument("--plate-width", type=_positive_count, required=True)
    add.add_argument("--plate-height", type=_positive_count, required=True)
    add.add_argument("--character-height", type=_positive)
    add.add_argument("--notes", default="")

    report = subparsers.add_parser("report", help="validate annotations and report")
    report.add_argument(
        "--annotations", type=Path, default=Path("data/anpr/annotations.csv")
    )
    report.add_argument("--output", type=Path, default=Path("data/anpr/report.md"))
    report.add_argument("--minimum-samples", type=_positive_count, default=20)
    report.add_argument("--minimum-readable-rate", type=_rate, default=0.8)
    report.add_argument("--stop-readable-rate", type=_rate, default=0.2)
    report.add_argument("--minimum-character-height", type=_positive, default=16.0)
    report.add_argument(
        "--required-lighting",
        type=_lighting_list,
        default=(),
        help="comma-separated relevant conditions, such as day,shade,glare",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "init":
        write_template(args.output, overwrite=args.overwrite)
        print(f"Created ANPR feasibility template: {args.output}")
        return 0
    if args.command == "add":
        append_sample(
            args.annotations,
            FeasibilitySample(
                sample_id=args.sample_id.strip(),
                image_path=args.image_path.strip(),
                direction=args.direction,
                lighting=args.lighting,
                speed=args.speed,
                plate_side=args.plate_side,
                human_readable=args.human_readable,
                plate_width_px=args.plate_width,
                plate_height_px=args.plate_height,
                character_height_px=args.character_height,
                notes=args.notes.strip(),
            ),
        )
        print(f"Added sample {args.sample_id} to {args.annotations}")
        return 0

    policy = FeasibilityPolicy(
        minimum_samples=args.minimum_samples,
        minimum_readable_rate=args.minimum_readable_rate,
        stop_readable_rate=args.stop_readable_rate,
        minimum_median_character_height_px=args.minimum_character_height,
        required_lighting=args.required_lighting,
    )
    samples = load_samples(args.annotations)
    assessment = assess_feasibility(samples, policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        render_report(assessment, policy, annotations_path=args.annotations),
        encoding="utf-8",
    )
    print(f"Recommendation: {assessment.recommendation.value}")
    print(f"Report: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
