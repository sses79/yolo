from __future__ import annotations

import csv
from pathlib import Path

import pytest

import roundabout_ai.anpr_app
from roundabout_ai.anpr_app import LABEL_FIELDS, load_evaluation_labels


def test_loads_and_normalizes_local_evaluation_labels(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(LABEL_FIELDS)
        writer.writerow(("vehicle-1", "ab12 cde"))
    assert load_evaluation_labels(path) == {"vehicle-1": "AB12CDE"}


def test_rejects_duplicate_evaluation_vehicle_ids(tmp_path: Path) -> None:
    path = tmp_path / "labels.csv"
    path.write_text(
        "vehicle_id,expected_text\nvehicle-1,AB12CDE\nvehicle-1,XY99XYZ\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate vehicle_id"):
        load_evaluation_labels(path)


def test_image_path_shorthand_routes_to_single_image_command(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    image = tmp_path / "vehicle.jpg"
    image.touch()
    seen: list[Path] = []

    def fake_command(args: object) -> int:
        seen.append(args.image_path)  # type: ignore[attr-defined]
        return 0

    monkeypatch.setattr(roundabout_ai.anpr_app, "_command_image", fake_command)

    assert roundabout_ai.anpr_app.main(("--image-path", str(image))) == 0
    assert seen == [image]
