from __future__ import annotations

import base64
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from roundabout_ai.detector import Detection
from roundabout_ai.event_images import (
    VehicleCandidateBuffer,
    event_preview_data_url,
    save_event_candidates,
    save_event_snapshot,
)
from roundabout_ai.events import EventRecord


def detection(
    track_id: int,
    xyxy: tuple[int, int, int, int],
    *,
    label: str = "car",
) -> Detection:
    return Detection(2, label, 0.9, xyxy, track_id=track_id)


def candidate_buffer(**kwargs: int) -> VehicleCandidateBuffer:
    return VehicleCandidateBuffer(
        minimum_vehicle_width=2,
        minimum_vehicle_height=2,
        horizontal_padding_ratio=0,
        vertical_padding_ratio=0,
        **kwargs,
    )


def test_buffer_selects_distinct_crossing_centred_and_sharpest_raw_crops() -> None:
    buffer = candidate_buffer()
    first = np.zeros((20, 30, 3), dtype=np.uint8)
    first_crop = first[2:10, 2:12]
    first_crop[::2, ::2] = 255
    centred = np.full((20, 30, 3), 80, dtype=np.uint8)
    crossing = np.full((20, 30, 3), 120, dtype=np.uint8)

    buffer.observe(first, (detection(7, (2, 2, 12, 10)),))
    buffer.observe(centred, (detection(7, (8, 1, 24, 18)),))
    buffer.observe(crossing, (detection(7, (3, 3, 18, 15)),))

    selected = buffer.select(7)
    assert [name for name, _candidate in selected] == [
        "crossing",
        "centred",
        "sharpest",
    ]
    assert selected[0][1].crop.shape == (12, 15, 3)
    assert selected[1][1].area == 16 * 17
    assert selected[2][1].sharpness > selected[1][1].sharpness
    assert np.array_equal(selected[0][1].crop, crossing[3:15, 3:18])


def test_buffer_ignores_people_untracked_boxes_and_expires_tracks() -> None:
    buffer = candidate_buffer(maximum_missing_frames=1)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    buffer.observe(
        frame,
        (
            detection(1, (1, 1, 8, 8), label="person"),
            Detection(2, "car", 0.9, (1, 1, 8, 8), track_id=None),
            detection(2, (-2, -2, 8, 8)),
        ),
    )
    assert buffer.track_count == 1
    assert buffer.select(2)[0][1].clipped

    buffer.observe(frame, ())
    buffer.observe(frame, ())
    assert buffer.select(2) == ()


def test_buffer_bounds_track_count() -> None:
    buffer = candidate_buffer(maximum_tracks=2)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)

    for track_id in (1, 2, 3):
        buffer.observe(frame, (detection(track_id, (1, 1, 8, 8)),))

    assert buffer.track_count == 2
    assert buffer.select(1) == ()


def test_save_event_candidates_uses_event_specific_names_and_raw_crops(
    tmp_path: Path,
) -> None:
    buffer = candidate_buffer()
    frame = np.full((10, 12, 3), 117, dtype=np.uint8)
    buffer.observe(frame, (detection(57, (2, 3, 10, 9)),))
    event = EventRecord(
        "2026-08-17T12:00:00.000Z",
        "line_crossing",
        "north entry",
        "car",
        "entering",
        57,
        0.9,
    )

    paths = save_event_candidates(
        buffer.select(57),
        tmp_path,
        event,
        clock=lambda: datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
    )

    assert len(paths) == 1
    assert paths[0].name == (
        "event-20260817-120000-000000+0000-track-57-north-entry-crossing.jpg"
    )
    saved = cv2.imread(str(paths[0]))
    assert saved is not None
    assert saved.shape == (6, 8, 3)
    assert np.allclose(saved, 117, atol=2)


def test_save_event_snapshot_preserves_full_annotated_frame(tmp_path: Path) -> None:
    frame = np.full((18, 24, 3), 93, dtype=np.uint8)
    event = EventRecord(
        "2026-08-17T12:00:00.000Z",
        "line_crossing",
        "north entry",
        "car",
        "entering",
        57,
        0.9,
    )

    path = save_event_snapshot(
        frame,
        tmp_path,
        event,
        clock=lambda: datetime(2026, 8, 17, 12, 0, tzinfo=UTC),
    )

    assert path.name == (
        "event-20260817-120000-000000+0000-track-57-north-entry-snapshot.jpg"
    )
    saved = cv2.imread(str(path))
    assert saved is not None
    assert saved.shape == frame.shape


def test_event_preview_is_a_bounded_jpeg_data_url() -> None:
    frame = np.full((100, 400, 3), 93, dtype=np.uint8)

    preview = event_preview_data_url(frame, maximum_width=200)

    assert preview.startswith("data:image/jpeg;base64,")
    encoded = preview.removeprefix("data:image/jpeg;base64,")
    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(encoded), dtype=np.uint8),
        cv2.IMREAD_COLOR,
    )
    assert decoded is not None
    assert decoded.shape[:2] == (50, 200)


def test_buffer_pads_crops_marks_edges_and_rejects_small_vehicle_boxes() -> None:
    frame = np.zeros((200, 300, 3), dtype=np.uint8)
    buffer = VehicleCandidateBuffer(
        minimum_vehicle_width=100,
        minimum_vehicle_height=60,
    )

    buffer.observe(frame, (detection(1, (50, 40, 150, 100)),))
    candidate = buffer.select(1)[0][1]
    assert candidate.crop.shape == (72, 130, 3)
    assert not candidate.clipped

    buffer.observe(frame, (detection(2, (5, 40, 105, 100)),))
    assert buffer.select(2)[0][1].clipped

    buffer.observe(frame, (detection(3, (50, 40, 149, 100)),))
    assert buffer.select(3) == ()
