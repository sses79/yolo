"""Interactive Phase 2 ROI and count-line calibration tool."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Sequence

import cv2

from roundabout_ai.capture import CameraCapture, CaptureConfig, Frame, LatestFrameStore
from roundabout_ai.diagnostic import DEFAULT_URL
from roundabout_ai.geometry import CountLine, Point
from roundabout_ai.scene import Scene, save_scene

WINDOW_NAME = "Roundabout calibration"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture or open a reference frame and select an ROI and count lines."
    )
    parser.add_argument(
        "--url", default=os.environ.get("ROUNDABOUT_CAMERA_URL", DEFAULT_URL)
    )
    parser.add_argument("--image", type=Path, help="use an existing image instead of the camera")
    parser.add_argument(
        "--reference-image",
        type=Path,
        default=Path("data/calibration/reference.jpg"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/calibration/scene.yaml")
    )
    parser.add_argument("--first-frame-timeout", type=float, default=15.0)
    return parser


def capture_reference(url: str, timeout_seconds: float) -> Frame:
    store = LatestFrameStore()
    capture = CameraCapture(CaptureConfig(url=url), store)
    capture.start()
    deadline = time.monotonic() + timeout_seconds
    try:
        while time.monotonic() < deadline:
            packet = store.consume_latest()
            if packet is not None:
                return packet.frame
            time.sleep(0.01)
    finally:
        capture.stop()
    raise TimeoutError(f"no camera frame received within {timeout_seconds:g} seconds")


class CalibrationSession:
    """Mutable click state kept separate from OpenCV callbacks for testing."""

    def __init__(self) -> None:
        self.mode = "roi"
        self.roi: list[Point] = []
        self.pending_line: list[Point] = []
        self.lines: list[CountLine] = []

    def add_point(self, point: Point) -> None:
        if self.mode == "roi":
            self.roi.append(point)
            return
        self.pending_line.append(point)
        if len(self.pending_line) == 2:
            self.lines.append(
                CountLine(
                    name=f"line_{len(self.lines) + 1}",
                    start=self.pending_line[0],
                    end=self.pending_line[1],
                )
            )
            self.pending_line.clear()

    def finish_roi(self) -> None:
        if len(self.roi) < 3:
            raise ValueError("select at least three ROI points")
        self.mode = "lines"

    def undo(self) -> None:
        if self.mode == "roi" and self.roi:
            self.roi.pop()
        elif self.pending_line:
            self.pending_line.pop()
        elif self.lines:
            line = self.lines.pop()
            self.pending_line.extend((line.start, line.end))
            self.pending_line.pop()

    def scene(self, width: int, height: int) -> Scene:
        if len(self.roi) < 3:
            raise ValueError("select at least three ROI points")
        if self.pending_line:
            raise ValueError("finish the current count line with a second point")
        if not self.lines:
            raise ValueError("select at least one count line")
        return Scene((width, height), tuple(self.roi), tuple(self.lines))


def draw_session(frame: Frame, session: CalibrationSession) -> Frame:
    display = frame.copy()
    roi_points = [(round(x), round(y)) for x, y in session.roi]
    for point in roi_points:
        cv2.circle(display, point, 5, (0, 255, 255), -1)
    for start, end in zip(roi_points, roi_points[1:]):
        cv2.line(display, start, end, (0, 255, 255), 2)
    if session.mode == "lines" and len(roi_points) >= 3:
        cv2.line(display, roi_points[-1], roi_points[0], (0, 255, 255), 2)
    for line in session.lines:
        cv2.line(
            display,
            (round(line.start[0]), round(line.start[1])),
            (round(line.end[0]), round(line.end[1])),
            (0, 80, 255),
            3,
        )
    if session.pending_line:
        point = session.pending_line[0]
        cv2.circle(display, (round(point[0]), round(point[1])), 6, (0, 80, 255), -1)
    instruction = (
        "ROI: click 3+ points, Enter to finish"
        if session.mode == "roi"
        else "Lines: click endpoint pairs, S save"
    )
    cv2.rectangle(display, (0, 0), (display.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(
        display,
        f"{instruction} | U undo | Q cancel",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return display


def run(args: argparse.Namespace) -> int:
    if args.image:
        frame = cv2.imread(str(args.image))
        if frame is None:
            raise ValueError(f"could not read image: {args.image}")
    else:
        frame = capture_reference(args.url, args.first_frame_timeout)
    args.reference_image.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.reference_image), frame):
        raise RuntimeError(f"could not save reference image: {args.reference_image}")

    session = CalibrationSession()

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            session.add_point((float(x), float(y)))

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)
    try:
        while True:
            cv2.imshow(WINDOW_NAME, draw_session(frame, session))
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), 27):
                print("Calibration cancelled.")
                return 1
            if key in (10, 13) and session.mode == "roi":
                try:
                    session.finish_roi()
                except ValueError as exc:
                    print(exc)
            elif key == ord("u"):
                session.undo()
            elif key == ord("s"):
                try:
                    scene = session.scene(frame.shape[1], frame.shape[0])
                except ValueError as exc:
                    print(exc)
                    continue
                save_scene(scene, args.output)
                print(f"Reference image saved: {args.reference_image}")
                print(f"Scene configuration saved: {args.output}")
                return 0
    finally:
        cv2.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (TimeoutError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
