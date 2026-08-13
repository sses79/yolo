"""Command-line Phase 0 camera diagnostic."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Sequence

import cv2

from roundabout_ai.capture import (
    CameraCapture,
    CaptureConfig,
    CaptureStats,
    ConsumptionRate,
    Frame,
    FramePacket,
    LatestFrameStore,
)

DEFAULT_URL = "http://192.168.1.142:8080/video"
WINDOW_NAME = "Roundabout camera diagnostic — s snapshot, r record, q quit"


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def recording_seconds(value: str) -> float:
    parsed = positive_float(value)
    if parsed > 300:
        raise argparse.ArgumentTypeError("must not exceed 300 seconds")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose an Android IP Webcam MJPEG stream without AI inference."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("ROUNDABOUT_CAMERA_URL", DEFAULT_URL),
        help="MJPEG stream URL (default: %(default)s)",
    )
    parser.add_argument("--headless", action="store_true", help="do not open a GUI window")
    parser.add_argument(
        "--duration",
        type=positive_float,
        help="exit after this many seconds; useful for headless/soak tests",
    )
    parser.add_argument(
        "--reconnect-seconds", type=positive_float, default=2.0
    )
    parser.add_argument(
        "--open-timeout-seconds", type=positive_float, default=5.0
    )
    parser.add_argument(
        "--read-timeout-seconds", type=positive_float, default=5.0
    )
    parser.add_argument(
        "--stale-after-seconds", type=positive_float, default=2.0
    )
    parser.add_argument(
        "--metrics-interval", type=positive_float, default=2.0
    )
    parser.add_argument(
        "--snapshot-directory", type=Path, default=Path("data/snapshots")
    )
    parser.add_argument(
        "--snapshot-after",
        type=nonnegative_float,
        help="save one snapshot this many seconds after the first frame",
    )
    parser.add_argument(
        "--record-seconds",
        type=recording_seconds,
        help="start a recording automatically; valid range is >0 to 300 seconds",
    )
    parser.add_argument(
        "--manual-record-seconds",
        type=recording_seconds,
        default=120.0,
        help="maximum recording duration when toggled with r (default: %(default)s)",
    )
    parser.add_argument(
        "--recording-directory", type=Path, default=Path("data/recordings")
    )
    parser.add_argument("--record-fps", type=positive_float, default=15.0)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser


def timestamped_path(directory: Path, prefix: str, suffix: str) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f%z")
    return directory / f"{prefix}-{stamp}{suffix}"


def save_snapshot(frame: Frame, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = timestamped_path(directory, "snapshot", ".jpg")
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise RuntimeError(f"OpenCV failed to save snapshot to {path}")
    return path


class Recorder:
    def __init__(self, directory: Path, fps: float) -> None:
        self.directory = directory
        self.fps = fps
        self.path: Path | None = None
        self._writer: cv2.VideoWriter | None = None
        self._started_at: float | None = None
        self._limit_seconds: float | None = None
        self.frames_written = 0

    @property
    def active(self) -> bool:
        return self._writer is not None

    @property
    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        return time.monotonic() - self._started_at

    def start(self, frame: Frame, *, limit_seconds: float) -> Path:
        if self.active:
            assert self.path is not None
            return self.path
        self.directory.mkdir(parents=True, exist_ok=True)
        path = timestamped_path(self.directory, "camera-sample", ".mp4")
        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self.fps,
            (width, height),
        )
        if not writer.isOpened():
            writer.release()
            raise RuntimeError(f"OpenCV could not create recording {path}")
        self.path = path
        self._writer = writer
        self._started_at = time.monotonic()
        self._limit_seconds = limit_seconds
        self.frames_written = 0
        return path

    def write(self, frame: Frame) -> bool:
        """Write a frame and return True when the duration limit was reached."""
        if not self.active:
            return False
        assert self._writer is not None
        elapsed = self.elapsed
        # Capture can be faster than the requested file rate. Sampling by
        # monotonic time prevents a 30 FPS source written to a 15 FPS container
        # from becoming a slow-motion recording.
        next_frame_at = self.frames_written / self.fps
        if elapsed + 1e-9 >= next_frame_at:
            self._writer.write(frame)
            self.frames_written += 1
        if self._limit_seconds is not None and elapsed >= self._limit_seconds:
            self.stop()
            return True
        return False

    def stop(self) -> Path | None:
        if self._writer is None:
            return None
        path = self.path
        self._writer.release()
        self._writer = None
        self._started_at = None
        self._limit_seconds = None
        return path


@dataclass(slots=True)
class DiagnosticState:
    latest_packet: FramePacket | None = None
    first_frame_at: float | None = None
    snapshot_saved: bool = False
    auto_record_started: bool = False


def format_metrics(
    stats: CaptureStats,
    *,
    consumed_fps: float,
    frame_age_seconds: float | None,
    stale_after_seconds: float,
    recording: bool,
) -> str:
    stale = (
        stats.seconds_since_frame is None
        or stats.seconds_since_frame >= stale_after_seconds
    )
    health = "STALE" if stale else "LIVE"
    age = "n/a" if frame_age_seconds is None else f"{frame_age_seconds * 1000:.0f}ms"
    resolution = f"{stats.width}x{stats.height}" if stats.width else "unknown"
    return (
        f"[{health}] status={stats.status} resolution={resolution} "
        f"capture_fps={stats.capture_fps:.1f} consumed_fps={consumed_fps:.1f} "
        f"received={stats.frames_received} consumed={stats.frames_consumed} "
        f"overwritten={stats.frames_overwritten} read_failures={stats.read_failures} "
        f"reconnects={stats.reconnects} age={age} "
        f"recording={'yes' if recording else 'no'}"
        + (f" error={stats.last_error!r}" if stats.last_error else "")
    )


def draw_overlay(frame: Frame, text: str) -> Frame:
    display = frame.copy()
    lines = [text[i : i + 110] for i in range(0, len(text), 110)]
    overlay_height = 12 + len(lines) * 24
    cv2.rectangle(display, (0, 0), (display.shape[1], overlay_height), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (10, 24 + index * 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 80),
            1,
            cv2.LINE_AA,
        )
    return display


def run(args: argparse.Namespace) -> int:
    logger = logging.getLogger("roundabout_ai.diagnostic")
    store = LatestFrameStore()
    camera = CameraCapture(
        CaptureConfig(
            url=args.url,
            reconnect_seconds=args.reconnect_seconds,
            open_timeout_seconds=args.open_timeout_seconds,
            read_timeout_seconds=args.read_timeout_seconds,
        ),
        store,
        logger=logger,
    )
    recorder = Recorder(args.recording_directory, args.record_fps)
    consumed_rate = ConsumptionRate()
    state = DiagnosticState()
    stopping = False

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stopping
        stopping = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    previous_sigterm = signal.signal(signal.SIGTERM, request_stop)
    started_at = time.monotonic()
    next_metrics_at = started_at

    print(f"Camera URL: {args.url}", flush=True)
    if not args.headless:
        print("Controls: s=snapshot, r=toggle recording, q/Escape=quit", flush=True)

    camera.start()
    try:
        while not stopping:
            now = time.monotonic()
            if args.duration is not None and now - started_at >= args.duration:
                break

            packet = store.consume_latest()
            if packet is not None:
                state.latest_packet = packet
                consumed_rate.tick(now)
                if state.first_frame_at is None:
                    state.first_frame_at = now

                if (
                    args.record_seconds is not None
                    and not state.auto_record_started
                ):
                    path = recorder.start(packet.frame, limit_seconds=args.record_seconds)
                    state.auto_record_started = True
                    print(f"Recording started: {path}", flush=True)

                if recorder.active and recorder.write(packet.frame):
                    print(
                        f"Recording completed: {recorder.path} "
                        f"({recorder.frames_written} frames)",
                        flush=True,
                    )

                if (
                    args.snapshot_after is not None
                    and not state.snapshot_saved
                    and state.first_frame_at is not None
                    and now - state.first_frame_at >= args.snapshot_after
                ):
                    path = save_snapshot(packet.frame, args.snapshot_directory)
                    state.snapshot_saved = True
                    print(f"Snapshot saved: {path}", flush=True)

            stats = store.stats(now=now)
            frame_age = (
                None
                if state.latest_packet is None
                else max(0.0, now - state.latest_packet.captured_at)
            )
            metrics = format_metrics(
                stats,
                consumed_fps=consumed_rate.fps,
                frame_age_seconds=frame_age,
                stale_after_seconds=args.stale_after_seconds,
                recording=recorder.active,
            )
            if now >= next_metrics_at:
                print(metrics, flush=True)
                next_metrics_at = now + args.metrics_interval

            if not args.headless and state.latest_packet is not None:
                cv2.imshow(WINDOW_NAME, draw_overlay(state.latest_packet.frame, metrics))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    path = save_snapshot(
                        state.latest_packet.frame, args.snapshot_directory
                    )
                    print(f"Snapshot saved: {path}", flush=True)
                if key == ord("r"):
                    if recorder.active:
                        path = recorder.stop()
                        print(f"Recording stopped: {path}", flush=True)
                    else:
                        path = recorder.start(
                            state.latest_packet.frame,
                            limit_seconds=args.manual_record_seconds,
                        )
                        print(f"Recording started: {path}", flush=True)
                try:
                    if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                        break
                except cv2.error:
                    break
            else:
                time.sleep(0.005)
    finally:
        recording_path = recorder.stop()
        if recording_path is not None:
            print(f"Recording closed: {recording_path}", flush=True)
        camera.stop()
        if not args.headless:
            cv2.destroyAllWindows()
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        print(format_metrics(
            store.stats(),
            consumed_fps=consumed_rate.fps,
            frame_age_seconds=(
                None
                if state.latest_packet is None
                else max(0.0, time.monotonic() - state.latest_packet.captured_at)
            ),
            stale_after_seconds=args.stale_after_seconds,
            recording=False,
        ), flush=True)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.headless is False and sys.platform == "darwin" and not os.environ.get("DISPLAY"):
        # Native macOS OpenCV windows do not require DISPLAY. Keep this branch
        # as documentation rather than forcing headless mode.
        pass
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
