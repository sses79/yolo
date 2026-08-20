"""Streamlit presentation layer for the roundabout processing worker."""

from __future__ import annotations

import logging
import os
import statistics
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

import streamlit as st

from roundabout_ai.detector import ROAD_USER_CLASSES
from roundabout_ai.diagnostic import DEFAULT_URL
from roundabout_ai.events import EventRecord
from roundabout_ai.shared_state import DashboardSnapshot
from roundabout_ai.worker import DetectionWorker, OverlayControls, ProcessingConfig


@st.cache_resource(show_spinner=False)
def get_worker() -> DetectionWorker:
    """Return the process-wide worker retained across Streamlit reruns."""

    return DetectionWorker()


def crossing_chart_rows(snapshot: DashboardSnapshot) -> list[dict[str, str | int]]:
    totals: Counter[tuple[str, str]] = Counter()
    for key, count in snapshot.crossing_counts.items():
        _line, direction, object_class = key.split(":", maxsplit=2)
        totals[(object_class, direction)] += count
    return [
        {
            "category": f"{object_class} · {direction}",
            "count": count,
        }
        for (object_class, direction), count in sorted(totals.items())
    ]


def event_table_rows(
    snapshot: DashboardSnapshot,
) -> list[dict[str, str | int | float | bool | None]]:
    return [event.as_dict() for event in snapshot.recent_events]


def adaptive_camera_performance_rows(
    snapshot: DashboardSnapshot,
) -> list[dict[str, str | int | float | None]]:
    """Aggregate recent event outcomes without treating confidence as accuracy."""

    groups: dict[tuple[str, str, str, str], list[EventRecord]] = defaultdict(list)
    for event in snapshot.recent_events:
        groups[
            (
                event.camera_profile or "baseline",
                event.camera_condition or "unknown",
                event.direction,
                event.speed_class,
            )
        ].append(event)
    rows: list[dict[str, str | int | float | None]] = []
    for (profile, condition, direction, speed_class), events in sorted(groups.items()):
        eligible = [event for event in events if event.ocr_status != "not_run"]
        accepted = [event for event in eligible if event.ocr_status == "accepted"]
        plate_detected = [event for event in eligible if event.plate_detected is True]
        rows.append(
            {
                "profile": profile,
                "condition": condition,
                "direction": direction,
                "speed_class": speed_class,
                "crossings": len(events),
                "ocr_eligible": len(eligible),
                "plate_detection_rate": _ratio(len(plate_detected), len(eligible)),
                "ocr_accepted": len(accepted),
                "ocr_uncertain": sum(
                    event.ocr_status == "uncertain" for event in eligible
                ),
                "ocr_no_read": sum(event.ocr_status == "no_read" for event in eligible),
                "ocr_acceptance_rate": _ratio(len(accepted), len(eligible)),
                "mean_accepted_ocr_confidence": _mean(
                    event.ocr_confidence for event in accepted
                ),
                "mean_plate_detector_confidence": _mean(
                    event.plate_detector_confidence for event in eligible
                ),
                "mean_plate_sharpness": _mean(
                    event.plate_sharpness for event in eligible
                ),
            }
        )
    return rows


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _mean(values: Iterable[float | None]) -> float | None:
    present = tuple(value for value in values if value is not None)
    return None if not present else statistics.fmean(present)


def _status_text(snapshot: DashboardSnapshot) -> str:
    labels = {
        "loading_model": "Loading model",
        "starting": "Starting",
        "connecting": "Connecting",
        "connected": "Connected",
        "live": "Live",
        "offline": "Offline",
        "reconnecting": "Reconnecting",
        "stopping": "Stopping",
        "stopped": "Stopped",
        "error": "Error",
    }
    return labels.get(snapshot.status, snapshot.status.replace("_", " ").title())


def render_dashboard() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    st.set_page_config(
        page_title="Roundabout AI",
        page_icon="🚗",
        layout="wide",
    )
    st.title("Roundabout AI")
    st.caption(
        "Local vehicle/person tracking and directional crossing events. "
        "No raw video is retained by default."
    )

    worker = get_worker()
    initial = worker.snapshot()
    default_scene = (
        "data/calibration/scene.yaml"
        if Path("data/calibration/scene.yaml").exists()
        else ""
    )

    with st.sidebar:
        st.header("Processing")
        url = st.text_input(
            "Camera URL",
            value=os.environ.get("ROUNDABOUT_CAMERA_URL", DEFAULT_URL),
            disabled=initial.running,
        )
        model = st.text_input("Model", value="yolo26n.pt", disabled=initial.running)
        device = st.selectbox(
            "Device", ("cpu", "mps", "auto"), disabled=initial.running
        )
        scene_path = st.text_input(
            "Scene configuration",
            value=default_scene,
            placeholder="Leave blank to disable ROI/counting",
            disabled=initial.running,
        )
        event_file = st.text_input(
            "Event CSV",
            value="data/events/events.csv",
            disabled=initial.running,
        )
        save_event_images = st.checkbox(
            "Save event snapshot and vehicle crops",
            value=False,
            disabled=initial.running,
            help=(
                "Off by default. On a crossing, retain the full annotated event "
                "snapshot plus distinct raw crossing, centred, largest, and "
                "sharpest vehicle crops locally."
            ),
        )
        live_anpr = st.checkbox(
            "Live ANPR/OCR on crossings",
            value=False,
            disabled=initial.running,
            help=(
                "Runs the local plate model and OCR only after a confirmed vehicle "
                "crossing. Event data stores the first four characters and masks "
                "the remainder."
            ),
        )

        st.subheader("Live controls")
        confidence = st.slider(
            "Detection confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.35,
            step=0.05,
        )
        show_detections = st.checkbox("Detection boxes", value=True)
        show_scene = st.checkbox("ROI and count lines", value=True)
        show_metrics = st.checkbox("Frame metrics overlay", value=True)

        with st.expander("Advanced"):
            image_size = st.select_slider(
                "Inference image size",
                options=(320, 480, 640, 800, 960),
                value=640,
                disabled=initial.running,
            )
            minimum_track_age = st.number_input(
                "Minimum track observations",
                min_value=1,
                max_value=30,
                value=3,
                disabled=initial.running,
            )
            maximum_missing_frames = st.number_input(
                "Maximum missing frames",
                min_value=0,
                max_value=300,
                value=30,
                disabled=initial.running,
            )
            fast_speed_threshold = st.number_input(
                "Fast threshold (box heights/second)",
                min_value=0.05,
                max_value=20.0,
                value=1.0,
                step=0.05,
                disabled=initial.running,
                help=(
                    "Relative image speed, not mph or km/h. Recalibrate this value "
                    "against your re-collected tracks."
                ),
            )
            minimum_speed_observations = st.number_input(
                "Minimum speed observations",
                min_value=2,
                max_value=30,
                value=3,
                disabled=initial.running,
            )
            minimum_speed_duration_seconds = st.number_input(
                "Minimum speed duration (seconds)",
                min_value=0.1,
                max_value=5.0,
                value=0.5,
                step=0.1,
                disabled=initial.running,
            )
            event_crop_minimum_width = st.number_input(
                "Minimum vehicle crop width",
                min_value=1,
                max_value=1920,
                value=200,
                disabled=initial.running,
            )
            event_crop_minimum_height = st.number_input(
                "Minimum vehicle crop height",
                min_value=1,
                max_value=1080,
                value=100,
                disabled=initial.running,
            )
            event_crop_horizontal_padding = st.number_input(
                "Horizontal crop padding",
                min_value=0.0,
                max_value=0.5,
                value=0.35,
                step=0.05,
                disabled=initial.running,
                help=(
                    "Extra context on both sides of the vehicle box. The larger "
                    "default preserves plates on front bumpers that extend beyond "
                    "the general vehicle detector box."
                ),
            )
            event_crop_vertical_padding = st.number_input(
                "Vertical crop padding",
                min_value=0.0,
                max_value=0.5,
                value=0.10,
                step=0.05,
                disabled=initial.running,
            )
            st.markdown("**Live ANPR/OCR**")
            anpr_plate_model = st.text_input(
                "Plate detector model",
                value="models/license-plate.pt",
                disabled=initial.running,
            )
            anpr_detector_confidence = st.slider(
                "Plate detector confidence",
                min_value=0.05,
                max_value=0.95,
                value=0.35,
                step=0.05,
                disabled=initial.running,
            )
            anpr_image_size = st.select_slider(
                "Plate detector image size",
                options=(640, 960, 1280),
                value=1280,
                disabled=initial.running,
            )
            anpr_minimum_ocr_confidence = st.slider(
                "Minimum OCR confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                disabled=initial.running,
            )
            anpr_minimum_agreement = st.number_input(
                "Required agreeing OCR frames",
                min_value=1,
                max_value=3,
                value=2,
                disabled=initial.running,
            )
            refresh_seconds = st.number_input(
                "Dashboard refresh seconds",
                min_value=0.1,
                max_value=5.0,
                value=0.2,
                step=0.1,
            )
            st.markdown("**Adaptive camera (Phases 6-7)**")
            camera_adaptation_mode = st.selectbox(
                "Camera adaptation",
                ("off", "recommend", "automatic"),
                format_func=lambda value: {
                    "off": "Off",
                    "recommend": "Recommend only",
                    "automatic": "Automatic (experimental)",
                }[value],
                disabled=initial.running,
                help=(
                    "Recommendations never write settings. Automatic mode uses "
                    "only allowlisted presets after repeated ROI measurements."
                ),
            )
            camera_control_url = st.text_input(
                "IP Webcam control URL",
                value=os.environ.get("ROUNDABOUT_CAMERA_CONTROL_URL", ""),
                placeholder="http://192.168.1.142:8080",
                disabled=initial.running,
                help="Explicit base URL; do not include /video.",
            )
            camera_quality_interval_seconds = st.number_input(
                "Quality evaluation interval (seconds)",
                min_value=1.0,
                max_value=300.0,
                value=5.0,
                disabled=initial.running,
            )
            camera_minimum_dwell_seconds = st.number_input(
                "Minimum profile dwell (seconds)",
                min_value=0.0,
                max_value=3600.0,
                value=300.0,
                disabled=initial.running,
            )
            camera_switch_cooldown_seconds = st.number_input(
                "Failed-switch cooldown (seconds)",
                min_value=0.0,
                max_value=3600.0,
                value=60.0,
                disabled=initial.running,
            )
            camera_automatic_confirmed = st.checkbox(
                "I reviewed the presets and accept automatic camera changes",
                value=False,
                disabled=initial.running,
                help=(
                    "Required for automatic mode. Test each preset manually under "
                    "representative road conditions first."
                ),
            )
            camera_validated_profiles_file = st.text_input(
                "Validated OCR profile mapping",
                value=os.environ.get("ROUNDABOUT_CAMERA_PROFILE_MAPPING", ""),
                placeholder="Optional: data/camera/validated_profiles.json",
                disabled=initial.running,
                help=(
                    "Optional Phase 7 mapping backed by minimum sample coverage, "
                    "operator approval, and non-regressing held-out false reads."
                ),
            )
            camera_minimum_profile_samples = st.number_input(
                "Minimum samples per validated profile",
                min_value=1,
                max_value=1000,
                value=30,
                disabled=initial.running,
            )

        worker.set_controls(
            OverlayControls(
                confidence=confidence,
                show_detections=show_detections,
                show_scene=show_scene,
                show_metrics=show_metrics,
            )
        )
        start_column, stop_column = st.columns(2)
        start_clicked = start_column.button(
            "Start", type="primary", disabled=initial.running, width="stretch"
        )
        stop_clicked = stop_column.button(
            "Stop", disabled=not initial.running, width="stretch"
        )

        if start_clicked:
            try:
                started = worker.start(
                    ProcessingConfig(
                        url=url,
                        model=model,
                        device=device,
                        confidence=confidence,
                        image_size=int(image_size),
                        classes=ROAD_USER_CLASSES,
                        scene_config=Path(scene_path) if scene_path.strip() else None,
                        minimum_track_age=int(minimum_track_age),
                        maximum_missing_frames=int(maximum_missing_frames),
                        fast_speed_threshold=float(fast_speed_threshold),
                        minimum_speed_observations=int(minimum_speed_observations),
                        minimum_speed_duration_seconds=float(
                            minimum_speed_duration_seconds
                        ),
                        event_file=Path(event_file),
                        save_event_images=save_event_images,
                        event_crop_horizontal_padding=float(
                            event_crop_horizontal_padding
                        ),
                        event_crop_vertical_padding=float(event_crop_vertical_padding),
                        event_crop_minimum_width=int(event_crop_minimum_width),
                        event_crop_minimum_height=int(event_crop_minimum_height),
                        live_anpr=live_anpr,
                        anpr_plate_model=Path(anpr_plate_model),
                        anpr_detector_confidence=float(anpr_detector_confidence),
                        anpr_image_size=int(anpr_image_size),
                        anpr_minimum_ocr_confidence=float(anpr_minimum_ocr_confidence),
                        anpr_minimum_agreement=int(anpr_minimum_agreement),
                        camera_adaptation_mode=camera_adaptation_mode,
                        camera_control_url=camera_control_url.strip() or None,
                        camera_quality_interval_seconds=float(
                            camera_quality_interval_seconds
                        ),
                        camera_minimum_dwell_seconds=float(
                            camera_minimum_dwell_seconds
                        ),
                        camera_switch_cooldown_seconds=float(
                            camera_switch_cooldown_seconds
                        ),
                        camera_automatic_confirmed=camera_automatic_confirmed,
                        camera_validated_profiles_file=Path(
                            camera_validated_profiles_file
                        )
                        if camera_validated_profiles_file.strip()
                        else None,
                        camera_minimum_profile_samples=int(
                            camera_minimum_profile_samples
                        ),
                    )
                )
                if started:
                    st.rerun()
            except ValueError as exc:
                st.error(str(exc))
        if stop_clicked:
            if worker.stop():
                st.rerun()

    @st.fragment(run_every=float(refresh_seconds))
    def live_panel() -> None:
        snapshot = worker.snapshot()
        status = _status_text(snapshot)
        if snapshot.status in {"error", "offline", "reconnecting"}:
            message = snapshot.status_message or status
            st.warning(f"{status}: {message}")
        else:
            st.info(status)
        if snapshot.person_visible:
            st.warning("Person detected in the configured road ROI.")

        if snapshot.camera_adaptation_mode != "off":
            st.subheader("Adaptive camera")
            profile_columns = st.columns(4)
            profile_columns[0].metric("Mode", snapshot.camera_adaptation_mode.title())
            profile_columns[1].metric(
                "Current profile", snapshot.camera_current_profile or "Baseline"
            )
            profile_columns[2].metric(
                "Recommended", snapshot.camera_recommended_profile or "Collecting"
            )
            quality = snapshot.camera_quality or {}
            profile_columns[3].metric(
                "ROI brightness",
                "n/a"
                if "luminance_median" not in quality
                else f"{quality['luminance_median']:.0f}/255",
            )
            st.caption(snapshot.camera_control_status)
            if snapshot.camera_last_changed_at:
                st.caption(f"Last verified change: {snapshot.camera_last_changed_at}")
            if snapshot.camera_current_settings:
                st.caption(
                    "Verified settings: "
                    + ", ".join(
                        f"{name}={value}"
                        for name, value in sorted(
                            snapshot.camera_current_settings.items()
                        )
                    )
                )
            if quality:
                st.caption(
                    f"ROI sharpness {quality['sharpness']:.1f} · "
                    f"underexposed {quality['underexposed_ratio']:.1%} · "
                    f"overexposed {quality['overexposed_ratio']:.1%} · "
                    f"noise {quality['noise']:.1f}"
                )
            if snapshot.running:
                selected_profile = st.selectbox(
                    "Manual camera profile",
                    ("day", "glare", "dusk", "night"),
                    key="manual_camera_profile",
                )
                apply_column, rollback_column = st.columns(2)
                if apply_column.button("Apply profile", width="stretch"):
                    worker.request_camera_profile(selected_profile)
                if rollback_column.button("Roll back", width="stretch"):
                    worker.request_camera_rollback()
            performance_rows = adaptive_camera_performance_rows(snapshot)
            st.markdown("**Recent OCR performance by verified profile**")
            if performance_rows:
                st.dataframe(
                    performance_rows,
                    column_config={
                        "plate_detection_rate": st.column_config.NumberColumn(
                            "Plate detected", format="percent"
                        ),
                        "ocr_acceptance_rate": st.column_config.NumberColumn(
                            "OCR accepted", format="percent"
                        ),
                        "mean_accepted_ocr_confidence": (
                            st.column_config.NumberColumn(
                                "Mean accepted OCR confidence", format="%.3f"
                            )
                        ),
                        "mean_plate_detector_confidence": (
                            st.column_config.NumberColumn(
                                "Mean plate confidence", format="%.3f"
                            )
                        ),
                        "mean_plate_sharpness": st.column_config.NumberColumn(
                            "Mean plate sharpness", format="%.1f"
                        ),
                    },
                    hide_index=True,
                    width="stretch",
                )
                st.caption(
                    "Recent in-memory events only. Acceptance coverage is a live "
                    "proxy; held-out labels are still required to measure false reads."
                )
            else:
                st.caption("No crossing evidence for profile comparison yet.")

        status_columns = st.columns(6)
        status_columns[0].metric("Capture FPS", f"{snapshot.capture_fps:.1f}")
        status_columns[1].metric("Inference FPS", f"{snapshot.inference_fps:.1f}")
        status_columns[2].metric("Inference", f"{snapshot.inference_ms:.1f} ms")
        status_columns[3].metric("Crossings", snapshot.total_crossings)
        status_columns[4].metric("Processed", snapshot.frames_processed)
        status_columns[5].metric("Reconnects", snapshot.reconnects)

        frame_column, summary_column = st.columns((2, 1))
        with frame_column:
            st.subheader("Live annotated frame")
            if snapshot.latest_frame is None:
                st.info("Start processing and wait for the first camera frame.")
            else:
                st.image(snapshot.latest_frame, channels="BGR", width="stretch")
                age = (
                    "n/a"
                    if snapshot.frame_age_ms is None
                    else f"{snapshot.frame_age_ms:.0f} ms"
                )
                st.caption(
                    f"Internal frame age: {age} · received: {snapshot.frames_received} "
                    f"· overwritten: {snapshot.frames_overwritten} "
                    f"· read failures: {snapshot.read_failures}"
                )
        with summary_column:
            st.subheader("Current objects")
            if snapshot.object_counts:
                st.dataframe(
                    [
                        {"object_class": name, "visible": count}
                        for name, count in sorted(snapshot.object_counts.items())
                    ],
                    hide_index=True,
                    width="stretch",
                )
            else:
                st.caption("No road users in the latest processed frame.")
            st.subheader("Traffic totals")
            chart_rows = crossing_chart_rows(snapshot)
            if chart_rows:
                st.bar_chart(chart_rows, x="category", y="count")
            else:
                st.caption("No confirmed line crossings in this run.")

        st.subheader("Recent events")
        rows = event_table_rows(snapshot)
        if rows:
            st.dataframe(
                rows,
                column_order=(
                    "timestamp",
                    "preview_image",
                    "object_class",
                    "direction",
                    "speed_class",
                    "normalized_speed",
                    "ocr_plate",
                    "ocr_status",
                    "ocr_confidence",
                    "ocr_observation_count",
                    "ocr_agreement",
                    "ocr_reasons",
                    "plate_detected",
                    "plate_detector_confidence",
                    "plate_width",
                    "plate_height",
                    "plate_sharpness",
                    "camera_profile",
                    "camera_condition",
                    "line_name",
                    "track_id",
                    "detection_confidence",
                ),
                column_config={
                    "preview_image": st.column_config.ImageColumn(
                        "Crossing image",
                        width="small",
                        help=(
                            "Raw crossing crop when suitable; annotated event "
                            "snapshot otherwise. Available when event images are enabled."
                        ),
                    ),
                    "direction": "Crossing direction",
                    "speed_class": "Speed class",
                    "normalized_speed": st.column_config.NumberColumn(
                        "Relative speed", format="%.2f"
                    ),
                    "ocr_plate": "OCR plate (masked)",
                    "ocr_confidence": st.column_config.NumberColumn(
                        "OCR confidence", format="%.3f"
                    ),
                },
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption(
                "Events will appear after a tracked object crosses a count line."
            )

    live_panel()


def main() -> None:
    """Launch the packaged dashboard on localhost through Streamlit's CLI."""

    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("dashboard_app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false",
        *sys.argv[1:],
    ]
    streamlit_cli.main()
