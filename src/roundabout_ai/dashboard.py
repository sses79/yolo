"""Streamlit presentation layer for the roundabout processing worker."""

from __future__ import annotations

import logging
import os
import sys
from collections import Counter
from pathlib import Path

import streamlit as st

from roundabout_ai.detector import ROAD_USER_CLASSES
from roundabout_ai.diagnostic import DEFAULT_URL
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
) -> list[dict[str, str | int | float | None]]:
    return [event.as_dict() for event in snapshot.recent_events]


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
            "Save event snapshots",
            value=False,
            disabled=initial.running,
            help="Off by default. When enabled, crossing frames are retained locally.",
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
            refresh_seconds = st.number_input(
                "Dashboard refresh seconds",
                min_value=0.1,
                max_value=5.0,
                value=0.2,
                step=0.1,
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
                        event_file=Path(event_file),
                        save_event_images=save_event_images,
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
            st.dataframe(rows, hide_index=True, width="stretch")
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
