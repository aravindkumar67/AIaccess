"""Streamlit dashboard for the Falling Object Detection POC."""
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import streamlit as st

from fallwatch.config import load_settings
from fallwatch.pipeline import FallPipeline

st.set_page_config(page_title="AIaccess | FallWatch", page_icon="📡", layout="wide")
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {background:#0b1020;color:#e8edf7}
[data-testid="stSidebar"] {background:#111a2e}
div[data-testid="stMetric"] {background:#151f35;border:1px solid #283652;padding:12px;border-radius:10px}
h1,h2,h3 {color:#e8edf7}
</style>
""", unsafe_allow_html=True)

st.title("📡 FallWatch")
st.caption("AIaccess · Fixed-camera object fall detection & tracking · Local POC (no robot/API control)")

with st.sidebar:
    st.subheader("SOURCE")
    mode = st.radio("Input", ["Video file", "Camera"], horizontal=True)
    uploaded = st.file_uploader("Choose recorded video", type=["mp4", "avi", "mov", "mkv"]) if mode == "Video file" else None
    camera = int(st.number_input("Camera index", min_value=0, max_value=10, value=0)) if mode == "Camera" else 0
    model_path = st.text_input("YOLO weights path", os.getenv("YOLO_MODEL_PATH", "yolo26n.pt"))
    conf = st.slider("Confidence", 0.05, 0.95, 0.30, 0.05)
    st.divider()
    start = st.button("▶ Start / process", type="primary", use_container_width=True)
    stop = st.button("■ Stop", use_container_width=True)
    st.caption("Calibrate normalized conveyor/ground/arm polygons in config.yaml.")

if "run" not in st.session_state:
    st.session_state.run = False
if "events" not in st.session_state:
    st.session_state.events = []
if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False

left, right = st.columns([2.15, 1])
with left:
    st.subheader("LIVE VIEW")
    view = st.empty()
    state_line = st.empty()
with right:
    st.subheader("EVENT MONITOR")
    metrics = st.empty()
    flag_view = st.empty()
    event_list = st.empty()

if not st.session_state.run:
    view.info("Choose a video or camera source, then press Start / process.")
    metrics.metric("Confirmed falls", len(st.session_state.events))
    flag_view.code("object_fallen = False\narm_approach = False")
    state_line.caption("Waiting for input")
    event_list.dataframe(st.session_state.events[-10:][::-1], use_container_width=True)
    st.stop()

if mode == "Video file" and uploaded is None:
    st.warning("Select a video file first.")
    st.session_state.run = False
    st.stop()

source = camera
temp_path = None
if mode == "Video file":
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded.getbuffer())
        temp_path = temp_file.name
    source = temp_path

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    st.error("Cannot open the selected source.")
    st.session_state.run = False
    if temp_path:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
    st.stop()

stop_capture = threading.Event()
lock = threading.Lock()
shared = {"frame": None, "timestamp": 0.0, "done": False, "error": None}

def capture_loop():
    """Only this worker owns VideoCapture; it never reads Streamlit session state."""
    try:
        while not stop_capture.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            with lock:
                shared["frame"] = frame
                shared["timestamp"] = time.monotonic()
            # Avoid a tight, CPU-heavy loop when processing a prerecorded clip.
            if mode == "Video file":
                time.sleep(0.002)
    except Exception as exc:
        shared["error"] = str(exc)
    finally:
        shared["done"] = True

worker = None
try:
    settings = load_settings()
    settings.model_path = model_path
    settings.config["confidence"] = conf
    pipeline = FallPipeline(settings)

    worker = threading.Thread(target=capture_loop, daemon=True, name="opencv-capture")
    worker.start()
    last_timestamp = None

    while not shared["done"]:
        # A Stop click triggers a Streamlit rerun; that rerun flips run=False.
        # The current script is then interrupted by Streamlit, and finally sets the event.
        if not st.session_state.run:
            break

        with lock:
            frame = shared["frame"].copy() if shared["frame"] is not None else None
            timestamp = shared["timestamp"]

        if frame is None or timestamp == last_timestamp:
            time.sleep(0.01)
            continue

        last_timestamp = timestamp
        rendered, detections, new_events = pipeline.process(frame, timestamp)
        view.image(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        states = [memory.state for memory in pipeline.analyzer.tracks.values()]
        metrics.metric("Confirmed falls", len(st.session_state.events) + len(pipeline.events))
        # Flags are event pulses: True only on the frame where a new fall is confirmed.
        confirmed_now = bool(new_events)
        flag_view.code(f"object_fallen = {confirmed_now}\narm_approach = {confirmed_now}")
        state_line.caption(
            f"Frame {pipeline.frame_id} · active detections {len(detections)} · "
            f"states: {', '.join(states[-5:]) if states else 'waiting'}"
        )

        if new_events:
            st.session_state.events.extend(new_events)
            for event in new_events:
                st.toast(
                    f"Fall confirmed: {event.get('class_name', 'object')} · {event.get('event_id', '')}",
                    icon="⚠️",
                )
        event_list.dataframe(st.session_state.events[-10:][::-1], use_container_width=True)

    if shared["error"]:
        st.error(f"Camera/video capture error: {shared['error']}")
except Exception as exc:
    st.exception(exc)
finally:
    stop_capture.set()
    cap.release()
    if worker is not None:
        worker.join(timeout=2.0)
    st.session_state.run = False
    if temp_path:
        try:
            os.unlink(temp_path)
        except OSError:
            pass
