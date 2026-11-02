"""Streamlit UI; processing runs in a worker and the UI polls thread-safe state."""
import os
import tempfile
import threading
import time
from pathlib import Path

import cv2
import streamlit as st

from fallwatch.config import load_settings
from fallwatch.pipeline import FallPipeline
from fallwatch.state import RuntimeState

st.set_page_config(page_title="AIaccess | FallWatch", page_icon="📡", layout="wide")
st.title("FallWatch — Falling Object Detection")
st.caption("Pretrained YOLO26n · video upload · fixed-camera fall events · local-only POC")

with st.sidebar:
    uploaded = st.file_uploader("Recorded video", type=["mp4", "avi", "mov", "mkv"])
    model_path = st.text_input("YOLO weights", os.getenv("YOLO_MODEL_PATH", "yolo26n.pt"))
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.30, 0.05)
    start_clicked = st.button("Start processing", type="primary", disabled=uploaded is None)
    stop_clicked = st.button("Stop processing")
    reset_clicked = st.button("Reset event flags")
    st.caption("Calibrate conveyor, ground, and optional arm-exclusion polygons in config.yaml.")

if "fallwatch_runtime" not in st.session_state:
    st.session_state.fallwatch_runtime = RuntimeState()
if "fallwatch_worker" not in st.session_state:
    st.session_state.fallwatch_worker = None
if "fallwatch_stop" not in st.session_state:
    st.session_state.fallwatch_stop = None
if "fallwatch_temp" not in st.session_state:
    st.session_state.fallwatch_temp = None

runtime = st.session_state.fallwatch_runtime

def process_video(path, settings, state, stop_event):
    """Worker owns the model and processing loop; never accesses Streamlit APIs."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        state.update(status="error", error="OpenCV could not open the uploaded video.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else float(settings.config.get("fps", 20))
    pipeline = None
    try:
        pipeline = FallPipeline(settings)
        state.update(status="running", error=None)
        frame_id = 0
        started = time.perf_counter()
        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break
            frame_id += 1
            video_time = (frame_id - 1) / fps
            infer_started = time.perf_counter()
            annotated, detections, new_events, tracks = pipeline.process(frame, video_time)
            infer_ms = (time.perf_counter() - infer_started) * 1000
            for event in new_events:
                state.add_event(event)
            state.update(
                frame=annotated, frame_id=frame_id, video_time=video_time,
                tracks=tracks, inference_ms=infer_ms,
                fps=frame_id / max(time.perf_counter() - started, 1e-6),
            )
        state.update(status="stopped" if stop_event.is_set() else "completed")
    except Exception as exc:
        state.update(status="error", error=f"{type(exc).__name__}: {exc}")
    finally:
        cap.release()

if start_clicked and uploaded is not None:
    # Stop a previous worker before starting a new session.
    previous_stop = st.session_state.fallwatch_stop
    if previous_stop is not None:
        previous_stop.set()
    old_path = st.session_state.fallwatch_temp
    if old_path:
        try:
            os.unlink(old_path)
        except OSError:
            pass
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getvalue())
        video_path = tmp.name
    st.session_state.fallwatch_temp = video_path
    runtime = RuntimeState()
    st.session_state.fallwatch_runtime = runtime
    settings = load_settings()
    settings.model_path = model_path
    settings.config["confidence"] = confidence
    stop_event = threading.Event()
    worker = threading.Thread(
        target=process_video, args=(video_path, settings, runtime, stop_event),
        daemon=True, name="fallwatch-video-worker",
    )
    st.session_state.fallwatch_stop = stop_event
    st.session_state.fallwatch_worker = worker
    worker.start()

if stop_clicked and st.session_state.fallwatch_stop is not None:
    st.session_state.fallwatch_stop.set()
if reset_clicked:
    runtime.reset_flags()

@st.fragment(run_every=0.2)
def render_monitor():
    snapshot = st.session_state.fallwatch_runtime.snapshot()
    video_col, status_col = st.columns([2, 1])
    with video_col:
        st.subheader("Video")
        if snapshot["frame"] is not None:
            st.image(cv2.cvtColor(snapshot["frame"], cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        else:
            st.info("Start processing to see the annotated video.")
        st.caption(f"Status: {snapshot['status']} · Frame: {snapshot['frame_id']} · Video time: {snapshot['video_time']:.2f}s")
    with status_col:
        st.subheader("Fall event flags")
        st.metric("object_fallen_onto_ground", str(snapshot["object_fallen_onto_ground"]))
        st.metric("arm_object_approach", str(snapshot["arm_object_approach"]))
        st.metric("Confirmed events", len(snapshot["events"]))
        st.metric("Processing FPS", f"{snapshot['fps']:.1f}")
        st.metric("Inference time", f"{snapshot['inference_ms']:.1f} ms")
    if snapshot["error"]:
        st.error(snapshot["error"])
    st.subheader("Tracked objects")
    st.dataframe(snapshot["tracks"], use_container_width=True, hide_index=True)
    st.subheader("Event history")
    st.dataframe(snapshot["events"][::-1], use_container_width=True, hide_index=True)

render_monitor()
