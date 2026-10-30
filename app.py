import time
import tempfile
from pathlib import Path
import cv2
import streamlit as st
from fallwatch.config import load_settings
from fallwatch.pipeline import FallPipeline

st.set_page_config(page_title="AIaccess | Falling Object Monitor", layout="wide")
st.title("Falling Object Detection & Tracking")
st.caption("POC dashboard · fixed-camera conveyor and ground zones · configurable event API")

with st.sidebar:
    st.header("Input")
    mode = st.radio("Source", ["Video file", "Camera index"])
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"]) if mode == "Video file" else None
    camera_index = st.number_input("Camera index", min_value=0, value=0, step=1)
    start = st.button("Start / reset processing", type="primary")
    stop = st.button("Stop")
    st.markdown("**Calibration:** edit normalized conveyor/ground polygons in `config.yaml`.")

if start:
    st.session_state.running = True
    st.session_state.pipeline = None
if stop:
    st.session_state.running = False

if "running" not in st.session_state:
    st.session_state.running = False

if st.session_state.running:
    if mode == "Video file":
        if uploaded is None:
            st.warning("Upload a video first.")
            st.stop()
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            source_path = tmp.name
        cap = cv2.VideoCapture(source_path)
    else:
        cap = cv2.VideoCapture(int(camera_index))

    if not cap.isOpened():
        st.error("Could not open the selected video/camera source.")
        st.session_state.running = False
        st.stop()

    try:
        settings = load_settings()
        pipeline = FallPipeline(settings)
        st.session_state.pipeline = pipeline
        video_slot = st.empty()
        status = st.empty()
        event_slot = st.container()
        frame_no = 0
        while st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                status.info("Source ended or frame could not be read.")
                break
            processed, _, events = pipeline.process(frame, time.time())
            video_slot.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            status.write(f"Frame {frame_no} · confirmed events: {len(pipeline.events)}")
            for ev in events:
                with event_slot:
                    st.success(f"Fall confirmed: {ev['class_name']} · event {ev['event_id']}")
                    st.json(ev)
            frame_no += 1
    except Exception as exc:
        st.exception(exc)
    finally:
        cap.release()
        st.session_state.running = False

st.subheader("Recent confirmed events")
pipeline = st.session_state.get("pipeline")
if pipeline and pipeline.events:
    st.dataframe(pipeline.events[::-1], use_container_width=True)
else:
    st.info("Start a source to process frames. Calibrate zones for the camera before evaluating detection.")
