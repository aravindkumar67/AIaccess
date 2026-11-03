from pathlib import Path
import yaml
import streamlit as st
import pandas as pd

from core.pipeline import Pipeline
from core.state import SharedState

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "test_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AIAccess | Fallen Object POC", layout="wide")
st.title("AIAccess — Fallen Object Detection POC")
st.caption("YOLO + ByteTrack video analysis. Camera regions and thresholds require scene calibration.")

config_path = ROOT / "config.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
samples = ["— choose sample —"] + [p.name for p in sorted(VIDEO_DIR.glob("*.mp4"))]
sample = st.sidebar.selectbox("Or choose a video from test_videos", samples)
run = st.sidebar.button("Run analysis", type="primary", disabled=uploaded is None and sample.startswith("—"))

if uploaded is None and sample.startswith("—"):
    st.info("Upload a video or choose a sample, then click Run analysis.")
    st.stop()

if uploaded is not None:
    video_path = VIDEO_DIR / Path(uploaded.name).name
    video_path.write_bytes(uploaded.getvalue())
else:
    video_path = VIDEO_DIR / sample

st.success(f"Input: {video_path.name}")
if not run:
    st.video(str(video_path))
    st.info("Click Run analysis to start inference.")
    st.stop()

out_path = VIDEO_DIR / f"{video_path.stem}_annotated.mp4"
state = SharedState()
progress = st.progress(0, text="Loading model…")
preview = st.empty()
status = st.empty()
try:
    pipeline = Pipeline(str(video_path), config, state)
    def update_progress(fraction, frame):
        progress.progress(max(0, min(100, int(fraction * 100))), text=f"Processed {state.frames_processed} frames")
        preview.image(frame, channels="BGR", use_container_width=True)
    summary = pipeline.process_all(str(out_path), update_progress)
    progress.progress(100, text="Finished")
    status.success(f"Processed {summary['frames_processed']} frames.")
    st.subheader("Annotated output")
    if out_path.exists() and out_path.stat().st_size:
        st.video(str(out_path))
    st.subheader("Events")
    if state.events:
        df = pd.DataFrame(state.events)
        cols = ["event_id","event","Anomaly_Type","track_id","class_name","confidence","frame","time_sec","signal_level","fallen_onto_ground","amr_approach","linked_to","evidence_path"]
        st.dataframe(df[[c for c in cols if c in df.columns]], use_container_width=True)
    else:
        st.info("No confirmed events were generated.")
    a,b,c = st.columns(3)
    a.metric("Frames", summary["frames_processed"])
    b.metric("Tracked IDs", len(state.track_ids))
    c.metric("Events", len(state.events))
    a,b,c = st.columns(3)
    a.metric("fallen_onto_ground", str(state.fallen_onto_ground))
    b.metric("amr_approach", str(state.amr_approach))
    c.metric("Current signal", state.active_level.upper())
    if state.warnings:
        with st.expander("Warnings"):
            for warning in state.warnings:
                st.warning(warning)
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.exception(exc)
