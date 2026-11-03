from pathlib import Path
import yaml
import streamlit as st

from core.pipeline import Pipeline
from core.state import SharedState

ROOT = Path(__file__).resolve().parent
VIDEO_DIR = ROOT / "test_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="AIAccess | Fallen Object POC", layout="wide")
st.title("AIAccess — Fallen Object Detection POC")
st.caption("Video inference with YOLO + ByteTrack. Results are heuristic and require scene calibration.")

config_path = ROOT / "config.yaml"
config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
uploaded = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
sample_options = ["— choose uploaded video —"] + [p.name for p in sorted(VIDEO_DIR.glob("*.mp4"))]
sample = st.sidebar.selectbox("Or use a video in test_videos", sample_options)
run = st.sidebar.button("Run analysis", type="primary", disabled=uploaded is None and sample.startswith("—"))

if uploaded is None and sample.startswith("—"):
    st.info("Upload a video or select a sample video, then click Run analysis.")
    st.stop()

if uploaded is not None:
    video_path = VIDEO_DIR / Path(uploaded.name).name
    video_path.write_bytes(uploaded.getvalue())
else:
    video_path = VIDEO_DIR / sample

st.success(f"Input: {video_path.name}")
if not run:
    st.video(str(video_path))
    st.info("Click **Run analysis** to start YOLO inference and fall-event analysis.")
    st.stop()

out_path = VIDEO_DIR / f"{video_path.stem}_annotated.mp4"
state = SharedState()
bar = st.progress(0, text="Starting detector…")
preview = st.empty()
status = st.empty()
try:
    pipeline = Pipeline(str(video_path), config, state)
    def update_progress(fraction, frame):
        bar.progress(max(0, min(100, int(fraction * 100))), text=f"Processing frame {state.frames_processed}…")
        preview.image(frame, channels="BGR", use_container_width=True)
    summary = pipeline.process_all(str(out_path), update_progress)
    bar.progress(100, text="Complete")
    status.success(f"Processed {summary['frames_processed']} frames.")
    st.subheader("Annotated video")
    if out_path.exists() and out_path.stat().st_size:
        st.video(str(out_path))
    else:
        st.warning("No annotated output video was produced.")
    st.subheader("Fall events")
    if state.events:
        st.dataframe(state.events, use_container_width=True)
    else:
        st.info("No fall-like events were flagged by the configured heuristic.")
    a, b, c = st.columns(3)
    a.metric("Frames", summary["frames_processed"])
    b.metric("Unique tracked objects", len(state.track_ids))
    c.metric("Events", len(state.events))
    if state.warnings:
        with st.expander("Warnings"):
            for warning in state.warnings:
                st.warning(warning)
except Exception as exc:
    st.error(f"Analysis failed: {exc}")
    st.exception(exc)
