from pathlib import Path
import streamlit as st
from core.pipeline import Pipeline
from core.state import SharedState

PROJECT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = PROJECT_DIR / "test_videos"
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Fallen Object POC Dashboard", layout="wide")
st.title("Fallen Object POC - Detailed Implementation")
st.sidebar.header("Configuration & Stream Parameters")
video_file = st.sidebar.file_uploader("Upload Test Video", type=["mp4", "avi"])

if video_file is not None:
    safe_name = Path(video_file.name).name
    temp_path = VIDEO_DIR / safe_name
    temp_path.write_bytes(video_file.getbuffer())
    st.success(f"Loaded input video: {safe_name}")
    st.caption(f"Saved to: {temp_path}")

st.info("System Ready. Multi-threaded processor initialized to monitor corridors, fallen components, and AMR approach trajectories.")
