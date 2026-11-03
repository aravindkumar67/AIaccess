import streamlit as st
import cv2
import os
from core.state import SharedState
from core.pipeline import Pipeline

st.set_page_config(page_title="Fallen Object POC Dashboard", layout="wide")
st.title("Fallen Object POC - Detailed Implementation")

sidebar = st.sidebar
sidebar.header("Configuration & Stream Parameters")
video_file = sidebar.file_uploader("Upload Test Video", type=["mp4", "avi"])

if video_file:
    temp_path = os.path.join("test_videos", video_file.name)
    with open(temp_path, "wb") as f:
        f.write(video_file.getbuffer())
    st.success(f"Loaded input video: {video_file.name}")

st.info("System Ready. Multi-threaded processor initialized to monitor corridors, fallen components, and AMR approach trajectories.")