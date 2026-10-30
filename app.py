"""Streamlit dashboard for the Falling Object Detection POC."""
import os, tempfile, threading, time
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
.metric-card {background:#151f35;border:1px solid #283652;border-radius:12px;padding:14px}
div[data-testid="stMetric"] {background:#151f35;border:1px solid #283652;padding:12px;border-radius:10px}
h1,h2,h3 {color:#e8edf7}
</style>
""", unsafe_allow_html=True)
st.title("📡 FallWatch")
st.caption("AIaccess · Fixed-camera object fall detection & tracking · Local POC (no robot/API control)")
with st.sidebar:
    st.subheader("SOURCE")
    mode=st.radio("Input",["Video file","Camera"],horizontal=True)
    uploaded=st.file_uploader("Choose recorded video",type=["mp4","avi","mov","mkv"]) if mode=="Video file" else None
    camera=int(st.number_input("Camera index",0,10,0)) if mode=="Camera" else 0
    model_path=st.text_input("YOLO weights path",os.getenv("YOLO_MODEL_PATH","yolo26n.pt"))
    conf=st.slider("Confidence",0.05,0.95,0.30,0.05)
    st.divider()
    start=st.button("▶ Start / process",type="primary",use_container_width=True)
    stop=st.button("■ Stop",use_container_width=True)
    st.caption("Calibrate normalized conveyor/ground/arm polygons in config.yaml.")

if "run" not in st.session_state: st.session_state.run=False
if start: st.session_state.run=True
if stop: st.session_state.run=False

left,right=st.columns([2.15,1])
with left:
    st.subheader("LIVE VIEW")
    view=st.empty()
    state_line=st.empty()
with right:
    st.subheader("EVENT MONITOR")
    metrics=st.empty()
    flag_view=st.empty()
    event_list=st.empty()

if st.session_state.run:
    if mode=="Video file" and uploaded is None:
        st.warning("Select a video file first."); st.session_state.run=False; st.stop()
    if mode=="Video file":
        suffix=Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as f:
            f.write(uploaded.getbuffer()); source=f.name
    else: source=camera
    cap=cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Cannot open the selected source."); st.session_state.run=False; st.stop()
    try:
        settings=load_settings()
        settings.model_path=model_path
        settings.config["confidence"]=conf
        pipeline=FallPipeline(settings)
        # Capture thread owns VideoCapture; inference loop consumes the newest frame.
        lock=threading.Lock()
        shared={"frame":None,"timestamp":0.0,"done":False,"error":None}\n        stop_capture=threading.Event()
        def capture_loop():
            try:
                while st.session_state.run:
                    ok,frame=cap.read()
                    if not ok: break
                    with lock:
                        shared["frame"]=frame
                        shared["timestamp"]=time.time()
                    if mode=="Video file": time.sleep(0.002)
            except Exception as exc: shared["error"]=str(exc)
            finally: shared["done"]=True
        worker=threading.Thread(target=capture_loop,daemon=True,name="opencv-capture")
        worker.start()
        last=None; frames=0
        while not stop_capture.is_set() and not shared["done"]:\n            if stop:\n                stop_capture.set()
            with lock:
                frame=shared["frame"].copy() if shared["frame"] is not None else None
                ts=shared["timestamp"]
            if frame is None or ts==last:
                time.sleep(.01); continue
            last=ts
            rendered,detections,new_events=pipeline.process(frame,ts)
            frames+=1
            view.image(cv2.cvtColor(rendered,cv2.COLOR_BGR2RGB),channels="RGB",use_container_width=True)
            states=[m.state for m in pipeline.analyzer.tracks.values()]
            fall_count=sum(1 for e in pipeline.events)
            metrics.metric("Confirmed falls",fall_count)
            flag_view.code("object_fallen = "+str(bool(new_events))+"\narm_approach = "+str(bool(new_events)))
            state_line.caption(f"Frame {pipeline.frame_id} · tracks {len(detections)} · states: {', '.join(states[-5:]) if states else 'waiting'}")
            event_list.dataframe(pipeline.events[-10:][::-1],use_container_width=True)
            for ev in new_events:
                st.toast(f"Fall confirmed: {ev.get('class_name','object')} · {ev.get('event_id','')}",icon="⚠️")
        if shared["error"]: st.error(shared["error"])
    except Exception as exc: st.exception(exc)
    finally:
        st.session_state.run=False
        cap.release()
        if mode=="Video file":
            try: os.unlink(source)
            except OSError: pass
else:
    view.info("Choose a video or camera source, then press Start / process.")
    metrics.metric("Confirmed falls",len(st.session_state.get("events",[])))
    flag_view.code("object_fallen = False\narm_approach = False")
    state_line.caption("Waiting for input")
    event_list.caption("Confirmed event records appear here.")
