import yaml,streamlit as st
from core.pipeline import VideoPipeline
st.set_page_config(page_title='Fallen Object POC',layout='wide')
st.title('Conveyor Fallen-Object + AMR Approach POC')
st.caption('Provisional image-space prototype. Calibrate and validate against the real camera before operational use.')
with st.sidebar:
    config_path=st.text_input('Config file','config.yaml'); weights=st.text_input('YOLO weights','yolo26n.pt'); video=st.text_input('Video file path',''); go=st.button('Run analysis',type='primary',disabled=not video)
if go:
    with open(config_path,encoding='utf-8') as f: cfg=yaml.safe_load(f)
    framebox=st.empty(); eventbox=st.empty()
    try:
        pipe=VideoPipeline(cfg,weights,video)
        def update(frame,events): framebox.image(frame,channels='BGR',use_container_width=True); eventbox.json(events)
        events=pipe.run(update); st.success(f'Analysis complete: {len(events)} tracked fall event(s).'); st.dataframe(events,use_container_width=True)
    except Exception as e: st.error(f'Pipeline failed: {e}')
else: st.info('Enter a video path and click Run analysis. Evidence writes to outputs/events/.')
