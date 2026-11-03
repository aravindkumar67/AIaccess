# AIaccess — Fallen Object POC (v2.2)

Streamlit proof of concept for monitoring fallen objects, conveyor regions, and AMR approach risk. This repository uses the agreed layout.

## Layout

- `app.py`: Streamlit dashboard entrypoint.
- `core/`: state, perception, geometry, event logic, evidence, and pipeline modules.
- `config.yaml`: detection and event thresholds.
- `camera_intrinsics.yaml` and `calibration.yaml`: camera/scene calibration placeholders.
- `test_videos/`: sample clips and uploaded videos.
- `eval/evaluate.py`: evidence metric summary.
- `evidence/`: categorized evidence output.

## Run

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Calibration files are placeholders, not measured camera calibration. Populate them before treating coordinates or risk metrics as real-world measurements.
