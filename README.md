# Conveyor Fall + AMR Approach POC

Prototype video pipeline: object detection/tracking, fall confirmation, independent severity grading, AMR approach confirmation, and persisted event evidence (snapshot, MP4 clip, JSON).

## Install / run

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Choose video path and YOLO weights (for example yolo26n.pt or custom weights). Evidence output: `outputs/events/<event_id>/`.

## Calibration warning

Values in `config.yaml` are provisional image-space pixels. Replace camera intrinsics/extrinsics and region polygons with actual calibrated values. This POC is not safety-rated and must not be used to control an AMR without field validation. Detection labels must match your model's class names.

Run tests: `pytest -q`.
