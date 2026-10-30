# AIaccess — Falling Object Detection & Tracking

A configurable computer-vision proof of concept for a fixed-camera scene containing a conveyor, common objects/geometric shapes, a robot arm, and ground. The POC supports two fall patterns: an object falling from height and an object tipping/falling from the conveyor.

## Decisions
- Fixed camera; video-file and camera-index input.
- YOLO26n inference with object tracking (ByteTrack through Ultralytics).
- OpenCV capture thread plus an inference worker; Streamlit presents the latest frame and event state.
- Temporal state machine: candidate → airborne → ground contact → settling → confirmed. A disappearance alone is not a confirmed fall.
- Configurable local flags: `object_fallen` and `arm_approach` become true only on confirmed event. `arm_approach` is a software signal only; no robot control is issued.
- Per-event evidence: event images, a video clip, and JSON metadata under `fallenobjects/<event_id>/`.
- No external API calls in this POC.

## Setup
```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
# Set model_path in config.yaml or environment to your YOLO26n weights
streamlit run app.py
```

Calibrate normalized regions in `config.yaml` (x/y coordinates from 0.0 to 1.0) for conveyor, ground, and optional arm exclusion. Thresholds are camera-specific and must be validated with representative footage.

## Test
```bash
pytest -q
python scripts/generate_test_videos.py
```

Generated clips are synthetic test fixtures for checking pipeline behavior and event timing; they are not realistic camera footage and cannot establish real-world detection accuracy. Fine-tune the detector for object classes that are not represented by the pretrained model. This is a POC, not a production safety system.