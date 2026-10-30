# AIaccess — Falling Object Detection & Tracking

A configurable computer-vision proof of concept for a fixed camera observing a conveyor, objects, and an arm. The project uses a YOLO tracking pipeline, normalized conveyor/ground regions, temporal fall confirmation, evidence capture, and a modular HTTP event adapter.

## Quick start

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set YOLO_MODEL_PATH to your YOLO26n weights path in .env
streamlit run app.py
```

Select a video or camera index in the Streamlit sidebar. Adjust the normalized polygons in `config.yaml` (coordinates range from 0 to 1) to fit your fixed-camera view.

## Event API

Set `API_ENABLED=true`, `API_URL`, optional `API_BEARER_TOKEN`, and `API_TIMEOUT_SECONDS` in `.env`. The adapter emits a payload for each locally confirmed fall. It is deliberately modular for integration with a separately developed API POC.

## Evidence and event flags

Artifacts are written to `fallenobjects/<event_id>/`: fall-motion frame, ground-confirmation frame, and `metadata.json`. Confirmed event payloads include `object_fallen=true` and `arm_approach=true`.

## Tests

Run `pytest -q`. The current tests cover basic polygon geometry and API-adapter configuration behavior.

## Calibration and limitations

This is a POC, not a production accuracy guarantee. The pretrained model's class coverage depends on its training data; custom shapes/objects may need fine-tuning. Calibrate conveyor, ground, and optional arm-exclusion zones and temporal thresholds against representative footage. The API call is synchronous; add durable queuing/retry and authentication policy as required for production deployment. Validate camera performance and event accuracy in the target environment before operational use.
