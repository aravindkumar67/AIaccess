# AIaccess — FallWatch

A fixed-camera falling-object detection and tracking POC using pretrained YOLO26n (COCO classes), Ultralytics tracking, OpenCV, and Streamlit. The initial workflow processes a user-uploaded video. It does not send robot commands or require an external API.

## Flow

1. Upload a recorded video in Streamlit.
2. A worker reads frames, runs YOLO tracking, and analyzes each track's motion against configured scene regions.
3. A fall candidate progresses through airborne and ground-contact states.
4. A fall is confirmed promptly after the configured short ground-contact validation window. Settling is monitored afterward and does not delay the initial event notification.
5. Each confirmed event receives a unique event ID and local evidence record.
6. The worker publishes a thread-safe state snapshot. Streamlit's periodic fragment refresh displays the latest frame, flags, tracks, metrics, and event history.
7. The two event flags remain true until the user explicitly resets them. Resetting flags does not remove event history or evidence.

## Model

- Default weights: `yolo26n.pt` (pretrained COCO model).
- The model detects only classes it was pretrained to recognize. It does not automatically detect arbitrary industrial objects or custom geometric shape classes.
- Fine-tuning is not included in this POC.
- Set `YOLO_MODEL_PATH` or enter the weights path in the sidebar.

## Setup

The VM is assumed to have the project's YOLO/PyTorch environment and GPU setup ready. Install application dependencies if needed:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Camera calibration

Edit normalized polygons in `config.yaml` for the video view:
- `conveyor`: region where normal conveyor travel occurs.
- `ground`: region used to validate ground contact.
- `arm_exclusion`: optional area excluded from fall logic.

Thresholds are image-space and camera-dependent. They must be checked on the reference footage; the default polygons are examples, not universal calibration.

## Validation

```bash
pytest -q
python scripts/generate_test_videos.py
```

The unit tests cover fast ground confirmation, conveyor-motion rejection, and sticky frontend state. Synthetic videos are smoke-test fixtures only. They cannot establish real-world precision/recall or event latency. Run the POC on the actual reference video and record false positives, missed events, confirmation latency, frontend update latency, and evidence integrity before calling it validated.

## Evidence

Each confirmed event is saved under `fallenobjects/<event_id>/` with annotated event images, a video clip, and metadata. The app is a POC, not a production safety system.
