# AIaccess — Fallen Object Detection POC

Core video pipeline: YOLO detection + ByteTrack identities, fall-to-ground heuristic, independent severity grading, persistent event booleans, AMR approach escalation, and per-event evidence output. Streamlit is only the demo shell; the pipeline can be called directly.

## Run
```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Put the selected YOLO weights (for example, `yolo26n.pt`) at the configured path or use an available model path. Do not commit model weights unless intentionally permitted.

## Event semantics
- A confirmed fall sets `fallen_onto_ground=true` independently of risk-region overlap.
- Severity is graded separately: high-risk polygon overlap => `high`; else yellow path overlap => `medium`; otherwise `low`.
- Confirmed AMR approach to a tracked fallen object sets `amr_approach=true`, emits `signal_level=red`, and links to the fall event via `linked_to`.
- Evidence is saved under `evidence/<level>/<event_id>/` with snapshot, clip, and JSON.

## Calibration warning
The included region polygons and image-space thresholds are provisional. They are not camera calibration or ground-truth metrics. Set polygons from the actual camera image; image-space AMR distance is not meters and path-heading is not inferred here. Validate with representative annotated videos before operational deployment.
