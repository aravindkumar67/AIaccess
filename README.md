# Fallen Object POC - Detailed Implementation Plan (v2.2)

Complete implementation of the Fallen Object POC pipeline strictly following Design v2.2 without any deviations.

## Folder Structure
- `core/`: State machines, detector wrappers, zone manager, path network graph, geometry transform, validation, signal evaluation, evidence writers.
- `config.yaml`: Frozen configuration thresholds.
- `camera_intrinsics.yaml` & `calibration.yaml`: Real-world intrinsics and homography point pairs.
- `app.py`: Streamlit dashboard entrypoint.
- `test_videos/`: Test input video clips (`VID-20261102-WA0002.mp4` and `istockphoto-2204511529-640_adpp_is.mp4`).
- `eval/`: Offline performance and accuracy evaluation.
- `evidence/`: Structured evidence folders (`low`, `medium`, `high`, `red`, `uncertain`).

## Quick Start
```bash
pip install streamlit opencv-python pyyaml numpy
streamlit run app.py
```