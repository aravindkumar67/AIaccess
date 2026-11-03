from collections import defaultdict, deque


class FallStateMachine:
    """Conservative image-space fall heuristic; calibrate for camera and scene."""

    def __init__(self, config):
        self.config = config
        self.history = defaultdict(lambda: deque(maxlen=max(3, int(config.get("history_size", 24)))))
        self.emitted = set()
        self.min_observations = int(config.get("min_track_observations", 5))
        self.min_drop = float(config.get("departure_vertical_pixels", 18))
        self.ground_ratio = float(config.get("ground_y_ratio", 0.63))

    def update_track(self, detection, frame_index, fps, frame_height):
        tid = detection.get("track_id")
        if tid is None:
            return None
        cx, cy = detection["center"]
        bottom_y = detection["bottom_center"][1]
        hist = self.history[tid]
        hist.append((frame_index, cx, cy, bottom_y))
        if len(hist) < self.min_observations:
            return None

        recent = list(hist)
        displacement = recent[-1][2] - recent[0][2]
        lower_region = bottom_y >= frame_height * self.ground_ratio
        if displacement < self.min_drop or not lower_region or tid in self.emitted:
            return None

        self.emitted.add(tid)
        return {
            "event_id": f"FALL-{tid}-{frame_index}",
            "event": "fall_like_motion",
            "track_id": int(tid),
            "class_name": detection.get("class_name", "object"),
            "confidence": round(float(detection.get("confidence", 0.0)), 3),
            "frame": int(frame_index),
            "time_sec": round(frame_index / max(float(fps), 1.0), 2),
            "bbox": [round(float(v), 1) for v in detection["bbox"]],
            "risk": "medium",
            "basis": "downward image-space displacement into configured lower region",
        }
