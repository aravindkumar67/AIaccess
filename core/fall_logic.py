from collections import defaultdict, deque

class FallStateMachine:
    """Per-track fall-to-floor confirmation with persistent one-shot event emission.

    The heuristic is camera-specific. Confirming a fall requires evidence of
    vertical descent and then a stable track in the configured ground zone.
    """
    def __init__(self, config):
        self.config = config or {}
        self.history = defaultdict(lambda: deque(maxlen=max(4, int(self.config.get("history_size", 24)))))
        self.emitted = set()
        self.min_observations = int(self.config.get("min_track_observations", 5))
        self.min_drop = float(self.config.get("departure_vertical_pixels", 18))
        self.ground_y_ratio = float(self.config.get("ground_y_ratio", 0.63))
        self.stable_frames = int(self.config.get("ground_stable_frames", 5))
        self.stable_motion = float(self.config.get("ground_motion_pixels", 7))
        self.conveyor_residency = int(self.config.get("conveyor_residency_frames", 8))

    def update_track(self, detection, frame_index, fps, frame_height, frame_width=None):
        tid = detection.get("track_id")
        if tid is None:
            return None
        cx, cy = detection["center"]
        bottom_y = detection["bottom_center"][1]
        hist = self.history[int(tid)]
        hist.append((int(frame_index), float(cx), float(cy), float(bottom_y)))
        if len(hist) < self.min_observations or int(tid) in self.emitted:
            return None

        data = list(hist)
        # Look at a short window so an object moving along the belt does not
        # accumulate a false "fall" merely from long-term image perspective.
        window = min(len(data), max(self.min_observations, self.stable_frames + 2))
        recent = data[-window:]
        displacement = recent[-1][2] - recent[0][2]
        in_ground_band = bottom_y >= frame_height * self.ground_y_ratio

        stable = False
        if len(data) >= self.stable_frames:
            stable_data = data[-self.stable_frames:]
            xs = [p[1] for p in stable_data]
            ys = [p[2] for p in stable_data]
            motion = max(
                ((xs[i]-xs[0])**2 + (ys[i]-ys[0])**2)**0.5
                for i in range(len(xs))
            )
            stable = motion <= self.stable_motion

        if displacement < self.min_drop or not in_ground_band or not stable:
            return None

        self.emitted.add(int(tid))
        return {
            "event_id": f"FALL-{int(tid)}-{int(frame_index)}",
            "event": "Fall",
            "Anomaly_Type": "Object_Fallen_From_Conveyor",
            "track_id": int(tid),
            "class_name": detection.get("class_name", "object"),
            "confidence": round(float(detection.get("confidence", 0.0)), 4),
            "frame": int(frame_index),
            "time_sec": round(int(frame_index) / max(float(fps), 1.0), 3),
            "bbox": [round(float(v), 1) for v in detection["bbox"]],
            "fallen_onto_ground": True,
            "amr_approach": False,
            "uncertain": False,
            "coords_source": "image",
            "pixel": [round(float(cx), 1), round(float(bottom_y), 1)],
            "risk": "pending",
            "basis": "downward displacement followed by stable ground-band observations",
        }
