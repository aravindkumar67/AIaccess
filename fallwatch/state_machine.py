"""Temporal fall analysis for tracked detections."""
from collections import deque
from dataclasses import dataclass, field
import math
from .geometry import bbox_center, bbox_bottom_center, point_in_polygon

@dataclass
class TrackMemory:
    history: deque = field(default_factory=lambda: deque(maxlen=24))
    state: str = "UNKNOWN"
    last_seen_frame: int = -1
    ground_frames: int = 0
    ground_anchor: tuple | None = None
    fall_frame: object = None
    emitted: bool = False
    class_name: str = "unknown"
    confidence: float = 0.0
    bbox: tuple = (0, 0, 0, 0)
    event_started_at: float | None = None

class FallAnalyzer:
    """Detect a fall promptly at validated ground contact; settling is tracked separately."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}

    def update(self, detection, frame_id, timestamp, frame):
        size = max(3, int(self.cfg.get("history_size", 24)))
        mem = self.tracks.setdefault(detection.track_id, TrackMemory(history=deque(maxlen=size)))
        mem.last_seen_frame = frame_id
        mem.class_name, mem.confidence, mem.bbox = detection.class_name, detection.confidence, detection.bbox
        center, bottom = bbox_center(detection.bbox), bbox_bottom_center(detection.bbox)
        regions = self.cfg.get("regions", {})
        conveyor, ground, arm = regions.get("conveyor", []), regions.get("ground", []), regions.get("arm_exclusion", [])
        if arm and point_in_polygon(center, arm, frame.shape):
            mem.state, mem.ground_frames, mem.ground_anchor = "ARM_EXCLUDED", 0, None
            return mem, None

        previous = mem.history[-1] if mem.history else None
        mem.history.append((frame_id, timestamp, center, bottom, detection.bbox))
        on_belt = point_in_polygon(bottom, conveyor, frame.shape)
        on_ground = point_in_polygon(bottom, ground, frame.shape)
        if mem.state == "ARM_EXCLUDED":
            mem.state = "ON_CONVEYOR" if on_belt else "UNKNOWN"
            previous = None

        min_drop = float(self.cfg.get("departure_vertical_pixels", 18))
        free_drop = float(self.cfg.get("free_fall_vertical_pixels", 12))
        if mem.state in {"UNKNOWN", "ON_CONVEYOR"}:
            if on_belt:
                mem.state = "ON_CONVEYOR"
            elif previous is not None:
                dy = center[1] - previous[2][1]
                threshold = min_drop if mem.state == "ON_CONVEYOR" else free_drop
                if dy >= threshold:
                    mem.state, mem.event_started_at = "AIRBORNE", timestamp
                    mem.fall_frame = frame.copy()
                    mem.ground_frames, mem.ground_anchor = 0, None
                elif mem.state == "ON_CONVEYOR":
                    # A belt exit without downward evidence is not a fall.
                    mem.state = "UNKNOWN"

        if mem.state in {"AIRBORNE", "GROUND_CANDIDATE"}:
            if not on_ground:
                mem.state, mem.ground_frames, mem.ground_anchor = "AIRBORNE", 0, None
                return mem, None
            limit = float(self.cfg.get("ground_motion_pixels", 7))
            if mem.ground_anchor is None or math.dist(center, mem.ground_anchor) > limit:
                mem.ground_anchor, mem.ground_frames = center, 1
            else:
                mem.ground_frames += 1
            mem.state = "GROUND_CANDIDATE"
            # Short configurable confirmation window: avoids a long post-landing wait.
            if mem.ground_frames >= max(1, int(self.cfg.get("ground_confirm_frames", 2))):
                mem.state = "FALL_CONFIRMED"
                if not mem.emitted:
                    mem.emitted = True
                    return mem, {
                        "frame_id": frame_id, "timestamp": timestamp,
                        "ground_frame": frame.copy(), "bbox": list(detection.bbox),
                        "fall_started_at": mem.event_started_at,
                    }
        elif mem.state == "FALL_CONFIRMED":
            # Confirmation is sticky; a separate status can mark later settling.
            if on_ground:
                if mem.ground_anchor is None or math.dist(center, mem.ground_anchor) > float(self.cfg.get("settle_motion_pixels", 5)):
                    mem.ground_anchor, mem.ground_frames = center, 1
                else:
                    mem.ground_frames += 1
                if mem.ground_frames >= int(self.cfg.get("settled_frames", 8)):
                    mem.state = "SETTLED"
        return mem, None

    def prune(self, frame_id):
        grace = max(1, int(self.cfg.get("lost_track_grace_frames", 12)))
        for track_id in [tid for tid, mem in self.tracks.items() if frame_id - mem.last_seen_frame > max(grace * 5, 100)]:
            del self.tracks[track_id]
