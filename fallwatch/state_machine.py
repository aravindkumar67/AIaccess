from collections import deque
from dataclasses import dataclass, field
import math

from .geometry import bbox_center, bbox_bottom_center, point_in_polygon


@dataclass
class TrackMemory:
    history: deque = field(default_factory=lambda: deque(maxlen=24))
    state: str = "UNKNOWN"
    last_seen_frame: int = -1
    departure_frame: int | None = None
    ground_frames: int = 0
    ground_anchor: tuple[float, float] | None = None
    fall_frame: object = None
    emitted: bool = False
    class_name: str = "unknown"
    confidence: float = 0.0
    bbox: tuple = (0, 0, 0, 0)


class FallAnalyzer:
    """Confirm falls from conveyor departure or sustained downward motion, then ground settling."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.tracks = {}

    def update(self, detection, frame_id, timestamp, frame):
        history_size = max(2, int(self.cfg.get("history_size", 24)))
        mem = self.tracks.setdefault(
            detection.track_id,
            TrackMemory(history=deque(maxlen=history_size)),
        )
        mem.last_seen_frame = frame_id
        mem.class_name = detection.class_name
        mem.confidence = detection.confidence
        mem.bbox = detection.bbox

        center = bbox_center(detection.bbox)
        bottom = bbox_bottom_center(detection.bbox)
        regions = self.cfg.get("regions", {})
        conveyor = regions.get("conveyor", [])
        ground = regions.get("ground", [])
        arm = regions.get("arm_exclusion", [])

        if arm and point_in_polygon(center, arm, frame.shape):
            # Keep this track out of fall logic while it is in the arm's exclusion region.
            mem.state = "ARM_EXCLUDED"
            mem.ground_frames = 0
            mem.ground_anchor = None
            return mem, None

        previous = mem.history[-1] if mem.history else None
        mem.history.append((frame_id, timestamp, center, bottom, detection.bbox))
        in_conveyor = point_in_polygon(bottom, conveyor, frame.shape)
        in_ground = point_in_polygon(bottom, ground, frame.shape)

        # Re-entering from an exclusion region starts a fresh baseline.
        if mem.state == "ARM_EXCLUDED":
            mem.state = "ON_CONVEYOR" if in_conveyor else "UNKNOWN"
            previous = None

        # A conveyor departure needs both leaving the belt polygon and downward movement.
        if mem.state in {"UNKNOWN", "ON_CONVEYOR"}:
            if in_conveyor:
                mem.state = "ON_CONVEYOR"
            elif previous is not None:
                dy = center[1] - previous[2][1]
                minimum_drop = float(self.cfg.get("departure_vertical_pixels", 18))
                if dy >= minimum_drop:
                    mem.state = "AIRBORNE"
                    mem.departure_frame = frame_id
                    mem.fall_frame = frame.copy()
                    mem.ground_frames = 0
                    mem.ground_anchor = None
                elif mem.state == "ON_CONVEYOR":
                    # A horizontal belt exit without evidence of a drop is not yet a fall.
                    mem.state = "UNKNOWN"

        # Also support objects first observed in free fall (not previously on the conveyor).
        if mem.state == "UNKNOWN" and previous is not None:
            dy = center[1] - previous[2][1]
            if dy >= float(self.cfg.get("free_fall_vertical_pixels", 12)):
                mem.state = "AIRBORNE"
                mem.departure_frame = frame_id
                mem.fall_frame = frame.copy()
                mem.ground_frames = 0
                mem.ground_anchor = None

        if mem.state in {"AIRBORNE", "GROUND_CANDIDATE"}:
            if not in_ground:
                mem.state = "AIRBORNE"
                mem.ground_frames = 0
                mem.ground_anchor = None
                return mem, None

            motion_limit = float(self.cfg.get("ground_motion_pixels", 7))
            if mem.ground_anchor is None:
                mem.ground_anchor = center
                mem.ground_frames = 1
            elif math.dist(center, mem.ground_anchor) <= motion_limit:
                mem.ground_frames += 1
            else:
                mem.ground_anchor = center
                mem.ground_frames = 1

            mem.state = "GROUND_CANDIDATE"
            if mem.ground_frames >= int(self.cfg.get("ground_stable_frames", 5)):
                mem.state = "FALL_CONFIRMED"
                return mem, {
                    "frame_id": frame_id,
                    "timestamp": timestamp,
                    "ground_frame": frame.copy(),
                    "bbox": list(detection.bbox),
                }

        return mem, None

    def prune(self, frame_id):
        grace = int(self.cfg.get("lost_track_grace_frames", 12))
        stale = [
            track_id
            for track_id, memory in self.tracks.items()
            if frame_id - memory.last_seen_frame > max(grace * 5, 100)
        ]
        for track_id in stale:
            del self.tracks[track_id]
