import numpy as np
from enum import Enum

class FallState(Enum):
    ON_CONVEYOR = 1
    FALLING = 2
    LANDED = 3
    UNCERTAIN = 4

class TrackFallStateMachine:
    def __init__(self, track_id: int, start_frame: int, start_box: list, fps: float, config: dict):
        self.track_id = track_id
        self.state = FallState.ON_CONVEYOR
        self.frames_on_conveyor = 1
        self.fall_start_frame = None
        self.fall_start_y = None
        self.resting_frames = 0
        self.recent_boxes = []
        self.fps = fps
        self.config = config
        self.landing_box = None
        self.event_id = f"FALL_{track_id:04d}"

    def update(self, frame_idx: int, bbox: list, is_in_conveyor_zone: bool, frame_height: int):
        center_y = (bbox[1] + bbox[3]) / 2.0

        if self.state == FallState.ON_CONVEYOR:
            if is_in_conveyor_zone:
                self.frames_on_conveyor += 1
            else:
                min_conveyor_frames = int(self.config['min_time_on_conveyor_sec'] * self.fps)
                if self.frames_on_conveyor >= min_conveyor_frames:
                    self.state = FallState.FALLING
                    self.fall_start_frame = frame_idx
                    self.fall_start_y = center_y
                else:
                    self.state = FallState.UNCERTAIN

        elif self.state == FallState.FALLING:
            elapsed_sec = (frame_idx - self.fall_start_frame) / self.fps
            if elapsed_sec > self.config['max_fall_duration_sec']:
                self.state = FallState.UNCERTAIN
                return

            drop_pct = (center_y - self.fall_start_y) / float(frame_height)
            if drop_pct >= self.config['vertical_drop_threshold_pct']:
                self.recent_boxes.append(bbox)
                req_resting_frames = max(1, int(self.config['resting_duration_sec'] * self.fps))
                if len(self.recent_boxes) > req_resting_frames:
                    self.recent_boxes.pop(0)

                if len(self.recent_boxes) >= 2:
                    dy = abs(self.recent_boxes[-1][3] - self.recent_boxes[-2][3]) / float(frame_height)
                    if dy <= self.config['resting_speed_threshold_pct']:
                        self.resting_frames += 1
                    else:
                        self.resting_frames = 0

                    if self.resting_frames >= req_resting_frames:
                        self.state = FallState.LANDED
                        self.landing_box = np.median(self.recent_boxes, axis=0).tolist()