from types import SimpleNamespace
import numpy as np
from fallwatch.state_machine import FallAnalyzer

def det(track_id, bbox):
    return SimpleNamespace(track_id=track_id, class_name="person", confidence=.9, bbox=bbox)

def test_fall_confirms_at_ground_contact_without_long_wait():
    cfg = {
        "history_size": 8, "departure_vertical_pixels": 8,
        "free_fall_vertical_pixels": 8, "ground_confirm_frames": 2,
        "ground_motion_pixels": 10,
        "regions": {
            "conveyor": [[0, .1], [1, .1], [1, .55], [0, .55]],
            "ground": [[0, .6], [1, .6], [1, 1], [0, 1]],
            "arm_exclusion": [],
        },
    }
    analyzer = FallAnalyzer(cfg)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    analyzer.update(det(7, (40, 20, 60, 40)), 1, 0.0, frame)
    analyzer.update(det(7, (40, 35, 60, 55)), 2, .05, frame)
    analyzer.update(det(7, (40, 50, 60, 70)), 3, .10, frame)
    _, event = analyzer.update(det(7, (40, 60, 60, 80)), 4, .15, frame)
    assert event is not None
    assert event["frame_id"] == 4
    assert event["ground_frame"] is not None
    _, duplicate = analyzer.update(det(7, (40, 62, 60, 82)), 5, .20, frame)
    assert duplicate is None

def test_normal_conveyor_motion_does_not_confirm_fall():
    cfg = {
        "departure_vertical_pixels": 10, "free_fall_vertical_pixels": 10,
        "ground_confirm_frames": 2,
        "regions": {
            "conveyor": [[0, .1], [1, .1], [1, .7], [0, .7]],
            "ground": [[0, .75], [1, .75], [1, 1], [0, 1]],
            "arm_exclusion": [],
        },
    }
    analyzer = FallAnalyzer(cfg)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(1, 10):
        analyzer.update(det(2, (10+i, 25, 30+i, 45)), i, i/20, frame)
    assert all(memory.state != "FALL_CONFIRMED" for memory in analyzer.tracks.values())
