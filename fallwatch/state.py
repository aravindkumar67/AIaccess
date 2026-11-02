"""Thread-safe state shared by the video worker and Streamlit renderer."""
from __future__ import annotations
from threading import RLock
from copy import deepcopy
import time

class RuntimeState:
    def __init__(self):
        self._lock = RLock()
        self._data = {
            "status": "idle", "frame": None, "frame_id": 0, "video_time": 0.0,
            "tracks": [], "events": [], "error": None, "fps": 0.0,
            "inference_ms": 0.0, "object_fallen_onto_ground": False,
            "arm_object_approach": False, "updated_at": time.time(),
        }

    def update(self, **values):
        with self._lock:
            self._data.update(values)
            self._data["updated_at"] = time.time()

    def add_event(self, event):
        with self._lock:
            self._data["events"].append(deepcopy(event))
            self._data["object_fallen_onto_ground"] = True
            self._data["arm_object_approach"] = True
            self._data["updated_at"] = time.time()

    def reset_flags(self):
        with self._lock:
            self._data["object_fallen_onto_ground"] = False
            self._data["arm_object_approach"] = False
            self._data["updated_at"] = time.time()

    def snapshot(self):
        with self._lock:
            result = dict(self._data)
            if result["frame"] is not None:
                result["frame"] = result["frame"].copy()
            result["events"] = deepcopy(result["events"])
            result["tracks"] = deepcopy(result["tracks"])
            return result
