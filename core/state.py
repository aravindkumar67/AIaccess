import threading
from typing import Any, Dict, List

class SharedState:
    """Thread-safe UI and event state. Confirmed booleans persist until reset."""
    def __init__(self):
        self._lock = threading.RLock()
        self.latest_frame = None
        self.active_level = "low"
        self.fallen_onto_ground = False
        self.amr_approach = False
        self.events: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.coords_source = "uncalibrated"
        self.frames_processed = 0
        self.track_ids = set()
        self.track_states: Dict[int, Dict[str, Any]] = {}

    def update_frame(self, frame, level="low", fallen=None, approach=None):
        with self._lock:
            self.latest_frame = frame.copy() if frame is not None else None
            self.active_level = str(level)
            # None means preserve event-derived boolean, rather than clearing it.
            if fallen is not None:
                self.fallen_onto_ground = bool(fallen) or self.fallen_onto_ground
            if approach is not None:
                self.amr_approach = bool(approach) or self.amr_approach
            self.frames_processed += 1

    def add_event(self, event_data):
        with self._lock:
            self.events.append(dict(event_data))

    def add_warning(self, msg):
        with self._lock:
            if msg not in self.warnings:
                self.warnings.append(msg)

    def add_track_id(self, track_id):
        if track_id is not None:
            with self._lock:
                self.track_ids.add(int(track_id))

    def update_track_state(self, track_id, patch):
        with self._lock:
            tid = int(track_id)
            current = self.track_states.setdefault(tid, {
                "fallen_onto_ground": False,
                "amr_approach": False,
                "signal_level": "low",
            })
            current.update(dict(patch))
            current["fallen_onto_ground"] = bool(current.get("fallen_onto_ground"))
            current["amr_approach"] = bool(current.get("amr_approach"))
            return dict(current)

    def get_track_state(self, track_id):
        with self._lock:
            return dict(self.track_states.get(int(track_id), {}))
