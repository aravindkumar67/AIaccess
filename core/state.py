import threading
from typing import Any, Dict, List


class SharedState:
    """Thread-safe state for processing summary and dashboard results."""

    def __init__(self):
        self._lock = threading.Lock()
        self.latest_frame = None
        self.active_level = "low"
        self.fallen_onto_ground = False
        self.amr_approach = False
        self.events: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.coords_source = "uncalibrated"
        self.frames_processed = 0
        self.track_ids = set()

    def update_frame(self, frame, level="low", fallen=False, approach=False):
        with self._lock:
            self.latest_frame = frame.copy() if frame is not None else None
            self.active_level = level
            self.fallen_onto_ground = bool(fallen)
            self.amr_approach = bool(approach)
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
