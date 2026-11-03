import threading
from typing import Dict, Any, List

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.latest_frame = None
        self.active_level = "low"
        self.fallen_onto_ground = False
        self.amr_approach = False
        self.events: List[Dict[str, Any]] = []
        self.warnings: List[str] = []
        self.coords_source = "calibrated"

    def update_frame(self, frame, level: str, fallen: bool, approach: bool):
        with self._lock:
            self.latest_frame = frame
            self.active_level = level
            self.fallen_onto_ground = fallen
            self.amr_approach = approach

    def add_event(self, event_data: Dict[str, Any]):
        with self._lock:
            self.events.append(event_data)

    def add_warning(self, msg: str):
        with self._lock:
            if msg not in self.warnings:
                self.warnings.append(msg)