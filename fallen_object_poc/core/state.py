import threading
from typing import Dict, Any, List

class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.latest_frame = None
        self.fallen_onto_ground: bool = False
        self.amr_approach: bool = False
        self.current_signal_level: str = "normal"
        self.events: List[Dict[str, Any]] = []
        self.warnings: List[str] = []

    def update_frame(self, frame):
        with self._lock:
            self.latest_frame = frame

    def update_booleans(self, fallen: bool, amr_approach: bool, signal_level: str):
        with self._lock:
            self.fallen_onto_ground = fallen
            self.amr_approach = amr_approach
            self.current_signal_level = signal_level

    def add_event(self, event_data: Dict[str, Any]):
        with self._lock:
            self.events.append(event_data)

    def add_warning(self, warning_msg: str):
        with self._lock:
            if warning_msg not in self.warnings:
                self.warnings.append(warning_msg)

    def get_snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "fallen_onto_ground": self.fallen_onto_ground,
                "amr_approach": self.amr_approach,
                "current_signal_level": self.current_signal_level,
                "events": list(self.events),
                "warnings": list(self.warnings)
            }