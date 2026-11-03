import os
import json
import cv2

class EvidenceWriter:
    def __init__(self, output_dir="evidence"):
        self.output_dir = output_dir

    def write_event(self, level: str, event_payload: dict, snapshot_frame, clip_frames, fps=24.0):
        event_id = event_payload.get("event_id", "EVENT_0000")
        folder_path = os.path.join(self.output_dir, level, f"{event_id}")
        os.makedirs(folder_path, exist_ok=True)
        
        cv2.imwrite(os.path.join(folder_path, "snapshot.jpg"), snapshot_frame)
        
        with open(os.path.join(folder_path, "data.json"), "w") as f:
            json.dump(event_payload, f, indent=2)
            
        return folder_path