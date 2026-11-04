import os
import json
import cv2
from datetime import datetime

class EvidenceManager:
    def __init__(self, base_evidence_dir: str = "evidence"):
        self.base_dir = base_evidence_dir

    def save_evidence(self, event_id: str, level: str, snapshot_frame, video_frames: list, payload: dict, justification: dict):
        timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        event_folder = os.path.join(self.base_dir, level, f"{event_id}_{timestamp_str}")
        justification_folder = os.path.join(self.base_dir, "justification")

        os.makedirs(event_folder, exist_ok=True)
        os.makedirs(justification_folder, exist_ok=True)

        # 1. Snapshot
        if snapshot_frame is not None:
            cv2.imwrite(os.path.join(event_folder, "snapshot.jpg"), snapshot_frame)

        # 2. Data JSON
        with open(os.path.join(event_folder, "data.json"), 'w') as f:
            json.dump(payload, f, indent=2)

        # 3. Combined/Standalone evidence.json inside the event folder itself
        evidence_summary = {
            "event_id": event_id,
            "signal_level": level,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "payload_summary": payload,
            "justification_summary": justification
        }
        with open(os.path.join(event_folder, "evidence.json"), 'w') as f:
            json.dump(evidence_summary, f, indent=2)

        # 4. Separate Justification File (in justification folder)
        with open(os.path.join(justification_folder, f"{event_id}_justification.json"), 'w') as f:
            json.dump(justification, f, indent=2)

        # 5. Video Clip
        if video_frames and len(video_frames) > 0:
            clip_path = os.path.join(event_folder, "clip.mp4")
            h, w, _ = video_frames[0].shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(clip_path, fourcc, 24.0, (w, h))
            for f in video_frames:
                out.write(f)
            out.release()