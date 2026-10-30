from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone
import json, uuid, cv2

@dataclass
class FallEvent:
    event_id: str
    track_id: int
    class_name: str
    confidence: float
    object_fallen: bool
    arm_approach: bool
    state: str
    frame_id: int
    video_timestamp_sec: float
    bbox_xyxy: list[float]
    evidence: dict[str,str]

class EvidenceStore:
    def __init__(self, root="fallenobjects"): self.root=Path(root)
    def save(self,event_id,fall_frame,ground_frame,metadata):
        folder=self.root/event_id; folder.mkdir(parents=True,exist_ok=True)
        fall_path=folder/"fall_motion.jpg"; ground_path=folder/"ground_confirmed.jpg"
        if fall_frame is not None: cv2.imwrite(str(fall_path),fall_frame)
        if ground_frame is not None: cv2.imwrite(str(ground_path),ground_frame)
        metadata["evidence"]={"fall_frame":str(fall_path),"ground_frame":str(ground_path)}
        with (folder/"metadata.json").open("w",encoding="utf-8") as f: json.dump(metadata,f,indent=2)
        return metadata["evidence"]

def new_event_id(): return str(uuid.uuid4())
def utc_now(): return datetime.now(timezone.utc).isoformat()
