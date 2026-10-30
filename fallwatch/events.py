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
    """Persist a fall's still frames, buffered video, and metadata as one local record."""
    def __init__(self, root="fallenobjects"):
        self.root=Path(root)

    def save(self,event_id,fall_frame,ground_frame,metadata,clip_frames=None,fps=20):
        folder=self.root/event_id
        folder.mkdir(parents=True,exist_ok=True)
        fall_path=folder/"fall_motion.jpg"
        ground_path=folder/"ground_confirmed.jpg"
        if fall_frame is not None: cv2.imwrite(str(fall_path),fall_frame)
        if ground_frame is not None: cv2.imwrite(str(ground_path),ground_frame)
        clip_path=folder/"fall_clip.mp4"
        clip_frames=clip_frames or []
        if clip_frames:
            h,w=clip_frames[0].shape[:2]
            writer=cv2.VideoWriter(str(clip_path),cv2.VideoWriter_fourcc(*"mp4v"),max(1,float(fps)),(w,h))
            try:
                for frame in clip_frames:
                    if frame.shape[:2] != (h,w): frame=cv2.resize(frame,(w,h))
                    writer.write(frame)
            finally: writer.release()
        evidence={"fall_frame":str(fall_path) if fall_frame is not None else "",
                  "ground_frame":str(ground_path) if ground_frame is not None else "",
                  "video_clip":str(clip_path) if clip_frames else ""}
        metadata["evidence"]=evidence
        metadata["created_at_utc"]=datetime.now(timezone.utc).isoformat()
        with (folder/"metadata.json").open("w",encoding="utf-8") as f:
            json.dump(metadata,f,indent=2,default=str)
        return evidence

def new_event_id():
    return "fall_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"_"+uuid.uuid4().hex[:8]

def utc_now():
    return datetime.now(timezone.utc).isoformat()
