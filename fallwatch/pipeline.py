from collections import deque
from dataclasses import asdict
import cv2
from .perception import YoloTracker
from .state_machine import FallAnalyzer
from .events import EvidenceStore, FallEvent, new_event_id

class FallPipeline:
    """Detection/tracking + temporal confirmation + local evidence only (no external API)."""
    def __init__(self, settings):
        self.settings=settings
        cfg=settings.config
        self.detector=YoloTracker(settings.model_path,cfg)
        self.analyzer=FallAnalyzer(cfg)
        self.evidence=EvidenceStore(cfg.get("evidence_root","fallenobjects"))
        self.frame_id=0
        self.events=[]
        self.recent_frames=deque(maxlen=int(cfg.get("evidence_clip_frames",90)))

    def process(self, frame, timestamp_sec=None):
        self.frame_id+=1
        timestamp_sec=float(timestamp_sec if timestamp_sec is not None else self.frame_id/30.0)
        self.recent_frames.append(frame.copy())
        detections=self.detector.infer(frame)
        rendered=frame.copy()
        current_events=[]
        for det in detections:
            mem,confirmed=self.analyzer.update(det,self.frame_id,timestamp_sec,frame)
            x1,y1,x2,y2=map(int,det.bbox)
            color=(0,0,255) if mem.state in {"FALL_CONFIRMED","EVENT_PROCESSED"} else (30,210,130)
            cv2.rectangle(rendered,(x1,y1),(x2,y2),color,2)
            label=f"{det.class_name} ID:{det.track_id} {det.confidence:.2f} | {mem.state}"
            cv2.putText(rendered,label,(x1,max(20,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
            if confirmed and not mem.emitted:
                eid=new_event_id()
                metadata={"event_id":eid,"track_id":int(det.track_id),"class_name":det.class_name,
                    "confidence":float(det.confidence),"bbox_xyxy":confirmed["bbox"],
                    "frame_id":confirmed["frame_id"],"video_timestamp_sec":confirmed["timestamp"],
                    "state":"FALL_CONFIRMED","object_fallen":True,"arm_approach":True,
                    "flag_semantics":"Both local flags assert on confirmed fall; no robot command is sent."}
                evidence=self.evidence.save(eid,mem.fall_frame,confirmed["ground_frame"],metadata,
                    list(self.recent_frames),float(cfg_fps(self.settings.config)))
                event=FallEvent(eid,int(det.track_id),det.class_name,float(det.confidence),True,True,
                    "FALL_CONFIRMED",int(confirmed["frame_id"]),float(confirmed["timestamp"]),
                    list(confirmed["bbox"]),evidence)
                payload=asdict(event)
                self.events.append(payload); current_events.append(payload)
                mem.emitted=True; mem.state="EVENT_PROCESSED"
        self.analyzer.prune(self.frame_id)
        return rendered,detections,current_events

def cfg_fps(config):
    return config.get("fps",20)
