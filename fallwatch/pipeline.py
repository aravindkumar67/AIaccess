from dataclasses import asdict
from .perception import YoloTracker
from .state_machine import FallAnalyzer
from .events import EvidenceStore,FallEvent,new_event_id
from .api import EventApi

class FallPipeline:
    """Single inference and event pipeline used by video or camera sources."""
    def __init__(self,settings):
        self.settings=settings; cfg=settings.config; self.detector=YoloTracker(settings.model_path,cfg); self.analyzer=FallAnalyzer(cfg)
        self.evidence=EvidenceStore(cfg.get("evidence_root","fallenobjects")); self.api=EventApi(settings.api_enabled,settings.api_url,settings.api_method,settings.api_token,settings.api_timeout)
        self.frame_id=0; self.events=[]
    def process(self,frame,timestamp_sec=None):
        import cv2
        self.frame_id+=1; timestamp_sec=float(timestamp_sec if timestamp_sec is not None else self.frame_id/30.0)
        detections=self.detector.infer(frame); rendered=frame.copy(); new_events=[]
        for det in detections:
            mem,confirmed=self.analyzer.update(det,self.frame_id,timestamp_sec,frame); x1,y1,x2,y2=map(int,det.bbox)
            color=(0,0,255) if mem.state in {"FALL_CONFIRMED","EVENT_PROCESSED"} else (0,200,0)
            cv2.rectangle(rendered,(x1,y1),(x2,y2),color,2)
            cv2.putText(rendered,f"{det.class_name} #{det.track_id} {det.confidence:.2f} | {mem.state}",(x1,max(18,y1-8)),cv2.FONT_HERSHEY_SIMPLEX,.48,color,2)
            if confirmed and not mem.emitted:
                eid=new_event_id(); meta={"event_id":eid,"track_id":det.track_id,"class_name":det.class_name,"confidence":det.confidence,"bbox_xyxy":confirmed["bbox"],"frame_id":confirmed["frame_id"],"video_timestamp_sec":confirmed["timestamp"],"state":"FALL_CONFIRMED","object_fallen":True,"arm_approach":True}
                evidence=self.evidence.save(eid,mem.fall_frame,confirmed["ground_frame"],meta)
                event=FallEvent(eid,det.track_id,det.class_name,det.confidence,True,True,"FALL_CONFIRMED",confirmed["frame_id"],confirmed["timestamp"],confirmed["bbox"],evidence)
                payload=asdict(event); payload["api_result"]=self.api.emit(payload); self.events.append(payload); new_events.append(payload)
                mem.emitted=True; mem.state="EVENT_PROCESSED"
        self.analyzer.prune(self.frame_id); return rendered,detections,new_events
