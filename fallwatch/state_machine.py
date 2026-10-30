from collections import deque
from dataclasses import dataclass,field
import math
from .geometry import bbox_center,bbox_bottom_center,point_in_polygon

@dataclass
class TrackMemory:
    history: deque=field(default_factory=lambda:deque(maxlen=24))
    state: str="UNKNOWN"
    last_seen_frame: int=-1
    departure_frame: int|None=None
    ground_frames: int=0
    ground_anchor: tuple[float,float]|None=None
    fall_frame: object=None
    emitted: bool=False
    class_name: str="unknown"
    confidence: float=0.0
    bbox: tuple=(0,0,0,0)

class FallAnalyzer:
    """Temporal zone/motion logic: conveyor departure -> ground stabilization."""
    def __init__(self,cfg): self.cfg=cfg; self.tracks={}
    def update(self,detection,frame_id,timestamp,frame):
        mem=self.tracks.setdefault(detection.track_id,TrackMemory(history=deque(maxlen=int(self.cfg.get("history_size",24)))))
        mem.last_seen_frame=frame_id; mem.class_name=detection.class_name; mem.confidence=detection.confidence; mem.bbox=detection.bbox
        center=bbox_center(detection.bbox); bottom=bbox_bottom_center(detection.bbox)
        regions=self.cfg.get("regions",{}); conveyor=regions.get("conveyor",[]); ground=regions.get("ground",[]); arm=regions.get("arm_exclusion",[])
        if arm and point_in_polygon(center,arm,frame.shape): mem.state="ARM_EXCLUDED"; return mem,None
        mem.history.append((frame_id,timestamp,center,bottom,detection.bbox))
        in_conveyor=point_in_polygon(bottom,conveyor,frame.shape); in_ground=point_in_polygon(bottom,ground,frame.shape)
        if mem.state in {"UNKNOWN","ARM_EXCLUDED"}: mem.state="ON_CONVEYOR" if in_conveyor else "UNKNOWN"
        if mem.state=="ON_CONVEYOR" and not in_conveyor:
            if len(mem.history)>=2 and center[1]-mem.history[0][2][1]>=float(self.cfg.get("departure_vertical_pixels",18)):
                mem.state="DEPARTURE_CANDIDATE"; mem.departure_frame=frame_id; mem.fall_frame=frame.copy()
            else: mem.state="UNKNOWN"
        if mem.state=="DEPARTURE_CANDIDATE": mem.state="AIRBORNE"; mem.ground_frames=0; mem.ground_anchor=None
        if mem.state in {"AIRBORNE","GROUND_CANDIDATE"}:
            if in_ground:
                if mem.ground_anchor is None: mem.ground_anchor=center; mem.ground_frames=1; mem.state="GROUND_CANDIDATE"
                else:
                    if math.dist(center,mem.ground_anchor)<=float(self.cfg.get("ground_motion_pixels",7)): mem.ground_frames+=1
                    else: mem.ground_anchor=center; mem.ground_frames=1
                if mem.ground_frames>=int(self.cfg.get("ground_stable_frames",5)):
                    mem.state="FALL_CONFIRMED"
                    return mem,{"frame_id":frame_id,"timestamp":timestamp,"ground_frame":frame.copy(),"bbox":list(detection.bbox)}
            else: mem.state="AIRBORNE"; mem.ground_frames=0; mem.ground_anchor=None
        return mem,None
    def prune(self,frame_id):
        grace=int(self.cfg.get("lost_track_grace_frames",12)); stale=[tid for tid,m in self.tracks.items() if frame_id-m.last_seen_frame>max(grace*5,100)]
        for tid in stale: del self.tracks[tid]
