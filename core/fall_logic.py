from collections import deque
from dataclasses import dataclass,field

@dataclass
class TrackState:
    track_id:int
    label:str
    history:deque=field(default_factory=lambda:deque(maxlen=40))
    down_count:int=0
    settled_count:int=0
    fallen:bool=False
    fall_frame:int|None=None
    fall_bbox:list|None=None
    last_seen:int=0

class FallDetector:
    def __init__(self,cfg): self.cfg=cfg; self.tracks={}
    def update(self,track_id,label,bbox,frame_index):
        s=self.tracks.setdefault(int(track_id),TrackState(int(track_id),str(label)))
        cx=(bbox[0]+bbox[2])/2; cy=(bbox[1]+bbox[3])/2
        s.history.append((frame_index,cx,cy,list(map(float,bbox)))); s.last_seen=frame_index
        if len(s.history)<2: return s,False
        prev=s.history[-2]; dy=cy-prev[2]; speed=((cx-prev[1])**2+(cy-prev[2])**2)**0.5
        if not s.fallen and dy>=float(self.cfg.get('fall_vertical_px',35)): s.down_count+=1
        elif not s.fallen: s.down_count=max(0,s.down_count-1)
        if not s.fallen and s.down_count>=int(self.cfg.get('fall_confirm_frames',2)):
            s.fallen=True; s.fall_frame=frame_index; s.fall_bbox=list(map(float,bbox)); return s,True
        if s.fallen and speed<=float(self.cfg.get('settled_speed_px_per_frame',3.5)): s.settled_count+=1
        elif s.fallen: s.settled_count=0
        return s,False
    def prune(self,frame_index):
        expiry=int(self.cfg.get('track_expiry_frames',45))
        for tid in list(self.tracks):
            if frame_index-self.tracks[tid].last_seen>expiry: del self.tracks[tid]
