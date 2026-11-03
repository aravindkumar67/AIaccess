from collections import deque
from pathlib import Path
import json,cv2

class EvidenceRecorder:
    def __init__(self,cfg,fps):
        self.cfg=cfg; self.fps=max(float(fps),1.0); self.buffer=deque(maxlen=max(1,int(self.fps*float(cfg.get('pre_event_seconds',3)))))
        self.pending={}; self.root=Path(cfg.get('root','outputs/events'))
    def push(self,frame):
        self.buffer.append(frame.copy()); completed=[]
        for eid,item in self.pending.items():
            item['frames'].append(frame.copy()); item['remaining']-=1
            if item['remaining']<=0:
                self._clip(item['clip'],item['frames']); item['payload']['evidence']['clip_path']=str(item['clip'])
                self._json(item['json'],item['payload']); completed.append(eid)
        for eid in completed: del self.pending[eid]
    def start(self,payload,frame):
        eid=payload['event_id']; folder=self.root/eid; folder.mkdir(parents=True,exist_ok=True)
        snap=folder/'snapshot.jpg'; cv2.imwrite(str(snap),frame,[cv2.IMWRITE_JPEG_QUALITY,int(self.cfg.get('jpeg_quality',92))])
        clip=folder/'event.mp4'; js=folder/'event.json'; post=max(0,int(self.fps*float(self.cfg.get('post_event_seconds',4))))
        payload['evidence']={'snapshot_path':str(snap),'clip_path':None,'json_path':str(js)}
        frames=list(self.buffer)+[frame.copy()]
        if post==0: self._clip(clip,frames); payload['evidence']['clip_path']=str(clip); self._json(js,payload)
        else: self.pending[eid]={'clip':clip,'json':js,'payload':payload,'frames':frames,'remaining':post}
    @staticmethod
    def _clip(path,frames):
        if not frames:return
        h,w=frames[0].shape[:2]; writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),24,(w,h))
        try:
            for f in frames: writer.write(f)
        finally: writer.release()
    @staticmethod
    def _json(path,payload): path.write_text(json.dumps(payload,indent=2,default=str),encoding='utf-8')
