from datetime import datetime,timezone
from uuid import uuid4
import cv2
from ultralytics import YOLO
from .fall_logic import FallDetector
from .signal_logic import grade_severity
from .path_graph import PathGraph
from .evidence import EvidenceRecorder

class VideoPipeline:
    def __init__(self,config,weights,source):
        self.cfg=config; self.model=YOLO(weights); self.source=source; self.fall=FallDetector(config['logic']); self.path=PathGraph(config['regions'],config['logic']); self.events={}; self.frame_index=0
    def run(self,on_frame=None):
        cap=cv2.VideoCapture(self.source)
        if not cap.isOpened(): raise RuntimeError(f'Cannot open video source: {self.source}')
        fps=cap.get(cv2.CAP_PROP_FPS) or 24.0; recorder=EvidenceRecorder(self.cfg['evidence'],fps)
        try:
            while True:
                ok,frame=cap.read()
                if not ok: break
                self.frame_index+=1; result=self.model.track(frame,persist=True,tracker='bytetrack.yaml',verbose=False)[0]; det=[]
                if result.boxes is not None:
                    for b in result.boxes:
                        if b.id is None: continue
                        cid=int(b.cls[0].cpu().item()); det.append((int(b.id[0].cpu().item()),str(result.names.get(cid,cid)).lower(),b.xyxy[0].cpu().tolist(),float(b.conf[0].cpu().item())))
                obs=[d for d in det if d[1] in self.cfg['classes']['obstacle']]; robots=[d for d in det if d[1] in self.cfg['classes']['amr']]
                for tid,label,bbox,conf in obs:
                    state,just_fell=self.fall.update(tid,label,bbox,self.frame_index)
                    if just_fell:
                        sev=grade_severity(frame.shape,bbox,self.cfg['regions'],self.cfg['logic']); eid='FALL-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'-'+uuid4().hex[:8]
                        p={'event_name':'fallen_object_detected','event_id':eid,'event_type':'FALL','timestamp_utc':datetime.now(timezone.utc).isoformat(),'fallen_onto_ground':True,'severity':sev['severity'],'amr_approach':False,'object':{'track_id':tid,'class_name':label,'confidence':conf,'bbox_xyxy_px':bbox,'center_xy_px':[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]},'amr':{'track_id':None,'distance_px':None},'geometry':sev,'calibration':{'camera_id':self.cfg['camera']['camera_id'],'quality':self.cfg['camera']['calibration_quality'],'coordinate_frame':'image_px'},'flags':['image-space-only','provisional-calibration']}
                        self.events[tid]=p; recorder.start(p,frame)
                for objid,p in list(self.events.items()):
                    if p['amr_approach']: continue
                    for robotid,robotlabel,robotbox,robotconf in robots:
                        confirmed,metrics=self.path.check_amr_approach(robotid,robotbox,p['object']['bbox_xyxy_px'],frame.shape)
                        if confirmed:
                            p['amr_approach']=True; p['amr'].update({'track_id':robotid,'distance_px':metrics['distance_px']}); p['event_name']='amr_approach_to_fallen_object'; p['event_type']='AMR_APPROACH'; p['timestamp_amr_approach_utc']=datetime.now(timezone.utc).isoformat(); p['amr']['metrics']=metrics; break
                recorder.push(frame)
                if on_frame: on_frame(result.plot(),list(self.events.values()))
                self.fall.prune(self.frame_index)
        finally: cap.release()
        return list(self.events.values())
