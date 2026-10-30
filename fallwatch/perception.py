from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO

@dataclass
class Detection:
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[float,float,float,float]

class YoloTracker:
    """Persistent YOLO tracking wrapper; invoke once for each frame."""
    def __init__(self, model_path, cfg):
        if not Path(model_path).exists():
            raise FileNotFoundError(f"YOLO model missing: {model_path}. Set YOLO_MODEL_PATH.")
        self.model = YOLO(model_path); self.cfg = cfg; self.names = self.model.names
    def infer(self, frame):
        args = {"source":frame,"persist":True,"tracker":self.cfg.get("tracker","bytetrack.yaml"),
            "conf":float(self.cfg.get("confidence",.3)),"iou":float(self.cfg.get("iou",.5)),
            "imgsz":int(self.cfg.get("image_size",640)),"verbose":False}
        if self.cfg.get("device"): args["device"] = self.cfg["device"]
        if self.cfg.get("classes"): args["classes"] = self.cfg["classes"]
        result = self.model.track(**args)[0]; output=[]
        if result.boxes is None: return output
        for box in result.boxes:
            if box.id is None: continue
            cid=int(box.cls[0].cpu().item()); xyxy=box.xyxy[0].cpu().tolist()
            output.append(Detection(int(box.id[0].cpu().item()),cid,str(self.names.get(cid,cid)),float(box.conf[0].cpu().item()),tuple(map(float,xyxy))))
        return output
