from ultralytics import YOLO


class YOLOByteTrackDetector:
    """Run YOLO inference and persistent ByteTrack tracking on consecutive frames."""

    def __init__(self, model_name="yolo26n.pt", conf=0.25, iou=0.5, imgsz=640, device=None):
        self.model = YOLO(model_name)
        self.conf = float(conf)
        self.iou = float(iou)
        self.imgsz = int(imgsz)
        self.device = device or None

    def detect_and_track(self, frame):
        results = self.model.track(
            source=frame, persist=True, tracker="bytetrack.yaml",
            conf=self.conf, iou=self.iou, imgsz=self.imgsz,
            device=self.device, verbose=False,
        )
        if not results:
            return [], frame.copy()
        result = results[0]
        annotated = result.plot()
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return [], annotated
        coords = boxes.xyxy.cpu().tolist()
        scores = boxes.conf.cpu().tolist()
        classes = boxes.cls.int().cpu().tolist()
        ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(coords)
        names = result.names
        detections = []
        for box, score, cls_id, track_id in zip(coords, scores, classes, ids):
            x1, y1, x2, y2 = map(float, box)
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(score),
                "class_id": int(cls_id),
                "class_name": str(names.get(int(cls_id), cls_id)),
                "track_id": int(track_id) if track_id is not None else None,
                "center": [(x1 + x2) / 2, (y1 + y2) / 2],
                "bottom_center": [(x1 + x2) / 2, y2],
            })
        return detections, annotated
