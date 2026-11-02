"""YOLO tracking, fall analysis, and evidence orchestration."""
from collections import deque
from dataclasses import asdict
import cv2
from .perception import YoloTracker
from .state_machine import FallAnalyzer
from .events import EvidenceStore, FallEvent, new_event_id

class FallPipeline:
    def __init__(self, settings):
        self.settings = settings
        cfg = settings.config
        self.detector = YoloTracker(settings.model_path, cfg)
        self.analyzer = FallAnalyzer(cfg)
        self.evidence = EvidenceStore(cfg.get("evidence_root", "fallenobjects"))
        self.frame_id = 0
        self.events = []
        self.recent_frames = deque(maxlen=max(1, int(cfg.get("evidence_clip_frames", 90))))

    def process(self, frame, timestamp_sec=None):
        self.frame_id += 1
        timestamp_sec = float(timestamp_sec if timestamp_sec is not None else self.frame_id / float(self.settings.config.get("fps", 20)))
        self.recent_frames.append(frame.copy())
        detections = self.detector.infer(frame)
        rendered = frame.copy()
        current_events = []
        for det in detections:
            mem, confirmation = self.analyzer.update(det, self.frame_id, timestamp_sec, frame)
            x1, y1, x2, y2 = map(int, det.bbox)
            color = (0, 0, 255) if mem.state in {"FALL_CONFIRMED", "SETTLED"} else (30, 210, 130)
            cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
            label = f"{det.class_name} ID:{det.track_id} {det.confidence:.2f} | {mem.state}"
            cv2.putText(rendered, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2)
            if confirmation:
                event_id = new_event_id()
                metadata = {
                    "event_id": event_id, "track_id": int(det.track_id), "class_name": det.class_name,
                    "confidence": float(det.confidence), "bbox_xyxy": confirmation["bbox"],
                    "frame_id": confirmation["frame_id"], "video_timestamp_sec": confirmation["timestamp"],
                    "fall_started_at": confirmation.get("fall_started_at"), "state": "FALL_CONFIRMED",
                    "object_fallen_onto_ground": True, "arm_object_approach": True,
                    "flag_semantics": "Sticky local indicators; reset explicitly; no robot command is issued.",
                }
                evidence = self.evidence.save(event_id, mem.fall_frame, confirmation["ground_frame"], metadata,
                    list(self.recent_frames), float(self.settings.config.get("fps", 20)))
                event = FallEvent(event_id, int(det.track_id), det.class_name, float(det.confidence), True, True,
                    "FALL_CONFIRMED", int(confirmation["frame_id"]), float(confirmation["timestamp"]),
                    list(confirmation["bbox"]), evidence)
                payload = asdict(event)
                self.events.append(payload)
                current_events.append(payload)
            cv2.putText(rendered, f"FALL FLAG: {bool(self.events)}", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, .7, (0, 0, 255) if self.events else (30, 210, 130), 2)
        self.analyzer.prune(self.frame_id)
        tracks = [{"track_id": int(d.track_id), "class_name": d.class_name,
                   "confidence": float(d.confidence), "bbox": list(d.bbox),
                   "state": self.analyzer.tracks[d.track_id].state} for d in detections]
        return rendered, detections, current_events, tracks
