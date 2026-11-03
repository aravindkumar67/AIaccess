import cv2

from core.detector import YOLOByteTrackDetector
from core.fall_logic import FallStateMachine
from core.preprocess import Preprocessor
from core.source import FrameSource


class Pipeline:
    def __init__(self, video_path, config, state):
        self.config = config or {}
        self.source = FrameSource(video_path)
        self.preprocess = Preprocessor()
        self.detector = YOLOByteTrackDetector(
            self.config.get("model", "yolo26n.pt"),
            self.config.get("conf", 0.25),
            self.config.get("iou", 0.5),
            self.config.get("imgsz", 640),
            self.config.get("device", ""),
        )
        self.fall_logic = FallStateMachine(self.config)
        self.state = state

    def process_all(self, output_path=None, on_progress=None):
        fps = max(float(self.source.fps or 24.0), 1.0)
        frame_count = int(self.source.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        writer = None
        frame_index = 0
        try:
            while True:
                ok, frame = self.source.read()
                if not ok:
                    break
                frame = self.preprocess.process(frame)
                detections, annotated = self.detector.detect_and_track(frame)
                fallen = False
                for det in detections:
                    self.state.add_track_id(det.get("track_id"))
                    event = self.fall_logic.update_track(det, frame_index, fps, frame.shape[0])
                    if event:
                        self.state.add_event(event)
                        fallen = True

                if output_path:
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(
                            output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Cannot create annotated video: {output_path}")
                    writer.write(annotated)

                self.state.update_frame(annotated, "medium" if fallen else "low", fallen, False)
                frame_index += 1
                if on_progress:
                    fraction = frame_index / frame_count if frame_count > 0 else 0.0
                    on_progress(min(1.0, fraction), annotated)
        finally:
            self.source.release()
            if writer is not None:
                writer.release()
        return {"frames_processed": frame_index, "events": len(self.state.events)}
