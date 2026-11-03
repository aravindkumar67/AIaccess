import time
from core.source import FrameSource
from core.preprocess import Preprocessor
from core.detector import YOLOByteTrackDetector
from core.zones import ZoneManager
from core.state import SharedState

class Pipeline:
    def __init__(self, video_path: str, config: dict, state: SharedState):
        self.source = FrameSource(video_path)
        self.preprocess = Preprocessor()
        self.detector = YOLOByteTrackDetector(config.get("model", "yolo26n.pt"), config.get("conf", 0.25), config.get("iou", 0.5))
        self.zone_mgr = ZoneManager(config)
        self.state = state

    def run_setup(self):
        frames = []
        for _ in range(30):
            ret, frame = self.source.read()
            if ret:
                frames.append(frame)
        self.zone_mgr.initialize_setup(frames)

    def process_all(self):
        self.run_setup()
        while True:
            ret, frame = self.source.read()
            if not ret:
                break
            proc_frame = self.preprocess.process(frame)
            self.state.update_frame(proc_frame, "low", False, False)
        self.source.release()