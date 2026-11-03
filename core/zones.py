import cv2
import numpy as np

class ZoneManager:
    def __init__(self, config):
        self.config = config
        self.corridor_mask = None
        self.strip_mask = None
        self.conveyor_roi = None

    def initialize_setup(self, setup_frames):
        if not setup_frames:
            return
        h, w = setup_frames[0].shape[:2]
        self.corridor_mask = np.zeros((h, w), dtype=np.uint8)
        self.strip_mask = np.zeros((h, w), dtype=np.uint8)
        self.conveyor_roi = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(self.corridor_mask, (int(w*0.2), int(h*0.3)), (int(w*0.8), int(h*0.9)), 255, -1)