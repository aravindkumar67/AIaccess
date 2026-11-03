import cv2

class FrameSource:
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.cap = cv2.VideoCapture(source_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source_path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self):
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()