from abc import ABC, abstractmethod
import cv2

class FrameSource(ABC):
    @abstractmethod
    def read(self): raise NotImplementedError
    @abstractmethod
    def release(self): raise NotImplementedError

class OpenCVSource(FrameSource):
    def __init__(self, source): self.capture=cv2.VideoCapture(source)
    def is_opened(self): return self.capture.isOpened()
    def read(self):
        ok, frame=self.capture.read()
        return frame if ok else None
    def release(self): self.capture.release()
