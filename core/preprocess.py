import cv2
import numpy as np

class Preprocessor:
    def __init__(self, camera_matrix=None, dist_coeffs=None, enable_denoise=False):
        self.camera_matrix = np.array(camera_matrix) if camera_matrix is not None else None
        self.dist_coeffs = np.array(dist_coeffs) if dist_coeffs is not None else None
        self.enable_denoise = enable_denoise

    def process(self, frame):
        if frame is None:
            return None
        out = frame.copy()
        if self.camera_matrix is not None and self.dist_coeffs is not None:
            out = cv2.undistort(out, self.camera_matrix, self.dist_coeffs)
        if self.enable_denoise:
            out = cv2.fastNlMeansDenoisingColored(out, None, 10, 10, 7, 21)
        return out