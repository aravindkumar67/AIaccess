import cv2
import numpy as np

class GeometryMapper:
    def __init__(self, calibration_data=None):
        self.coords_source = "estimated"
        self.H = np.eye(3)
        if calibration_data:
            self.coords_source = "calibrated"

    def get_footprint(self, bbox, bottom_ratio=0.35):
        x1, y1, x2, y2 = bbox
        footprint_y1 = y2 - int((y2 - y1) * bottom_ratio)
        return [x1, footprint_y1, x2, y2]

    def pixel_to_meters(self, px, py):
        pt = np.array([px, py, 1.0])
        res = self.H @ pt
        res /= res[2]
        return [round(float(res[0]), 2), round(float(res[1]), 2)]