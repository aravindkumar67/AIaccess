import cv2
import numpy as np

class SignalEvaluator:
    """Grade a confirmed floor obstacle using its image-space footprint.
    A High-risk interior hit takes precedence over strip overlap.
    """
    def __init__(self, config):
        self.config = config or {}
        self.regions = self.config.get("regions", {})

    @staticmethod
    def _mask(poly, width, height):
        mask = np.zeros((height, width), dtype=np.uint8)
        if not poly or len(poly) < 3:
            return mask
        points = np.asarray([[int(x * width), int(y * height)] for x, y in poly], dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        return mask

    @staticmethod
    def _overlap(footprint, zone):
        denom = int(np.count_nonzero(footprint))
        if denom == 0:
            return 0.0
        return float(np.count_nonzero(cv2.bitwise_and(footprint, zone))) / denom

    def evaluate_detection(self, bbox, width, height):
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        # Lower part of the box approximates floor contact; avoid grading its full body.
        ratio = float(self.config.get("footprint_bottom_ratio", 0.35))
        fy1 = max(y1, int(y2 - max(1, y2-y1) * ratio))
        footprint = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(footprint, (max(0,x1), max(0,fy1)), (min(width-1,x2), min(height-1,y2)), 255, -1)

        interior = self._mask(self.regions.get("high_risk", []), width, height)
        strip = self._mask(self.regions.get("yellow_path", []), width, height)
        interior_overlap = self._overlap(footprint, interior)
        strip_overlap = self._overlap(footprint, strip)
        high_min = float(self.config.get("high_interior_overlap_min", 0.01))
        strip_min = float(self.config.get("medium_strip_overlap_min", 0.01))

        # High must win whenever the object footprint overlaps the high-risk interior.
        if interior_overlap >= high_min:
            level, reason = "high", "footprint overlaps configured high-risk interior"
        elif strip_overlap >= strip_min:
            level, reason = "medium", "footprint overlaps configured yellow path"
        else:
            level, reason = "low", "confirmed ground object outside configured risk regions"

        return {
            "signal_level": level,
            "risk": level,
            "interior_overlap": round(interior_overlap, 4),
            "strip_overlap": round(strip_overlap, 4),
            "basis": reason,
        }

    def evaluate_level(self, interior_overlap: float, strip_overlap: float) -> str:
        if interior_overlap >= float(self.config.get("high_interior_overlap_min", 0.01)):
            return "high"
        if strip_overlap >= float(self.config.get("medium_strip_overlap_min", 0.01)):
            return "medium"
        return "low"
