import json
import numpy as np
import cv2

class PathNetwork:
    def __init__(self, path_network_file: str):
        with open(path_network_file, 'r') as f:
            self.data = json.load(f)
        self.corridors = self.data.get("corridors", [])
        self.conveyor_polygon = self.data.get("conveyor_zone_polygon", [])

    def is_in_conveyor_zone(self, bbox: list) -> bool:
        if not self.conveyor_polygon:
            return False
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        pts = np.array(self.conveyor_polygon, dtype=np.int32)
        return cv2.pointPolygonTest(pts, (float(cx), float(cy)), False) >= 0

    def evaluate_footprint(self, footprint_box: list):
        # footprint_box: [[x1, y1], [x2, y2]]
        x1, y1 = footprint_box[0]
        x2, y2 = footprint_box[1]

        sample_pts = [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2),
            ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        ]

        interior_hits = 0
        matched_corridor = "Corridor_A1"

        for corr in self.corridors:
            polygon = np.array(corr["interior_polygon"], dtype=np.int32)
            for pt in sample_pts:
                if cv2.pointPolygonTest(polygon, (float(pt[0]), float(pt[1])), False) >= 0:
                    interior_hits += 1
            matched_corridor = corr.get("id", "Corridor_A1")

        interior_ratio = (interior_hits / len(sample_pts)) * 100.0
        strip_ratio = 100.0 - interior_ratio

        if interior_ratio >= 1.0:
            level = "high"
        elif interior_hits > 0:
            level = "medium"
        else:
            level = "low"

        return level, interior_ratio, strip_ratio, matched_corridor