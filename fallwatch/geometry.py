import cv2
import numpy as np

def polygon_mask(shape, normalized_polygon):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    if not normalized_polygon: return mask
    pts = np.array([[round(x*w), round(y*h)] for x,y in normalized_polygon], dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask

def point_in_polygon(point_xy, normalized_polygon, frame_shape):
    if not normalized_polygon: return False
    h,w = frame_shape[:2]
    pts = np.array([[round(x*w), round(y*h)] for x,y in normalized_polygon], dtype=np.int32)
    return cv2.pointPolygonTest(pts, (float(point_xy[0]), float(point_xy[1])), False) >= 0

def bbox_center(bbox):
    x1,y1,x2,y2 = bbox
    return ((x1+x2)/2.0, (y1+y2)/2.0)

def bbox_bottom_center(bbox):
    x1,_,x2,y2 = bbox
    return ((x1+x2)/2.0, y2)

def bbox_overlaps_polygon(bbox, normalized_polygon, frame_shape):
    return bool(normalized_polygon) and (point_in_polygon(bbox_center(bbox), normalized_polygon, frame_shape) or point_in_polygon(bbox_bottom_center(bbox), normalized_polygon, frame_shape))
