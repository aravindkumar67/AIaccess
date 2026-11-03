import cv2
import numpy as np

def polygon_mask(shape, polygon):
    h,w=shape[:2]; mask=np.zeros((h,w),dtype=np.uint8)
    if polygon and len(polygon)>=3: cv2.fillPoly(mask,[np.asarray(polygon,dtype=np.int32)],1)
    return mask

def bbox_mask(shape,bbox):
    h,w=shape[:2]; x1,y1,x2,y2=[int(round(v)) for v in bbox]
    x1=max(0,min(w,x1)); x2=max(0,min(w,x2)); y1=max(0,min(h,y1)); y2=max(0,min(h,y2))
    mask=np.zeros((h,w),dtype=np.uint8)
    if x2>x1 and y2>y1: mask[y1:y2,x1:x2]=1
    return mask

def overlap_ratio(shape,bbox,polygon):
    obj=bbox_mask(shape,bbox); area=int(obj.sum())
    return float((obj & polygon_mask(shape,polygon)).sum()/area) if area else 0.0

def center(bbox):
    x1,y1,x2,y2=bbox; return ((x1+x2)/2,(y1+y2)/2)

def distance(a,b):
    return float(np.hypot(a[0]-b[0],a[1]-b[1]))
