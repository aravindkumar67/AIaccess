"""Create deterministic synthetic clips for smoke-testing the pipeline (not accuracy validation)."""
from pathlib import Path
import cv2
import numpy as np

OUT=Path("test_videos")
OUT.mkdir(exist_ok=True)

def make_clip(name,shape="ball",fall_from_height=True):
    w,h=960,540; fps=20; frames=160
    path=OUT/f"{name}.mp4"
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
    for i in range(frames):
        img=np.full((h,w,3),(38,42,48),dtype=np.uint8)
        # Fixed-camera industrial mockup: conveyor and floor bands.
        cv2.rectangle(img,(30,210),(930,345),(75,80,86),-1)
        cv2.line(img,(30,345),(930,345),(180,180,180),3)
        cv2.rectangle(img,(0,430),(960,540),(52,54,58),-1)
        # Fixed arm silhouette at right, deliberately excluded by region config.
        cv2.rectangle(img,(820,120),(855,310),(95,100,108),-1)
        if fall_from_height:
            y=int(80+min(i,90)*3.2) if i<90 else 368
            x=460
        else:
            # object is initially on conveyor, then tips/falls from its edge
            x=440+max(0,i-45)*4
            y=280+max(0,i-45)*2.4
        x=min(900,int(x)); y=min(410,int(y))
        if shape=="ball":
            cv2.circle(img,(x,y),26,(35,145,245),-1)
        elif shape=="box":
            cv2.rectangle(img,(x-26,y-25),(x+26,y+25),(30,190,90),-1)
        elif shape=="cuboid":
            pts=np.array([[x-30,y-18],[x+20,y-18],[x+30,y+20],[x-20,y+20]],np.int32)
            cv2.fillPoly(img,[pts],(210,90,45))
        writer.write(img)
    writer.release()
    print(path)

if __name__=="__main__":
    make_clip("ball_fall_from_height","ball",True)
    make_clip("box_fall_from_height","box",True)
    make_clip("cuboid_tip_off_conveyor","cuboid",False)
