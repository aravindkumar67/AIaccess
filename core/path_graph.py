from .geometry import center,distance,overlap_ratio

class PathGraph:
    """Provisional image-space approach check. Real deployment needs calibrated path coordinates."""
    def __init__(self,regions,logic): self.regions=regions; self.logic=logic; self.counts={}; self.last_distance={}
    def check_amr_approach(self,track_id,amr_bbox,fallen_bbox,frame_shape):
        d=distance(center(amr_bbox),center(fallen_bbox)); on_path=overlap_ratio(frame_shape,amr_bbox,self.regions.get('amr_path',[]))>0
        prev=self.last_distance.get(int(track_id)); moving_closer=prev is None or d<=prev+2.0
        self.last_distance[int(track_id)]=d
        candidate=on_path and d<=float(self.logic.get('amr_near_px',180)) and moving_closer
        tid=int(track_id); self.counts[tid]=self.counts.get(tid,0)+1 if candidate else 0
        return self.counts[tid]>=int(self.logic.get('amr_confirm_frames',3)),{'distance_px':d,'on_path':bool(on_path),'moving_closer':bool(moving_closer)}
