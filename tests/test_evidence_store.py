import json
import numpy as np
from fallwatch.events import EvidenceStore

def test_evidence_store_writes_metadata_images_and_clip(tmp_path):
    frame=np.zeros((64,64,3),dtype=np.uint8)
    store=EvidenceStore(tmp_path)
    result=store.save("fall_test",frame,frame,
        {"object_fallen":True,"arm_approach":True},[frame,frame],fps=5)
    folder=tmp_path/"fall_test"
    assert (folder/"fall_motion.jpg").exists()
    assert (folder/"ground_confirmed.jpg").exists()
    assert (folder/"fall_clip.mp4").exists()
    meta=json.loads((folder/"metadata.json").read_text())
    assert meta["object_fallen"] is True
    assert meta["arm_approach"] is True
    assert result["video_clip"].endswith("fall_clip.mp4")
