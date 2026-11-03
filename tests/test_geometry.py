from core.geometry import overlap_ratio
from core.signal_logic import grade_severity

def test_full_overlap(): assert overlap_ratio((100,100,3),[10,10,30,30],[[0,0],[50,0],[50,50],[0,50]])==1.0

def test_high_priority():
    r={'medium_risk':[[0,0],[99,0],[99,99],[0,99]],'high_risk':[[0,0],[99,0],[99,99],[0,99]]}
    assert grade_severity((100,100,3),[10,10,30,30],r,{'medium_overlap':.05,'high_overlap':.2})['severity']=='High'
