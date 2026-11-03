from core.fall_logic import FallDetector

def test_fall_confirmation():
    d=FallDetector({'fall_vertical_px':20,'fall_confirm_frames':2,'settled_speed_px_per_frame':4})
    assert not d.update(1,'box',[0,0,20,20],1)[1]
    d.update(1,'box',[0,25,20,45],2); state,event=d.update(1,'box',[0,50,20,70],3)
    assert event and state.fallen
