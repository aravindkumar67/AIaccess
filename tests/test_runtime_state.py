from fallwatch.state import RuntimeState

def test_flags_stay_true_until_explicit_reset_and_history_is_kept():
    state = RuntimeState()
    state.add_event({"event_id": "fall_1"})
    snap = state.snapshot()
    assert snap["object_fallen_onto_ground"] is True
    assert snap["arm_object_approach"] is True
    state.update(status="running")
    assert state.snapshot()["object_fallen_onto_ground"] is True
    state.reset_flags()
    snap = state.snapshot()
    assert snap["object_fallen_onto_ground"] is False
    assert snap["arm_object_approach"] is False
    assert len(snap["events"]) == 1
