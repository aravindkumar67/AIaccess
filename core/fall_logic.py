class FallStateMachine:
    def __init__(self, config):
        self.config = config
        self.state = "ON_CONVEYOR"
        self.conveyor_timer = 0.0

    def update(self, in_conveyor: bool, center_y: float, fps: float):
        return self.state