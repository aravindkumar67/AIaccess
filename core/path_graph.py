import numpy as np

class PathGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def build_from_mask(self, mask):
        pass

    def check_amr_approach(self, amr_box, object_box, frame_history):
        return False