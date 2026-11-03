class SignalEvaluator:
    def __init__(self, config):
        self.config = config

    def evaluate_level(self, interior_overlap: float, strip_overlap: float) -> str:
        if interior_overlap >= self.config.get("high_interior_overlap_min", 0.01):
            return "high"
        elif strip_overlap >= self.config.get("medium_strip_overlap_min", 0.01):
            return "medium"
        return "low"