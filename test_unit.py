import unittest
from core.signal_logic import SignalEvaluator

class TestSignalLogic(unittest.TestCase):
    def test_risk_grading(self):
        evaluator = SignalEvaluator({"high_interior_overlap_min": 0.01, "medium_strip_overlap_min": 0.01})
        self.assertEqual(evaluator.evaluate_level(0.05, 0.0), "high")
        self.assertEqual(evaluator.evaluate_level(0.0, 0.05), "medium")
        self.assertEqual(evaluator.evaluate_level(0.0, 0.0), "low")

if __name__ == "__main__":
    unittest.main()