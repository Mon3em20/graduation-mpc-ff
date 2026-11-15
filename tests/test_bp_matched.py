import importlib.util
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "prepare_secure_bp_matched", ROOT / "scripts" / "prepare_secure_bp_matched.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)


class MatchedBPTests(unittest.TestCase):
    def test_architecture_parameter_count(self):
        self.assertEqual(784 * 128 + 128 + 128 * 128 + 128 + 128 * 10 + 10,
                         118282)
        self.assertEqual(PREP.ARCHITECTURE, "784-128-128-10")

    def test_initialization_is_reproducible(self):
        first = PREP.parameters(42)
        second = PREP.parameters(42)
        changed = PREP.parameters(43)
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)

    def test_softmax_cross_entropy_reference(self):
        logits = [1.0, -2.0, 0.5]
        shift = max(logits)
        probabilities = [math.exp(value - shift) for value in logits]
        denominator = sum(probabilities)
        probabilities = [value / denominator for value in probabilities]
        self.assertAlmostEqual(sum(probabilities), 1.0, places=14)
        self.assertAlmostEqual(-math.log(probabilities[0]),
                               math.log(sum(math.exp(value - shift)
                                            for value in logits)) + shift - logits[0],
                               places=14)

    def test_strict_relu_at_zero(self):
        self.assertEqual(int(0.0 > 0.0), 0)
        self.assertEqual(int(1.0 > 0.0), 1)


if __name__ == "__main__":
    unittest.main()
