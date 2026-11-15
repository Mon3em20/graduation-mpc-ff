import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_secure_bp_program",
    ROOT / "scripts" / "generate_secure_bp_program.py")
GENERATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GENERATOR)


class SecureBPProgramTests(unittest.TestCase):
    def test_profile_dimensions_and_schedule(self):
        bounded = GENERATOR.PROFILES["bounded"]
        matched = GENERATOR.PROFILES["matched"]
        self.assertEqual(bounded["layer_sizes"], [784, 8, 10])
        self.assertEqual(matched["layer_sizes"], [784, 128, 128, 10])
        self.assertEqual(sum(a * b + b for a, b in
                             zip(bounded["layer_sizes"], bounded["layer_sizes"][1:])),
                         6370)
        self.assertEqual(sum(a * b + b for a, b in
                             zip(matched["layer_sizes"], matched["layer_sizes"][1:])),
                         118282)
        self.assertEqual(len(bounded["batch_sizes"]), 30)
        self.assertEqual(len(matched["batch_sizes"]), 40)
        self.assertEqual(matched["batch_sizes"][-1], 8)

    def test_rendered_program_has_shared_operations(self):
        for profile in ("bounded", "matched"):
            source = GENERATOR.render(profile)
            self.assertNotIn("__LAYER_SIZES__", source)
            self.assertIn("deltas[-1]", source)
            self.assertIn("ml.softmax", source)
            self.assertIn("weight_gradients", source)
            self.assertIn("batch_size", source)

    def test_bounded_and_matched_use_same_template(self):
        bounded = GENERATOR.render("bounded")
        matched = GENERATOR.render("matched")
        shared = ("def forward", "def cross_entropy", "def train_epoch",
                  "def evaluate", "strict")
        for marker in shared[:-1]:
            self.assertIn(marker, bounded)
            self.assertIn(marker, matched)

    def test_training_uses_runtime_loops(self):
        source = GENERATOR.render("matched")
        self.assertIn("LOOP_BUDGET = 100", source)
        self.assertIn("@for_range_opt(batch_size, budget=LOOP_BUDGET)", source)
        self.assertIn("@for_range_opt(EPOCHS, budget=LOOP_BUDGET)", source)
        self.assertNotIn("for sample in range(batch_size)", source)
        self.assertNotIn("for epoch in range(EPOCHS)", source)


if __name__ == "__main__":
    unittest.main()
