import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_secure_bp_program",
    ROOT / "scripts" / "generate_secure_bp_program.py")
GEN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GEN)


class DiagnosticProfileTests(unittest.TestCase):
    def test_full_architecture_smoke_profile(self):
        profile = GEN.PROFILES["diagnostic"]
        self.assertEqual(profile["layer_sizes"], [784, 128, 128, 10])
        self.assertEqual(profile["counts"], [9, 3, 3])
        self.assertEqual(profile["batch_sizes"], [3, 3, 3])
        self.assertEqual(profile["epochs"], 1)

    def test_diagnostic_source_is_distinct(self):
        self.assertNotEqual(GEN.PROFILES["diagnostic"]["output"],
                            GEN.PROFILES["matched"]["output"])
        source = GEN.render("diagnostic")
        self.assertIn("TRAIN, VALIDATION, TEST = [9, 3, 3]", source)
        self.assertIn("EPOCHS = 1", source)


if __name__ == "__main__":
    unittest.main()
