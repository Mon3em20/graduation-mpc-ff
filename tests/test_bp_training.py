import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_bp_training", ROOT / "scripts/verify/verify_bp_training.py")
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class BPTrainingTests(unittest.TestCase):
    def test_aggregate_parser_rejects_per_example_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.log"
            path.write_text(
                "validation loss=1.25 correct=4 total=30\n"
                "test loss=1.5 correct=3 total=30\n", encoding="utf-8")
            values = CHECKER.parse(path)
        self.assertEqual(values["validation"]["total"], 30)
        self.assertEqual(values["test"]["correct"], 3)

    def test_parser_requires_both_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.log"
            path.write_text("validation loss=1 correct=0 total=30\n", encoding="utf-8")
            self.assertEqual(set(CHECKER.parse(path)), {"validation"})


if __name__ == "__main__":
    unittest.main()
