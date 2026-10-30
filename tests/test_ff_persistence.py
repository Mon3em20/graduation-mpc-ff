import csv
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ff_mnist_under_test", ROOT / "src" / "baselines" / "ff.py")
FF = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FF)
BP_SPEC = importlib.util.spec_from_file_location("bp_mnist_under_test", ROOT / "src" / "baselines" / "bp.py")
BP = importlib.util.module_from_spec(BP_SPEC)
BP_SPEC.loader.exec_module(BP)


class FFPersistenceTests(unittest.TestCase):
    def test_canonical_defaults_and_explicit_paths(self):
        self.assertEqual(FF.parse_output_path(["ff_mnist.py"]), ROOT / "results" / "baselines" / "ff.csv")
        self.assertEqual(BP.parse_output_path(["bp_mnist.py"]), ROOT / "results" / "baselines" / "bp.csv")
        self.assertEqual(
            FF.parse_output_path(["ff_mnist.py", "--reference-check", "--output", "/tmp/ff.csv"]),
            Path("/tmp/ff.csv"),
        )
        self.assertEqual(
            BP.parse_output_path(["bp_mnist.py", "--output", "/tmp/bp.csv"]),
            Path("/tmp/bp.csv"),
        )

    def test_preservation_filters_only_regenerated_record_types(self):
        rows = [
            {"record_type": "epoch", "id": "replace"},
            {"record_type": "reference_ff_epoch", "id": "keep-reference"},
            {"record_type": "negative_comparison_epoch", "id": "keep-negative"},
            {"record_type": "historical_note", "id": "keep-other"},
        ]
        self.assertEqual(
            FF.preserve_historical_rows(rows),
            rows[1:],
        )

    def test_write_preserves_unrelated_records_and_uses_requested_path(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic.csv"
            rows = [
                {"record_type": "reference_ff_epoch", "reference_id": "ref"},
                {"record_type": "negative_comparison_aggregate", "experiment_id": "neg"},
            ]
            FF.write_csv_atomic(output, rows)

            with output.open(newline="") as stream:
                written = list(csv.DictReader(stream))

            self.assertEqual(
                [{key: row.get(key, "") for key in ("record_type", "reference_id", "experiment_id")}
                 for row in written],
                [
                    {"record_type": "reference_ff_epoch", "reference_id": "ref", "experiment_id": ""},
                    {"record_type": "negative_comparison_aggregate",
                     "reference_id": "", "experiment_id": "neg"},
                ],
            )
            self.assertNotEqual(output, ROOT / "results" / "baselines" / "ff.csv")
            self.assertFalse(list(output.parent.glob(".*.tmp")))

    def test_failed_replace_keeps_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "synthetic.csv"
            output.write_text("record_type,value\nold,1\n", encoding="utf-8")
            with mock.patch.object(os, "replace", side_effect=OSError("simulated failure")):
                with self.assertRaises(OSError):
                    FF.write_csv_atomic(output, [{"record_type": "new", "value": "2"}])
            self.assertEqual(output.read_text(encoding="utf-8"), "record_type,value\nold,1\n")
            self.assertFalse(list(output.parent.glob(".*.tmp")))

    def test_hard_negative_requires_auxiliary_predictor(self):
        with self.assertRaisesRegex(ValueError, "auxiliary_predictor=True"):
            FF.run(None, None, [], [], [], 42, 1,
                   negative_strategy="neutral_softmax_hard")

    def test_unknown_negative_strategy_is_rejected_early(self):
        with self.assertRaisesRegex(ValueError, "unsupported negative strategy"):
            FF.run(None, None, [], [], [], 42, 1, negative_strategy="unknown")

    def test_bp_baseline_preserves_non_baseline_records(self):
        rows = [
            {"record_type": "epoch", "id": "replace"},
            {"record_type": "reference_bp_epoch", "id": "keep-reference"},
            {"record_type": "historical_note", "id": "keep-other"},
        ]
        preserved = [
            row for row in rows
            if row.get("record_type") not in BP.BASELINE_RECORD_TYPES
        ]
        self.assertEqual(preserved, rows[1:])


if __name__ == "__main__":
    unittest.main()
