import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("benchmark", ROOT / "scripts" / "benchmark.py")
BENCHMARK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCHMARK)


class BenchmarkHelpersTests(unittest.TestCase):
    def test_benchmark_delegates_algorithm_training(self):
        source = (ROOT / "scripts" / "benchmark.py").read_text()
        self.assertIn("BP.train_epoch(", source)
        self.assertIn("FF.train_layer_epoch(", source)
        self.assertNotIn("for x, y in train_loader:", source)
        self.assertNotIn("for images, labels in train_loader:", source)

    def test_benchmark_ff_uses_documented_inference_profile(self):
        self.assertEqual(BENCHMARK.FF_BENCHMARK_INFERENCE_VARIANT, "all_layers")

    def test_targets_use_validation_overlap(self):
        rows = [
            {"algorithm": "BP", "validation_accuracy_pct": 45},
            {"algorithm": "BP", "validation_accuracy_pct": 86},
            {"algorithm": "FF", "validation_accuracy_pct": 51},
            {"algorithm": "FF", "validation_accuracy_pct": 71},
        ]
        self.assertEqual(BENCHMARK.derive_targets(rows), [55, 60, 65, 70])

    def test_unreachable_target_is_explicit(self):
        rows = [
            {"algorithm": "BP", "seed": 42, "repetition": 1, "epoch": 1, "layer": "",
             "validation_accuracy_pct": 55, "training_time_seconds": 1,
             "validation_time_seconds": 0},
            {"algorithm": "FF", "seed": 42, "repetition": 1, "epoch": 1, "layer": 1,
             "validation_accuracy_pct": 45, "training_time_seconds": 2,
             "validation_time_seconds": 0},
        ]
        result = BENCHMARK.matched_accuracy(rows, [50])
        self.assertEqual({row["status"] for row in result}, {"reached", "not_reached"})

    def test_csv_export_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.csv"
            BENCHMARK.update_csv(path, [{"algorithm": "BP", "value": 1}])
            self.assertIn("algorithm,value", path.read_text())

    def test_bp_training_time_and_updates_are_rebuilt_from_epochs(self):
        runs = [{
            "algorithm": "BP", "seed": "42", "repetition": "1",
            "data_loader_seconds": "1", "model_initialization_seconds": "2",
            "training_time_seconds": "999", "validation_time_seconds": "999",
            "inference_time_seconds": "4", "optimizer_updates": "40",
            "train_samples": "5000", "epochs": "5",
        }]
        epochs = [
            {"algorithm": "BP", "seed": "42", "repetition": "1",
             "training_time_seconds": "3", "validation_time_seconds": "0.5",
             "optimizer_updates": "40", "epoch": "1", "layer": ""},
            {"algorithm": "BP", "seed": "42", "repetition": "1",
             "training_time_seconds": "4", "validation_time_seconds": "0.6",
             "optimizer_updates": "40", "epoch": "2", "layer": ""},
        ]
        repaired = BENCHMARK.rebuild_runs(runs, epochs)[0]
        self.assertEqual(float(repaired["training_time_seconds"]), 7)
        self.assertEqual(repaired["optimizer_updates"], 80)
        self.assertEqual(float(repaired["total_experiment_seconds"]), 15.1)

    def test_ff_matched_time_includes_previous_layer_and_repetition(self):
        rows = [
            {"algorithm": "FF", "seed": "42", "repetition": "1",
             "layer": "1", "epoch": str(epoch), "validation_accuracy_pct": "",
             "training_time_seconds": "1", "validation_time_seconds": "0"}
            for epoch in range(1, 6)
        ]
        rows.extend([
            {"algorithm": "FF", "seed": "42", "repetition": "1",
             "layer": "2", "epoch": "1", "validation_accuracy_pct": "60",
             "training_time_seconds": "2", "validation_time_seconds": "0.2"},
            {"algorithm": "FF", "seed": "42", "repetition": "2",
             "layer": "1", "epoch": "1", "validation_accuracy_pct": "",
             "training_time_seconds": "10", "validation_time_seconds": "0"},
            {"algorithm": "FF", "seed": "42", "repetition": "2",
             "layer": "2", "epoch": "1", "validation_accuracy_pct": "60",
             "training_time_seconds": "20", "validation_time_seconds": "0.2"},
        ])
        result = BENCHMARK.matched_accuracy(rows, [60])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["repetition"], "1")
        self.assertEqual(float(result[0]["training_only_seconds"]), 7)
        self.assertEqual(float(result[1]["training_only_seconds"]), 30)

    def test_summary_uses_one_utility_outcome_per_seed(self):
        runs = []
        for repetition in ("1", "2", "3"):
            runs.append({
                "algorithm": "BP", "seed": "42", "repetition": repetition,
                "test_accuracy_pct": "80", "training_time_seconds": "2",
                "optimizer_updates": "200", "training_presentations": "25000",
                "inference_time_seconds": "1",
            })
        summary = BENCHMARK.summarize_runs(runs, 3)[0]
        self.assertEqual(summary["mean_test_accuracy_pct"], 80)
        self.assertEqual(summary["independent_seeds"], 1)
        self.assertEqual(summary["timing_runs"], 3)

    def test_corrected_figure_set_has_no_memory_comparison(self):
        figures = BENCHMARK.OUT / "figures"
        self.assertTrue((figures / "accuracy.png").is_file())
        self.assertTrue((figures / "time_to_target.png").is_file())
        self.assertFalse((figures / "memory.png").exists())


if __name__ == "__main__":
    unittest.main()
