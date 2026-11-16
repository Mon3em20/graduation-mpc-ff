"""Offline tests for the batched-matrix matched implementation (no MPC executed)."""
import importlib.util
import re
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for sub in ("src", "scripts", "scripts/verify"):
    sys.path.insert(0, str(ROOT / sub))
import matrix_bp_reference as R  # noqa: E402
import prepare_secure_bp_matched as P  # noqa: E402
import run_matrix_reference as RR  # noqa: E402
from baselines.bp import datasets_and_indices  # noqa: E402

MATRIX = (ROOT / "mpc/programs/bp/bp_matched_matrix.mpc").read_text()
SMOKE_SPEC = importlib.util.spec_from_file_location(
    "gen_smoke", ROOT / "scripts/generate_matrix_bp_smoke.py")
GEN = importlib.util.module_from_spec(SMOKE_SPEC)
SMOKE_SPEC.loader.exec_module(GEN)

_CACHE = {}


def setup():
    if not _CACHE:
        train_ds, test_ds, tr, va, te = datasets_and_indices()
        _, mnist = P.training_orders(tr, 42)
        _CACHE.update(
            ds=(train_ds, test_ds, tr, va, te),
            plan=list(P.input_plan(mnist, va, te)),
            arrays=RR.arrays(train_ds, test_ds),
            params=P.serialized_parameters(42))
        _CACHE["torch"] = RR.torch_baseline(train_ds, test_ds, tr, va, te)
        _CACHE["sim"] = R.simulate(_CACHE["params"], _CACHE["plan"],
                                   *_CACHE["arrays"], "float")
    return _CACHE


class MatrixProgramContractTests(unittest.TestCase):
    def test_schedule_constants_match_scalar_program(self):
        for line in ("TRAIN, VALIDATION, TEST = 5000, 1000, 1000",
                     "FULL_BATCH, FINAL_BATCH = 128, 8", "FULL_BATCHES = 39",
                     "EPOCHS = 5", "LR = cfix(0.05)", "sfix.set_precision(16, 31)"):
            self.assertIn(line, MATRIX)
        scalar = (ROOT / "mpc/programs/bp/bp_matched.mpc").read_text()
        self.assertIn("LAYER_SIZES = [784, 128, 128, 10]", scalar)
        self.assertIn("INPUTS, HIDDEN_1, HIDDEN_2, OUTPUTS = 784, 128, 128, 10", MATRIX)

    def test_input_consumption_order(self):
        # parameters (W[in,out] then bias per layer) then per-sample x, y with owner formula
        self.assertRegex(MATRIX, r"for parameter in \(W1, b1, W2, b2, W3, b3\):\s+parameter\.input_from\(0, binary=False\)")
        self.assertIn("owner = (start + sample) % 3", MATRIX)
        self.assertLess(MATRIX.index("sample_x.input_from(owner"),
                        MATRIX.index("sample_y.input_from(owner"))
        self.assertIn("train_batch(full_buffers, batch * FULL_BATCH, loss_total)", MATRIX)
        self.assertIn("train_batch(final_buffers, FULL_BATCHES * FULL_BATCH, loss_total)", MATRIX)
        self.assertIn("load_rows(buf, chunk * EVAL_CHUNK)", MATRIX)
        # sequential (non-parallel) loop for input reads
        self.assertRegex(MATRIX, r"@for_range\(buf\.rows\)\s+def _\(sample\):\s+owner")

    def test_update_avoids_single_quantized_constant(self):
        # 0.05/128 is not representable with 16 fractional bits (+1.56 % LR bias)
        quantized = round(0.05 / 128 * 65536) / 65536
        self.assertAlmostEqual(quantized / (0.05 / 128) - 1, 0.015625, places=6)
        self.assertIn("cfix(1.0 / batch_rows)", MATRIX)
        self.assertIn("(gradient.get_vector() * LR) * inverse_batch", MATRIX)
        self.assertNotIn("LR / ", MATRIX.replace("LR / batch", "X"))  # no LR/B constant
        for rows in (128, 8):
            self.assertEqual((1.0 / rows) * 65536, int((1.0 / rows) * 65536))

    def test_only_aggregate_reveals(self):
        reveals = re.findall(r"\.reveal\(\)", MATRIX)
        self.assertEqual(len(reveals), 3)  # epoch loss; evaluation loss and correct count
        self.assertNotIn("print_ln(\"%s\", W", MATRIX)

    def test_strict_relu_mask_reused_for_derivative(self):
        self.assertIn("buf.M1.assign_vector(buf.Z1.get_vector() > 0)", MATRIX)
        self.assertIn("buf.D1.get_vector() * buf.M1.get_vector()", MATRIX)

    def test_smoke_profile_is_constants_only(self):
        self.assertEqual((ROOT / "mpc/programs/bp/bp_matched_matrix_smoke.mpc").read_text(),
                         GEN.render())
        smoke = GEN.render().splitlines()
        full = MATRIX.splitlines()
        diff = [l for l in smoke[1:] if l not in full]
        self.assertEqual(sorted(diff), sorted([
            "TRAIN, VALIDATION, TEST = 264, 100, 100", "FULL_BATCHES = 2", "EPOCHS = 1"]))


class ReferenceEquivalenceTests(unittest.TestCase):
    def test_float_matrix_algorithm_equals_pytorch_float64_all_epochs(self):
        c = setup()
        t_losses, t_snaps, t_val, t_test = c["torch"]
        sim = c["sim"]
        for e in range(5):
            self.assertAlmostEqual(sim["epoch_losses"][e], t_losses[e], places=12)
            self.assertLess(RR.maxdiff(sim["snapshots"][e], t_snaps[e]), 1e-12)
        self.assertEqual(sim["validation"][1], t_val[1])
        self.assertEqual(sim["test"][1], t_test[1])
        self.assertAlmostEqual(sim["validation"][0], t_val[0], places=10)

    def test_matches_recorded_float32_baseline(self):
        sim = setup()["sim"]
        expected = [2.2747019737243654, 2.161566234970093, 1.8416416954040526,
                    1.3057574975967408, 0.9091569893836975]  # results/baselines/bp.csv
        for got, want in zip(sim["epoch_losses"], expected):
            self.assertAlmostEqual(got, want, places=6)
        self.assertEqual((sim["validation"][1], sim["test"][1]), (766, 772))

    def test_streams_fully_and_exactly_consumed(self):
        sim = setup()["sim"]
        self.assertTrue(sim["fully_consumed"])
        self.assertEqual(sim["consumed"], [9003, 9001, 8996])
        tokens = [118282 + 794 * sim["consumed"][0], 794 * sim["consumed"][1],
                  794 * sim["consumed"][2]]
        self.assertEqual(tokens, [7266664, 7146794, 7142824])

    def test_consumption_order_is_dataloader_order_per_epoch(self):
        c = setup()
        trace = []
        R.simulate(c["params"], c["plan"], *c["arrays"], "float", trace=trace)
        _, _, tr, _, _ = c["ds"]
        reference = P.baseline_style_orders(tr, 42)
        self.assertEqual(len(trace), 25000)
        self.assertEqual([trace[e * 5000:(e + 1) * 5000] for e in range(5)], reference)

    def test_partial_and_multiple_batches_persist_parameters(self):
        c = setup()
        plan = c["plan"][:10000] + c["plan"][25000:]  # two epochs, then evaluation
        short = R.simulate(c["params"], plan, *c["arrays"], "float", epochs=2)
        t_losses, t_snaps, _, _ = c["torch"]
        self.assertLess(RR.maxdiff(short["snapshots"][1], t_snaps[1]), 1e-12)

    def test_fixed_point_emulation_stays_close(self):
        c = setup()
        fx = R.simulate(c["params"], c["plan"], *c["arrays"], "fixed", seed=0)
        for got, want in zip(fx["epoch_losses"], c["sim"]["epoch_losses"]):
            self.assertLess(abs(got - want), 2e-3)
        self.assertLess(abs(fx["validation"][1] - c["sim"]["validation"][1]), 5)
        self.assertLess(abs(fx["test"][1] - c["sim"]["test"][1]), 5)
        drift = max(np.abs(fx["W"][l] - c["sim"]["W"][l]).max() for l in range(3))
        self.assertLess(drift, 2e-2)

    def test_single_constant_update_is_measurably_biased(self):
        """Regression: the rejected 0.05/B single-constant update biases training."""
        c = setup()
        original = R.run_batch

        def single(az, W, b, X, Y, lr, rows):
            (Z1, Z2, Z3), (M1, M2), (X, H1, H2) = R.forward(az, W, b, X)
            Pm, loss, _ = R.softmax_loss(az, Z3, Y)
            D3 = Pm - Y
            D2 = az.mm(D3, W[2].T) * M2
            D1 = az.mm(D2, W[1].T) * M1
            gw = [az.mm(X.T, D1), az.mm(H1.T, D2), az.mm(H2.T, D3)]
            gb = [D1.sum(0), D2.sum(0), D3.sum(0)]
            k = az.const(lr / rows)
            for l in range(3):
                W[l] = W[l] - az.mul_const(gw[l], k)
                b[l] = b[l] - az.mul_const(gb[l], k)
            return loss.sum()
        R.run_batch = single
        try:
            biased = R.simulate(c["params"], c["plan"], *c["arrays"], "fixed", seed=0)
        finally:
            R.run_batch = original
        ref = c["sim"]["epoch_losses"][-1]
        self.assertGreater(abs(biased["epoch_losses"][-1] - ref), 0.01)


if __name__ == "__main__":
    unittest.main()
