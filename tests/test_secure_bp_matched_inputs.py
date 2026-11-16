"""Reproducibility tests for SECURE-BP-MATCHED-v1 input generation."""
import importlib.util
import itertools
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "prepare_secure_bp_matched_under_test",
    ROOT / "scripts" / "prepare_secure_bp_matched.py")
PREP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREP)
BP_SPEC = importlib.util.spec_from_file_location(
    "bp_baseline_under_test", ROOT / "src" / "baselines" / "bp.py")
BP = importlib.util.module_from_spec(BP_SPEC)
BP_SPEC.loader.exec_module(BP)
FF_SPEC = importlib.util.spec_from_file_location(
    "ff_baseline_under_test", ROOT / "src" / "baselines" / "ff.py")
FF = importlib.util.module_from_spec(FF_SPEC)
FF_SPEC.loader.exec_module(FF)

_CACHE = {}


def data():
    if not _CACHE:
        _CACHE["data"] = BP.datasets_and_indices()
        _CACHE["orders"] = PREP.training_orders(_CACHE["data"][2], 42)
    return _CACHE["data"], _CACHE["orders"]


class SplitTests(unittest.TestCase):
    def test_split_sizes_and_bp_ff_equality(self):
        (_, _, tr, va, te), _ = data()
        self.assertEqual((len(tr), len(va), len(te)), (5000, 1000, 1000))
        self.assertFalse(set(tr) & set(va))
        self.assertEqual(len(set(te)), 1000)
        _, _, ftr, fva, fte = FF.get_data()
        self.assertEqual((tr, va, te), (ftr, fva, fte))


class OrderTests(unittest.TestCase):
    def test_five_epoch_dataloader_order_equals_baseline(self):
        (train_ds, _, tr, _, _), (positions, mnist) = data()
        # Independent baseline loader over the real MNIST Subset: compare labels
        # sample by sample for every epoch, exactly as bp.py trains.
        loader = DataLoader(Subset(train_ds, tr), batch_size=128, shuffle=True,
                            generator=torch.Generator().manual_seed(42),
                            num_workers=0)
        for epoch in range(5):
            labels = [int(y) for _, y_batch in loader for y in y_batch]
            expected = [int(train_ds.targets[index]) for index in mnist[epoch]]
            self.assertEqual(labels, expected, f"epoch {epoch + 1}")
        reference = PREP.baseline_style_orders(tr, 42)
        self.assertEqual(mnist, reference)

    def test_epochs_are_permutations_and_distinct(self):
        (_, _, tr, _, _), (positions, mnist) = data()
        for epoch in range(5):
            self.assertEqual(sorted(positions[epoch]), list(range(5000)))
            self.assertEqual(sorted(mnist[epoch]), sorted(tr))
            self.assertEqual([tr[p] for p in positions[epoch]], mnist[epoch])
        self.assertEqual(len({tuple(order) for order in positions}), 5)

    def test_plain_randperm_would_differ(self):
        _, (positions, _) = data()
        generator = torch.Generator().manual_seed(42)
        naive = torch.randperm(5000, generator=generator).tolist()
        self.assertNotEqual(naive, positions[0])

    def test_hash_is_stable_and_order_sensitive(self):
        _, (_, mnist) = data()
        self.assertEqual(PREP.order_hash(mnist[0]), PREP.order_hash(list(mnist[0])))
        self.assertNotEqual(PREP.order_hash(mnist[0]), PREP.order_hash(mnist[1]))
        self.assertNotEqual(PREP.order_hash([1, 2]), PREP.order_hash([2, 1]))


class BatchAndOwnershipTests(unittest.TestCase):
    def test_batch_schedule(self):
        self.assertEqual(PREP.BATCH_SIZES, [128] * 39 + [8])
        self.assertEqual(sum(PREP.BATCH_SIZES), 5000)
        batches = PREP.dataloader_epoch_orders(PREP.IndexDataset(5000), 42)
        for epoch in batches:
            self.assertEqual([len(b) for b in epoch], PREP.BATCH_SIZES)

    def test_mpc_source_agrees(self):
        source = (ROOT / "mpc/programs/bp/bp_matched.mpc").read_text()
        self.assertIn("BATCH_SIZES = " + repr(PREP.BATCH_SIZES), source)
        self.assertIn("EPOCHS = 5", source)
        self.assertIn("(batch_start + sample) % 3", source)
        self.assertIn("owner = sample % 3", source)
        self.assertIn("LAYER_SIZES = [784, 128, 128, 10]", source)

    def test_training_ownership_matches_mpc_formula(self):
        (_, _, tr, va, te), (_, mnist) = data()
        plan = list(PREP.input_plan(mnist, va, te))
        train_plan = plan[:25000]
        for epoch in range(5):
            start = 0
            for batch_index, size in enumerate(PREP.BATCH_SIZES):
                batch_start = batch_index * 128
                for sample in range(size):
                    entry = train_plan[epoch * 5000 + start + sample]
                    self.assertEqual(entry[0], (batch_start + sample) % 3)
                    self.assertEqual(entry[2], mnist[epoch][batch_start + sample])
                start += size
        for offset, indices in ((25000, va), (26000, te)):
            for sample, index in enumerate(indices):
                owner, split, mnist_index = plan[offset + sample]
                self.assertEqual((owner, mnist_index), (sample % 3, index))
        self.assertEqual({s for _, s, _ in plan[25000:26000]}, {"validation"})
        self.assertEqual({s for _, s, _ in plan[26000:]}, {"test"})

    def test_expected_token_counts(self):
        (_, _, _, va, te), (_, mnist) = data()
        plan = list(PREP.input_plan(mnist, va, te))
        self.assertEqual(len(plan), 27000)
        self.assertEqual(PREP.expected_token_counts(plan),
                         [7266664, 7146794, 7142824])


class InitializationTests(unittest.TestCase):
    def reference_params(self):
        torch.manual_seed(42)
        model = BP.make_model()
        return [model[1].weight, model[1].bias, model[3].weight, model[3].bias,
                model[5].weight, model[5].bias], model

    def test_initial_parameters_match_baseline_exactly(self):
        reference, _ = self.reference_params()
        actual = PREP.parameters(42)
        for tensor, values in zip(reference, actual):
            self.assertEqual(tensor.detach().flatten().tolist(), values)
        self.assertEqual(sum(len(v) for v in actual), 118282)

    def test_weight_transpose_and_bias_untouched(self):
        reference, _ = self.reference_params()
        serialized = PREP.serialized_parameters(42)
        for layer, (n_in, n_out) in enumerate(PREP.LAYER_SHAPES):
            weight = reference[2 * layer].detach()
            self.assertEqual(tuple(weight.shape), (n_out, n_in))
            flat = serialized[2 * layer]
            self.assertEqual(len(flat), n_in * n_out)
            self.assertEqual(flat, weight.t().contiguous().flatten().tolist())
            # row-major [input, output]: element (i, j) at i * n_out + j
            for i, j in ((0, 0), (n_in - 1, n_out - 1), (1, n_out - 1), (n_in - 1, 1)):
                self.assertEqual(flat[i * n_out + j], weight[j, i].item())
            self.assertEqual(serialized[2 * layer + 1],
                             reference[2 * layer + 1].detach().tolist())
        # nonsquare layers must differ from the untransposed order
        self.assertNotEqual(serialized[0], PREP.parameters(42)[0])

    def test_serialized_layout_reproduces_forward_pass(self):
        (train_ds, _, tr, _, _), _ = data()
        _, model = self.reference_params()
        serialized = PREP.serialized_parameters(42)
        images = torch.stack([train_ds[tr[i]][0] for i in range(4)])
        with torch.no_grad():
            expected = model(images).double()
        for row in range(4):
            activation = images[row].flatten().double().tolist()
            for layer, (n_in, n_out) in enumerate(PREP.LAYER_SHAPES):
                weights, bias = serialized[2 * layer], serialized[2 * layer + 1]
                out = [bias[j] + sum(weights[i * n_out + j] * activation[i]
                                     for i in range(n_in)) for j in range(n_out)]
                activation = out if layer == 2 else [max(v, 0.0) for v in out]
            for got, want in zip(activation, expected[row].tolist()):
                self.assertAlmostEqual(got, want, places=4)

    def test_naive_layout_would_fail_forward_pass(self):
        (train_ds, _, tr, _, _), _ = data()
        _, model = self.reference_params()
        naive = PREP.parameters(42)
        x = train_ds[tr[0]][0].flatten().double().tolist()
        j = 0
        wrong = naive[1][j] + sum(naive[0][i * 128 + j] * x[i] for i in range(784))
        right = model[1](train_ds[tr[0]][0].unsqueeze(0).flatten(1))[0, j].item()
        self.assertGreater(abs(wrong - right), 1e-3)


class SerializationTests(unittest.TestCase):
    def test_write_and_parse_roundtrip(self):
        (train_ds, test_ds, tr, va, te), (_, mnist) = data()
        plan = list(itertools.islice(PREP.input_plan(mnist, va, te), 7))
        plan += [(2, "test", te[0]), (1, "validation", va[0])]
        params = PREP.serialized_parameters(42)
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / f"Input-P{p}-0" for p in range(3)]
            streams = [p.open("w") for p in paths]
            try:
                PREP.write_inputs(streams, params, plan, train_ds, test_ds)
            finally:
                for stream in streams:
                    stream.close()
            tokens = [p.read_text().split() for p in paths]
        flat_params = [v for group in params for v in group]
        self.assertEqual([float(t) for t in tokens[0][:118282]], flat_params)
        cursors = [118282, 0, 0]
        for owner, split, index in plan:
            image, label = (test_ds if split == "test" else train_ds)[index]
            values = [float(t) for t in tokens[owner][cursors[owner]:cursors[owner] + 794]]
            self.assertEqual(values[:784], image.flatten().tolist())
            self.assertEqual(values[784:], [float(i == label) for i in range(10)])
            cursors[owner] += 794
        self.assertEqual(cursors, [len(t) for t in tokens])


def mpc_mirror_step(params, batch, lr):
    """Plain-Python mirror of train_batch() in bp_matched.mpc (serialized layout)."""
    sizes = [784, 128, 128, 10]
    weights = [[params[2 * l][i * sizes[l + 1]:(i + 1) * sizes[l + 1]]
                for i in range(sizes[l])] for l in range(3)]
    biases = [list(params[2 * l + 1]) for l in range(3)]
    gw = [[[0.0] * sizes[l + 1] for _ in range(sizes[l])] for l in range(3)]
    gb = [[0.0] * sizes[l + 1] for l in range(3)]
    loss_total = 0.0
    for x, target in batch:
        acts, pre = [x], []
        for l in range(3):
            z = [biases[l][j] + sum(weights[l][i][j] * acts[l][i]
                                    for i in range(sizes[l])) for j in range(sizes[l + 1])]
            pre.append(z)
            acts.append(z if l == 2 else [v if v > 0 else 0.0 for v in z])
        logits = acts[3]
        shift = max(logits)
        exps = [pow(2.718281828459045, v - shift) for v in logits]
        total = sum(exps)
        probs = [e / total for e in exps]
        import math
        loss_total += -(sum(t * v for t, v in zip(target, logits))
                        - (shift + math.log(total)))
        deltas = [None, None, [p - t for p, t in zip(probs, target)]]
        for l in (1, 0):
            deltas[l] = [sum(deltas[l + 1][k] * weights[l + 1][j][k]
                             for k in range(sizes[l + 2])) * (1.0 if pre[l][j] > 0 else 0.0)
                         for j in range(sizes[l + 1])]
        for l in range(3):
            for j in range(sizes[l + 1]):
                gb[l][j] += deltas[l][j]
                for i in range(sizes[l]):
                    gw[l][i][j] += deltas[l][j] * acts[l][i]
    n = len(batch)
    new = []
    for l in range(3):
        new.append([weights[l][i][j] - lr * gw[l][i][j] / n
                    for i in range(sizes[l]) for j in range(sizes[l + 1])])
        new.append([biases[l][j] - lr * gb[l][j] / n for j in range(sizes[l + 1])])
    return new, loss_total


class MPCAlgorithmMirrorTests(unittest.TestCase):
    def test_mirror_matches_pytorch_sgd_for_full_and_partial_batches(self):
        (train_ds, _, tr, _, _), (_, mnist) = data()
        torch.manual_seed(42)
        model = BP.make_model().double()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
        criterion = torch.nn.CrossEntropyLoss()
        params = [[float(v) for v in group] for group in PREP.serialized_parameters(42)]
        for size, offset in ((6, 0), (3, 6)):  # a "full" and a smaller final batch
            samples = [train_ds[i] for i in mnist[0][offset:offset + size]]
            x = torch.stack([s[0].flatten() for s in samples]).double()
            y = torch.tensor([s[1] for s in samples])
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            batch = [(x[k].tolist(), [float(i == y[k]) for i in range(10)])
                     for k in range(size)]
            params, loss_sum = mpc_mirror_step(params, batch, 0.05)
            self.assertAlmostEqual(loss_sum / size, loss.item(), places=9)
        expected = [model[1].weight.detach().t().flatten().tolist(),
                    model[1].bias.tolist(),
                    model[3].weight.detach().t().flatten().tolist(),
                    model[3].bias.tolist(),
                    model[5].weight.detach().t().flatten().tolist(),
                    model[5].bias.tolist()]
        for got, want in zip(params, expected):
            self.assertLess(max(abs(a - b) for a, b in zip(got, want)), 1e-9)


if __name__ == "__main__":
    unittest.main()
