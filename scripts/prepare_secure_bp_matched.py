"""Generate immutable inputs for SECURE-BP-MATCHED-v1.

The training order reproduces the plaintext baseline's
``DataLoader(Subset(train, tr), batch_size=128, shuffle=True,
generator=torch.Generator().manual_seed(seed), num_workers=0)`` exactly,
including the generator draws PyTorch makes internally per epoch.  Weights are
serialized as ``[input, output]`` row-major to match ``sfix.Matrix(in, out)``.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
sys.path.insert(0, str(ROOT / "src"))
from baselines.bp import datasets_and_indices, make_model  # noqa: E402
from secure_bp_input import input_hashes, write_values  # noqa: E402

EXPERIMENT_ID = "SECURE-BP-MATCHED-v1"
ARCHITECTURE = "784-128-128-10"
PARTIES = 3
BATCH_SIZE = 128
EPOCHS = 5
BATCH_SIZES = [128] * 39 + [8]
LAYER_SHAPES = [(784, 128), (128, 128), (128, 10)]  # (in, out)
ORDER_HASH_RULE = (
    "sha256 over UTF-8 text: each index as a decimal integer followed by a "
    "newline, in training order for one epoch (5000 lines)")


class IndexDataset(Dataset):
    """Index-only dataset: item ``i`` is the integer ``i``."""

    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return index


def parameters(seed):
    """Initial parameters in PyTorch layout (weights flattened [out, in])."""
    torch.manual_seed(seed)
    model = make_model()
    layers = [model[1], model[3], model[5]]
    return ([value for value in layers[0].weight.detach().flatten().tolist()],
            layers[0].bias.detach().tolist(),
            [value for value in layers[1].weight.detach().flatten().tolist()],
            layers[1].bias.detach().tolist(),
            [value for value in layers[2].weight.detach().flatten().tolist()],
            layers[2].bias.detach().tolist())


def transpose_linear(values, rows, columns):
    """Flattened [rows, columns] row-major -> flattened [columns, rows]."""
    return [values[row * columns + column]
            for column in range(columns)
            for row in range(rows)]


def serialized_parameters(seed):
    """Parameters in MPC input order: W1[in,out], b1, W2[in,out], b2, W3[in,out], b3."""
    w1, b1, w2, b2, w3, b3 = parameters(seed)
    return (transpose_linear(w1, 128, 784), b1,
            transpose_linear(w2, 128, 128), b2,
            transpose_linear(w3, 10, 128), b3)


def dataloader_epoch_orders(source, seed, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Per-epoch batches (lists of ints) drawn from ONE DataLoader instance.

    ``source`` is any map-style dataset whose items are plain integers; it is
    iterated with the same arguments as the plaintext baseline loader.
    """
    loader = DataLoader(source, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        num_workers=0)
    return [[[int(value) for value in batch] for batch in loader]
            for _ in range(epochs)]


def training_orders(train_indices, seed, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Return (position_orders, mnist_index_orders), one flat list per epoch.

    Positions index into the 5,000-sample training subset; MNIST indices are
    ``train_indices[position]``.
    """
    n = len(train_indices)
    batches = dataloader_epoch_orders(IndexDataset(n), seed, epochs, batch_size)
    expected_sizes = [min(batch_size, n - start) for start in range(0, n, batch_size)]
    positions = []
    for epoch_batches in batches:
        assert [len(batch) for batch in epoch_batches] == expected_sizes
        flat = [position for batch in epoch_batches for position in batch]
        assert sorted(flat) == list(range(n)), "epoch is not a permutation"
        positions.append(flat)
    mnist = [[train_indices[position] for position in epoch] for epoch in positions]
    return positions, mnist


def baseline_style_orders(train_indices, seed, epochs=EPOCHS, batch_size=BATCH_SIZE,
                          dataset_length=60000):
    """Independent construction mirroring bp.py: DataLoader(Subset(ds, tr), ...).

    The underlying dataset yields its own MNIST index, so the loader returns the
    MNIST index of each drawn sample exactly as the baseline would load it.
    """
    loader = DataLoader(Subset(IndexDataset(dataset_length), train_indices),
                        batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    return [[int(value) for batch in loader for value in batch]
            for _ in range(epochs)]


def order_hash(indices):
    text = "".join(f"{int(index)}\n" for index in indices)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def owner_of(global_position):
    return global_position % PARTIES


def input_plan(mnist_orders, validation_indices, test_indices):
    """Yield (owner, split, mnist_index) in exact MPC consumption order."""
    for order in mnist_orders:
        for global_position, index in enumerate(order):
            yield owner_of(global_position), "train", index
    for split, indices in (("validation", validation_indices), ("test", test_indices)):
        for sample, index in enumerate(indices):
            yield owner_of(sample), split, index


PARAMETER_COUNT = 118282
TOKENS_PER_SAMPLE = 784 + 10


def expected_token_counts(plan):
    """Numeric tokens per party: P0 also holds all model parameters."""
    counts = [PARAMETER_COUNT, 0, 0]
    for owner, _, _ in plan:
        counts[owner] += TOKENS_PER_SAMPLE
    return counts


def write_inputs(streams, params, plan, train_ds, test_ds):
    for values in params:
        write_values(streams, 0, values)
    for owner, split, index in plan:
        image, label = (test_ds if split == "test" else train_ds)[index]
        write_values(streams, owner, image.flatten().tolist())
        write_values(streams, owner, [float(i == label) for i in range(10)])


def assert_matches_baseline(train_indices, positions, mnist_orders, seed):
    reference = baseline_style_orders(train_indices, seed, epochs=len(mnist_orders))
    for epoch, (ours, theirs) in enumerate(zip(mnist_orders, reference), 1):
        assert ours == theirs, f"epoch {epoch} order differs from baseline DataLoader"
        assert len(set(ours)) == len(ours) == len(train_indices)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=42)
    args = parser.parse_args()
    output = args.output_dir.expanduser().resolve()
    if "Player-Data" in output.parts:
        raise SystemExit("Refusing to write inside Player-Data; choose a new run directory")
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"Refusing to use non-empty output directory: {output}")

    train_ds, test_ds, train_indices, validation_indices, test_indices = (
        datasets_and_indices())
    assert (len(train_indices), len(validation_indices), len(test_indices)) == (
        5000, 1000, 1000)
    positions, mnist_orders = training_orders(train_indices, args.seed)
    assert_matches_baseline(train_indices, positions, mnist_orders, args.seed)
    params = serialized_parameters(args.initialization_seed)
    plan = list(input_plan(mnist_orders, validation_indices, test_indices))
    token_counts = expected_token_counts(plan)

    output.mkdir(parents=True, exist_ok=True)
    streams = [(output / f"Input-P{party}-0").open("w") for party in range(PARTIES)]
    try:
        write_inputs(streams, params, plan, train_ds, test_ds)
    finally:
        for stream in streams:
            stream.close()

    hashes = input_hashes(output)
    metadata = {
        "experiment_id": EXPERIMENT_ID,
        "architecture": ARCHITECTURE,
        "parameter_count": 118282,
        "dataset": "MNIST",
        "split_policy": "src.baselines.bp.datasets_and_indices",
        "split_seed": 42,
        "utility_seed": args.seed,
        "initialization_seed": args.initialization_seed,
        "data_order": (
            "One torch.utils.data.DataLoader(index-only dataset of length 5000, "
            "batch_size=128, shuffle=True, generator=torch.Generator().manual_seed("
            "seed), num_workers=0) reused for all five epochs; per epoch PyTorch "
            "consumes the generator for a base-seed draw and the sampler's "
            "randperm(5000) plus an empty-remainder randperm(5000), so state "
            "continues across epochs. Positions are mapped to MNIST indices via "
            "train_indices[position]; verified equal to a baseline-style "
            "DataLoader(Subset(...))."),
        "epoch_order_hash_rule": ORDER_HASH_RULE,
        "epoch_order_mnist_index_sha256": [order_hash(o) for o in mnist_orders],
        "epoch_order_position_sha256": [order_hash(o) for o in positions],
        "weight_layout": "[input, output] row-major (transposed from PyTorch "
                         "[output, input]); biases unchanged",
        "batch_sizes": BATCH_SIZES,
        "epochs": EPOCHS,
        "learning_rate": 0.05,
        "ownership": "global position within epoch modulo three for training; "
                     "sample position modulo three for validation and test",
        "input_token_counts": {f"Input-P{p}-0": token_counts[p] for p in range(PARTIES)},
        "protocol": "Rep3 Ring, semi-honest honest majority, 64-bit ring",
        "fixed_point": "sfix(16,31)",
        "sha256": hashes,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "source_tree_dirty": bool(subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--",
             "scripts", "src", "mpc"], text=True).strip()),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "parameters.json").write_text(
        json.dumps({"architecture": ARCHITECTURE, "seed": args.initialization_seed,
                    "weight_layout": "[input, output] row-major",
                    "parameters": params}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
