"""Prepare isolated inputs for the batched-matrix BP prototype."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baselines.bp import datasets_and_indices  # noqa: E402
from prepare_secure_bp_matched import parameters  # noqa: E402
from secure_bp_input import input_hashes, write_values  # noqa: E402

BATCH_SIZE = 3
PARTIES = 3


def transpose_linear(values, rows, columns):
    return [values[row * columns + column]
            for column in range(columns)
            for row in range(rows)]


def matrix_parameters(seed):
    w1, b1, w2, b2, w3, b3 = parameters(seed)
    return (
        transpose_linear(w1, 128, 784), b1,
        transpose_linear(w2, 128, 128), b2,
        transpose_linear(w3, 10, 128), b3,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=42)
    args = parser.parse_args()

    train_ds, _, train_indices, _, _ = datasets_and_indices()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    streams = [(output / f"Input-P{party}-0").open("w")
               for party in range(PARTIES)]
    order = torch.randperm(
        BATCH_SIZE,
        generator=torch.Generator().manual_seed(args.seed)).tolist()
    try:
        for values in matrix_parameters(args.initialization_seed):
            write_values(streams, 0, values)
        for position, source_position in enumerate(order):
            owner = position % PARTIES
            image, label = train_ds[train_indices[source_position]]
            write_values(streams, owner, image.flatten().tolist())
            write_values(streams, owner,
                         [float(index == label) for index in range(10)])
    finally:
        for stream in streams:
            stream.close()

    (output / "metadata.json").write_text(json.dumps({
        "experiment_id": "SECURE-BP-MATRIX-PROTOTYPE-v1",
        "architecture": "784-128-128-10",
        "parameter_count": 118282,
        "batch_size": BATCH_SIZE,
        "utility_seed": args.seed,
        "initialization_seed": args.initialization_seed,
        "weight_order": "[input, output], transposed from PyTorch [output, input]",
        "ownership": "global position modulo three",
        "sha256": input_hashes(output),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
