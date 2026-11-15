"""Prepare a small full-architecture MPC smoke-test input set."""
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

PARTIES = 3
TRAIN, VALIDATION, TEST = 9, 3, 3
BATCH_SIZES = [3, 3, 3]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initialization-seed", type=int, default=42)
    args = parser.parse_args()
    train_ds, test_ds, train_indices, validation_indices, test_indices = (
        datasets_and_indices())
    params = parameters(args.initialization_seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    streams = [(output / f"Input-P{party}-0").open("w")
               for party in range(PARTIES)]
    selected_train = train_indices[:TRAIN]
    try:
        for values in params:
            write_values(streams, 0, values)
        order_generator = torch.Generator().manual_seed(args.seed)
        order = torch.randperm(TRAIN, generator=order_generator).tolist()
        position = 0
        for batch_size in BATCH_SIZES:
            for local in range(batch_size):
                source_position = order[position + local]
                owner = (position + local) % PARTIES
                image, label = train_ds[selected_train[source_position]]
                write_values(streams, owner, image.flatten().tolist())
                write_values(streams, owner,
                            [float(i == label) for i in range(10)])
            position += batch_size
        for indices, dataset in ((validation_indices[:VALIDATION], train_ds),
                                 (test_indices[:TEST], test_ds)):
            for sample, index in enumerate(indices):
                image, label = dataset[index]
                owner = sample % PARTIES
                write_values(streams, owner, image.flatten().tolist())
                write_values(streams, owner,
                            [float(i == label) for i in range(10)])
    finally:
        for stream in streams:
            stream.close()
    metadata = {
        "experiment_id": "SECURE-BP-MATCHED-DIAGNOSTIC-v1",
        "architecture": "784-128-128-10",
        "parameter_count": 118282,
        "counts": {"train": TRAIN, "validation": VALIDATION, "test": TEST},
        "batch_sizes": BATCH_SIZES,
        "epochs": 1,
        "split_policy": "first records of authoritative plaintext split",
        "split_seed": 42,
        "utility_seed": args.seed,
        "initialization_seed": args.initialization_seed,
        "ownership": "global position modulo three",
        "sha256": input_hashes(output),
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
